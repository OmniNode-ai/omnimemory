#!/usr/bin/env python3
"""ONEX Secret Detection.

Detects potential hardcoded secrets in Python files.
Catches common patterns like API keys, passwords, tokens.

Usage:
    python scripts/validation/validate_secrets.py [files...]
    python scripts/validation/validate_secrets.py src/
    python scripts/validation/validate_secrets.py --verbose src/  # Log skipped patterns
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path
from typing import NamedTuple

# Configure logging for skip pattern visibility
logger = logging.getLogger(__name__)


class Violation(NamedTuple):
    """A validation violation."""

    file: str
    line: int
    message: str


# Patterns that suggest hardcoded secrets
SECRET_PATTERNS = [
    (
        re.compile(r'(?i)(api[_-]?key|apikey)\s*=\s*["\'][^"\']{10,}["\']'),
        "Potential hardcoded API key",
    ),
    (
        re.compile(r'(?i)(secret[_-]?key|secretkey)\s*=\s*["\'][^"\']{10,}["\']'),
        "Potential hardcoded secret key",
    ),
    (
        re.compile(r'(?i)password\s*=\s*["\'][^"\']{4,}["\']'),
        "Potential hardcoded password",
    ),
    (
        re.compile(
            r'(?i)(auth[_-]?token|access[_-]?token)\s*=\s*["\'][^"\']{10,}["\']'
        ),
        "Potential hardcoded auth token",
    ),
    (
        re.compile(r"(?i)bearer\s+[a-zA-Z0-9_\-\.]{20,}"),
        "Potential hardcoded bearer token",
    ),
    (
        re.compile(r'(?i)private[_-]?key\s*=\s*["\'][^"\']{20,}["\']'),
        "Potential hardcoded private key",
    ),
]

# Lines to skip (false positives)
# Each pattern must have a clear, documented reason for exclusion.
# Be conservative: prefer false positives over missing real secrets.
#
# SECURITY NOTE: Skip patterns must be as PRECISE as possible.
# Over-broad patterns can hide real secrets in unexpected locations.
# Use exact path matching and word boundaries where possible.
#
# PATTERN DESIGN RULES:
# 1. Use word boundaries (\b) to prevent substring matching
# 2. Match exact function call syntax, not partial strings
# 3. Require specific markers (like nosec) rather than generic words
# 4. Document the exact scenario each pattern catches
SKIP_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # -------------------------------------------------------------------------
    # CATEGORY: Runtime environment variable access
    # SAFE BECAUSE: Values come from environment at runtime, not hardcoded
    # PATTERN PRECISION: Requires exact os.environ/os.getenv call syntax
    # -------------------------------------------------------------------------
    (
        re.compile(r"\bos\.environ\s*\["),
        "os.environ[] dict access - value from runtime environment",
    ),
    (
        re.compile(r"\bos\.environ\.get\s*\("),
        "os.environ.get() call - value from runtime environment",
    ),
    (
        re.compile(r"\bos\.getenv\s*\("),
        "os.getenv() call - value from runtime environment",
    ),
    # -------------------------------------------------------------------------
    # CATEGORY: Pydantic configuration patterns
    # SAFE BECAUSE: These use factory functions or env vars, not literals
    # PATTERN PRECISION: Requires Field() call with specific parameters
    # -------------------------------------------------------------------------
    (
        re.compile(r"\bField\s*\([^)]*\bdefault_factory\s*="),
        "Pydantic Field with default_factory - value generated at runtime",
    ),
    (
        re.compile(r"\bField\s*\([^)]*\benv\s*="),
        "Pydantic Field with env= parameter - value from environment",
    ),
    # -------------------------------------------------------------------------
    # CATEGORY: Explicit placeholder strings
    # SAFE BECAUSE: These are clearly marked as placeholders to replace
    # PATTERN PRECISION: Matches exact placeholder conventions only
    # -------------------------------------------------------------------------
    (
        # Matches: "your-api-key", "your_secret", "your-password", "your_token"
        # Does NOT match: "yourname", "your_data", "your-file"
        re.compile(
            r'["\']your[-_]?(api[-_]?key|secret[-_]?key|password|token|credential)["\']',
            re.IGNORECASE,
        ),
        "Explicit placeholder: 'your-{secret-type}'",
    ),
    (
        # Matches: "<API_KEY>", "<your-secret>", "<INSERT_TOKEN>"
        # Does NOT match: "<html>", "<div>", "<span>" (HTML tags)
        # Requires the placeholder to contain secret-related words
        re.compile(
            r'["\']<[a-zA-Z_-]*(key|secret|password|token|credential)[a-zA-Z_-]*>["\']',
            re.IGNORECASE,
        ),
        "Explicit placeholder: '<...-key/secret/password/token-...>'",
    ),
    (
        # Matches only strings that are entirely placeholder x's
        # Pattern: "xxxx", "XXXX", "xxxxxxxx" (3+ x characters, nothing else)
        re.compile(r'["\']x{3,}["\']', re.IGNORECASE),
        "Explicit placeholder: string of only 'x' characters",
    ),
    (
        # Exact match for common placeholder values
        re.compile(r'["\']CHANGEME["\']'),
        "Explicit placeholder: 'CHANGEME' (exact match)",
    ),
    (
        re.compile(r'["\']REPLACE_ME["\']'),
        "Explicit placeholder: 'REPLACE_ME' (exact match)",
    ),
    (
        # Matches: "TODO_ADD_REAL_KEY", "TODO-replace-secret"
        # Requires TODO followed by separator, not just containing TODO
        re.compile(r'["\']TODO[-_][A-Z_-]+["\']', re.IGNORECASE),
        "Explicit placeholder: 'TODO_...' or 'TODO-...'",
    ),
    # -------------------------------------------------------------------------
    # CATEGORY: Inline security annotations
    # SAFE BECAUSE: Developer explicitly marked as non-secret
    # PATTERN PRECISION: Requires specific security annotation syntax
    # -------------------------------------------------------------------------
    (
        # Standard security tool annotation (bandit, semgrep, etc.)
        # Requires: # nosec at word boundary (not "nosecret" or similar)
        re.compile(r"#\s*nosec\b", re.IGNORECASE),
        "Security annotation: # nosec (explicit security tool marker)",
    ),
    (
        # Explicit "not a secret" or "not a real" annotation
        # More specific than just "fake" which could appear in other contexts
        re.compile(
            r"#.*\bnot\s+a\s+(real\s+)?(secret|password|key|token)\b", re.IGNORECASE
        ),
        "Security annotation: # not a (real) secret/password/key/token",
    ),
    (
        # Explicit example/placeholder documentation in comments
        # Requires "example" followed by secret-related word to avoid false matches
        re.compile(
            r"#.*\b(example|placeholder|dummy|sample)\s+(api[-_]?key|secret|password|token|credential)\b",
            re.IGNORECASE,
        ),
        "Documentation: # example/placeholder/dummy/sample {secret-type}",
    ),
]

# Additional patterns that only skip in test files
# SECURITY NOTE: These patterns are ONLY applied to files identified as tests.
# This prevents test fixtures from accidentally allowing secrets in prod code.
#
# Test file detection: test_*.py, *_test.py, or in tests/ directory
TEST_ONLY_SKIP_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # -------------------------------------------------------------------------
    # CATEGORY: Mock/fake values in test fixtures
    # SAFE BECAUSE: Test files intentionally use fake credentials
    # PATTERN PRECISION: Uses word boundaries and requires secret-related context
    # -------------------------------------------------------------------------
    (
        # Matches: mock_api_key, mock_password, mock_secret_key
        # Does NOT match: hammock_key (no word boundary), mock_data (not a secret)
        re.compile(
            r"\bmock[-_]?(api[-_]?key|secret[-_]?key|password|token|credential)\b",
            re.IGNORECASE,
        ),
        "Test fixture: mock_{secret-type}",
    ),
    (
        # Matches: api_key_mock, password_mock, secret_mock
        re.compile(
            r"\b(api[-_]?key|secret[-_]?key|password|token|credential)[-_]?mock\b",
            re.IGNORECASE,
        ),
        "Test fixture: {secret-type}_mock",
    ),
    (
        # Matches: fake_api_key, fake_password, fake_secret
        re.compile(
            r"\bfake[-_]?(api[-_]?key|secret[-_]?key|password|token|credential)\b",
            re.IGNORECASE,
        ),
        "Test fixture: fake_{secret-type}",
    ),
    (
        # Matches: api_key_fake, password_fake, token_fake
        re.compile(
            r"\b(api[-_]?key|secret[-_]?key|password|token|credential)[-_]?fake\b",
            re.IGNORECASE,
        ),
        "Test fixture: {secret-type}_fake",
    ),
    (
        # Matches: = "test_secret_value", = 'test_api_key_12345'
        # Requires "test_" prefix in the string value itself
        re.compile(r'=\s*["\']test[-_][a-zA-Z0-9_-]+["\']', re.IGNORECASE),
        "Test fixture: literal 'test_...' value assignment",
    ),
    (
        # Matches pytest fixtures and unittest.mock patterns
        re.compile(r"@pytest\.fixture|@mock\.patch|MagicMock|Mock\(", re.IGNORECASE),
        "Test framework: pytest fixture or mock decorator/class",
    ),
]


def is_test_file(filepath: Path) -> bool:
    """Check if a file is a test file based on path or name."""
    name = filepath.name
    path_str = str(filepath)
    return (
        name.startswith("test_")
        or name.endswith("_test.py")
        or "/tests/" in path_str
        or "\\tests\\" in path_str
        or "/test/" in path_str
        or "\\test\\" in path_str
    )


def _check_skip_patterns(
    line: str,
    patterns: list[tuple[re.Pattern[str], str]],
    filepath: Path,
    line_num: int,
    verbose: bool,
) -> tuple[bool, str | None]:
    """Check if a line matches any skip pattern.

    Returns:
        Tuple of (should_skip, reason) where reason is None if not skipped.
    """
    for pattern, reason in patterns:
        if pattern.search(line):
            if verbose:
                logger.debug(
                    "SKIP: %s:%d matched pattern '%s' - %s",
                    filepath,
                    line_num,
                    pattern.pattern[:50],
                    reason,
                )
            return True, reason
    return False, None


def validate_file(filepath: Path, verbose: bool = False) -> list[Violation]:
    """Validate a single Python file.

    Args:
        filepath: Path to the Python file to validate.
        verbose: If True, log skipped patterns for visibility.

    Returns:
        List of Violation objects for detected potential secrets.
    """
    try:
        content = filepath.read_text(encoding="utf-8")
    except Exception as e:
        if verbose:
            logger.warning("Could not read file %s: %s", filepath, e)
        return []

    violations: list[Violation] = []
    is_test = is_test_file(filepath)

    for line_num, line in enumerate(content.splitlines(), start=1):
        # First, check if line potentially contains a secret
        potential_secret = None
        for pattern, message in SECRET_PATTERNS:
            if pattern.search(line):
                potential_secret = message
                break

        # If no potential secret, skip further analysis
        if potential_secret is None:
            continue

        # Check if line matches general skip patterns
        skipped, reason = _check_skip_patterns(
            line, SKIP_PATTERNS, filepath, line_num, verbose
        )
        if skipped:
            if verbose:
                logger.info(
                    "SKIP (general): %s:%d - Would have flagged '%s' but matched: %s",
                    filepath,
                    line_num,
                    potential_secret,
                    reason,
                )
            continue

        # Check test-only patterns if in test file
        if is_test:
            skipped, reason = _check_skip_patterns(
                line, TEST_ONLY_SKIP_PATTERNS, filepath, line_num, verbose
            )
            if skipped:
                if verbose:
                    logger.info(
                        "SKIP (test-only): %s:%d - Would have flagged '%s' but matched: %s",
                        filepath,
                        line_num,
                        potential_secret,
                        reason,
                    )
                continue

        # Line contains potential secret and wasn't skipped
        violations.append(Violation(str(filepath), line_num, potential_secret))

    return violations


def main() -> int:
    """Main entry point."""
    # Parse arguments
    args = sys.argv[1:]

    # Handle --verbose flag
    verbose = False
    if "--verbose" in args:
        verbose = True
        args.remove("--verbose")
        # Configure logging for verbose output
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(levelname)s: %(message)s",
        )
    elif "-v" in args:
        verbose = True
        args.remove("-v")
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(levelname)s: %(message)s",
        )

    if not args:
        print("Usage: validate_secrets.py [--verbose|-v] [files or directories...]")
        print()
        print("Options:")
        print("  --verbose, -v  Log skipped patterns for debugging false negatives")
        print()
        print("Examples:")
        print("  validate_secrets.py src/")
        print("  validate_secrets.py --verbose src/myfile.py")
        return 1

    files_to_check: list[Path] = []

    for arg in args:
        path = Path(arg)
        if path.is_file() and path.suffix == ".py":
            files_to_check.append(path)
        elif path.is_dir():
            files_to_check.extend(path.rglob("*.py"))

    if verbose:
        print(f"Scanning {len(files_to_check)} Python file(s)...")
        print(
            f"Skip patterns: {len(SKIP_PATTERNS)} general, {len(TEST_ONLY_SKIP_PATTERNS)} test-only"
        )
        print()

    all_violations: list[Violation] = []

    for filepath in files_to_check:
        violations = validate_file(filepath, verbose=verbose)
        all_violations.extend(violations)

    if all_violations:
        print(f"Found {len(all_violations)} potential secret(s):")
        for v in all_violations:
            print(f"  {v.file}:{v.line}: {v.message}")
        return 1

    if verbose:
        print("No potential secrets detected.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
