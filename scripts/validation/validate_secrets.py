#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

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
    # -------------------------------------------------------------------------
    # CATEGORY: Direct assignment patterns (var = "secret")
    # -------------------------------------------------------------------------
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
    # -------------------------------------------------------------------------
    # CATEGORY: Hardcoded secrets in os.getenv/os.environ.get defaults
    # These catch os.getenv/os.environ.get with hardcoded string defaults
    # Note: The skip patterns only match SAFE uses (no literal default)
    # -------------------------------------------------------------------------
    (
        # Catches getenv/environ.get with literal string defaults (10+ chars)
        # Matches any os.getenv/os.environ.get with a string literal default 10+ chars
        re.compile(
            r'(?i)(?:os\.getenv|os\.environ\.get)\s*\(\s*["\'][^"\']*'
            r"(?:API[_-]?KEY|SECRET[_-]?KEY|PASSWORD|TOKEN|CREDENTIAL|PRIVATE[_-]?KEY)"
            r'[^"\']*["\']\s*,\s*["\'][^"\']{10,}["\']'
        ),
        "Potential hardcoded secret in env var default",
    ),
    (
        # Catches any getenv with a suspicious-looking default (sk-, secret-, etc.)
        re.compile(
            r'(?i)(?:os\.getenv|os\.environ\.get)\s*\(\s*["\'][^"\']*["\']\s*,\s*'
            r'["\'](?:sk-|secret-|password-|key-|token-|api-)[^"\']{8,}["\']'
        ),
        "Potential hardcoded secret prefix in env var default",
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
    # CATEGORY: Runtime environment variable access (STRICT safe patterns only)
    # SAFE BECAUSE: Values come from environment at runtime, not hardcoded
    #
    # SECURITY NOTE: These patterns are INTENTIONALLY NARROW.
    # We only skip the most obviously safe patterns:
    #   1. os.environ["VAR"] - dict access, no default possible
    #   2. os.getenv("VAR") - single arg, no default
    #   3. os.environ.get("VAR") - single arg, no default
    #   4. os.getenv("VAR", None) - explicit None default
    #   5. os.getenv("VAR", "") - empty string default
    #
    # We DO NOT skip calls with:
    #   - Variable references as defaults (could hide secrets in variables)
    #   - Function calls as defaults (could return secrets)
    #   - Any non-empty string literal (caught by SECRET_PATTERNS)
    #
    # UNSAFE examples (NOT matched, will be checked by SECRET_PATTERNS):
    #   api_key = os.getenv("API_KEY", "sk-xxx...")  # literal secret
    #   api_key = os.getenv("API_KEY", default_var)  # variable could be secret
    #   api_key = os.getenv("API_KEY", get_key())    # function could return secret
    # -------------------------------------------------------------------------
    (
        # os.environ[] dict access - no default possible, always safe
        re.compile(r"\bos\.environ\s*\["),
        "os.environ[] dict access - value from runtime environment",
    ),
    (
        # os.getenv() with NO default argument at all
        # Only matches: os.getenv("VAR") with closing paren immediately after
        re.compile(r"\bos\.getenv\s*\(\s*[\"'][^\"']*[\"']\s*\)"),
        "os.getenv() with no default argument",
    ),
    (
        # os.environ.get() with NO default argument at all
        # Only matches: os.environ.get("VAR") with closing paren immediately after
        re.compile(r"\bos\.environ\.get\s*\(\s*[\"'][^\"']*[\"']\s*\)"),
        "os.environ.get() with no default argument",
    ),
    (
        # os.getenv() or os.environ.get() with explicit None default
        # Only matches: os.getenv("VAR", None) - the word None, not a variable
        re.compile(
            r"\b(?:os\.getenv|os\.environ\.get)\s*\(\s*[\"'][^\"']*[\"']\s*,\s*None\s*\)"
        ),
        "os.getenv/environ.get() with explicit None default",
    ),
    (
        # os.getenv() or os.environ.get() with EMPTY string default - safe
        # Only matches: os.getenv("VAR", "") or os.getenv("VAR", '')
        re.compile(
            r"\b(?:os\.getenv|os\.environ\.get)\s*\(\s*[\"'][^\"']*[\"']\s*,\s*(?:\"\"|'')\s*\)"
        ),
        "os.getenv/environ.get() with empty string default",
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
]

# -------------------------------------------------------------------------
# LINE-LEVEL SKIP PATTERNS - These suppress the ENTIRE line regardless of position
# -------------------------------------------------------------------------
# These patterns are checked separately and don't require overlap with the secret match.
# Use these for inline annotations that apply to the whole line (like # nosec).
#
# SECURITY NOTE: Be conservative with these patterns. They suppress ALL detections
# on a line, so they should only be used for explicit security annotations.
LINE_LEVEL_SKIP_PATTERNS: list[tuple[re.Pattern[str], str]] = [
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

# NOTE: Test files are skipped entirely at file level in validate_file().
# No line-level TEST_ONLY_SKIP_PATTERNS are needed - this is more reliable
# and prevents any possibility of test fixture patterns masking real secrets
# in production code that happens to use test-like naming.


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


def _check_line_level_skip_patterns(
    line: str,
    filepath: Path,
    line_num: int,
    verbose: bool,
) -> tuple[bool, str | None]:
    """Check if a line matches any line-level skip pattern (no overlap required).

    These patterns suppress the ENTIRE line regardless of where on the line they match.
    Used for explicit security annotations like # nosec that apply to the whole line.

    Args:
        line: The line of code to check.
        filepath: Path to the file (for logging).
        line_num: Line number (for logging).
        verbose: Whether to log skip decisions.

    Returns:
        Tuple of (should_skip, reason) where reason is None if not skipped.
    """
    for pattern, reason in LINE_LEVEL_SKIP_PATTERNS:
        if pattern.search(line):
            if verbose:
                logger.debug(
                    "SKIP (line-level): %s:%d matched pattern '%s' - %s",
                    filepath,
                    line_num,
                    pattern.pattern[:50],
                    reason,
                )
            return True, reason
    return False, None


def _check_skip_patterns(  # stub-ok: docstring uses 'xxxx' as example secret string, not a FIXME marker
    line: str,
    patterns: list[tuple[re.Pattern[str], str]],
    filepath: Path,
    line_num: int,
    verbose: bool,
    secret_match: re.Match[str] | None = None,
) -> tuple[bool, str | None]:
    """Check if a line matches any skip pattern that covers the detected secret.

    SECURITY: Only skip if the skip pattern match OVERLAPS with the secret match.
    This prevents safe patterns (like os.getenv("VAR")) from masking secrets
    elsewhere on the same line (like "; password = 'xxxx'").

    Args:
        line: The line of code to check.
        patterns: List of (pattern, reason) tuples to check against.
        filepath: Path to the file (for logging).
        line_num: Line number (for logging).
        verbose: Whether to log skip decisions.
        secret_match: The regex match object for the detected secret.
                     If provided, skip patterns must overlap with this match.

    Returns:
        Tuple of (should_skip, reason) where reason is None if not skipped.
    """
    for pattern, reason in patterns:
        skip_match = pattern.search(line)
        if skip_match:
            # If we have a secret_match, verify the skip pattern overlaps with it
            # This prevents safe patterns from masking unrelated secrets on same line
            if secret_match is not None:
                # Check if the skip pattern match overlaps with secret match
                skip_start, skip_end = skip_match.span()
                secret_start, secret_end = secret_match.span()

                # Overlap exists if: NOT (skip ends before secret starts OR
                #                        skip starts after secret ends)
                overlaps = not (skip_end <= secret_start or skip_start >= secret_end)

                if not overlaps:
                    if verbose:
                        logger.debug(
                            "SKIP REJECTED: %s:%d - skip pattern at [%d:%d] does not "
                            "overlap with secret at [%d:%d]",
                            filepath,
                            line_num,
                            skip_start,
                            skip_end,
                            secret_start,
                            secret_end,
                        )
                    continue  # Skip pattern doesn't cover the secret, try next pattern

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
    # Skip test files entirely at the file level
    # This is more reliable than line-level pattern matching for test fixtures
    # Uses is_test_file() for consistent detection of all test file patterns
    if is_test_file(filepath):
        if verbose:
            logger.debug("SKIP FILE: %s - test file skipped at file level", filepath)
        return []

    try:
        content = filepath.read_text(encoding="utf-8")
    except PermissionError:
        logger.warning("Permission denied reading file: %s", filepath)
        return []
    except FileNotFoundError:
        logger.warning("File not found (possibly deleted during scan): %s", filepath)
        return []
    except OSError as e:
        logger.warning("OS error reading file %s: %s", filepath, e)
        return []
    except UnicodeDecodeError as e:
        logger.warning("Unicode decode error in file %s: %s", filepath, e)
        return []

    violations: list[Violation] = []

    for line_num, line in enumerate(content.splitlines(), start=1):
        # First, check if line potentially contains a secret
        potential_secret = None
        secret_match: re.Match[str] | None = None
        for pattern, message in SECRET_PATTERNS:
            match = pattern.search(line)
            if match:
                potential_secret = message
                secret_match = match
                break

        # If no potential secret, skip further analysis
        if potential_secret is None:
            continue

        # First, check line-level skip patterns (no overlap required)
        # These are explicit security annotations like # nosec
        skipped, reason = _check_line_level_skip_patterns(
            line, filepath, line_num, verbose
        )
        if skipped:
            if verbose:
                logger.info(
                    "SKIP (line-level): %s:%d - Flagged '%s' but matched: %s",
                    filepath,
                    line_num,
                    potential_secret,
                    reason,
                )
            continue

        # Check if line matches general skip patterns (overlap required)
        # SECURITY: Pass the secret_match to ensure skip patterns only apply
        # if they overlap with the detected secret, not just anywhere on the line
        skipped, reason = _check_skip_patterns(
            line, SKIP_PATTERNS, filepath, line_num, verbose, secret_match
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
            f"Skip patterns: {len(SKIP_PATTERNS)} overlap-based, "
            f"{len(LINE_LEVEL_SKIP_PATTERNS)} line-level "
            "(test files skipped at file level)"
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
