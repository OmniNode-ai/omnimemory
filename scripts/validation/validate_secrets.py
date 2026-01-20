#!/usr/bin/env python3
"""ONEX Secret Detection.

Detects potential hardcoded secrets in Python files.
Catches common patterns like API keys, passwords, tokens.

Usage:
    python scripts/validation/validate_secrets.py [files...]
    python scripts/validation/validate_secrets.py src/
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import NamedTuple


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
        re.compile(r'(?i)(auth[_-]?token|access[_-]?token)\s*=\s*["\'][^"\']{10,}["\']'),
        "Potential hardcoded auth token",
    ),
    (
        re.compile(r'(?i)bearer\s+[a-zA-Z0-9_\-\.]{20,}'),
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
SKIP_PATTERNS = [
    # REASON: Environment variable lookups retrieve values at runtime, not hardcoded
    # These patterns require the actual os module call syntax
    (re.compile(r"os\.environ\["), "Environment variable dict access"),
    (re.compile(r"os\.environ\.get\("), "Environment variable get() access"),
    (re.compile(r"os\.getenv\("), "Environment variable getenv() access"),
    # REASON: Pydantic Field() with default_factory or env= is config, not secrets
    (re.compile(r"Field\s*\([^)]*default_factory"), "Pydantic Field with factory"),
    (re.compile(r"Field\s*\([^)]*env\s*="), "Pydantic Field with env parameter"),
    # REASON: Placeholder values that are clearly not real secrets
    # Use stricter patterns to avoid over-matching legitimate code
    (
        re.compile(r'["\']your[-_](api[-_]?key|secret|password|token)["\']', re.IGNORECASE),
        "Placeholder 'your-*' for common secret types",
    ),
    (re.compile(r'["\']<[a-zA-Z_-]+>["\']'), "Placeholder like '<your-key>'"),
    (
        re.compile(r'["\']x{3,}["\']', re.IGNORECASE),
        "Placeholder string of only x characters",
    ),
    (re.compile(r'["\']CHANGEME["\']'), "Placeholder CHANGEME (exact)"),
    (re.compile(r'["\']REPLACE_ME["\']'), "Placeholder REPLACE_ME (exact)"),
    (re.compile(r'["\']TODO[_-]'), "Placeholder TODO marker"),
    # REASON: Comments that explicitly mark lines as examples
    (re.compile(r"#\s*example", re.IGNORECASE), "Explicit example comment"),
    (re.compile(r"#\s*placeholder", re.IGNORECASE), "Explicit placeholder comment"),
    (re.compile(r"#\s*fake", re.IGNORECASE), "Explicit fake comment"),
    (re.compile(r"#\s*nosec\b", re.IGNORECASE), "Explicit nosec marker"),
    # REASON: Mock/test values should only skip in test files - handled below
]

# Additional patterns that only skip in test files
# SECURITY NOTE: These use word boundaries (\b) to prevent over-matching.
# E.g., "mock" should match "mock_password" but not "hammock_key".
TEST_ONLY_SKIP_PATTERNS = [
    (re.compile(r"\bmock[_-]", re.IGNORECASE), "Mock value prefix in test file"),
    (re.compile(r"[_-]mock\b", re.IGNORECASE), "Mock value suffix in test file"),
    (re.compile(r"\bfake[_-]", re.IGNORECASE), "Fake value prefix in test file"),
    (re.compile(r"[_-]fake\b", re.IGNORECASE), "Fake value suffix in test file"),
    (re.compile(r'=\s*["\']mock["\']', re.IGNORECASE), "Literal 'mock' assignment"),
    (re.compile(r'=\s*["\']fake["\']', re.IGNORECASE), "Literal 'fake' assignment"),
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


def validate_file(filepath: Path) -> list[Violation]:
    """Validate a single Python file."""
    try:
        content = filepath.read_text(encoding="utf-8")
    except Exception:
        return []

    violations: list[Violation] = []
    is_test = is_test_file(filepath)

    for line_num, line in enumerate(content.splitlines(), start=1):
        # Skip if line matches general skip patterns (tuple format: pattern, reason)
        if any(pattern.search(line) for pattern, _reason in SKIP_PATTERNS):
            continue

        # Skip test-only patterns if in test file
        if is_test and any(
            pattern.search(line) for pattern, _reason in TEST_ONLY_SKIP_PATTERNS
        ):
            continue

        for pattern, message in SECRET_PATTERNS:
            if pattern.search(line):
                violations.append(Violation(str(filepath), line_num, message))
                break  # Only one violation per line

    return violations


def main() -> int:
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: validate_secrets.py [files or directories...]")
        return 1

    files_to_check: list[Path] = []

    for arg in sys.argv[1:]:
        path = Path(arg)
        if path.is_file() and path.suffix == ".py":
            files_to_check.append(path)
        elif path.is_dir():
            files_to_check.extend(path.rglob("*.py"))

    all_violations: list[Violation] = []
    for filepath in files_to_check:
        violations = validate_file(filepath)
        all_violations.extend(violations)

    if all_violations:
        print(f"Found {len(all_violations)} potential secret(s):")
        for v in all_violations:
            print(f"  {v.file}:{v.line}: {v.message}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
