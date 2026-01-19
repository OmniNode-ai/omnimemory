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
SKIP_PATTERNS = [
    re.compile(r"os\.environ"),  # Environment variable access
    re.compile(r"os\.getenv"),  # Environment variable access
    re.compile(r"\.get\s*\("),  # Dict/config gets
    re.compile(r"Field\s*\("),  # Pydantic Field definitions
    re.compile(r"#.*example"),  # Example comments
    re.compile(r'["\']your[_-]'),  # Placeholder values
    re.compile(r'["\']<'),  # Placeholder values like <your-key>
    re.compile(r"test_"),  # Test files/functions
    re.compile(r"mock"),  # Mock values
]


def validate_file(filepath: Path) -> list[Violation]:
    """Validate a single Python file."""
    try:
        content = filepath.read_text(encoding="utf-8")
    except Exception:
        return []

    violations: list[Violation] = []

    for line_num, line in enumerate(content.splitlines(), start=1):
        # Skip if line matches skip patterns
        if any(skip.search(line) for skip in SKIP_PATTERNS):
            continue

        for pattern, message in SECRET_PATTERNS:
            if pattern.search(line):
                violations.append(
                    Violation(str(filepath), line_num, message)
                )
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
