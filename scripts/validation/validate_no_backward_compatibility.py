#!/usr/bin/env python3
"""ONEX No Backward Compatibility Anti-Pattern Detection.

Detects patterns that suggest backward compatibility hacks:
- Deprecated decorators
- Aliases for old names
- "# deprecated" or "# backwards compatibility" comments
- Re-exports of old names

Usage:
    python scripts/validation/validate_no_backward_compatibility.py -d src/
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import NamedTuple


class Violation(NamedTuple):
    """A validation violation."""

    file: str
    line: int
    message: str


# Patterns that suggest backward compatibility hacks
PATTERNS = [
    (
        re.compile(r"@deprecated", re.IGNORECASE),
        "Deprecated decorator found - remove deprecated code instead",
    ),
    (
        re.compile(r"#\s*(backwards?|backward)\s*compat", re.IGNORECASE),
        "Backward compatibility comment found - remove old code instead",
    ),
    (
        re.compile(r"#\s*deprecated", re.IGNORECASE),
        "Deprecated comment found - remove deprecated code instead",
    ),
    (
        re.compile(r"#\s*legacy", re.IGNORECASE),
        "Legacy comment found - migrate to new patterns",
    ),
    (
        re.compile(r"#\s*TODO:\s*(remove|delete).*deprecated", re.IGNORECASE),
        "TODO to remove deprecated code - do it now",
    ),
    (
        re.compile(r"=\s*\w+\s*#\s*alias", re.IGNORECASE),
        "Alias assignment found - avoid maintaining old names",
    ),
]

# Lines to skip (false positives)
SKIP_PATTERNS = [
    re.compile(r"from omnibase_core"),  # Importing from dependencies is fine
    re.compile(r"\"\"\".*deprecated.*\"\"\"", re.IGNORECASE),  # Docstrings explaining why something is NOT deprecated
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

        for pattern, message in PATTERNS:
            if pattern.search(line):
                violations.append(
                    Violation(str(filepath), line_num, message)
                )
                break  # Only one violation per line

    return violations


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Detect backward compatibility anti-patterns"
    )
    parser.add_argument(
        "-d", "--directory",
        default="src/",
        help="Directory to scan",
    )
    args = parser.parse_args()

    directory = Path(args.directory)
    if not directory.exists():
        print(f"Directory not found: {directory}")
        return 1

    files_to_check = list(directory.rglob("*.py"))

    all_violations: list[Violation] = []
    for filepath in files_to_check:
        violations = validate_file(filepath)
        all_violations.extend(violations)

    if all_violations:
        print(f"Found {len(all_violations)} backward compatibility anti-pattern(s):")
        for v in all_violations:
            print(f"  {v.file}:{v.line}: {v.message}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
