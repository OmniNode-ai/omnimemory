#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

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
import logging
import re
import sys
from pathlib import Path
from typing import NamedTuple

# Configure logging - default to WARNING so scripts are quiet unless debugging
logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")


class Violation(NamedTuple):
    """A validation violation."""

    file: str
    line: int
    message: str


# Patterns that suggest backward compatibility hacks
# IMPORTANT: Use word boundaries (\b) to prevent false positives from
# substring matching. For example, @deprecated\b matches "@deprecated"
# but not "@deprecated_feature" or "@my_deprecated"
PATTERNS = [
    (
        # @deprecated decorator - followed by word boundary (end of word, paren)
        # Matches: @deprecated, @Deprecated, @deprecated(...)
        # Does NOT match: @deprecated_feature, @my_deprecated_decorator
        re.compile(r"@deprecated\b", re.IGNORECASE),
        "Deprecated decorator found - remove deprecated code instead",
    ),
    (
        # Backward compatibility comment - requires "compat" word
        # Matches: # backward compat, # backwards compatibility
        # Does NOT match: # backward_compat_layer (variable name)
        re.compile(r"#\s*(backwards?|backward)\s+compat(ibility)?\b", re.IGNORECASE),
        "Backward compatibility comment found - remove old code instead",
    ),
    (
        # Deprecated comment - must be standalone word after #
        # Matches: # deprecated, # DEPRECATED, # deprecated:
        # Does NOT match: # deprecated_field, # not_deprecated, # undeprecated
        re.compile(r"#\s*deprecated\b(?!\s*[_a-z])", re.IGNORECASE),
        "Deprecated comment found - remove deprecated code instead",
    ),
    (
        # Legacy comment - must be standalone word after #
        # Matches: # legacy, # LEGACY, # legacy:, # legacy code
        # Does NOT match: # legacy_helper, # non_legacy, # mylegacy
        re.compile(r"#\s*legacy\b(?!\s*[_a-z])", re.IGNORECASE),
        "Legacy comment found - migrate to new patterns",
    ),
    (
        # TODO to remove deprecated code
        re.compile(r"#\s*TODO:\s*(remove|delete).*\bdeprecated\b", re.IGNORECASE),
        "TODO to remove deprecated code - do it now",
    ),
    (
        # Alias assignment with explicit comment
        # Matches: old_name = new_name  # alias
        # Does NOT match: alias_manager = ... or aliased = ...
        re.compile(r"=\s*\w+\s*#\s*alias\b", re.IGNORECASE),
        "Alias assignment found - avoid maintaining old names",
    ),
]

# Lines to skip (false positives)
SKIP_PATTERNS = [
    re.compile(r"from omnibase_core"),  # Importing from dependencies is fine
    re.compile(
        r"\"\"\".*deprecated.*\"\"\"", re.IGNORECASE
    ),  # Docstrings explaining why something is NOT deprecated
]


def validate_file(filepath: Path) -> list[Violation]:
    """Validate a single Python file."""
    try:
        content = filepath.read_text(encoding="utf-8")
    except PermissionError:
        logging.warning("Permission denied reading file: %s", filepath)
        return []
    except FileNotFoundError:
        logging.warning("File not found (possibly deleted during scan): %s", filepath)
        return []
    except OSError as e:
        logging.warning("OS error reading file %s: %s", filepath, e)
        return []
    except UnicodeDecodeError as e:
        logging.warning("Unicode decode error in file %s: %s", filepath, e)
        return []

    violations: list[Violation] = []

    for line_num, line in enumerate(content.splitlines(), start=1):
        # Skip if line matches skip patterns
        if any(skip.search(line) for skip in SKIP_PATTERNS):
            continue

        for pattern, message in PATTERNS:
            if pattern.search(line):
                violations.append(Violation(str(filepath), line_num, message))
                break  # Only one violation per line

    return violations


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Detect backward compatibility anti-patterns"
    )
    parser.add_argument(
        "-d",
        "--directory",
        default="src/",
        help="Directory to scan",
    )
    args = parser.parse_args()

    directory = Path(args.directory)
    if not directory.is_dir():
        print(f"Directory not found or not a directory: {directory}")
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
