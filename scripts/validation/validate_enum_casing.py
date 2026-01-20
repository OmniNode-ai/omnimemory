#!/usr/bin/env python3
"""ONEX Enum Member Casing Validation.

Enforces UPPER_SNAKE_CASE for all enum member names.
This is an ONEX architectural standard for consistency.

Usage:
    python scripts/validation/validate_enum_casing.py [files...]
    python scripts/validation/validate_enum_casing.py src/omnimemory/enums/
"""

from __future__ import annotations

import ast
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


UPPER_SNAKE_CASE = re.compile(r"^[A-Z][A-Z0-9]*(_[A-Z0-9]+)*$")


class EnumCasingVisitor(ast.NodeVisitor):
    """AST visitor to check enum member casing."""

    def __init__(self, filepath: str) -> None:
        self.filepath = filepath
        self.violations: list[Violation] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Visit class definition to find enums."""
        # Check if this is an Enum subclass
        # Include all standard enum types: Enum, StrEnum, IntEnum, IntFlag, Flag
        enum_types = ("Enum", "StrEnum", "IntEnum", "IntFlag", "Flag")
        is_enum = any(
            (isinstance(base, ast.Name) and base.id in enum_types)
            or (isinstance(base, ast.Attribute) and base.attr in enum_types)
            for base in node.bases
        )

        if is_enum:
            for item in node.body:
                # Handle regular assignment: MEMBER = "value"
                if isinstance(item, ast.Assign):
                    for target in item.targets:
                        if isinstance(target, ast.Name):
                            name = target.id
                            # Skip dunder and private attrs
                            if name.startswith("_"):
                                continue
                            if not UPPER_SNAKE_CASE.match(name):
                                self.violations.append(
                                    Violation(
                                        self.filepath,
                                        item.lineno,
                                        f"Enum member '{name}' in {node.name} "
                                        "should be UPPER_SNAKE_CASE",
                                    )
                                )
                # Handle annotated assignment: MEMBER: str = "value"
                elif isinstance(item, ast.AnnAssign):
                    if isinstance(item.target, ast.Name):
                        name = item.target.id
                        # Skip dunder and private attrs
                        if name.startswith("_"):
                            continue
                        if not UPPER_SNAKE_CASE.match(name):
                            self.violations.append(
                                Violation(
                                    self.filepath,
                                    item.lineno,
                                    f"Enum member '{name}' in {node.name} "
                                    "should be UPPER_SNAKE_CASE",
                                )
                            )

        self.generic_visit(node)


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

    try:
        tree = ast.parse(content, filename=str(filepath))
    except SyntaxError as e:
        logging.debug("Skipping file with syntax error: %s (%s)", filepath, e)
        return []

    visitor = EnumCasingVisitor(str(filepath))
    visitor.visit(tree)
    return visitor.violations


def main() -> int:
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: validate_enum_casing.py [files or directories...]")
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
        print(f"Found {len(all_violations)} enum casing violation(s):")
        for v in all_violations:
            print(f"  {v.file}:{v.line}: {v.message}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
