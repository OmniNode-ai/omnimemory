#!/usr/bin/env python3
"""ONEX Single Class Per File Validation.

Enforces the ONEX architectural rule: one model/enum per file.
This promotes clean imports and reduces circular dependency issues.

Usage:
    python scripts/validation/validate_single_class_per_file.py [files...]
    python scripts/validation/validate_single_class_per_file.py src/omnimemory/models/
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import NamedTuple


class Violation(NamedTuple):
    """A validation violation."""

    file: str
    line: int
    message: str


# Files that are allowed to have multiple classes
EXEMPTIONS = {
    "__init__.py",  # Package exports
    "base.py",  # Base classes often group related items
    "base_protocols.py",  # Protocols often group related items
    "data_models.py",  # Data models often group related items
    "error_models.py",  # Error models often group related items
}

# Directories where multiple classes per file is allowed (foundation types)
EXEMPT_DIRECTORIES = {
    "core",  # Core models often contain related types
    "foundation",  # Foundation models often contain related types
    "protocols",  # Protocols often contain related types
    "compat",  # Compatibility modules
}


def is_enum_class(node: ast.ClassDef) -> bool:
    """Check if a class definition is an Enum subclass."""
    enum_bases = {"Enum", "StrEnum", "IntEnum", "IntFlag", "Flag", "auto"}
    for base in node.bases:
        if isinstance(base, ast.Name) and base.id in enum_bases:
            return True
        if isinstance(base, ast.Attribute) and base.attr in enum_bases:
            return True
    return False


def count_classes(filepath: Path) -> tuple[int, list[str], int, list[str]]:
    """Count top-level class definitions in a file.

    Returns:
        Tuple of (non_enum_count, non_enum_names, enum_count, enum_names)
    """
    try:
        content = filepath.read_text(encoding="utf-8")
        tree = ast.parse(content, filename=str(filepath))
    except SyntaxError:
        return 0, [], 0, []
    except Exception:
        return 0, [], 0, []

    non_enum_names: list[str] = []
    enum_names: list[str] = []

    # Only check top-level classes
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            if is_enum_class(node):
                enum_names.append(node.name)
            else:
                non_enum_names.append(node.name)

    return len(non_enum_names), non_enum_names, len(enum_names), enum_names


def validate_file(filepath: Path) -> list[Violation]:
    """Validate a single Python file.

    Rules:
    - Only one non-enum class per file (enforced)
    - Multiple enums in one file are allowed
    """
    if filepath.name in EXEMPTIONS:
        return []

    # Check if file is in an exempt directory
    for part in filepath.parts:
        if part in EXEMPT_DIRECTORIES:
            return []

    non_enum_count, non_enum_names, enum_count, enum_names = count_classes(filepath)

    # Only enforce single-class rule for non-enum classes
    # Multiple enums in one file are explicitly allowed
    if non_enum_count > 1:
        return [
            Violation(
                str(filepath),
                1,
                f"Multiple non-enum classes in file ({non_enum_count}): "
                f"{', '.join(non_enum_names)}. ONEX requires one model per file. "
                f"(Note: {enum_count} enum(s) also present, which are allowed)",
            )
        ]

    return []


def main() -> int:
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: validate_single_class_per_file.py [files or directories...]")
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
        print(f"Found {len(all_violations)} single-class-per-file violation(s):")
        for v in all_violations:
            print(f"  {v.file}:{v.line}: {v.message}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
