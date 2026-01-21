#!/usr/bin/env python3
"""ONEX Single Class Per File Validation.

Enforces the ONEX architectural rule: one non-enum class per file.
Multiple enums in a single file are explicitly allowed since enums are
simple value types that don't create the same import/dependency issues.

This promotes clean imports and reduces circular dependency issues.

Usage:
    python scripts/validation/validate_single_class_per_file.py [files...]
    python scripts/validation/validate_single_class_per_file.py src/omnimemory/models/
"""

from __future__ import annotations

import ast
import logging
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
    """Check if a class definition is an Enum subclass.

    Note: We check for actual Enum base classes only.
    'auto' is a function used for auto-numbering, not a base class.
    """
    enum_bases = {"Enum", "StrEnum", "IntEnum", "IntFlag", "Flag"}
    for base in node.bases:
        if isinstance(base, ast.Name) and base.id in enum_bases:
            return True
        if isinstance(base, ast.Attribute) and base.attr in enum_bases:
            return True
    return False


def count_classes(filepath: Path) -> tuple[int, list[str], int, list[str]]:
    """Count top-level class definitions in a file.

    Handles file reading and parsing errors gracefully by emitting warnings
    and returning zero counts rather than crashing.

    Returns:
        Tuple of (non_enum_count, non_enum_names, enum_count, enum_names)
    """
    try:
        content = filepath.read_text(encoding="utf-8")
    except PermissionError:
        logging.warning("Permission denied reading file: %s", filepath)
        return 0, [], 0, []
    except FileNotFoundError:
        logging.warning("File not found (possibly deleted during scan): %s", filepath)
        return 0, [], 0, []
    except OSError as e:
        logging.warning("OS error reading file %s: %s", filepath, e)
        return 0, [], 0, []
    except UnicodeDecodeError as e:
        logging.warning("Unicode decode error in file %s: %s", filepath, e)
        return 0, [], 0, []

    try:
        tree = ast.parse(content, filename=str(filepath))
    except SyntaxError as e:
        logging.warning("Syntax error parsing file %s: %s", filepath, e)
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

    # Check if file is in an exempt directory.
    #
    # Design Decision: We check ALL path components, not just the immediate parent.
    # This is intentional because:
    #
    # 1. Nested structures: Files in subdirectories of exempt directories should
    #    inherit the exemption. For example, if "core" is exempt, then both
    #    "models/core/base.py" AND "models/core/utils/helpers.py" are exempt.
    #
    # 2. Flexible project layouts: Different projects may nest exempt directories
    #    at varying depths (e.g., "src/pkg/core/", "pkg/core/", or just "core/").
    #    Checking any component handles all these cases uniformly.
    #
    # 3. Transitive exemption: Foundation/core modules often have internal
    #    organization with subdirectories. The entire subtree should be treated
    #    as a cohesive unit where related types may coexist in single files.
    #
    # Trade-off: This is more permissive than checking only immediate parent.
    # If a non-exempt directory happens to be named "core" elsewhere in the path,
    # it would incorrectly be exempt. However, this is rare in practice and the
    # benefit of simpler, more robust exemption logic outweighs this edge case.
    for part in filepath.parts:
        if part in EXEMPT_DIRECTORIES:
            return []

    non_enum_count, non_enum_names, enum_count, enum_names = count_classes(filepath)

    # Only enforce single-class rule for non-enum classes
    # Multiple enums in one file are explicitly allowed
    if non_enum_count > 1:
        enum_note = ""
        if enum_count > 0:
            enum_list = ", ".join(enum_names)
            enum_note = f" (Also {enum_count} enum(s): {enum_list}, allowed)"
        return [
            Violation(
                str(filepath),
                1,
                f"Multiple non-enum classes ({non_enum_count}): "
                f"{', '.join(non_enum_names)}. ONEX requires one per file.{enum_note}",
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
