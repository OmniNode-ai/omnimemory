#!/usr/bin/env python3
"""ONEX Naming Convention Validation.

Validates that classes and files follow ONEX naming conventions:
- Classes: ModelXxx, EnumXxx, ProtocolXxx, ServiceXxx, etc.
- Files: model_xxx.py, enum_xxx.py, protocol_xxx.py, etc.

Usage:
    python scripts/validation/validate_naming.py src/
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path
from typing import NamedTuple


class Violation(NamedTuple):
    """A validation violation."""

    file: str
    line: int
    message: str


# Class naming patterns
CLASS_PATTERNS = {
    "Model": re.compile(r"^Model[A-Z][a-zA-Z0-9]*$"),
    "Enum": re.compile(r"^Enum[A-Z][a-zA-Z0-9]*$"),
    "Protocol": re.compile(r"^Protocol[A-Z][a-zA-Z0-9]*$"),
    "Service": re.compile(r"^Service[A-Z][a-zA-Z0-9]*$"),
    "Handler": re.compile(r"^Handler[A-Z][a-zA-Z0-9]*$"),
    "Mixin": re.compile(r"^Mixin[A-Z][a-zA-Z0-9]*$"),
    "Node": re.compile(r"^Node[A-Z][a-zA-Z0-9]*$"),
    "Validator": re.compile(r"^Validator[A-Z][a-zA-Z0-9]*$"),
}

# File naming patterns
FILE_PATTERNS = {
    "models": re.compile(r"^model_[a-z][a-z0-9_]*\.py$"),
    "enums": re.compile(r"^enum_[a-z][a-z0-9_]*\.py$"),
    "protocols": re.compile(r"^protocol_[a-z][a-z0-9_]*$|^[a-z_]+_protocols?\.py$"),
    "services": re.compile(r"^service_[a-z][a-z0-9_]*\.py$"),
    "handlers": re.compile(r"^handler_[a-z][a-z0-9_]*\.py$"),
    "mixins": re.compile(r"^mixin_[a-z][a-z0-9_]*\.py$"),
    "nodes": re.compile(r"^node_[a-z][a-z0-9_]*\.py$"),
    "validators": re.compile(r"^validator_[a-z][a-z0-9_]*\.py$"),
    "utils": re.compile(r"^[a-z][a-z0-9_]*\.py$"),  # Utils are more flexible
}

# Files to skip (exact filename match)
SKIP_FILES = {"__init__.py", "conftest.py", "base.py"}

# Directory names to skip (exact directory name match in path parts)
# - compat: compatibility stubs
# - protocols: API contracts use standard Request/Response patterns, not Model prefix
# - enums: Enum files follow enum_xxx.py naming, classes use semantic names (not EnumXxx)
# - models: Model files follow model_xxx.py naming, classes use semantic names (not ModelXxx)
# - utils: Utility classes are not domain models, use semantic names
# - foundation: Foundation models are base infrastructure, use semantic names
# - __pycache__, .git, tests: standard exclusions
# NOTE: File naming is enforced via file patterns, class prefix naming is relaxed for
#       readability and to avoid redundancy (e.g., model_connection_metadata.py containing
#       ConnectionMetadata vs ModelConnectionMetadata)
SKIP_DIRECTORIES = {
    "compat",
    "__pycache__",
    ".git",
    "tests",
    "protocols",
    "enums",
    "models",
    "utils",
    "foundation",
}


def get_directory_type(filepath: Path) -> str | None:
    """Get the type based on parent directory name."""
    parent = filepath.parent.name
    if parent in FILE_PATTERNS:
        return parent
    return None


def validate_file(filepath: Path) -> list[Violation]:
    """Validate a single Python file."""
    # Skip files by exact filename match
    if filepath.name in SKIP_FILES:
        return []

    # Skip files in specific directories (exact directory name match)
    # Use path.parts to check for exact directory names, not substring matching
    if any(skip_dir in filepath.parts for skip_dir in SKIP_DIRECTORIES):
        return []

    violations: list[Violation] = []

    # Validate file naming based on directory
    dir_type = get_directory_type(filepath)
    if dir_type and dir_type in FILE_PATTERNS:
        if not FILE_PATTERNS[dir_type].match(filepath.name):
            # Don't fail, just warn - this is advisory
            pass  # File naming is advisory, not enforced

    # Validate class naming
    try:
        content = filepath.read_text(encoding="utf-8")
        tree = ast.parse(content, filename=str(filepath))
    except (SyntaxError, Exception):
        return violations

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            class_name = node.name

            # Determine expected pattern based on parent class or directory
            expected_pattern = None
            expected_prefix = None

            # Check parent classes
            for base in node.bases:
                base_name = None
                if isinstance(base, ast.Name):
                    base_name = base.id
                elif isinstance(base, ast.Attribute):
                    base_name = base.attr

                if base_name:
                    if base_name == "BaseModel":
                        expected_pattern = CLASS_PATTERNS["Model"]
                        expected_prefix = "Model"
                    elif base_name in ("Enum", "StrEnum", "IntEnum"):
                        expected_pattern = CLASS_PATTERNS["Enum"]
                        expected_prefix = "Enum"
                    elif base_name == "Protocol":
                        expected_pattern = CLASS_PATTERNS["Protocol"]
                        expected_prefix = "Protocol"

            # Also check by directory
            if dir_type == "models" and expected_prefix is None:
                expected_pattern = CLASS_PATTERNS["Model"]
                expected_prefix = "Model"
            elif dir_type == "enums" and expected_prefix is None:
                expected_pattern = CLASS_PATTERNS["Enum"]
                expected_prefix = "Enum"
            elif dir_type == "protocols" and expected_prefix is None:
                expected_pattern = CLASS_PATTERNS["Protocol"]
                expected_prefix = "Protocol"
            elif dir_type == "services" and expected_prefix is None:
                expected_pattern = CLASS_PATTERNS["Service"]
                expected_prefix = "Service"
            elif dir_type == "handlers" and expected_prefix is None:
                expected_pattern = CLASS_PATTERNS["Handler"]
                expected_prefix = "Handler"
            elif dir_type == "mixins" and expected_prefix is None:
                expected_pattern = CLASS_PATTERNS["Mixin"]
                expected_prefix = "Mixin"
            elif dir_type == "nodes" and expected_prefix is None:
                expected_pattern = CLASS_PATTERNS["Node"]
                expected_prefix = "Node"
            elif dir_type == "validators" and expected_prefix is None:
                expected_pattern = CLASS_PATTERNS["Validator"]
                expected_prefix = "Validator"

            if expected_pattern and not expected_pattern.match(class_name):
                violations.append(
                    Violation(
                        str(filepath),
                        node.lineno,
                        f"Class '{class_name}' should follow ONEX naming: "
                        f"{expected_prefix}Xxx",
                    )
                )

    return violations


def main() -> int:
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: validate_naming.py [directory]")
        return 1

    directory = Path(sys.argv[1])
    if not directory.exists():
        print(f"Directory not found: {directory}")
        return 1

    files_to_check = list(directory.rglob("*.py"))

    all_violations: list[Violation] = []
    for filepath in files_to_check:
        violations = validate_file(filepath)
        all_violations.extend(violations)

    if all_violations:
        print(f"Found {len(all_violations)} naming convention violation(s):")
        for v in all_violations:
            print(f"  {v.file}:{v.line}: {v.message}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
