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

# Directories to skip entirely - no validation at all
# These are either non-source directories or special cases
SKIP_DIRECTORIES = {
    "__pycache__",
    ".git",
    ".venv",
    ".tox",
    "tests",  # Test files have different naming conventions
    "compat",  # Compatibility stubs
}

# Directories where class prefix naming is relaxed for readability
# File naming is still enforced, but classes can use semantic names
# (e.g., ConnectionMetadata instead of ModelConnectionMetadata)
RELAXED_CLASS_PREFIX_DIRECTORIES = {
    "utils",  # Utility classes are not domain models
    "foundation",  # Foundation models are base infrastructure
}

# Directories where ONEX naming conventions are strictly enforced
# Both file naming (e.g., model_xxx.py) and class naming (e.g., ModelXxx) are validated
STRICT_NAMING_DIRECTORIES = {
    "models",
    "enums",
    "protocols",
    "services",
    "handlers",
    "mixins",
    "nodes",
    "validators",
}

# Exact relative paths to skip (for specific files that don't follow conventions)
# Use forward slashes for cross-platform compatibility
SKIP_PATHS_PATTERNS: list[re.Pattern[str]] = [
    # Example: re.compile(r"src/omnimemory/legacy/.*"),
]


def get_directory_type(filepath: Path) -> str | None:
    """Get the type based on parent directory name.

    Returns the immediate parent directory name if it's a known type directory.
    """
    parent = filepath.parent.name
    if parent in FILE_PATTERNS:
        return parent
    return None


def should_skip_file(filepath: Path) -> bool:
    """Determine if a file should be skipped entirely.

    Uses precise matching:
    - Exact filename match for SKIP_FILES
    - Immediate parent directory match for SKIP_DIRECTORIES
    - Regex patterns for SKIP_PATHS_PATTERNS
    """
    # Skip by exact filename
    if filepath.name in SKIP_FILES:
        return True

    # Skip by immediate parent directory (not arbitrary ancestor)
    # This is more precise than checking if any path part matches
    parent_dir = filepath.parent.name
    if parent_dir in SKIP_DIRECTORIES:
        return True

    # Also check for nested skip directories (e.g., tests/unit/)
    # But only match the directory itself, not files that happen to have similar names
    for part in filepath.parts[:-1]:  # Exclude the filename
        if part in SKIP_DIRECTORIES:
            return True

    # Check against explicit path patterns (regex)
    filepath_str = str(filepath).replace("\\", "/")  # Normalize for cross-platform
    for pattern in SKIP_PATHS_PATTERNS:
        if pattern.search(filepath_str):
            return True

    return False


def is_relaxed_naming_directory(filepath: Path) -> bool:
    """Check if file is in a directory with relaxed class prefix naming.

    Only the immediate parent directory is checked, not ancestors.
    """
    return filepath.parent.name in RELAXED_CLASS_PREFIX_DIRECTORIES


def validate_file(filepath: Path) -> list[Violation]:
    """Validate a single Python file.

    Validates:
    - File naming conventions based on directory type
    - Class naming conventions (prefix patterns like ModelXxx, ServiceXxx)

    Files in SKIP_DIRECTORIES are skipped entirely.
    Files in RELAXED_CLASS_PREFIX_DIRECTORIES skip class prefix validation.
    """
    if should_skip_file(filepath):
        return []

    violations: list[Violation] = []
    dir_type = get_directory_type(filepath)
    relaxed_prefix = is_relaxed_naming_directory(filepath)

    # Validate file naming based on directory type
    # This applies to all files in typed directories
    if dir_type and dir_type in FILE_PATTERNS:
        if not FILE_PATTERNS[dir_type].match(filepath.name):
            violations.append(
                Violation(
                    str(filepath),
                    0,
                    f"File '{filepath.name}' in {dir_type}/ should follow naming: "
                    f"{dir_type[:-1]}_xxx.py (e.g., {dir_type[:-1]}_example.py)",
                )
            )

    # Validate class naming
    try:
        content = filepath.read_text(encoding="utf-8")
        tree = ast.parse(content, filename=str(filepath))
    except (SyntaxError, UnicodeDecodeError):
        return violations

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            class_name = node.name

            # Skip private classes (single underscore prefix)
            if class_name.startswith("_"):
                continue

            # Determine expected pattern based on parent class or directory
            expected_pattern = None
            expected_prefix = None

            # Check parent classes to infer expected naming
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

                    # Found a pattern match from parent class, stop checking
                    if expected_prefix:
                        break

            # Check by directory type if no parent class determined the pattern
            if expected_prefix is None and dir_type:
                # Map directory type to class pattern
                dir_to_pattern = {
                    "models": ("Model", CLASS_PATTERNS["Model"]),
                    "enums": ("Enum", CLASS_PATTERNS["Enum"]),
                    "protocols": ("Protocol", CLASS_PATTERNS["Protocol"]),
                    "services": ("Service", CLASS_PATTERNS["Service"]),
                    "handlers": ("Handler", CLASS_PATTERNS["Handler"]),
                    "mixins": ("Mixin", CLASS_PATTERNS["Mixin"]),
                    "nodes": ("Node", CLASS_PATTERNS["Node"]),
                    "validators": ("Validator", CLASS_PATTERNS["Validator"]),
                }
                if dir_type in dir_to_pattern:
                    expected_prefix, expected_pattern = dir_to_pattern[dir_type]

            # Skip class prefix validation for relaxed directories
            # (utils, foundation - these use semantic names for readability)
            if relaxed_prefix:
                continue

            # Validate class naming convention
            if expected_pattern and not expected_pattern.match(class_name):
                violations.append(
                    Violation(
                        str(filepath),
                        node.lineno,
                        f"Class '{class_name}' should follow ONEX naming: "
                        f"{expected_prefix}Xxx (e.g., {expected_prefix}{class_name})",
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
