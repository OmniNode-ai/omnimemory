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

# Base class names that indicate expected naming prefix
# Maps base class name -> expected prefix
BASE_CLASS_TO_PREFIX = {
    # Model base classes (Pydantic)
    "BaseModel": "Model",
    # Enum base classes
    "Enum": "Enum",
    "StrEnum": "Enum",
    "IntEnum": "Enum",
    "Flag": "Enum",
    "IntFlag": "Enum",
    # Protocol base classes
    "Protocol": "Protocol",
    # Node base classes (ONEX 4-node architecture)
    "BaseNode": "Node",
    "BaseEffectNode": "Node",
    "BaseComputeNode": "Node",
    "BaseReducerNode": "Node",
    "BaseOrchestratorNode": "Node",
    # Common base classes from omnibase
    "NodeBase": "Node",
    "EffectNode": "Node",
    "ComputeNode": "Node",
    "ReducerNode": "Node",
    "OrchestratorNode": "Node",
    # Service base classes (ONEX service layer)
    "BaseService": "Service",
    "ServiceBase": "Service",
    "AbstractService": "Service",
    # Handler base classes (ONEX handler layer)
    "BaseHandler": "Handler",
    "HandlerBase": "Handler",
    "AbstractHandler": "Handler",
    # Mixin base classes
    "BaseMixin": "Mixin",
    "MixinBase": "Mixin",
    # Validator base classes
    "BaseValidator": "Validator",
    "ValidatorBase": "Validator",
    "AbstractValidator": "Validator",
}

# File naming patterns - applied when file is in a typed directory
FILE_PATTERNS = {
    "models": re.compile(r"^model_[a-z][a-z0-9_]*\.py$"),
    "enums": re.compile(r"^enum_[a-z][a-z0-9_]*\.py$"),
    "protocols": re.compile(r"^protocol_[a-z][a-z0-9_]*\.py$|^[a-z_]+_protocols?\.py$"),
    "services": re.compile(r"^service_[a-z][a-z0-9_]*\.py$"),
    "handlers": re.compile(r"^handler_[a-z][a-z0-9_]*\.py$"),
    "mixins": re.compile(r"^mixin_[a-z][a-z0-9_]*\.py$"),
    "nodes": re.compile(r"^node_[a-z][a-z0-9_]*\.py$"),
    "validators": re.compile(r"^validator_[a-z][a-z0-9_]*\.py$"),
    "adapters": re.compile(r"^adapter_[a-z][a-z0-9_]*\.py$"),
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
    "adapters",  # Adapter classes wrap external dependencies
}

# Directories where ONEX naming conventions are strictly enforced
# Both file naming (e.g., model_xxx.py) and class naming (e.g., ModelXxx) are validated
# Note: These apply only to IMMEDIATE parent directory, not ancestors
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
    # Skip data_models.py and error_models.py in protocols/ - they contain data types
    re.compile(r".*/protocols/(data_models|error_models)\.py$"),
]


def get_directory_type(filepath: Path) -> str | None:
    """Get the type based on immediate parent directory name.

    Returns the immediate parent directory name if it's a known STRICT type directory.
    This only returns a type for directories that enforce ONEX naming conventions.

    Note: We check against STRICT_NAMING_DIRECTORIES first, then FILE_PATTERNS
    to ensure we only enforce naming in directories that require it.
    """
    parent = filepath.parent.name

    # First check if it's a strict naming directory
    if parent in STRICT_NAMING_DIRECTORIES:
        return parent

    # Also check adapters for file naming (but not class naming - see RELAXED)
    if parent in FILE_PATTERNS:
        return parent

    return None


def get_ancestor_typed_directory(filepath: Path) -> str | None:
    """Check if any ancestor is a typed directory.

    This helps identify files that are deeply nested within a typed directory
    (e.g., nodes/memory_storage_effect/adapters/) where the top-level type
    should still influence validation behavior.

    Returns the first ancestor typed directory name, or None.
    """
    for part in filepath.parts[:-1]:  # Exclude filename
        if part in STRICT_NAMING_DIRECTORIES:
            return part
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


def detect_expected_prefix_from_bases(
    bases: list[ast.expr],
) -> tuple[str | None, re.Pattern[str] | None]:
    """Detect expected naming prefix from base classes.

    Examines the class's base classes and returns the expected prefix
    based on known base class names.

    Args:
        bases: List of AST base class expressions

    Returns:
        Tuple of (expected_prefix, pattern) or (None, None) if not determined
    """
    for base in bases:
        base_name = None
        if isinstance(base, ast.Name):
            base_name = base.id
        elif isinstance(base, ast.Attribute):
            base_name = base.attr
        elif isinstance(base, ast.Subscript):
            # Handle generic types like Generic[T]
            if isinstance(base.value, ast.Name):
                base_name = base.value.id
            elif isinstance(base.value, ast.Attribute):
                base_name = base.value.attr

        if base_name and base_name in BASE_CLASS_TO_PREFIX:
            prefix = BASE_CLASS_TO_PREFIX[base_name]
            return prefix, CLASS_PATTERNS[prefix]

    return None, None


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
    ancestor_typed_dir = get_ancestor_typed_directory(filepath)

    # Validate file naming based on immediate parent directory type
    # Only enforce file naming in typed directories, not in nested subdirectories
    if dir_type and dir_type in FILE_PATTERNS:
        if not FILE_PATTERNS[dir_type].match(filepath.name):
            # Get the singular form for the example (models -> model, enums -> enum)
            singular = dir_type.rstrip("s") if dir_type.endswith("s") else dir_type
            violations.append(
                Violation(
                    str(filepath),
                    0,
                    f"File '{filepath.name}' in {dir_type}/ should follow naming: "
                    f"{singular}_xxx.py (e.g., {singular}_example.py)",
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

            # Determine expected pattern based on parent class first
            expected_prefix, expected_pattern = detect_expected_prefix_from_bases(
                node.bases
            )

            # If no parent class determined the pattern, check by directory type
            # This ensures files in typed directories enforce naming even without
            # inheriting from known base classes
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
            # (utils, foundation, adapters - these use semantic names for readability)
            if relaxed_prefix:
                continue

            # For files deeply nested within a typed directory (e.g., nodes/xxx/internal/),
            # only skip validation if we couldn't determine the expected prefix from parent class.
            # If parent class detection found a known base (e.g., BaseModel, Enum, Protocol),
            # we should still enforce naming even in nested directories.
            if expected_prefix is None and ancestor_typed_dir and dir_type is None:
                # No parent class detected AND file is nested within a typed directory
                # Skip class naming enforcement - these are typically implementation details
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
