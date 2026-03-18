#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""ONEX Naming Convention Validation.

Validates that classes and files follow ONEX naming conventions:
- Classes: ModelXxx, EnumXxx, ProtocolXxx, ServiceXxx, etc.
- Files: model_xxx.py, enum_xxx.py, protocol_xxx.py, etc.

Usage:
    python scripts/validation/validate_naming.py src/
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


# =============================================================================
# NAMING CONVENTIONS - Single Source of Truth
# =============================================================================
# All naming prefixes are defined here. CLASS_PATTERNS, DIR_PREFIXES, and
# STRICT_NAMING_DIRECTORIES are derived from this list to avoid duplication.

# List of naming prefixes for ONEX conventions
# All use standard PrefixXxx pattern except Settings which uses XxxSettings suffix
NAMING_PREFIXES = [
    "Model",
    "Enum",
    "Protocol",
    "Service",
    "Handler",
    "Mixin",
    "Node",
    "Validator",
    "Settings",  # Special case: uses suffix pattern, has no typed directory
]

# Class naming patterns - derived from NAMING_PREFIXES
# Settings uses suffix pattern (XxxSettings), all others use prefix pattern (PrefixXxx)
CLASS_PATTERNS = {
    prefix: (
        re.compile(r"^[A-Z][a-zA-Z0-9]*Settings$")
        if prefix == "Settings"
        else re.compile(rf"^{prefix}[A-Z][a-zA-Z0-9]*$")
    )
    for prefix in NAMING_PREFIXES
}

# Directory name to class prefix mapping - derived from NAMING_PREFIXES
# Convention: directory is lowercase plural of prefix (e.g., Model -> models)
# Settings has no typed directory (configuration classes live alongside their domain)
DIR_PREFIXES: dict[str, str] = {
    f"{prefix.lower()}s": prefix for prefix in NAMING_PREFIXES if prefix != "Settings"
}

# Base class names that indicate expected naming prefix
# Maps base class name -> expected prefix
# Note: This mapping is used by detect_expected_prefix_from_bases() to determine
# the required naming convention for a class based on its inheritance hierarchy.
BASE_CLASS_TO_PREFIX = {
    # Model base classes (Pydantic and similar)
    "BaseModel": "Model",
    "RootModel": "Model",  # Pydantic RootModel
    # Settings base classes (pydantic-settings configuration)
    # These use XxxSettings suffix naming, NOT ModelXxx prefix
    # Settings classes are configuration containers, not data models
    "BaseSettings": "Settings",
    # Enum base classes (stdlib and extensions)
    "Enum": "Enum",
    "StrEnum": "Enum",
    "IntEnum": "Enum",
    "Flag": "Enum",
    "IntFlag": "Enum",
    "auto": None,  # Explicitly ignore auto() - not a base class
    # Protocol base classes (typing)
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
    # Additional node patterns
    "AbstractNode": "Node",
    "NodeInterface": "Node",
    # Service base classes (ONEX service layer)
    "BaseService": "Service",
    "ServiceBase": "Service",
    "AbstractService": "Service",
    "ServiceInterface": "Service",
    "Service": "Service",  # Direct Service inheritance
    # Handler base classes (ONEX handler layer)
    "BaseHandler": "Handler",
    "HandlerBase": "Handler",
    "AbstractHandler": "Handler",
    "HandlerInterface": "Handler",
    "Handler": "Handler",  # Direct Handler inheritance
    "RequestHandler": "Handler",
    "EventHandler": "Handler",
    "MessageHandler": "Handler",
    # Mixin base classes
    "BaseMixin": "Mixin",
    "MixinBase": "Mixin",
    "Mixin": "Mixin",  # Direct Mixin inheritance
    # Validator base classes
    "BaseValidator": "Validator",
    "ValidatorBase": "Validator",
    "AbstractValidator": "Validator",
    "ValidatorInterface": "Validator",
    "Validator": "Validator",  # Direct Validator inheritance
    # Pydantic field validators (these should be ignored, not classes)
    # "field_validator": None,  # Not a base class
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
    "migrations",  # Database migrations have different naming
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
# Derived from DIR_PREFIXES to maintain single source of truth
STRICT_NAMING_DIRECTORIES = set(DIR_PREFIXES.keys())

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

    Special case: If the file is in models/*/adapters/, it should follow model_*.py
    naming, not adapter_*.py naming, because models take precedence in the
    models/ directory hierarchy.
    """
    parent = filepath.parent.name
    path_parts = filepath.parts

    # Special case: files in models/*/adapters/ should use model_*.py naming
    # because they're model files organized by adapter domain, not adapter implementations
    if parent == "adapters" and "models" in path_parts:
        models_idx = path_parts.index("models")
        parent_idx = len(path_parts) - 2  # Index of parent directory
        # Check if "models" is an ancestor of "adapters"
        if models_idx < parent_idx:
            return "models"

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
    (e.g., nodes/node_memory_storage_effect/adapters/) where the top-level type
    should still influence validation behavior.

    Returns the first ancestor typed directory name, or None.
    """
    for part in filepath.parts[:-1]:  # Exclude filename
        if part in STRICT_NAMING_DIRECTORIES:
            return part
    return None


def should_skip_file(filepath: Path) -> bool:
    """Determine if a file should be skipped entirely.

    Uses precise Path-based matching (not substring matching):

    1. Exact filename match against SKIP_FILES set
       - Matches: "__init__.py", "conftest.py", "base.py"
       - Does NOT match filenames containing these as substrings

    2. Directory component match against SKIP_DIRECTORIES set
       - Each path component is checked as a complete directory name
       - Matches: "src/tests/test_foo.py" (contains "tests" component)
       - Does NOT match: "src/attestations/foo.py" (no "tests" component)

    3. Regex patterns for specific path patterns (SKIP_PATHS_PATTERNS)
       - Used for precise path pattern matching when needed

    Note: This function uses Path.parts which splits on path separators,
    ensuring we match complete directory names, not substrings.
    For example, "tests" will NOT match in "src/my_tests_helper/foo.py"
    because "my_tests_helper" is a single path component that doesn't
    equal "tests".

    Args:
        filepath: Path to the file to check

    Returns:
        True if the file should be skipped, False otherwise
    """
    # Skip by exact filename (set membership, not substring)
    if filepath.name in SKIP_FILES:
        return True

    # Check each directory component against SKIP_DIRECTORIES
    # Path.parts splits on separators: ('src', 'tests', 'unit', 'test_foo.py')
    # We check each component (except filename) for exact match
    # This is NOT substring matching - 'tests' won't match 'my_tests_dir'
    path_parts = filepath.parts[:-1]  # Exclude the filename
    for part in path_parts:
        if part in SKIP_DIRECTORIES:
            return True

    # Check against explicit path patterns (regex)
    # Convert to forward slashes for cross-platform consistency
    filepath_str = str(filepath).replace("\\", "/")
    for pattern in SKIP_PATHS_PATTERNS:
        if pattern.search(filepath_str):
            return True

    return False


def is_relaxed_naming_directory(filepath: Path) -> bool:
    """Check if file is in a directory with relaxed class prefix naming.

    Only the immediate parent directory is checked, not ancestors.
    """
    return filepath.parent.name in RELAXED_CLASS_PREFIX_DIRECTORIES


def _extract_base_class_name(base: ast.expr) -> str | None:
    """Extract the class name from various AST expression types.

    Handles:
    - ast.Name: simple name like `BaseModel`
    - ast.Attribute: qualified name like `pydantic.BaseModel` or `a.b.c.BaseModel`
    - ast.Subscript: generic types like `Generic[T]` or `List[str]`
    - ast.Call: constructor calls like `SomeBase()` or `TypedDict("Name", ...)`

    Args:
        base: AST expression representing a base class

    Returns:
        The extracted class name, or None if unable to extract
    """
    if isinstance(base, ast.Name):
        return base.id
    elif isinstance(base, ast.Attribute):
        # For chained attributes like a.b.c.BaseModel, extract the final attr
        return base.attr
    elif isinstance(base, ast.Subscript):
        # Handle Generic[T], Optional[X], etc.
        return _extract_base_class_name(base.value)
    elif isinstance(base, ast.Call):
        # Handle SomeBase(), TypedDict("Name", ...), etc.
        return _extract_base_class_name(base.func)
    return None


def detect_expected_prefix_from_bases(
    bases: list[ast.expr],
) -> tuple[str | None, re.Pattern[str] | None]:
    """Detect expected naming prefix from base classes.

    Examines the class's base classes and returns the expected prefix
    based on known base class names. Checks ALL base classes to handle
    multiple inheritance correctly.

    Args:
        bases: List of AST base class expressions

    Returns:
        Tuple of (expected_prefix, pattern) or (None, None) if not determined
    """
    for base in bases:
        base_name = _extract_base_class_name(base)

        if base_name and base_name in BASE_CLASS_TO_PREFIX:
            prefix = BASE_CLASS_TO_PREFIX[base_name]
            # Handle explicitly ignored base classes (value is None)
            if prefix is None:
                continue
            return prefix, CLASS_PATTERNS[prefix]

    return None, None


def validate_file(  # stub-ok: docstring uses ModelXxx/ServiceXxx as naming examples, not TODO markers
    filepath: Path,
) -> list[Violation]:
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
    except PermissionError:
        logging.warning("Permission denied reading file: %s", filepath)
        return violations
    except FileNotFoundError:
        logging.warning("File not found (possibly deleted during scan): %s", filepath)
        return violations
    except OSError as e:
        logging.warning("OS error reading file %s: %s", filepath, e)
        return violations
    except UnicodeDecodeError as e:
        logging.warning("Unicode decode error in file %s: %s", filepath, e)
        return violations

    try:
        tree = ast.parse(content, filename=str(filepath))
    except SyntaxError as e:
        logging.warning("Syntax error parsing file %s: %s", filepath, e)
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
                # Use module-level DIR_PREFIXES mapping for consistency
                if dir_type in DIR_PREFIXES:
                    expected_prefix = DIR_PREFIXES[dir_type]
                    expected_pattern = CLASS_PATTERNS[expected_prefix]

            # Skip class prefix validation for relaxed directories
            # (utils, foundation, adapters - these use semantic names for readability)
            if relaxed_prefix:
                continue

            # For files nested within a typed directory (e.g., nodes/xxx/internal/),
            # only skip validation if we couldn't determine expected prefix from parent.
            # If parent class detection found a known base (BaseModel, Enum, Protocol),
            # we should still enforce naming in nested directories.
            if expected_prefix is None and ancestor_typed_dir and dir_type is None:
                # No parent class detected AND nested within typed directory
                # Skip naming enforcement - these are typically implementation details
                continue

            # Validate class naming convention
            if expected_pattern and not expected_pattern.match(class_name):
                # Settings classes use suffix pattern (XxxSettings), not prefix
                if expected_prefix == "Settings":
                    pattern_desc = "XxxSettings"
                    example = f"{class_name}Settings"
                else:
                    pattern_desc = f"{expected_prefix}Xxx"
                    example = f"{expected_prefix}{class_name}"
                violations.append(
                    Violation(
                        str(filepath),
                        node.lineno,
                        f"Class '{class_name}' should follow ONEX naming: "
                        f"{pattern_desc} (e.g., {example})",
                    )
                )

    return violations


def main() -> int:
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: validate_naming.py [directory]")
        return 1

    directory = Path(sys.argv[1])
    if not directory.is_dir():
        print(f"Directory not found or not a directory: {directory}")
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
