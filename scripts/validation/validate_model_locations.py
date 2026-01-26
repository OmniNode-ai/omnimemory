#!/usr/bin/env python3
"""ONEX Pydantic Model Location Enforcement.

Ensures that Pydantic models (classes inheriting from BaseModel) are located
in approved directories. This enforces the ONEX architectural pattern that
models should be organized in specific locations for maintainability.

Allowed locations for Pydantic models:
- src/omnimemory/models/ (primary models directory)
- src/omnimemory/nodes/*/models/ (node-specific models)
- src/omnimemory/protocols/ (protocol definitions and base models)
- tests/ (test fixtures and mocks)

All other locations should use models from the approved directories or
explicitly exempt themselves with an annotation.

Exemption Annotation:
    If a file legitimately needs a Pydantic model outside the allowed
    locations (rare), add a comment with the explicit exemption annotation
    on the SAME LINE as the class definition:

        class MyConfig(BaseModel):  # omnimemory-model-exempt: <reason>

    Example:
        class AdapterConfig(BaseModel):  # omnimemory-model-exempt: Adapter-local config

    This annotation is intentionally specific to prevent accidental matches
    with unrelated comments.

Usage:
    python scripts/validation/validate_model_locations.py src/
    python scripts/validation/validate_model_locations.py src/omnimemory/

Added for ONEX compliance: Model organization enforcement
"""

from __future__ import annotations

import argparse
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
    class_name: str
    message: str


# Paths where Pydantic models ARE allowed
ALLOWED_PATHS = [
    # Primary models directory
    "src/omnimemory/models/",
    # Node-specific models (ONEX pattern)
    "nodes/memory_retrieval_effect/models/",
    "nodes/memory_storage_effect/models/",
    "nodes/similarity_compute/models/",
    # Protocol definitions (interfaces and base models)
    "src/omnimemory/protocols/",
    # Tests can have models for fixtures/mocks
    "tests/",
]

# Pattern for node-specific models (dynamic matching)
NODE_MODELS_PATTERN = re.compile(r"nodes/[^/]+/models/")

# Base classes that indicate Pydantic model inheritance
PYDANTIC_BASE_CLASSES = {
    "BaseModel",
    "BaseMemoryModel",  # Project-specific base
}

# Exemption pattern
EXEMPTION_PATTERN = re.compile(r"#\s*omnimemory-model-exempt:", re.IGNORECASE)


def is_path_allowed(filepath: Path) -> bool:
    """Check if the file path is in an allowed location for Pydantic models.

    Uses path normalization and explicit boundary checking to avoid
    false positives from substring matching (e.g., 'models_extra' matching 'models/').
    """
    # Normalize to forward slashes for consistent matching
    filepath_str = str(filepath).replace("\\", "/")

    # Check static allowed paths with proper boundary matching
    for allowed in ALLOWED_PATHS:
        # Normalize allowed path
        allowed_normalized = allowed.replace("\\", "/")
        # Check if the path contains this allowed segment
        if allowed_normalized in filepath_str:
            # Verify it's a proper directory boundary, not a prefix of another name
            # e.g., "models/" should not match "models_extra/"
            idx = filepath_str.find(allowed_normalized)
            if idx != -1:
                # Either at the end or followed by a path separator
                end_idx = idx + len(allowed_normalized)
                if end_idx >= len(filepath_str) or allowed_normalized.endswith("/"):
                    return True

    # Check dynamic node models pattern
    return bool(NODE_MODELS_PATTERN.search(filepath_str))


class PydanticModelVisitor(ast.NodeVisitor):
    """AST visitor that finds Pydantic model class definitions."""

    def __init__(self, source_lines: list[str]) -> None:
        self.source_lines = source_lines
        self.models: list[tuple[str, int, bool]] = []  # (class_name, line, is_exempt)
        self.in_type_checking = False

    def visit_If(self, node: ast.If) -> None:
        """Track TYPE_CHECKING blocks.

        Handles multiple patterns for TYPE_CHECKING:
        - Direct: if TYPE_CHECKING:
        - Module-qualified: if typing.TYPE_CHECKING:
        - Any module prefix: if <module>.TYPE_CHECKING:
        """
        # Check if this is a TYPE_CHECKING block using various patterns
        if self._is_type_checking_test(node.test):
            # Don't visit inside TYPE_CHECKING blocks
            return

        self.generic_visit(node)

    def _is_type_checking_test(self, test: ast.expr) -> bool:
        """Check if an if-test is checking TYPE_CHECKING.

        Handles:
        - Direct name: TYPE_CHECKING
        - Attribute access: typing.TYPE_CHECKING, foo.TYPE_CHECKING
        """
        # Pattern 1: Direct name (from typing import TYPE_CHECKING)
        if isinstance(test, ast.Name) and test.id == "TYPE_CHECKING":
            return True

        # Pattern 2: Attribute access (typing.TYPE_CHECKING or any_module.TYPE_CHECKING)
        return isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Visit class definitions to find Pydantic models."""
        # Check if this class inherits from a Pydantic base
        is_pydantic_model = False

        for base in node.bases:
            base_name = self._get_base_name(base)
            if base_name in PYDANTIC_BASE_CLASSES:
                is_pydantic_model = True
                break

        if is_pydantic_model:
            # Check for exemption annotation on class definition lines
            # For multi-line class definitions, check from lineno to the line with ":"
            is_exempt = False
            start_line = node.lineno - 1  # 0-indexed
            # Scan from class start to the line ending with ":" (class signature end)
            for line_idx in range(start_line, min(start_line + 10, len(self.source_lines))):
                line = self.source_lines[line_idx]
                if EXEMPTION_PATTERN.search(line):
                    is_exempt = True
                    break
                # Stop AFTER checking the line that ends the class signature
                # (the signature always ends with ":")
                if line.rstrip().endswith(":"):
                    break

            self.models.append((node.name, node.lineno, is_exempt))

        # Continue visiting nested classes
        self.generic_visit(node)

    def _get_base_name(self, base: ast.expr) -> str:
        """Extract the base class name from an AST node."""
        if isinstance(base, ast.Name):
            return base.id
        if isinstance(base, ast.Attribute):
            return base.attr
        if isinstance(base, ast.Subscript):
            # Handle Generic[T] style bases
            if isinstance(base.value, ast.Name):
                return base.value.id
            if isinstance(base.value, ast.Attribute):
                return base.value.attr
        return ""


def validate_file(filepath: Path) -> list[Violation]:
    """Validate a single Python file for Pydantic model location violations."""
    # Skip files in allowed locations
    if is_path_allowed(filepath):
        return []

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

    # Quick check: skip files that don't import BaseModel
    if "BaseModel" not in content and "BaseMemoryModel" not in content:
        return []

    try:
        tree = ast.parse(content)
    except SyntaxError as e:
        logging.warning("Syntax error in file %s: %s", filepath, e)
        return []

    source_lines = content.splitlines()
    visitor = PydanticModelVisitor(source_lines)
    visitor.visit(tree)

    violations: list[Violation] = []
    for class_name, line_num, is_exempt in visitor.models:
        if not is_exempt:
            violations.append(
                Violation(
                    str(filepath),
                    line_num,
                    class_name,
                    f"Pydantic model '{class_name}' found outside allowed locations. "
                    f"Move to src/omnimemory/models/ or add exemption annotation.",
                )
            )

    return violations


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Enforce Pydantic model locations - detect models outside allowed directories"
    )
    parser.add_argument(
        "paths",
        nargs="*",
        default=["src/"],
        help="Files or directories to check",
    )
    args = parser.parse_args()

    files_to_check: list[Path] = []

    for path_str in args.paths:
        path = Path(path_str)
        if path.is_file() and path.suffix == ".py":
            files_to_check.append(path)
        elif path.is_dir():
            files_to_check.extend(path.rglob("*.py"))
        else:
            logging.warning("Skipping invalid path: %s", path)

    all_violations: list[Violation] = []
    for filepath in files_to_check:
        violations = validate_file(filepath)
        all_violations.extend(violations)

    if all_violations:
        print(f"Found {len(all_violations)} Pydantic model location violation(s):")
        print()
        for v in all_violations:
            print(f"  {v.file}:{v.line}:")
            print(f"    class {v.class_name}")
            print(f"    {v.message}")
            print()
        print("To fix either:")
        print(
            "  1. Move the model to src/omnimemory/models/ or a node's models/ directory"
        )
        print(
            "  2. Add exemption: class MyModel(BaseModel):  # omnimemory-model-exempt: <reason>"
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
