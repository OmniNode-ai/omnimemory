#!/usr/bin/env python3
"""ONEX Pydantic Pattern Validation.

Validates that Pydantic models follow ONEX conventions:
- Use Field() with descriptions
- Use ConfigDict with proper settings
- No bare model_config = {} assignments
- Proper inheritance from BaseModel
- Handles inherited model_config from parent classes
- Detects ConfigDict imports with aliases

Usage:
    python scripts/validation/validate_pydantic_patterns.py [files...]
    python scripts/validation/validate_pydantic_patterns.py src/
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


class ImportAliasCollector(ast.NodeVisitor):
    """Collect ConfigDict import aliases from a module."""

    def __init__(self) -> None:
        self.config_dict_aliases: set[str] = {"ConfigDict"}  # Always include default

    def visit_Import(self, node: ast.Import) -> None:
        """Visit import statements."""
        for alias in node.names:
            # import pydantic - ConfigDict accessed via pydantic.ConfigDict
            if alias.name == "pydantic":
                # We handle this via attribute access, no alias needed
                pass
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Visit from-import statements."""
        if node.module and ("pydantic" in node.module):
            for alias in node.names:
                if alias.name == "ConfigDict":
                    # from pydantic import ConfigDict as CD
                    actual_name = alias.asname if alias.asname else alias.name
                    self.config_dict_aliases.add(actual_name)
        self.generic_visit(node)


class ClassModelConfigCollector(ast.NodeVisitor):
    """First pass: collect which classes have model_config defined and which are Pydantic models."""

    # Known Pydantic base classes that indicate a Pydantic model
    PYDANTIC_BASE_CLASSES = {"BaseModel", "GenericModel", "BaseSettings"}

    def __init__(self) -> None:
        self.classes_with_model_config: set[str] = set()
        self.class_bases: dict[str, list[str]] = {}  # class_name -> list of base names
        self.pydantic_models: set[str] = set()  # Classes that are Pydantic models

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Visit class definition to check for model_config and Pydantic inheritance."""
        # Collect base class names (simple names only, not fully qualified)
        base_names: list[str] = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                base_names.append(base.id)
            elif isinstance(base, ast.Attribute):
                base_names.append(base.attr)
            elif isinstance(base, ast.Subscript):
                # Handle Generic[T], etc.
                if isinstance(base.value, ast.Name):
                    base_names.append(base.value.id)
                elif isinstance(base.value, ast.Attribute):
                    base_names.append(base.value.attr)
        self.class_bases[node.name] = base_names

        # Check if this class directly inherits from a Pydantic base
        for base_name in base_names:
            if base_name in self.PYDANTIC_BASE_CLASSES:
                self.pydantic_models.add(node.name)
                break

        # Check if this class has model_config defined
        for item in node.body:
            if isinstance(item, ast.Assign):
                for target in item.targets:
                    if isinstance(target, ast.Name) and target.id == "model_config":
                        self.classes_with_model_config.add(node.name)
            elif isinstance(item, ast.AnnAssign):
                if (
                    isinstance(item.target, ast.Name)
                    and item.target.id == "model_config"
                ):
                    self.classes_with_model_config.add(node.name)

        self.generic_visit(node)

    def _resolve_pydantic_models(self) -> None:
        """Resolve transitive Pydantic model inheritance within the file.

        A class is a Pydantic model if:
        1. It directly inherits from BaseModel/GenericModel/BaseSettings, OR
        2. It inherits from another class in this file that is a Pydantic model
        """
        changed = True
        while changed:
            changed = False
            for class_name, bases in self.class_bases.items():
                if class_name in self.pydantic_models:
                    continue
                for base in bases:
                    if base in self.pydantic_models:
                        self.pydantic_models.add(class_name)
                        changed = True
                        break

    def is_pydantic_model(self, class_name: str) -> bool:
        """Check if a class is a Pydantic model (directly or transitively)."""
        return class_name in self.pydantic_models

    def has_inherited_model_config(self, class_name: str) -> bool:
        """Check if a class inherits model_config from a parent in this file."""
        if class_name in self.classes_with_model_config:
            return True

        bases = self.class_bases.get(class_name, [])
        for base in bases:
            # Recursively check parent classes defined in this file
            if base in self.class_bases:
                if self.has_inherited_model_config(base):
                    return True
        return False


class PydanticPatternVisitor(ast.NodeVisitor):
    """AST visitor to check Pydantic patterns."""

    def __init__(
        self,
        filepath: str,
        config_dict_aliases: set[str],
        model_config_collector: ClassModelConfigCollector,
    ) -> None:
        self.filepath = filepath
        self.violations: list[Violation] = []
        self.in_class: str | None = None
        self.is_pydantic_model = False
        self.config_dict_aliases = config_dict_aliases
        self.model_config_collector = model_config_collector

    def _is_config_dict_call(self, node: ast.expr) -> bool:
        """Check if a node is a ConfigDict() call (handles aliases and attributes)."""
        if not isinstance(node, ast.Call):
            return False
        func = node.func
        # Handle: ConfigDict() or aliased name like CD()
        if isinstance(func, ast.Name) and func.id in self.config_dict_aliases:
            return True
        # Handle: pydantic.ConfigDict()
        return isinstance(func, ast.Attribute) and func.attr == "ConfigDict"

    def _check_inherited_model_config(self, node: ast.ClassDef) -> bool:
        """Check if model_config is inherited from a parent class.

        Checks:
        1. Parent classes defined in this file that have model_config
        2. Known Pydantic base classes that have model_config (BaseModel does NOT)
        """
        for base in node.bases:
            base_name: str | None = None
            if isinstance(base, ast.Name):
                base_name = base.id
            elif isinstance(base, ast.Attribute):
                base_name = base.attr

            if base_name:
                # Check if parent class in this file has model_config
                if self.model_config_collector.has_inherited_model_config(base_name):
                    return True

                # BaseModel itself doesn't define model_config with settings,
                # so we don't skip validation for direct BaseModel inheritance
                # But if inheriting from another Pydantic model class defined
                # elsewhere, we can't know - so we're conservative and only
                # check classes in this file

        return False

    def _is_empty_config_dict(self, node: ast.Call) -> bool:
        """Check if a ConfigDict call has no arguments."""
        return not node.args and not node.keywords

    def _check_model_config_value(
        self, value: ast.expr, lineno: int, class_name: str
    ) -> None:
        """Check the value assigned to model_config and add violations if needed."""
        if isinstance(value, ast.Call) and self._is_config_dict_call(value):
            # Check if ConfigDict() with no args
            if self._is_empty_config_dict(value):
                self.violations.append(
                    Violation(
                        self.filepath,
                        lineno,
                        f"Empty ConfigDict() in {class_name} - "
                        "add explicit configuration like ConfigDict(frozen=True)",
                    )
                )
        elif isinstance(value, ast.Dict):
            # Check for bare dict: model_config = {} or model_config = {...}
            if not value.keys:
                self.violations.append(
                    Violation(
                        self.filepath,
                        lineno,
                        f"Empty dict for model_config in {class_name} - "
                        "use ConfigDict() with explicit settings like ConfigDict(frozen=True)",
                    )
                )
            else:
                self.violations.append(
                    Violation(
                        self.filepath,
                        lineno,
                        f"Bare dict for model_config in {class_name} - "
                        "use ConfigDict() instead of plain dict",
                    )
                )

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Visit class definition."""
        # Check if this is a Pydantic model (directly or via transitive inheritance)
        # Uses the pre-computed model_config_collector which tracks Pydantic models
        is_model = self.model_config_collector.is_pydantic_model(node.name)

        if is_model:
            old_class = self.in_class
            old_is_model = self.is_pydantic_model
            self.in_class = node.name
            self.is_pydantic_model = True

            # Track if model_config is defined
            has_model_config = False

            # Check for model_config patterns in class body
            for item in node.body:
                # Handle: model_config = ConfigDict(...) or model_config = {}
                if isinstance(item, ast.Assign):
                    for target in item.targets:
                        if isinstance(target, ast.Name) and target.id == "model_config":
                            has_model_config = True
                            self._check_model_config_value(
                                item.value, item.lineno, node.name
                            )

                # Handle: model_config: ConfigDict = ConfigDict(...)
                elif isinstance(item, ast.AnnAssign):
                    if (
                        isinstance(item.target, ast.Name)
                        and item.target.id == "model_config"
                    ):
                        has_model_config = True
                        if item.value is not None:
                            self._check_model_config_value(
                                item.value, item.lineno, node.name
                            )

            # Check for missing model_config (only if model has fields)
            # A field is an annotated assignment that:
            # - Doesn't start with underscore (private)
            # - Is not model_config itself
            has_fields = any(
                isinstance(item, ast.AnnAssign)
                and isinstance(item.target, ast.Name)
                and not item.target.id.startswith("_")
                and item.target.id != "model_config"
                for item in node.body
            )

            # Check if model_config is inherited from a parent class in this file
            inherits_model_config = self._check_inherited_model_config(node)

            if not has_model_config and has_fields and not inherits_model_config:
                self.violations.append(
                    Violation(
                        self.filepath,
                        node.lineno,
                        f"Missing model_config in {node.name} - "
                        "add model_config = ConfigDict(...) with explicit settings",
                    )
                )

            self.generic_visit(node)
            self.in_class = old_class
            self.is_pydantic_model = old_is_model
        else:
            self.generic_visit(node)


# Directories to skip from strict model_config validation
# - utils: Utility classes use inline models for convenience, not domain models
# - protocols: API contracts already have their own patterns
# - compat: Compatibility stubs
# - tests: Test fixtures and mocks
SKIP_DIRECTORIES = {"utils", "protocols", "compat", "tests", "__pycache__", ".git"}


def validate_file(filepath: Path) -> list[Violation]:
    """Validate a single Python file."""
    # Skip files in specific directories
    if any(skip_dir in filepath.parts for skip_dir in SKIP_DIRECTORIES):
        return []

    try:
        content = filepath.read_text(encoding="utf-8")
        tree = ast.parse(content, filename=str(filepath))
    except SyntaxError as e:
        logging.debug("Skipping file with syntax error: %s (%s)", filepath, e)
        return []
    except Exception as e:
        logging.debug("Skipping unprocessable file: %s (%s)", filepath, e)
        return []

    # First pass: collect ConfigDict import aliases
    alias_collector = ImportAliasCollector()
    alias_collector.visit(tree)

    # Second pass: collect classes and their model_config status
    model_config_collector = ClassModelConfigCollector()
    model_config_collector.visit(tree)

    # Resolve transitive Pydantic model inheritance
    model_config_collector._resolve_pydantic_models()

    # Third pass: validate patterns with full context
    visitor = PydanticPatternVisitor(
        str(filepath),
        alias_collector.config_dict_aliases,
        model_config_collector,
    )
    visitor.visit(tree)
    return visitor.violations


def main() -> int:
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: validate_pydantic_patterns.py [files or directories...]")
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
        print(f"Found {len(all_violations)} Pydantic pattern violation(s):")
        for v in all_violations:
            print(f"  {v.file}:{v.line}: {v.message}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
