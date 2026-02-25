#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""ONEX Pydantic Pattern Validation.

Validates that Pydantic models follow ONEX conventions:
- Use Field() with descriptions
- Use ConfigDict with proper settings
- No bare model_config = {} assignments
- Identifies Pydantic models via BaseModel/GenericModel/BaseSettings inheritance
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
    """Collect ConfigDict import aliases from a module.

    Recognizes both:
    - ConfigDict from pydantic (for BaseModel classes)
    - SettingsConfigDict from pydantic_settings (for BaseSettings classes)
    """

    def __init__(self) -> None:
        # Include both ConfigDict and SettingsConfigDict as valid config callables
        self.config_dict_aliases: set[str] = {"ConfigDict", "SettingsConfigDict"}

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
                # Handle ConfigDict and SettingsConfigDict (supports aliased imports)
                if alias.name in {"ConfigDict", "SettingsConfigDict"}:
                    actual_name = alias.asname if alias.asname else alias.name
                    self.config_dict_aliases.add(actual_name)
        self.generic_visit(node)


class ClassModelConfigCollector(ast.NodeVisitor):
    """Collect which classes have model_config and are Pydantic models."""

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
        # Use explicit flag for clarity instead of error-prone for-else-break pattern
        model_config_found = False
        for item in node.body:
            if model_config_found:
                break
            if isinstance(item, ast.Assign):
                for target in item.targets:
                    if isinstance(target, ast.Name) and target.id == "model_config":
                        self.classes_with_model_config.add(node.name)
                        model_config_found = True
                        break
            elif isinstance(item, ast.AnnAssign):
                if (
                    isinstance(item.target, ast.Name)
                    and item.target.id == "model_config"
                ):
                    # Only count as having model_config if there's an actual value
                    # Annotation-only (e.g., model_config: ConfigDict) is incomplete
                    if item.value is not None:
                        self.classes_with_model_config.add(node.name)
                        model_config_found = True
                        break

        self.generic_visit(node)

    def _trace_inheritance_chain(
        self, class_name: str, visited: set[str] | None = None
    ) -> tuple[list[str], bool]:
        """Trace the inheritance chain for a class.

        Args:
            class_name: The class to trace inheritance for.
            visited: Set of already visited classes (for cycle detection).

        Returns:
            A tuple of (chain, is_cycle) where:
            - chain: List of class names in the inheritance path
            - is_cycle: True if a cycle was detected
        """
        if visited is None:
            visited = set()

        chain = [class_name]

        if class_name in visited:
            return chain, True

        visited.add(class_name)

        bases = self.class_bases.get(class_name, [])
        for base in bases:
            if base in self.class_bases:
                # Base is defined in this file, trace it
                sub_chain, is_cycle = self._trace_inheritance_chain(
                    base, visited.copy()
                )
                if is_cycle:
                    return chain + sub_chain, True
            # If base not in class_bases, it's external (not traceable in this file)

        return chain, False

    def _format_inheritance_chains(self, unresolved: list[str]) -> str:
        """Format inheritance chain information for unresolved classes.

        Args:
            unresolved: List of unresolved class names.

        Returns:
            Formatted string showing inheritance chains for debugging.
        """
        lines = ["Unresolved classes with inheritance chains:"]

        for class_name in unresolved:
            bases = self.class_bases.get(class_name, [])
            chain, is_cycle = self._trace_inheritance_chain(class_name)

            if is_cycle:
                # Format: ClassA -> ClassB -> ClassA (cycle detected)
                cycle_str = " -> ".join(chain)
                lines.append(f"  - {cycle_str} (cycle detected)")
            elif bases:
                # Check which bases are missing (not in this file)
                missing_bases = [b for b in bases if b not in self.class_bases]
                known_bases = [b for b in bases if b in self.class_bases]

                if missing_bases:
                    # Format: ClassD -> [ClassE] (base ClassE not found in file)
                    missing_str = ", ".join(missing_bases)
                    lines.append(
                        f"  - {class_name} -> [{missing_str}] "
                        f"(base(s) not found in file)"
                    )
                elif known_bases:
                    # Bases exist but aren't Pydantic models
                    known_str = ", ".join(known_bases)
                    lines.append(
                        f"  - {class_name} -> [{known_str}] "
                        f"(base(s) not resolved as Pydantic models)"
                    )
                else:
                    lines.append(f"  - {class_name} (no traceable bases)")
            else:
                lines.append(f"  - {class_name} (no base classes)")

        return "\n".join(lines)

    def resolve_pydantic_models(self) -> None:
        """Resolve transitive Pydantic model inheritance within the file.

        A class is a Pydantic model if:
        1. It directly inherits from BaseModel/GenericModel/BaseSettings, OR
        2. It inherits from another class in this file that is a Pydantic model

        Includes protection against circular inheritance to prevent infinite loops.
        Maximum iterations is bounded by the number of classes (each iteration must
        add at least one new class, or the loop terminates).
        """
        # Safety limit: at most one class can be added per iteration,
        # so max iterations equals the number of classes
        max_iterations = len(self.class_bases) + 1
        iteration_count = 0

        changed = True
        while changed:
            # Check iteration limit to protect against circular inheritance
            iteration_count += 1
            if iteration_count > max_iterations:
                # Detect which classes might be involved in a cycle
                unresolved = [
                    name
                    for name in self.class_bases
                    if name not in self.pydantic_models
                ]
                # Format detailed inheritance chain information
                chain_info = self._format_inheritance_chains(unresolved)
                logging.warning(
                    "Circular inheritance detected or resolution limit reached.\n%s\n"
                    "Stopping to prevent infinite loop.",
                    chain_info,
                )
                break

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
        """Check if a node is a ConfigDict() or SettingsConfigDict() call.

        Handles:
        - Direct calls: ConfigDict(), SettingsConfigDict()
        - Aliased calls: CD(), SCD() (if imported with alias)
        - Attribute access: pydantic.ConfigDict(), pydantic_settings...
        """
        if not isinstance(node, ast.Call):
            return False
        func = node.func
        # Handle: ConfigDict(), SettingsConfigDict(), or aliased names like CD()
        if isinstance(func, ast.Name) and func.id in self.config_dict_aliases:
            return True
        # Handle fully-qualified calls via attribute access
        if isinstance(func, ast.Attribute):
            return func.attr in {"ConfigDict", "SettingsConfigDict"}
        return False

    def _is_classvar_annotation(self, annotation: ast.expr | None) -> bool:
        """Check if an annotation is a ClassVar type.

        Handles:
        - ClassVar[T]
        - typing.ClassVar[T]
        """
        if annotation is None:
            return False
        # Handle ClassVar[T] - subscript with ClassVar as value
        if isinstance(annotation, ast.Subscript):
            value = annotation.value
            if isinstance(value, ast.Name) and value.id == "ClassVar":
                return True
            if isinstance(value, ast.Attribute) and value.attr == "ClassVar":
                return True
        # Handle bare ClassVar (without subscript, though less common)
        if isinstance(annotation, ast.Name) and annotation.id == "ClassVar":
            return True
        return isinstance(annotation, ast.Attribute) and annotation.attr == "ClassVar"

    def _is_pydantic_field(self, item: ast.stmt) -> bool:
        """Check if an AST statement is a Pydantic model field.

        A field is an annotated assignment that:
        - Is an ast.AnnAssign
        - Has an ast.Name target (simple variable name)
        - Doesn't start with underscore (private)
        - Is not model_config
        - Is not a ClassVar annotation
        """
        if not isinstance(item, ast.AnnAssign):
            return False
        if not isinstance(item.target, ast.Name):
            return False
        name = item.target.id
        if name.startswith("_"):
            return False
        if name == "model_config":
            return False
        return not self._is_classvar_annotation(item.annotation)

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
        """Check the value assigned to model_config and add violations if needed.

        Detects:
        - Empty ConfigDict() calls (no arguments)
        - Bare dict assignments (model_config = {} or model_config = {...})
        - Non-ConfigDict callables (model_config = SomeOtherCallable())
        - Other invalid value types (variables, constants, etc.)
        """
        if isinstance(value, ast.Call):
            if self._is_config_dict_call(value):
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
            else:
                # Non-ConfigDict callable - extract name for better error message
                func_name = "unknown"
                if isinstance(value.func, ast.Name):
                    func_name = value.func.id
                elif isinstance(value.func, ast.Attribute):
                    func_name = value.func.attr
                self.violations.append(
                    Violation(
                        self.filepath,
                        lineno,
                        f"model_config uses '{func_name}' in {class_name} - "
                        "use ConfigDict() from pydantic instead",
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
                        "use ConfigDict() with explicit settings",
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
        else:
            # Other invalid value types (variables, constants, etc.)
            self.violations.append(
                Violation(
                    self.filepath,
                    lineno,
                    f"model_config has invalid value type in {class_name} - "
                    "use ConfigDict() from pydantic",
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
                        if item.value is not None:
                            # Has annotation AND value assignment
                            has_model_config = True
                            self._check_model_config_value(
                                item.value, item.lineno, node.name
                            )
                        else:
                            # Annotation-only (model_config: ConfigDict) without value
                            # Must have an actual ConfigDict assignment
                            self.violations.append(
                                Violation(
                                    self.filepath,
                                    item.lineno,
                                    f"Annotation-only model_config in {node.name} - "
                                    "add assignment: model_config = ConfigDict(...)",
                                )
                            )

            # Check for missing model_config on ALL Pydantic models
            # ONEX requires explicit model_config even on empty models to ensure
            # configuration is intentional and documented, not just defaulted.
            #
            # Note: We previously had a has_fields gate here, but removed it because:
            # 1. Empty Pydantic models should still have explicit config
            # 2. Consistency - all models follow the same pattern
            # 3. Prevents accidental defaults when fields are added later

            # Check if model_config is inherited from a parent class in this file
            inherits_model_config = self._check_inherited_model_config(node)

            if not has_model_config and not inherits_model_config:
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

    # First pass: collect ConfigDict import aliases
    alias_collector = ImportAliasCollector()
    alias_collector.visit(tree)

    # Second pass: collect classes and their model_config status
    model_config_collector = ClassModelConfigCollector()
    model_config_collector.visit(tree)

    # Resolve transitive Pydantic model inheritance
    model_config_collector.resolve_pydantic_models()

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
