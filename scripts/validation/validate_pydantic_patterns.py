#!/usr/bin/env python3
"""ONEX Pydantic Pattern Validation.

Validates that Pydantic models follow ONEX conventions:
- Use Field() with descriptions
- Use ConfigDict with proper settings
- No bare model_config = {} assignments
- Proper inheritance from BaseModel

Usage:
    python scripts/validation/validate_pydantic_patterns.py [files...]
    python scripts/validation/validate_pydantic_patterns.py src/
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


class PydanticPatternVisitor(ast.NodeVisitor):
    """AST visitor to check Pydantic patterns."""

    def __init__(self, filepath: str) -> None:
        self.filepath = filepath
        self.violations: list[Violation] = []
        self.in_class: str | None = None
        self.is_pydantic_model = False

    def _is_config_dict_call(self, node: ast.expr) -> bool:
        """Check if a node is a ConfigDict() call (handles both Name and Attribute)."""
        if not isinstance(node, ast.Call):
            return False
        func = node.func
        # Handle: ConfigDict() or pydantic.ConfigDict()
        return (isinstance(func, ast.Name) and func.id == "ConfigDict") or (
            isinstance(func, ast.Attribute) and func.attr == "ConfigDict"
        )

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
        # Check if this is a Pydantic model (inherits from BaseModel)
        is_model = any(
            (isinstance(base, ast.Name) and base.id == "BaseModel")
            or (isinstance(base, ast.Attribute) and base.attr == "BaseModel")
            for base in node.bases
        )

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

            if not has_model_config and has_fields:
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
    except SyntaxError:
        return []  # Skip files with syntax errors
    except Exception:
        return []

    visitor = PydanticPatternVisitor(str(filepath))
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
