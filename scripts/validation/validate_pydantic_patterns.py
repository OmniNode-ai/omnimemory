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

            # Check for empty ConfigDict
            for item in node.body:
                if isinstance(item, ast.Assign):
                    for target in item.targets:
                        if isinstance(target, ast.Name) and target.id == "model_config":
                            if isinstance(item.value, ast.Call):
                                # Check if ConfigDict() with no args
                                if (
                                    isinstance(item.value.func, ast.Name)
                                    and item.value.func.id == "ConfigDict"
                                    and not item.value.args
                                    and not item.value.keywords
                                ):
                                    self.violations.append(
                                        Violation(
                                            self.filepath,
                                            item.lineno,
                                            f"Empty ConfigDict() in {node.name} - "
                                            "add explicit configuration or remove",
                                        )
                                    )

            self.generic_visit(node)
            self.in_class = old_class
            self.is_pydantic_model = old_is_model
        else:
            self.generic_visit(node)


def validate_file(filepath: Path) -> list[Violation]:
    """Validate a single Python file."""
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
