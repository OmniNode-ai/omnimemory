# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""I/O Audit implementation for ONEX node purity enforcement.

This module provides AST-based static analysis to detect I/O violations
in ONEX nodes, enforcing the "pure compute / no I/O" architectural invariant.

Forbidden patterns:
- net-client: Network/DB client imports
- env-access: Environment variable access
- file-io: File system operations

Whitelist Hierarchy
-------------------
The I/O audit uses a two-level whitelist system:

1. **YAML Whitelist (Primary Source of Truth)**:
   - Located at ``tests/audit/io_audit_whitelist.yaml``
   - Defines which files are allowed exceptions and which rules apply
   - ALL exceptions MUST be registered here first

2. **Inline Pragmas (Secondary, Line-Level Granularity)**:
   - Format: ``# io-audit: ignore-next-line <rule>``
   - Only work for files already in the YAML whitelist

Reference: omniintelligence PR #100
Added for OMN-2218: Phase 7 CI Infrastructure Alignment
"""

from __future__ import annotations

import ast
import re
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Literal, overload

import yaml

if TYPE_CHECKING:
    from collections.abc import Sequence

# Directories to audit for I/O violations
IO_AUDIT_TARGETS: list[str] = [
    "src/omnimemory/nodes",
]

# Default whitelist path (relative to repository root)
DEFAULT_WHITELIST_PATH: str = "tests/audit/io_audit_whitelist.yaml"

# Forbidden network/DB client imports (prefix match)
FORBIDDEN_IMPORTS: frozenset[str] = frozenset(
    {
        "aiokafka",
        "confluent_kafka",
        "kafka",
        "asyncpg",
        "httpx",
        "aiohttp",
        "aiofiles",
        "redis",
        "aioredis",
        "psycopg2",
        "psycopg",
    }
)

# Forbidden pathlib I/O method names
PATHLIB_IO_METHODS: frozenset[str] = frozenset(
    {
        "read_text",
        "write_text",
        "read_bytes",
        "write_bytes",
        "open",
    }
)

# Variable name patterns that suggest a Path object
PATHLIB_VARIABLE_PATTERNS: frozenset[str] = frozenset(
    {
        "path",
        "file_path",
        "filepath",
        "dir_path",
        "dirpath",
        "p",
        "fp",
        "source_path",
        "target_path",
        "config_path",
    }
)

# Forbidden logging handler classes
LOGGING_FILE_HANDLERS: frozenset[str] = frozenset(
    {
        "FileHandler",
        "RotatingFileHandler",
        "TimedRotatingFileHandler",
        "WatchedFileHandler",
    }
)

# os.environ mutation/access methods
ENVIRON_MUTATION_METHODS: frozenset[str] = frozenset(
    {
        "get",
        "pop",
        "setdefault",
        "clear",
        "update",
    }
)


class EnumIOAuditRule(Enum):
    """I/O audit rule identifiers."""

    NET_CLIENT = "net-client"
    ENV_ACCESS = "env-access"
    FILE_IO = "file-io"


VALID_RULE_IDS: frozenset[str] = frozenset(r.value for r in EnumIOAuditRule)

REMEDIATION_HINTS: dict[EnumIOAuditRule, str] = {
    EnumIOAuditRule.NET_CLIENT: (
        "Move to an Effect node or inject client via dependency injection."
    ),
    EnumIOAuditRule.ENV_ACCESS: (
        "Pass configuration via constructor parameters instead of reading env vars."
    ),
    EnumIOAuditRule.FILE_IO: (
        "Move file I/O to an Effect node or pass file content as input parameter."
    ),
}


@dataclass(frozen=True)
class ModelIOAuditViolation:
    """Represents a single I/O audit violation."""

    file: Path
    line: int
    column: int
    rule: EnumIOAuditRule
    message: str

    def __str__(self) -> str:
        hint = REMEDIATION_HINTS.get(self.rule, "")
        base = f"{self.file}:{self.line}: {self.rule.value}: {self.message}"
        if hint:
            return f"{base}\n  -> Hint: {hint}"
        return base


@dataclass
class ModelInlinePragma:
    """Represents a parsed inline pragma comment."""

    rule: EnumIOAuditRule
    scope: str
    line: int


@dataclass
class ModelWhitelistEntry:
    """A single whitelist entry for a file or pattern."""

    path: str
    reason: str
    allowed_rules: list[str] = field(default_factory=list)


@dataclass
class ModelWhitelistConfig:
    """Complete whitelist configuration."""

    files: list[ModelWhitelistEntry] = field(default_factory=list)
    schema_version: str = "1.0.0"


@dataclass
class ModelAuditResult:
    """Result of an audit run."""

    violations: list[ModelIOAuditViolation]
    files_scanned: int

    @property
    def is_clean(self) -> bool:
        """Return True if no violations found."""
        return len(self.violations) == 0


# Regex for inline pragma: # io-audit: ignore-next-line <rule>
PRAGMA_PATTERN = re.compile(
    r"#\s*io-audit:\s*ignore-next-line\s+(net-client|env-access|file-io)"
)


def parse_inline_pragma(line: str) -> ModelInlinePragma | None:
    """Parse an inline pragma comment."""
    match = PRAGMA_PATTERN.search(line)
    if match is None:
        return None

    rule_str = match.group(1)
    rule_map = {rule.value: rule for rule in EnumIOAuditRule}
    rule = rule_map.get(rule_str)
    if rule is None:
        return None

    return ModelInlinePragma(rule=rule, scope="next-line", line=0)


class IOAuditVisitor(ast.NodeVisitor):
    """AST visitor that detects I/O violations in Python source files."""

    def __init__(
        self,
        file_path: Path,
        source_lines: list[str] | None = None,
        *,
        honor_inline_pragmas: bool = False,
    ) -> None:
        self.file_path = file_path
        self.source_lines = source_lines or []
        self.violations: list[ModelIOAuditViolation] = []
        self._honor_inline_pragmas = honor_inline_pragmas
        self._pragmas: dict[int, ModelInlinePragma] = {}
        self._imported_names: dict[str, str] = {}
        self._in_type_checking_block: bool = False
        self._type_checking_module_aliases: set[str] = set()
        self._type_checking_constant_aliases: set[str] = set()

        self._parse_pragmas()

    def _parse_pragmas(self) -> None:
        for i, line in enumerate(self.source_lines, start=1):
            pragma = parse_inline_pragma(line)
            if pragma is not None:
                pragma = ModelInlinePragma(rule=pragma.rule, scope=pragma.scope, line=i)
                self._pragmas[i] = pragma

    def _is_whitelisted_by_pragma(self, line: int, rule: EnumIOAuditRule) -> bool:
        if not self._honor_inline_pragmas:
            return False
        pragma = self._pragmas.get(line - 1)
        return (
            pragma is not None and pragma.scope == "next-line" and pragma.rule == rule
        )

    def _add_violation(
        self,
        node: ast.AST,
        rule: EnumIOAuditRule,
        message: str,
    ) -> None:
        line = getattr(node, "lineno", 0)
        col = getattr(node, "col_offset", 0)

        if self._is_whitelisted_by_pragma(line, rule):
            return

        self.violations.append(
            ModelIOAuditViolation(
                file=self.file_path,
                line=line,
                column=col,
                rule=rule,
                message=message,
            )
        )

    def _is_type_checking_guard(self, node: ast.If) -> bool:
        """Detect if an If node is a TYPE_CHECKING guard.

        Handles:
        - ``if TYPE_CHECKING:`` (direct import)
        - ``if TC:`` (when ``from typing import TYPE_CHECKING as TC``)
        - ``if typing.TYPE_CHECKING:`` (module-qualified)
        - ``if t.TYPE_CHECKING:`` (when ``import typing as t``)
        """
        test = node.test

        if isinstance(test, ast.Name) and (
            test.id == "TYPE_CHECKING"
            or test.id in self._type_checking_constant_aliases
        ):
            return True

        if isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING":
            if isinstance(test.value, ast.Name):
                if (
                    test.value.id == "typing"
                    or test.value.id in self._type_checking_module_aliases
                ):
                    return True
            return False

        return False

    def visit_If(self, node: ast.If) -> None:
        """Handle If statements, detecting TYPE_CHECKING guards."""
        if self._is_type_checking_guard(node):
            old_state = self._in_type_checking_block
            self._in_type_checking_block = True
            for child in node.body:
                self.visit(child)
            self._in_type_checking_block = old_state
            for child in node.orelse:
                self.visit(child)
        else:
            self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            module = alias.name
            asname = alias.asname or alias.name
            self._imported_names[asname] = module

            # Track typing module aliases for TYPE_CHECKING detection
            if module == "typing" and alias.asname:
                self._type_checking_module_aliases.add(alias.asname)

            if self._in_type_checking_block:
                continue

            for forbidden in FORBIDDEN_IMPORTS:
                if module == forbidden or module.startswith(f"{forbidden}."):
                    self._add_violation(
                        node,
                        EnumIOAuditRule.NET_CLIENT,
                        f"Forbidden import: {module}",
                    )
                    break

        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""

        for alias in node.names:
            asname = alias.asname or alias.name
            self._imported_names[asname] = f"{module}.{alias.name}"

            # Track TYPE_CHECKING constant aliases (e.g., from typing import TYPE_CHECKING as TC)
            if alias.name == "TYPE_CHECKING" and alias.asname:
                self._type_checking_constant_aliases.add(alias.asname)

        # Skip violation reporting inside TYPE_CHECKING blocks
        if self._in_type_checking_block:
            self.generic_visit(node)
            return

        for forbidden in FORBIDDEN_IMPORTS:
            if module == forbidden or module.startswith(f"{forbidden}."):
                self._add_violation(
                    node,
                    EnumIOAuditRule.NET_CLIENT,
                    f"Forbidden import: from {module}",
                )
                self.generic_visit(node)
                return

        if module in ("logging", "logging.handlers"):
            for alias in node.names:
                if alias.name in LOGGING_FILE_HANDLERS:
                    self._add_violation(
                        node,
                        EnumIOAuditRule.FILE_IO,
                        f"Forbidden import: {alias.name} from {module}",
                    )

        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        self._check_call_for_open(node)
        self._check_call_for_env_access(node)
        self._check_call_for_pathlib_io(node)
        self._check_call_for_logging_handler(node)
        self.generic_visit(node)

    def _check_call_for_open(self, node: ast.Call) -> None:
        func = node.func
        if isinstance(func, ast.Name) and func.id == "open":
            self._add_violation(node, EnumIOAuditRule.FILE_IO, "Forbidden call: open()")
            return

        if isinstance(func, ast.Attribute):
            if func.attr == "open" and isinstance(func.value, ast.Name):
                if func.value.id == "io":
                    self._add_violation(
                        node, EnumIOAuditRule.FILE_IO, "Forbidden call: io.open()"
                    )

    def _check_call_for_env_access(self, node: ast.Call) -> None:
        func = node.func
        if isinstance(func, ast.Attribute):
            if isinstance(func.value, ast.Name) and func.value.id == "os":
                if func.attr == "getenv":
                    self._add_violation(
                        node, EnumIOAuditRule.ENV_ACCESS, "Forbidden call: os.getenv()"
                    )
                elif func.attr == "putenv":
                    self._add_violation(
                        node, EnumIOAuditRule.ENV_ACCESS, "Forbidden call: os.putenv()"
                    )
            elif isinstance(func.value, ast.Attribute):
                if (
                    isinstance(func.value.value, ast.Name)
                    and func.value.value.id == "os"
                    and func.value.attr == "environ"
                    and func.attr in ENVIRON_MUTATION_METHODS
                ):
                    self._add_violation(
                        node,
                        EnumIOAuditRule.ENV_ACCESS,
                        f"Forbidden call: os.environ.{func.attr}()",
                    )

    def _has_pathlib_import(self) -> bool:
        for alias, module in self._imported_names.items():
            if module == "pathlib" or module.startswith("pathlib."):
                return True
            if alias in {"Path", "pathlib"}:
                return True
        return False

    def _is_likely_path_object(self, node: ast.expr) -> bool:
        """Heuristically determine whether *node* likely refers to a Path object.

        This uses variable-name pattern matching (e.g. names ending in
        ``_path`` or matching known patterns like ``fp``, ``file_path``).
        Because it relies on naming conventions rather than type inference,
        it can produce false positives (a variable named ``path`` that is
        actually a ``str``) and false negatives (a ``Path`` object stored
        in a variable with an unconventional name like ``destination``).
        """
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == "Path":
                return True
        if isinstance(node, ast.Name):
            var_name = node.id.lower()
            if var_name in PATHLIB_VARIABLE_PATTERNS:
                return True
            if var_name.endswith(("_path", "path")):
                return True
        if isinstance(node, ast.Attribute):
            attr_name = node.attr.lower()
            if attr_name in PATHLIB_VARIABLE_PATTERNS:
                return True
            if attr_name.endswith(("_path", "path")):
                return True
        return False

    def _check_call_for_pathlib_io(self, node: ast.Call) -> None:
        func = node.func
        if isinstance(func, ast.Attribute):
            if func.attr in PATHLIB_IO_METHODS:
                if not self._has_pathlib_import():
                    return
                if not self._is_likely_path_object(func.value):
                    return
                self._add_violation(
                    node,
                    EnumIOAuditRule.FILE_IO,
                    f"Forbidden call: Path.{func.attr}()",
                )

    def _check_call_for_logging_handler(self, node: ast.Call) -> None:
        func = node.func
        if isinstance(func, ast.Name) and func.id in LOGGING_FILE_HANDLERS:
            self._add_violation(
                node, EnumIOAuditRule.FILE_IO, f"Forbidden call: {func.id}()"
            )
        elif isinstance(func, ast.Attribute) and func.attr in LOGGING_FILE_HANDLERS:
            self._add_violation(
                node, EnumIOAuditRule.FILE_IO, f"Forbidden call: {func.attr}()"
            )

    def visit_Subscript(self, node: ast.Subscript) -> None:
        value = node.value
        if isinstance(value, ast.Attribute):
            if (
                isinstance(value.value, ast.Name)
                and value.value.id == "os"
                and value.attr == "environ"
            ):
                self._add_violation(
                    node,
                    EnumIOAuditRule.ENV_ACCESS,
                    "Forbidden access: os.environ[...]",
                )
        self.generic_visit(node)


@overload
def audit_file(
    file_path: Path,
    *,
    return_source_lines: Literal[False] = False,
) -> list[ModelIOAuditViolation]: ...


@overload
def audit_file(
    file_path: Path,
    *,
    return_source_lines: Literal[True],
) -> tuple[list[ModelIOAuditViolation], list[str]]: ...


def audit_file(
    file_path: Path,
    *,
    return_source_lines: bool = False,
) -> list[ModelIOAuditViolation] | tuple[list[ModelIOAuditViolation], list[str]]:
    """Audit a single Python file for I/O violations."""
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    source = file_path.read_text(encoding="utf-8")
    source_lines = source.splitlines()

    tree = ast.parse(source, filename=str(file_path))

    visitor = IOAuditVisitor(file_path, source_lines)
    visitor.visit(tree)

    if return_source_lines:
        return visitor.violations, source_lines
    return visitor.violations


def load_whitelist(path: Path) -> ModelWhitelistConfig:
    """Load whitelist configuration from a YAML file."""
    if not path.exists():
        return ModelWhitelistConfig()

    content = path.read_text(encoding="utf-8")
    data = yaml.safe_load(content) or {}

    files: list[ModelWhitelistEntry] = []
    for entry in data.get("files") or []:
        whitelist_entry = ModelWhitelistEntry(
            path=entry.get("path", ""),
            reason=entry.get("reason", ""),
            allowed_rules=entry.get("allowed_rules", []),
        )
        files.append(whitelist_entry)

    return ModelWhitelistConfig(
        files=files,
        schema_version=data.get("schema_version", "1.0.0"),
    )


def _matches_whitelist_entry(file_str: str, file_path: Path, entry_path: str) -> bool:
    """Check if a file matches a whitelist entry."""
    if "*" in entry_path or "?" in entry_path:
        return file_path.match(entry_path) or file_path.match(f"**/{entry_path}")

    if file_str == entry_path:
        return True

    file_name = file_path.name
    entry_name = Path(entry_path).name
    if file_name == entry_name and entry_name == entry_path:
        return True

    return file_str.endswith((f"/{entry_path}", f"\\{entry_path}"))


def apply_whitelist(
    violations: list[ModelIOAuditViolation],
    whitelist: ModelWhitelistConfig,
    file_path: Path,
    source_lines: list[str] | None = None,
) -> list[ModelIOAuditViolation]:
    """Filter violations based on whitelist configuration."""
    if not violations:
        return violations

    file_str = str(file_path)

    allowed_rules: set[str] = set()
    file_in_whitelist = False

    for entry in whitelist.files:
        if _matches_whitelist_entry(file_str, file_path, entry.path):
            file_in_whitelist = True
            allowed_rules.update(entry.allowed_rules)

    if not file_in_whitelist:
        return violations

    pragma_whitelist: dict[int, EnumIOAuditRule] = {}
    if source_lines:
        for i, line in enumerate(source_lines, start=1):
            pragma = parse_inline_pragma(line)
            if pragma is not None:
                pragma_whitelist[i + 1] = pragma.rule

    remaining: list[ModelIOAuditViolation] = []

    for v in violations:
        if v.rule.value in allowed_rules:
            continue
        if v.line in pragma_whitelist and pragma_whitelist[v.line] == v.rule:
            continue
        remaining.append(v)

    return remaining


def discover_python_files(targets: Sequence[str]) -> list[Path]:
    """Discover Python files in the target directories."""
    files: set[Path] = set()

    for target in targets:
        target_path = Path(target)
        if target_path.exists() and target_path.is_dir():
            for py_file in target_path.rglob("*.py"):
                try:
                    canonical = py_file.resolve()
                    if canonical.is_file():
                        files.add(canonical)
                except (OSError, RuntimeError):
                    pass

    return sorted(files)


def run_audit(
    targets: Sequence[str] | None = None,
    whitelist_path: Path | None = None,
) -> ModelAuditResult:
    """Run the full I/O audit on target directories.

    Args:
        targets: List of directory paths to audit. Defaults to IO_AUDIT_TARGETS.
        whitelist_path: Path to whitelist YAML. Optional.

    Returns:
        Audit result with violations and metadata.
    """
    if targets is None:
        targets = IO_AUDIT_TARGETS

    files = discover_python_files(targets)

    whitelist = ModelWhitelistConfig()
    if whitelist_path is not None and whitelist_path.exists():
        whitelist = load_whitelist(whitelist_path)

    all_violations: list[ModelIOAuditViolation] = []

    for file_path in files:
        violations, source_lines = audit_file(file_path, return_source_lines=True)
        remaining = apply_whitelist(violations, whitelist, file_path, source_lines)
        all_violations.extend(remaining)

    return ModelAuditResult(
        violations=all_violations,
        files_scanned=len(files),
    )


if __name__ == "__main__":
    from omnimemory.audit.__main__ import main

    sys.exit(main())
