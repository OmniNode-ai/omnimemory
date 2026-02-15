# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for the I/O audit module (ONEX node purity enforcement).

Tests validate that ``IOAuditVisitor`` and associated functions from
``src/omnimemory/audit/io_audit.py`` correctly detect I/O violations via
AST-based static analysis, honour whitelist configuration and inline pragmas,
and that the CLI entry point in ``__main__.py`` produces correct output.

Coverage targets:
- IOAuditVisitor: forbidden imports, open() calls, env access, pathlib I/O,
  logging handlers, TYPE_CHECKING guard handling
- Whitelist: YAML loading, rule filtering, glob matching
- Inline pragmas: parsing, next-line suppression
- audit_file / run_audit: end-to-end single-file and directory auditing
- __main__: text output, JSON output, exit codes
"""

from __future__ import annotations

import ast
import json
import textwrap
from pathlib import Path

import pytest
import yaml

from omnimemory.audit.__main__ import (
    _format_json_output,
    _format_text_output,
    main,
)
from omnimemory.audit.io_audit import (
    FORBIDDEN_IMPORTS,
    LOGGING_FILE_HANDLERS,
    PATHLIB_IO_METHODS,
    EnumIOAuditRule,
    IOAuditVisitor,
    ModelAuditResult,
    ModelIOAuditViolation,
    ModelWhitelistConfig,
    ModelWhitelistEntry,
    apply_whitelist,
    audit_file,
    load_whitelist,
    parse_inline_pragma,
    run_audit,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _visit(
    source: str, file_name: str = "test.py", **kwargs: object
) -> list[ModelIOAuditViolation]:
    """Parse source and run IOAuditVisitor, returning violations."""
    source_lines = source.splitlines()
    tree = ast.parse(source, filename=file_name)
    visitor = IOAuditVisitor(
        Path(file_name),
        source_lines,
        **kwargs,  # type: ignore[arg-type]
    )
    visitor.visit(tree)
    return visitor.violations


def _write_py(tmp_path: Path, content: str, name: str = "target.py") -> Path:
    """Write a Python file under tmp_path and return its path."""
    target = tmp_path / name
    target.write_text(textwrap.dedent(content), encoding="utf-8")
    return target


# ============================================================================
# IOAuditVisitor -- Forbidden Imports (net-client)
# ============================================================================


@pytest.mark.unit
class TestForbiddenImports:
    """Detect forbidden network/DB client imports."""

    @pytest.mark.parametrize(
        ("source", "expected_module"),
        [
            pytest.param(f"import {mod}\n", mod, id=f"import-{mod}")
            for mod in sorted(FORBIDDEN_IMPORTS)
        ],
    )
    def test_import_statement_flagged(self, source: str, expected_module: str) -> None:
        violations = _visit(source)
        assert len(violations) == 1
        assert violations[0].rule == EnumIOAuditRule.NET_CLIENT
        assert expected_module in violations[0].message

    @pytest.mark.parametrize(
        ("source", "expected_module"),
        [
            pytest.param(
                f"from {mod} import something\n",
                mod,
                id=f"from-{mod}",
            )
            for mod in sorted(FORBIDDEN_IMPORTS)
        ],
    )
    def test_from_import_flagged(self, source: str, expected_module: str) -> None:
        violations = _visit(source)
        assert len(violations) == 1
        assert violations[0].rule == EnumIOAuditRule.NET_CLIENT
        assert expected_module in violations[0].message

    def test_submodule_import_flagged(self) -> None:
        violations = _visit("import httpx.auth\n")
        assert len(violations) == 1
        assert violations[0].rule == EnumIOAuditRule.NET_CLIENT

    def test_from_submodule_import_flagged(self) -> None:
        violations = _visit("from kafka.errors import KafkaError\n")
        assert len(violations) == 1
        assert violations[0].rule == EnumIOAuditRule.NET_CLIENT

    def test_non_forbidden_import_clean(self) -> None:
        source = "import os\nimport json\nfrom pathlib import Path\n"
        violations = _visit(source)
        assert violations == []


# ============================================================================
# IOAuditVisitor -- TYPE_CHECKING Guard
# ============================================================================


@pytest.mark.unit
class TestTypeCheckingGuard:
    """Imports inside TYPE_CHECKING blocks are allowed."""

    def test_direct_type_checking_guard(self) -> None:
        source = textwrap.dedent("""\
            from typing import TYPE_CHECKING

            if TYPE_CHECKING:
                import httpx
        """)
        violations = _visit(source)
        assert violations == []

    def test_aliased_type_checking_constant(self) -> None:
        source = textwrap.dedent("""\
            from typing import TYPE_CHECKING as TC

            if TC:
                import redis
        """)
        violations = _visit(source)
        assert violations == []

    def test_module_qualified_type_checking(self) -> None:
        source = textwrap.dedent("""\
            import typing

            if typing.TYPE_CHECKING:
                import aiohttp
        """)
        violations = _visit(source)
        assert violations == []

    def test_aliased_module_type_checking(self) -> None:
        source = textwrap.dedent("""\
            import typing as t

            if t.TYPE_CHECKING:
                from asyncpg import Connection
        """)
        violations = _visit(source)
        assert violations == []

    def test_else_branch_is_runtime(self) -> None:
        source = textwrap.dedent("""\
            from typing import TYPE_CHECKING

            if TYPE_CHECKING:
                import httpx
            else:
                import httpx
        """)
        violations = _visit(source)
        assert len(violations) == 1
        assert violations[0].rule == EnumIOAuditRule.NET_CLIENT
        assert violations[0].line == 6

    def test_outside_type_checking_flagged(self) -> None:
        source = textwrap.dedent("""\
            from typing import TYPE_CHECKING

            if TYPE_CHECKING:
                import httpx

            import redis
        """)
        violations = _visit(source)
        assert len(violations) == 1
        assert "redis" in violations[0].message


# ============================================================================
# IOAuditVisitor -- open() / io.open() (file-io)
# ============================================================================


@pytest.mark.unit
class TestOpenCallDetection:
    """Detect builtin open() and io.open() calls."""

    def test_builtin_open(self) -> None:
        violations = _visit("f = open('data.txt')\n")
        assert len(violations) == 1
        assert violations[0].rule == EnumIOAuditRule.FILE_IO
        assert "open()" in violations[0].message

    def test_io_open(self) -> None:
        violations = _visit("import io\nf = io.open('data.txt')\n")
        assert len(violations) == 1
        assert violations[0].rule == EnumIOAuditRule.FILE_IO
        assert "io.open()" in violations[0].message

    def test_method_named_open_not_flagged(self) -> None:
        """A method .open() on a non-io object should not flag (unless pathlib)."""
        source = textwrap.dedent("""\
            class Connection:
                def open(self): ...

            c = Connection()
            c.open()
        """)
        violations = _visit(source)
        assert violations == []


# ============================================================================
# IOAuditVisitor -- Environment Variable Access (env-access)
# ============================================================================


@pytest.mark.unit
class TestEnvAccessDetection:
    """Detect os.getenv, os.putenv, os.environ[...] and os.environ.get(...)."""

    def test_os_getenv(self) -> None:
        violations = _visit("import os\nv = os.getenv('KEY')\n")
        assert len(violations) == 1
        assert violations[0].rule == EnumIOAuditRule.ENV_ACCESS
        assert "os.getenv()" in violations[0].message

    def test_os_putenv(self) -> None:
        violations = _visit("import os\nos.putenv('KEY', 'val')\n")
        assert len(violations) == 1
        assert violations[0].rule == EnumIOAuditRule.ENV_ACCESS
        assert "os.putenv()" in violations[0].message

    def test_os_environ_subscript(self) -> None:
        violations = _visit("import os\nv = os.environ['KEY']\n")
        assert len(violations) == 1
        assert violations[0].rule == EnumIOAuditRule.ENV_ACCESS
        assert "os.environ[...]" in violations[0].message

    @pytest.mark.parametrize(
        "method",
        ["get", "pop", "setdefault", "clear", "update"],
        ids=lambda m: f"environ-{m}",
    )
    def test_os_environ_method(self, method: str) -> None:
        source = f"import os\nos.environ.{method}('KEY')\n"
        violations = _visit(source)
        assert len(violations) == 1
        assert violations[0].rule == EnumIOAuditRule.ENV_ACCESS
        assert f"os.environ.{method}()" in violations[0].message


# ============================================================================
# IOAuditVisitor -- Pathlib I/O Heuristic (file-io)
# ============================================================================


@pytest.mark.unit
class TestPathlibIODetection:
    """Detect pathlib read/write methods when Path is imported and variable looks path-like."""

    @pytest.mark.parametrize("method", sorted(PATHLIB_IO_METHODS))
    def test_pathlib_io_with_path_constructor(self, method: str) -> None:
        source = textwrap.dedent(f"""\
            from pathlib import Path
            data = Path("file.txt").{method}()
        """)
        violations = _visit(source)
        assert len(violations) == 1
        assert violations[0].rule == EnumIOAuditRule.FILE_IO
        assert f"Path.{method}()" in violations[0].message

    @pytest.mark.parametrize(
        "var_name",
        ["path", "file_path", "filepath", "config_path", "source_path"],
    )
    def test_pathlib_io_with_path_variable(self, var_name: str) -> None:
        source = textwrap.dedent(f"""\
            from pathlib import Path
            {var_name} = Path("file.txt")
            data = {var_name}.read_text()
        """)
        violations = _visit(source)
        assert len(violations) == 1
        assert violations[0].rule == EnumIOAuditRule.FILE_IO

    def test_pathlib_io_with_custom_path_suffix_variable(self) -> None:
        source = textwrap.dedent("""\
            from pathlib import Path
            output_path = Path("out.txt")
            output_path.write_text("data")
        """)
        violations = _visit(source)
        assert len(violations) == 1
        assert violations[0].rule == EnumIOAuditRule.FILE_IO

    def test_no_pathlib_import_skips_check(self) -> None:
        """Without pathlib import, heuristic does not fire."""
        source = "path = something()\npath.read_text()\n"
        violations = _visit(source)
        assert violations == []

    def test_non_path_variable_skips_check(self) -> None:
        """Variable with non-path name does not trigger."""
        source = textwrap.dedent("""\
            from pathlib import Path
            result = get_result()
            result.read_text()
        """)
        violations = _visit(source)
        assert violations == []


# ============================================================================
# IOAuditVisitor -- Logging File Handlers (file-io)
# ============================================================================


@pytest.mark.unit
class TestLoggingHandlerDetection:
    """Detect forbidden logging file handler imports and calls."""

    @pytest.mark.parametrize("handler", sorted(LOGGING_FILE_HANDLERS))
    def test_import_from_logging(self, handler: str) -> None:
        source = f"from logging import {handler}\n"
        violations = _visit(source)
        assert len(violations) == 1
        assert violations[0].rule == EnumIOAuditRule.FILE_IO
        assert handler in violations[0].message

    @pytest.mark.parametrize("handler", sorted(LOGGING_FILE_HANDLERS))
    def test_import_from_logging_handlers(self, handler: str) -> None:
        source = f"from logging.handlers import {handler}\n"
        violations = _visit(source)
        assert len(violations) == 1
        assert violations[0].rule == EnumIOAuditRule.FILE_IO

    @pytest.mark.parametrize("handler", sorted(LOGGING_FILE_HANDLERS))
    def test_direct_call(self, handler: str) -> None:
        source = f"from logging import {handler}\n{handler}('/var/log/app.log')\n"
        violations = _visit(source)
        # One for the import, one for the call
        assert len(violations) == 2
        assert all(v.rule == EnumIOAuditRule.FILE_IO for v in violations)


# ============================================================================
# Inline Pragma Parsing
# ============================================================================


@pytest.mark.unit
class TestInlinePragmaParsing:
    """Test parse_inline_pragma function."""

    @pytest.mark.parametrize(
        ("line", "expected_rule"),
        [
            pytest.param(
                "# io-audit: ignore-next-line net-client",
                EnumIOAuditRule.NET_CLIENT,
                id="net-client",
            ),
            pytest.param(
                "# io-audit: ignore-next-line env-access",
                EnumIOAuditRule.ENV_ACCESS,
                id="env-access",
            ),
            pytest.param(
                "# io-audit: ignore-next-line file-io",
                EnumIOAuditRule.FILE_IO,
                id="file-io",
            ),
            pytest.param(
                "  # io-audit: ignore-next-line file-io  ",
                EnumIOAuditRule.FILE_IO,
                id="with-whitespace",
            ),
        ],
    )
    def test_valid_pragma(self, line: str, expected_rule: EnumIOAuditRule) -> None:
        result = parse_inline_pragma(line)
        assert result is not None
        assert result.rule == expected_rule
        assert result.scope == "next-line"

    @pytest.mark.parametrize(
        "line",
        [
            pytest.param("# just a comment", id="plain-comment"),
            pytest.param(
                "# io-audit: ignore-next-line unknown-rule", id="invalid-rule"
            ),
            pytest.param("x = 42", id="code-line"),
            pytest.param("", id="empty"),
        ],
    )
    def test_invalid_pragma_returns_none(self, line: str) -> None:
        assert parse_inline_pragma(line) is None


# ============================================================================
# Inline Pragma Suppression in Visitor
# ============================================================================


@pytest.mark.unit
class TestInlinePragmaSuppression:
    """Pragmas on the line before a violation suppress that violation."""

    def test_pragma_suppresses_next_line(self) -> None:
        source = textwrap.dedent("""\
            # io-audit: ignore-next-line net-client
            import httpx
        """)
        violations = _visit(source, honor_inline_pragmas=True)
        assert violations == []

    def test_pragma_wrong_rule_does_not_suppress(self) -> None:
        source = textwrap.dedent("""\
            # io-audit: ignore-next-line file-io
            import httpx
        """)
        violations = _visit(source, honor_inline_pragmas=True)
        assert len(violations) == 1
        assert violations[0].rule == EnumIOAuditRule.NET_CLIENT

    def test_pragma_without_honor_flag_does_not_suppress(self) -> None:
        source = textwrap.dedent("""\
            # io-audit: ignore-next-line net-client
            import httpx
        """)
        violations = _visit(source, honor_inline_pragmas=False)
        assert len(violations) == 1

    def test_pragma_suppresses_file_io(self) -> None:
        source = textwrap.dedent("""\
            # io-audit: ignore-next-line file-io
            f = open('data.txt')
        """)
        violations = _visit(source, honor_inline_pragmas=True)
        assert violations == []

    def test_pragma_suppresses_env_access(self) -> None:
        source = textwrap.dedent("""\
            import os
            # io-audit: ignore-next-line env-access
            v = os.getenv('KEY')
        """)
        violations = _visit(source, honor_inline_pragmas=True)
        assert violations == []


# ============================================================================
# Whitelist Loading
# ============================================================================


@pytest.mark.unit
class TestWhitelistLoading:
    """Test load_whitelist from YAML files."""

    def test_load_valid_whitelist(self, tmp_path: Path) -> None:
        wl_path = tmp_path / "whitelist.yaml"
        data = {
            "schema_version": "1.0.0",
            "files": [
                {
                    "path": "src/my_module/effect.py",
                    "reason": "Effect node with expected I/O",
                    "allowed_rules": ["net-client", "file-io"],
                },
            ],
        }
        wl_path.write_text(yaml.dump(data), encoding="utf-8")

        config = load_whitelist(wl_path)

        assert config.schema_version == "1.0.0"
        assert len(config.files) == 1
        assert config.files[0].path == "src/my_module/effect.py"
        assert config.files[0].reason == "Effect node with expected I/O"
        assert config.files[0].allowed_rules == ["net-client", "file-io"]

    def test_load_missing_file_returns_empty(self, tmp_path: Path) -> None:
        config = load_whitelist(tmp_path / "nonexistent.yaml")
        assert config.files == []
        assert config.schema_version == "1.0.0"

    def test_load_empty_yaml_returns_empty(self, tmp_path: Path) -> None:
        wl_path = tmp_path / "empty.yaml"
        wl_path.write_text("", encoding="utf-8")

        config = load_whitelist(wl_path)
        assert config.files == []

    def test_load_whitelist_no_files_key(self, tmp_path: Path) -> None:
        wl_path = tmp_path / "no_files.yaml"
        wl_path.write_text("schema_version: '2.0.0'\n", encoding="utf-8")

        config = load_whitelist(wl_path)
        assert config.files == []
        assert config.schema_version == "2.0.0"


# ============================================================================
# Whitelist Application
# ============================================================================


@pytest.mark.unit
class TestWhitelistApplication:
    """Test apply_whitelist filtering logic."""

    @staticmethod
    def _make_violation(
        file_path: Path,
        rule: EnumIOAuditRule,
        line: int = 1,
    ) -> ModelIOAuditViolation:
        return ModelIOAuditViolation(
            file=file_path,
            line=line,
            column=0,
            rule=rule,
            message=f"Test {rule.value} violation",
        )

    def test_whitelist_removes_matching_rule(self) -> None:
        fp = Path("src/module/effect.py")
        violations = [self._make_violation(fp, EnumIOAuditRule.NET_CLIENT)]
        whitelist = ModelWhitelistConfig(
            files=[
                ModelWhitelistEntry(
                    path="src/module/effect.py",
                    reason="allowed",
                    allowed_rules=["net-client"],
                ),
            ],
        )

        remaining = apply_whitelist(violations, whitelist, fp)
        assert remaining == []

    def test_whitelist_keeps_non_matching_rule(self) -> None:
        fp = Path("src/module/effect.py")
        violations = [self._make_violation(fp, EnumIOAuditRule.FILE_IO)]
        whitelist = ModelWhitelistConfig(
            files=[
                ModelWhitelistEntry(
                    path="src/module/effect.py",
                    reason="allowed",
                    allowed_rules=["net-client"],  # Only net-client allowed
                ),
            ],
        )

        remaining = apply_whitelist(violations, whitelist, fp)
        assert len(remaining) == 1
        assert remaining[0].rule == EnumIOAuditRule.FILE_IO

    def test_whitelist_file_not_listed_keeps_all(self) -> None:
        fp = Path("src/module/unlisted.py")
        violations = [self._make_violation(fp, EnumIOAuditRule.NET_CLIENT)]
        whitelist = ModelWhitelistConfig(
            files=[
                ModelWhitelistEntry(
                    path="src/other/module.py",
                    reason="different file",
                    allowed_rules=["net-client"],
                ),
            ],
        )

        remaining = apply_whitelist(violations, whitelist, fp)
        assert len(remaining) == 1

    def test_empty_violations_returns_empty(self) -> None:
        fp = Path("src/module/clean.py")
        whitelist = ModelWhitelistConfig(
            files=[
                ModelWhitelistEntry(
                    path="src/module/clean.py",
                    reason="allowed",
                    allowed_rules=["net-client"],
                ),
            ],
        )

        remaining = apply_whitelist([], whitelist, fp)
        assert remaining == []

    def test_whitelist_with_inline_pragma(self) -> None:
        fp = Path("src/module/effect.py")
        violations = [self._make_violation(fp, EnumIOAuditRule.FILE_IO, line=3)]
        whitelist = ModelWhitelistConfig(
            files=[
                ModelWhitelistEntry(
                    path="src/module/effect.py",
                    reason="allowed for pragmas",
                    allowed_rules=[],  # No blanket rules
                ),
            ],
        )
        source_lines = [
            "import os",
            "# io-audit: ignore-next-line file-io",
            "f = open('data.txt')",
        ]

        remaining = apply_whitelist(violations, whitelist, fp, source_lines)
        assert remaining == []

    def test_whitelist_path_suffix_matching(self) -> None:
        fp = Path("/abs/src/module/effect.py")
        violations = [self._make_violation(fp, EnumIOAuditRule.NET_CLIENT)]
        whitelist = ModelWhitelistConfig(
            files=[
                ModelWhitelistEntry(
                    path="src/module/effect.py",
                    reason="suffix match",
                    allowed_rules=["net-client"],
                ),
            ],
        )

        remaining = apply_whitelist(violations, whitelist, fp)
        assert remaining == []


# ============================================================================
# audit_file Function
# ============================================================================


@pytest.mark.unit
class TestAuditFile:
    """Test the audit_file function with real files on disk."""

    def test_audit_file_with_violations(self, tmp_path: Path) -> None:
        source = "import httpx\nf = open('x.txt')\n"
        target = _write_py(tmp_path, source)

        violations = audit_file(target)

        assert len(violations) == 2
        rules = {v.rule for v in violations}
        assert EnumIOAuditRule.NET_CLIENT in rules
        assert EnumIOAuditRule.FILE_IO in rules

    def test_audit_file_clean(self, tmp_path: Path) -> None:
        source = "import os\nx = 42\n"
        target = _write_py(tmp_path, source)

        violations = audit_file(target)
        assert violations == []

    def test_audit_file_returns_source_lines(self, tmp_path: Path) -> None:
        source = "import httpx\n"
        target = _write_py(tmp_path, source)

        violations, lines = audit_file(target, return_source_lines=True)

        assert len(violations) == 1
        assert lines == ["import httpx"]

    def test_audit_file_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            audit_file(tmp_path / "missing.py")


# ============================================================================
# run_audit (Directory Auditing)
# ============================================================================


@pytest.mark.unit
class TestRunAudit:
    """Test run_audit with directory-level scanning."""

    def test_run_audit_on_directory(self, tmp_path: Path) -> None:
        target_dir = tmp_path / "nodes"
        target_dir.mkdir()
        (target_dir / "node_a.py").write_text("import httpx\n", encoding="utf-8")
        (target_dir / "node_b.py").write_text("x = 42\n", encoding="utf-8")

        result = run_audit(targets=[str(target_dir)])

        assert result.files_scanned == 2
        assert len(result.violations) == 1
        assert result.violations[0].rule == EnumIOAuditRule.NET_CLIENT
        assert not result.is_clean

    def test_run_audit_clean_directory(self, tmp_path: Path) -> None:
        target_dir = tmp_path / "clean_nodes"
        target_dir.mkdir()
        (target_dir / "pure.py").write_text("x = 42\n", encoding="utf-8")

        result = run_audit(targets=[str(target_dir)])

        assert result.files_scanned == 1
        assert result.violations == []
        assert result.is_clean

    def test_run_audit_empty_directory(self, tmp_path: Path) -> None:
        target_dir = tmp_path / "empty"
        target_dir.mkdir()

        result = run_audit(targets=[str(target_dir)])

        assert result.files_scanned == 0
        assert result.is_clean

    def test_run_audit_with_whitelist(self, tmp_path: Path) -> None:
        target_dir = tmp_path / "nodes"
        target_dir.mkdir()
        (target_dir / "effect.py").write_text("import httpx\n", encoding="utf-8")

        wl_path = tmp_path / "whitelist.yaml"
        # Use the resolved path so it matches
        effect_resolved = (target_dir / "effect.py").resolve()
        wl_data = {
            "schema_version": "1.0.0",
            "files": [
                {
                    "path": str(effect_resolved),
                    "reason": "effect node",
                    "allowed_rules": ["net-client"],
                },
            ],
        }
        wl_path.write_text(yaml.dump(wl_data), encoding="utf-8")

        result = run_audit(targets=[str(target_dir)], whitelist_path=wl_path)

        assert result.files_scanned == 1
        assert result.is_clean

    def test_run_audit_nonexistent_target(self, tmp_path: Path) -> None:
        result = run_audit(targets=[str(tmp_path / "nonexistent")])

        assert result.files_scanned == 0
        assert result.is_clean


# ============================================================================
# ModelAuditResult
# ============================================================================


@pytest.mark.unit
class TestModelAuditResult:
    """Test ModelAuditResult properties."""

    def test_is_clean_true(self) -> None:
        result = ModelAuditResult(violations=[], files_scanned=5)
        assert result.is_clean is True

    def test_is_clean_false(self) -> None:
        violation = ModelIOAuditViolation(
            file=Path("test.py"),
            line=1,
            column=0,
            rule=EnumIOAuditRule.FILE_IO,
            message="test",
        )
        result = ModelAuditResult(violations=[violation], files_scanned=5)
        assert result.is_clean is False


# ============================================================================
# ModelIOAuditViolation __str__
# ============================================================================


@pytest.mark.unit
class TestViolationStr:
    """Test string rendering of violations."""

    def test_str_includes_hint(self) -> None:
        v = ModelIOAuditViolation(
            file=Path("test.py"),
            line=10,
            column=5,
            rule=EnumIOAuditRule.FILE_IO,
            message="Forbidden call: open()",
        )
        text = str(v)
        assert "test.py:10" in text
        assert "file-io" in text
        assert "Hint:" in text

    def test_str_includes_rule_value(self) -> None:
        v = ModelIOAuditViolation(
            file=Path("mod.py"),
            line=3,
            column=0,
            rule=EnumIOAuditRule.NET_CLIENT,
            message="Forbidden import: httpx",
        )
        text = str(v)
        assert "net-client" in text
        assert "mod.py:3" in text


# ============================================================================
# __main__ -- Text Output Formatting
# ============================================================================


@pytest.mark.unit
class TestTextOutputFormatting:
    """Test _format_text_output from __main__.py."""

    def test_clean_result(self) -> None:
        result = ModelAuditResult(violations=[], files_scanned=3)
        output = _format_text_output(result)
        assert "No I/O violations found" in output
        assert "3 files scanned" in output

    def test_violations_grouped_by_file(self) -> None:
        v1 = ModelIOAuditViolation(
            file=Path("a.py"),
            line=1,
            column=0,
            rule=EnumIOAuditRule.NET_CLIENT,
            message="import httpx",
        )
        v2 = ModelIOAuditViolation(
            file=Path("a.py"),
            line=5,
            column=0,
            rule=EnumIOAuditRule.FILE_IO,
            message="open()",
        )
        result = ModelAuditResult(violations=[v1, v2], files_scanned=1)
        output = _format_text_output(result)

        assert "a.py:" in output
        assert "Line 1:" in output
        assert "Line 5:" in output
        assert "2 violation(s)" in output

    def test_verbose_adds_whitelist_hint(self) -> None:
        v = ModelIOAuditViolation(
            file=Path("a.py"),
            line=1,
            column=0,
            rule=EnumIOAuditRule.NET_CLIENT,
            message="import httpx",
        )
        result = ModelAuditResult(violations=[v], files_scanned=1)
        output = _format_text_output(result, verbose=True)
        assert "--whitelist" in output


# ============================================================================
# __main__ -- JSON Output Formatting
# ============================================================================


@pytest.mark.unit
class TestJsonOutputFormatting:
    """Test _format_json_output from __main__.py."""

    def test_clean_result_json(self) -> None:
        result = ModelAuditResult(violations=[], files_scanned=2)
        output = json.loads(_format_json_output(result))

        assert output["is_clean"] is True
        assert output["files_scanned"] == 2
        assert output["violations"] == []

    def test_violations_json(self) -> None:
        v = ModelIOAuditViolation(
            file=Path("test.py"),
            line=10,
            column=3,
            rule=EnumIOAuditRule.FILE_IO,
            message="open()",
        )
        result = ModelAuditResult(violations=[v], files_scanned=1)
        output = json.loads(_format_json_output(result))

        assert output["is_clean"] is False
        assert len(output["violations"]) == 1
        assert output["violations"][0]["rule"] == "file-io"
        assert output["violations"][0]["line"] == 10
        assert output["violations"][0]["column"] == 3


# ============================================================================
# __main__ -- CLI Entry Point
# ============================================================================


@pytest.mark.unit
class TestCLIMain:
    """Test the main() CLI entry point."""

    def test_clean_exit_code_zero(self, tmp_path: Path) -> None:
        target_dir = tmp_path / "clean_nodes"
        target_dir.mkdir()
        (target_dir / "pure.py").write_text("x = 42\n", encoding="utf-8")

        exit_code = main([str(target_dir)])
        assert exit_code == 0

    def test_violations_exit_code_one(self, tmp_path: Path) -> None:
        target_dir = tmp_path / "dirty_nodes"
        target_dir.mkdir()
        (target_dir / "bad.py").write_text("import httpx\n", encoding="utf-8")

        exit_code = main([str(target_dir)])
        assert exit_code == 1

    def test_json_output(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        target_dir = tmp_path / "json_nodes"
        target_dir.mkdir()
        (target_dir / "mod.py").write_text("import httpx\n", encoding="utf-8")

        exit_code = main([str(target_dir), "--json"])
        captured = capsys.readouterr()
        output = json.loads(captured.out)

        assert exit_code == 1
        assert output["is_clean"] is False
        assert len(output["violations"]) == 1

    def test_empty_dir_clean(self, tmp_path: Path) -> None:
        target_dir = tmp_path / "empty_nodes"
        target_dir.mkdir()

        exit_code = main([str(target_dir)])
        assert exit_code == 0

    def test_nonexistent_dir_clean(self, tmp_path: Path) -> None:
        exit_code = main([str(tmp_path / "nonexistent_dir")])
        assert exit_code == 0


# ============================================================================
# Multiple Violations in Single File
# ============================================================================


@pytest.mark.unit
class TestMultipleViolations:
    """Ensure multiple distinct violations in one file are all detected."""

    def test_net_client_plus_file_io_plus_env(self) -> None:
        source = textwrap.dedent("""\
            import httpx
            import os
            f = open('data.txt')
            v = os.getenv('KEY')
        """)
        violations = _visit(source)

        rules = [v.rule for v in violations]
        assert EnumIOAuditRule.NET_CLIENT in rules
        assert EnumIOAuditRule.FILE_IO in rules
        assert EnumIOAuditRule.ENV_ACCESS in rules
        assert len(violations) == 3

    def test_multiple_forbidden_imports(self) -> None:
        source = textwrap.dedent("""\
            import httpx
            import redis
            import aiohttp
        """)
        violations = _visit(source)
        assert len(violations) == 3
        assert all(v.rule == EnumIOAuditRule.NET_CLIENT for v in violations)
