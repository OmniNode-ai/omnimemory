# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Unit tests for the AST-based transport import validator (ARCH-002 enforcement).

Tests validate that ``TransportImportChecker`` and supporting functions from
``scripts/validate_no_transport_imports.py`` correctly detect banned transport
imports, respect ``TYPE_CHECKING`` guards (including aliased forms), honour
the YAML whitelist, handle exclusion paths, and compose correctly through the
``main()`` entry point.
"""

from __future__ import annotations

import importlib.util
import sys
import textwrap
from pathlib import Path

import pytest
import yaml

# ---------------------------------------------------------------------------
# Import the module under test from ``scripts/`` (not a Python package).
# ---------------------------------------------------------------------------
_SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "validate_no_transport_imports.py"
)
_spec = importlib.util.spec_from_file_location(
    "validate_no_transport_imports", _SCRIPT_PATH
)
assert _spec is not None
assert _spec.loader is not None
_module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _module
_spec.loader.exec_module(_module)

# Re-export symbols under test
TransportImportChecker = _module.TransportImportChecker
Violation = _module.Violation
FileProcessingError = _module.FileProcessingError
WhitelistConfig = _module.WhitelistConfig
WhitelistEntry = _module.WhitelistEntry
check_file = _module.check_file
load_whitelist = _module.load_whitelist
is_whitelisted = _module.is_whitelisted
iter_python_files = _module.iter_python_files
BANNED_MODULES = _module.BANNED_MODULES
main = _module.main


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _check_source(source: str, *, filename: str = "example.py") -> list[Violation]:
    """Parse *source* and return violations found by TransportImportChecker."""
    import ast

    tree = ast.parse(textwrap.dedent(source))
    checker = TransportImportChecker(textwrap.dedent(source), Path(filename))
    checker.visit(tree)
    return checker.violations


def _write_py(tmp_path: Path, content: str, filename: str = "mod.py") -> Path:
    """Write a Python file into *tmp_path* and return its path."""
    target = tmp_path / filename
    target.write_text(textwrap.dedent(content), encoding="utf-8")
    return target


def _write_whitelist(tmp_path: Path, data: dict) -> Path:
    """Write a YAML whitelist file and return its path."""
    wl_path = tmp_path / "whitelist.yaml"
    wl_path.write_text(yaml.dump(data), encoding="utf-8")
    return wl_path


# ===================================================================
# TransportImportChecker -- direct import violations
# ===================================================================


@pytest.mark.unit
class TestDirectImportViolations:
    """Bare ``import X`` and ``from X import ...`` for banned modules must
    produce violations."""

    @pytest.mark.parametrize(
        ("source", "expected_module"),
        [
            pytest.param("import httpx\n", "httpx", id="import-httpx"),
            pytest.param("import redis\n", "redis", id="import-redis"),
            pytest.param("import asyncpg\n", "asyncpg", id="import-asyncpg"),
            pytest.param("import aiokafka\n", "aiokafka", id="import-aiokafka"),
            pytest.param("import grpc\n", "grpc", id="import-grpc"),
            pytest.param("import websockets\n", "websockets", id="import-websockets"),
            pytest.param(
                "from requests import Session\n", "requests", id="from-requests"
            ),
            pytest.param(
                "from confluent_kafka import Consumer\n",
                "confluent_kafka",
                id="from-confluent_kafka",
            ),
            pytest.param("from celery import Celery\n", "celery", id="from-celery"),
            pytest.param(
                "from aiohttp import ClientSession\n",
                "aiohttp",
                id="from-aiohttp",
            ),
        ],
    )
    def test_banned_import_detected(self, source: str, expected_module: str) -> None:
        violations = _check_source(source)
        assert len(violations) == 1
        assert violations[0].module_name == expected_module

    def test_dotted_submodule_import(self) -> None:
        """``import kafka.errors`` should flag ``kafka`` as root module."""
        violations = _check_source("import kafka.errors\n")
        assert len(violations) == 1
        assert violations[0].module_name == "kafka"

    def test_from_dotted_submodule(self) -> None:
        """``from redis.asyncio import Redis`` should flag ``redis``."""
        violations = _check_source("from redis.asyncio import Redis\n")
        assert len(violations) == 1
        assert violations[0].module_name == "redis"

    def test_multiple_violations_in_single_file(self) -> None:
        source = """\
        import httpx
        import redis
        from aiokafka import AIOKafkaConsumer
        """
        violations = _check_source(source)
        modules = {v.module_name for v in violations}
        assert modules == {"httpx", "redis", "aiokafka"}


# ===================================================================
# Clean files -- no violations
# ===================================================================


@pytest.mark.unit
class TestCleanFiles:
    """Files without banned imports produce zero violations."""

    def test_stdlib_only(self) -> None:
        source = """\
        from __future__ import annotations
        import os
        import sys
        from pathlib import Path
        """
        assert _check_source(source) == []

    def test_relative_imports_ignored(self) -> None:
        """Relative imports (``from . import X``) must never flag."""
        source = """\
        from . import kafka
        from .redis import client
        """
        assert _check_source(source) == []

    def test_non_banned_third_party(self) -> None:
        source = """\
        import pydantic
        from fastapi import APIRouter
        """
        assert _check_source(source) == []


# ===================================================================
# TYPE_CHECKING guard detection
# ===================================================================


@pytest.mark.unit
class TestTypeCheckingGuard:
    """Imports inside ``if TYPE_CHECKING:`` blocks must not produce violations."""

    def test_standard_type_checking_guard(self) -> None:
        source = """\
        from __future__ import annotations
        from typing import TYPE_CHECKING

        if TYPE_CHECKING:
            import httpx
            from redis import Redis
        """
        assert _check_source(source) == []

    def test_typing_module_qualified(self) -> None:
        """``import typing`` then ``if typing.TYPE_CHECKING:``."""
        source = """\
        import typing

        if typing.TYPE_CHECKING:
            import httpx
        """
        assert _check_source(source) == []

    def test_typing_aliased(self) -> None:
        """``import typing as t`` then ``if t.TYPE_CHECKING:``."""
        source = """\
        import typing as t

        if t.TYPE_CHECKING:
            import httpx
        """
        assert _check_source(source) == []

    def test_type_checking_aliased_constant(self) -> None:
        """``from typing import TYPE_CHECKING as TC`` then ``if TC:``."""
        source = """\
        from typing import TYPE_CHECKING as TC

        if TC:
            import httpx
        """
        assert _check_source(source) == []

    def test_else_branch_is_runtime(self) -> None:
        """Imports in the ``else`` branch of a TYPE_CHECKING guard are runtime
        imports and must produce violations."""
        source = """\
        from typing import TYPE_CHECKING

        if TYPE_CHECKING:
            import httpx
        else:
            import httpx
        """
        violations = _check_source(source)
        assert len(violations) == 1
        assert violations[0].module_name == "httpx"
        # The else branch starts at line 6 in the dedented source
        assert violations[0].line_number == 6

    def test_mixed_guarded_and_unguarded(self) -> None:
        """A guarded import (no violation) plus an unguarded import (violation)."""
        source = """\
        from typing import TYPE_CHECKING

        if TYPE_CHECKING:
            import httpx

        import redis
        """
        violations = _check_source(source)
        assert len(violations) == 1
        assert violations[0].module_name == "redis"


# ===================================================================
# Whitelist loading and matching
# ===================================================================


@pytest.mark.unit
class TestWhitelistLoading:
    """Test ``load_whitelist`` YAML parsing."""

    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        config = load_whitelist(tmp_path / "nonexistent.yaml")
        assert config.files == []

    def test_non_dict_yaml_returns_empty(self, tmp_path: Path) -> None:
        wl = tmp_path / "bad.yaml"
        wl.write_text("- just a list\n", encoding="utf-8")
        config = load_whitelist(wl)
        assert config.files == []

    def test_valid_whitelist_parsed(self, tmp_path: Path) -> None:
        data = {
            "schema_version": "1.0.0",
            "files": [
                {
                    "path": "src/omnimemory/utils/health.py",
                    "reason": "Health probes need direct connectivity",
                    "allowed_modules": ["asyncpg", "redis"],
                },
                {
                    "path": "src/omnimemory/infra/kafka_bridge.py",
                    "reason": "Bridge layer",
                },
            ],
        }
        wl_path = _write_whitelist(tmp_path, data)
        config = load_whitelist(wl_path)

        assert len(config.files) == 2
        assert config.files[0].path == "src/omnimemory/utils/health.py"
        assert config.files[0].allowed_modules == ["asyncpg", "redis"]
        assert config.files[1].allowed_modules == []

    def test_non_dict_entries_skipped(self, tmp_path: Path) -> None:
        data = {
            "files": [
                "just-a-string",
                {"path": "real.py", "reason": "ok"},
            ],
        }
        wl_path = _write_whitelist(tmp_path, data)
        config = load_whitelist(wl_path)
        assert len(config.files) == 1
        assert config.files[0].path == "real.py"


@pytest.mark.unit
class TestIsWhitelisted:
    """Test ``is_whitelisted`` matching logic."""

    @pytest.fixture
    def whitelist(self) -> WhitelistConfig:
        return WhitelistConfig(
            files=[
                WhitelistEntry(
                    path="src/omnimemory/utils/health.py",
                    reason="Health probes",
                    allowed_modules=["asyncpg", "redis"],
                ),
                WhitelistEntry(
                    path="src/omnimemory/infra/bridge.py",
                    reason="Bridge layer (all modules)",
                    allowed_modules=[],
                ),
            ]
        )

    def test_matching_path_and_module(self, whitelist: WhitelistConfig) -> None:
        result = is_whitelisted(
            Path("/repo/src/omnimemory/utils/health.py"), "asyncpg", whitelist
        )
        assert result is True

    def test_matching_path_wrong_module(self, whitelist: WhitelistConfig) -> None:
        result = is_whitelisted(
            Path("/repo/src/omnimemory/utils/health.py"), "httpx", whitelist
        )
        assert result is False

    def test_all_modules_whitelisted_when_empty_list(
        self, whitelist: WhitelistConfig
    ) -> None:
        result = is_whitelisted(
            Path("/repo/src/omnimemory/infra/bridge.py"), "httpx", whitelist
        )
        assert result is True

    def test_no_match_returns_false(self, whitelist: WhitelistConfig) -> None:
        result = is_whitelisted(
            Path("/repo/src/omnimemory/nodes/some_node.py"), "redis", whitelist
        )
        assert result is False


# ===================================================================
# check_file -- file-level processing
# ===================================================================


@pytest.mark.unit
class TestCheckFile:
    """Test ``check_file`` against actual files on disk."""

    def test_file_with_violation(self, tmp_path: Path) -> None:
        fp = _write_py(tmp_path, "import httpx\n")
        violations, errors = check_file(fp)
        assert len(violations) == 1
        assert violations[0].module_name == "httpx"
        assert errors == []

    def test_clean_file(self, tmp_path: Path) -> None:
        fp = _write_py(tmp_path, "import os\n")
        violations, errors = check_file(fp)
        assert violations == []
        assert errors == []

    def test_empty_file(self, tmp_path: Path) -> None:
        fp = _write_py(tmp_path, "")
        violations, errors = check_file(fp)
        assert violations == []
        assert errors == []

    def test_syntax_error_produces_processing_error(self, tmp_path: Path) -> None:
        fp = _write_py(tmp_path, "def broken(\n")
        violations, errors = check_file(fp)
        assert violations == []
        assert len(errors) == 1
        assert errors[0].error_type == "SyntaxError"

    def test_type_checking_guard_in_file(self, tmp_path: Path) -> None:
        source = """\
        from typing import TYPE_CHECKING

        if TYPE_CHECKING:
            import httpx
        """
        fp = _write_py(tmp_path, source)
        violations, errors = check_file(fp)
        assert violations == []
        assert errors == []


# ===================================================================
# iter_python_files -- directory traversal
# ===================================================================


@pytest.mark.unit
class TestIterPythonFiles:
    """Test ``iter_python_files`` traversal and exclusion logic."""

    def test_finds_python_files(self, tmp_path: Path) -> None:
        (tmp_path / "a.py").write_text("pass\n")
        (tmp_path / "b.py").write_text("pass\n")
        (tmp_path / "c.txt").write_text("not python\n")

        files = list(iter_python_files(tmp_path, set()))
        names = {f.name for f in files}
        assert "a.py" in names
        assert "b.py" in names
        assert "c.txt" not in names

    def test_skips_pycache(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / "__pycache__"
        cache_dir.mkdir()
        (cache_dir / "cached.py").write_text("pass\n")

        files = list(iter_python_files(tmp_path, set()))
        assert files == []

    def test_skips_venv(self, tmp_path: Path) -> None:
        venv_dir = tmp_path / ".venv"
        venv_dir.mkdir()
        (venv_dir / "something.py").write_text("pass\n")

        files = list(iter_python_files(tmp_path, set()))
        assert files == []

    def test_skips_egg_info_suffix(self, tmp_path: Path) -> None:
        egg_dir = tmp_path / "mypkg.egg-info"
        egg_dir.mkdir()
        (egg_dir / "PKG-INFO.py").write_text("pass\n")

        files = list(iter_python_files(tmp_path, set()))
        assert files == []

    def test_excludes_specific_path(self, tmp_path: Path) -> None:
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "excluded.py").write_text("pass\n")
        (tmp_path / "included.py").write_text("pass\n")

        files = list(iter_python_files(tmp_path, {sub}))
        names = {f.name for f in files}
        assert "included.py" in names
        assert "excluded.py" not in names


# ===================================================================
# main() -- end-to-end via CLI entry point
# ===================================================================


@pytest.mark.unit
class TestMain:
    """Test ``main()`` CLI entry point with explicit args and filesystem."""

    def test_clean_directory_returns_zero(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        src.mkdir()
        (src / "clean.py").write_text("import os\n", encoding="utf-8")

        result = main(
            [
                "--src-dir",
                str(src),
                "--whitelist",
                str(tmp_path / "nonexistent_wl.yaml"),
            ]
        )
        assert result == 0

    def test_violation_returns_one(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        src.mkdir()
        (src / "bad.py").write_text("import httpx\n", encoding="utf-8")

        result = main(
            [
                "--src-dir",
                str(src),
                "--whitelist",
                str(tmp_path / "nonexistent_wl.yaml"),
            ]
        )
        assert result == 1

    def test_whitelisted_violation_returns_zero(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        src.mkdir()
        bad_file = src / "health.py"
        bad_file.write_text("import asyncpg\n", encoding="utf-8")

        wl_data = {
            "schema_version": "1.0.0",
            "files": [
                {
                    "path": "health.py",
                    "reason": "Health probes",
                    "allowed_modules": ["asyncpg"],
                },
            ],
        }
        wl_path = _write_whitelist(tmp_path, wl_data)

        result = main(
            [
                "--src-dir",
                str(src),
                "--whitelist",
                str(wl_path),
            ]
        )
        assert result == 0

    def test_nonexistent_src_dir_returns_one(self, tmp_path: Path) -> None:
        result = main(
            [
                "--src-dir",
                str(tmp_path / "does_not_exist"),
                "--whitelist",
                str(tmp_path / "wl.yaml"),
            ]
        )
        assert result == 1

    def test_exclude_prevents_scanning(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        infra = src / "infra"
        infra.mkdir(parents=True)
        (infra / "bridge.py").write_text("import aiokafka\n", encoding="utf-8")

        result = main(
            [
                "--src-dir",
                str(src),
                "--exclude",
                str(infra),
                "--whitelist",
                str(tmp_path / "wl.yaml"),
            ]
        )
        assert result == 0

    def test_type_checking_guard_passes_main(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        src.mkdir()
        source = textwrap.dedent("""\
            from typing import TYPE_CHECKING

            if TYPE_CHECKING:
                import httpx
        """)
        (src / "guarded.py").write_text(source, encoding="utf-8")

        result = main(
            [
                "--src-dir",
                str(src),
                "--whitelist",
                str(tmp_path / "wl.yaml"),
            ]
        )
        assert result == 0


# ===================================================================
# Violation / FileProcessingError formatting
# ===================================================================


@pytest.mark.unit
class TestDataclassFormatting:
    """Verify __str__ output of dataclasses."""

    def test_violation_str(self) -> None:
        v = Violation(
            file_path=Path("src/mod.py"),
            line_number=42,
            module_name="httpx",
            import_statement="import httpx",
        )
        s = str(v)
        assert "src/mod.py" in s
        assert "42" in s
        assert "httpx" in s

    def test_file_processing_error_str(self) -> None:
        e = FileProcessingError(
            file_path=Path("bad.py"),
            error_type="SyntaxError",
            error_message="unexpected EOF",
        )
        s = str(e)
        assert "bad.py" in s
        assert "SyntaxError" in s
        assert "unexpected EOF" in s
