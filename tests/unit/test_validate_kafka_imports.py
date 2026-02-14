# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for the Kafka import lint guard (ARCH-002 enforcement).

Tests validate that ``validate_file`` from
``scripts/validation/validate_kafka_imports.py`` correctly detects direct
Kafka imports inside the enforced ``omnimemory/nodes/`` directory, respects
``TYPE_CHECKING`` guards, honours the ``omnimemory-kafka-exempt`` annotation,
and ignores files with no Kafka imports.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Import ``validate_file`` from the validation script.
# ``scripts/`` is not a Python package, so we load the module by file path.
# ---------------------------------------------------------------------------
_SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "validation"
    / "validate_kafka_imports.py"
)
_spec = importlib.util.spec_from_file_location("validate_kafka_imports", _SCRIPT_PATH)
assert _spec is not None
assert _spec.loader is not None
_module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _module
_spec.loader.exec_module(_module)

validate_file = _module.validate_file


def _write_node_file(
    tmp_path: Path, content: str, filename: str = "example_node.py"
) -> Path:
    """Write content into a file under a fake ``omnimemory/nodes/`` tree.

    ``validate_file`` only checks files whose path contains
    ``omnimemory/nodes/``, so the helper creates that directory structure
    inside ``tmp_path``.

    Returns the path to the created ``.py`` file.
    """
    node_dir = tmp_path / "omnimemory" / "nodes"
    node_dir.mkdir(parents=True, exist_ok=True)
    target = node_dir / filename
    target.write_text(content, encoding="utf-8")
    return target


@pytest.mark.unit
class TestDirectKafkaImportViolation:
    """A bare ``from aiokafka import ...`` must produce a violation."""

    def test_from_aiokafka_import(self, tmp_path: Path) -> None:
        source = "from aiokafka import AIOKafkaConsumer\n"
        filepath = _write_node_file(tmp_path, source)
        violations = validate_file(filepath)

        assert len(violations) == 1
        assert "aiokafka" in violations[0].message
        assert violations[0].line == 1


@pytest.mark.unit
class TestTypeCheckingGuardAllowed:
    """Kafka imports inside ``if TYPE_CHECKING:`` must not produce violations."""

    def test_import_inside_type_checking_block(self, tmp_path: Path) -> None:
        source = (
            "from __future__ import annotations\n"
            "from typing import TYPE_CHECKING\n"
            "\n"
            "if TYPE_CHECKING:\n"
            "    from aiokafka import AIOKafkaConsumer\n"
        )
        filepath = _write_node_file(tmp_path, source)
        violations = validate_file(filepath)

        assert violations == []


@pytest.mark.unit
class TestExemptAnnotationSkipped:
    """Lines with ``# omnimemory-kafka-exempt:`` must be skipped."""

    def test_exempt_annotation_suppresses_violation(self, tmp_path: Path) -> None:
        source = (
            "from aiokafka import AIOKafkaConsumer"
            "  # omnimemory-kafka-exempt: needed for bridge layer\n"
        )
        filepath = _write_node_file(tmp_path, source)
        violations = validate_file(filepath)

        assert violations == []


@pytest.mark.unit
class TestNoKafkaImports:
    """A file with no Kafka imports at all must produce no violations."""

    def test_clean_file(self, tmp_path: Path) -> None:
        source = (
            "from __future__ import annotations\n"
            "import os\n"
            "\n"
            "def hello() -> str:\n"
            "    return 'world'\n"
        )
        filepath = _write_node_file(tmp_path, source)
        violations = validate_file(filepath)

        assert violations == []


@pytest.mark.unit
class TestAllKafkaLibraryPatterns:
    """All six Kafka import patterns must produce violations, not just aiokafka."""

    @pytest.mark.parametrize(
        ("source", "expected_fragment"),
        [
            pytest.param(
                "import aiokafka\n",
                "aiokafka",
                id="import-aiokafka",
            ),
            pytest.param(
                "from confluent_kafka import Consumer\n",
                "confluent_kafka",
                id="from-confluent_kafka",
            ),
            pytest.param(
                "from kafka.errors import KafkaError\n",
                "kafka",
                id="from-kafka-submodule",
            ),
            pytest.param(
                "import kafka\n",
                "kafka",
                id="import-kafka",
            ),
            pytest.param(
                "import confluent_kafka\n",
                "confluent_kafka",
                id="import-confluent_kafka",
            ),
        ],
    )
    def test_library_pattern_produces_violation(
        self,
        tmp_path: Path,
        source: str,
        expected_fragment: str,
    ) -> None:
        filepath = _write_node_file(tmp_path, source)
        violations = validate_file(filepath)

        assert len(violations) == 1
        assert expected_fragment in violations[0].message
        assert violations[0].line == 1


@pytest.mark.unit
class TestOutsideEnforcedPath:
    """Files outside ``omnimemory/nodes/`` must be ignored regardless of content."""

    def test_kafka_import_outside_nodes_is_ignored(self, tmp_path: Path) -> None:
        utils_dir = tmp_path / "omnimemory" / "utils"
        utils_dir.mkdir(parents=True, exist_ok=True)
        target = utils_dir / "kafka_helper.py"
        target.write_text(
            "from aiokafka import AIOKafkaConsumer\n",
            encoding="utf-8",
        )
        violations = validate_file(target)

        assert violations == []


@pytest.mark.unit
class TestElseBlockAfterTypeChecking:
    """Kafka imports in an ``else:`` block after ``if TYPE_CHECKING:`` are
    runtime imports and must produce a violation."""

    def test_else_branch_is_runtime(self, tmp_path: Path) -> None:
        source = (
            "from typing import TYPE_CHECKING\n"
            "\n"
            "if TYPE_CHECKING:\n"
            "    from aiokafka import AIOKafkaConsumer\n"
            "else:\n"
            "    from aiokafka import AIOKafkaConsumer\n"
        )
        filepath = _write_node_file(tmp_path, source)
        violations = validate_file(filepath)

        assert len(violations) == 1
        assert violations[0].line == 6
        assert "aiokafka" in violations[0].message
