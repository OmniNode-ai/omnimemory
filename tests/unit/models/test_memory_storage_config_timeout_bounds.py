# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The storage config's timeout bounds are injected from the node contract (OMN-21738).

``min_timeout_ms`` and ``max_timeout_ms`` are ``descriptor`` entries of the storage
node's contract.yaml with the ``${env.VAR:default}`` overlay convention, expanded by
``ModelMemoryStorageConfig.from_node_contract``; the model source holds neither.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from omnimemory.enums.enum_memory_storage_type import EnumMemoryStorageType
from omnimemory.models.memory.model_memory_storage_config import (
    ModelMemoryStorageConfig,
)

pytestmark = pytest.mark.unit

_BASE: dict[str, object] = {
    "storage_id": "s1",
    "storage_name": "store",
    "storage_type": next(iter(EnumMemoryStorageType)),
    "connection_string": "file:///tmp/x",
    "host": "localhost",
    "port": 1,
    "database_name": "db",
}


def test_bounds_come_from_the_node_contract_descriptor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OMNIMEMORY_STORAGE_MIN_TIMEOUT_MS", raising=False)
    monkeypatch.delenv("OMNIMEMORY_STORAGE_MAX_TIMEOUT_MS", raising=False)
    config = ModelMemoryStorageConfig.from_node_contract(**_BASE)
    assert (config.min_timeout_ms, config.max_timeout_ms) == (100, 300000)


def test_env_overlay_replaces_a_bound(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OMNIMEMORY_STORAGE_MAX_TIMEOUT_MS", "20000")
    with pytest.raises(ValidationError, match="cannot exceed 20,000ms"):
        ModelMemoryStorageConfig.from_node_contract(**_BASE, idle_timeout_ms=30000)


def test_invalid_overlay_fails_at_binding(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OMNIMEMORY_STORAGE_MIN_TIMEOUT_MS", "not-a-number")
    with pytest.raises(ValidationError):
        ModelMemoryStorageConfig.from_node_contract(**_BASE)


def test_bounds_have_no_default() -> None:
    with pytest.raises(ValidationError, match="min_timeout_ms"):
        ModelMemoryStorageConfig.model_validate(_BASE)


def test_timeout_below_injected_minimum_is_refused() -> None:
    with pytest.raises(ValidationError, match="at least 500ms"):
        ModelMemoryStorageConfig.model_validate(
            {
                **_BASE,
                "min_timeout_ms": 500,
                "max_timeout_ms": 60000,
                "connection_timeout_ms": 100,
            }
        )
