# SPDX-FileCopyrightText: 2026 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Runtime timeouts, intervals and bounds resolve from node contracts (OMN-21738)."""

from __future__ import annotations

import inspect

import pytest
from pydantic import ValidationError

from omnimemory.enums.enum_memory_storage_type import EnumMemoryStorageType
from omnimemory.models.config import ModelSemanticComputePolicyConfig
from omnimemory.models.config.model_memory_runtime_timeout_settings import (
    ModelMemoryRuntimeTimeoutSettings,
)
from omnimemory.models.memory.model_memory_storage_config import (
    ModelMemoryStorageConfig,
)
from omnimemory.runtime.introspection import publish_memory_introspection
from omnimemory.runtime.plugin import _probe_tcp_reachable


@pytest.mark.unit
class TestMemoryRuntimeTimeoutSettings:
    def test_contract_declares_every_value(self) -> None:
        settings = ModelMemoryRuntimeTimeoutSettings.from_node_contract()
        assert settings.memgraph_probe_timeout_seconds > 0
        assert settings.heartbeat_interval_seconds > 0
        assert 0 < settings.min_timeout_ms < settings.max_timeout_ms

    def test_overlay_replaces_the_contract_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OMNIMEMORY_HEARTBEAT_INTERVAL_SECONDS", "7.5")
        monkeypatch.setenv("OMNIMEMORY_MEMGRAPH_PROBE_TIMEOUT_SECONDS", "2.5")
        settings = ModelMemoryRuntimeTimeoutSettings.from_node_contract()
        assert settings.heartbeat_interval_seconds == 7.5
        assert settings.memgraph_probe_timeout_seconds == 2.5

    def test_non_positive_overlay_value_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OMNIMEMORY_HEARTBEAT_INTERVAL_SECONDS", "0")
        with pytest.raises(ValidationError):
            ModelMemoryRuntimeTimeoutSettings.from_node_contract()

    def test_callers_take_the_timeout_as_a_required_parameter(self) -> None:
        for function, name in (
            (_probe_tcp_reachable, "timeout"),
            (publish_memory_introspection, "heartbeat_interval_seconds"),
        ):
            parameter = inspect.signature(function).parameters[name]
            assert parameter.default is inspect.Parameter.empty


def _storage_config(**overrides: int) -> ModelMemoryStorageConfig:
    base: dict[str, object] = {
        "storage_id": "s",
        "storage_name": "s",
        "storage_type": next(iter(EnumMemoryStorageType)),
        "connection_string": "c",
        "host": "h",
        "port": 1,
        "database_name": "d",
    }
    base.update(overrides)
    return ModelMemoryStorageConfig.model_validate(base)


@pytest.mark.unit
class TestStorageConfigBoundsFromContract:
    def test_bounds_follow_the_overlay(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OMNIMEMORY_STORAGE_MIN_TIMEOUT_MS", "1000")
        monkeypatch.setenv("OMNIMEMORY_STORAGE_MAX_TIMEOUT_MS", "9000")
        with pytest.raises(ValidationError, match="at least 1000ms"):
            _storage_config(connection_timeout_ms=999)
        with pytest.raises(ValidationError, match="cannot exceed 9,000ms"):
            _storage_config(idle_timeout_ms=9001)
        assert _storage_config(connection_timeout_ms=5000).connection_timeout_ms == 5000


@pytest.mark.unit
class TestSemanticHealthCheckTimeout:
    def test_health_check_timeout_comes_from_the_descriptor(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OMNIMEMORY_SEMANTIC_HEALTH_CHECK_TIMEOUT_SECONDS", "3.5")
        assert ModelSemanticComputePolicyConfig().health_check_timeout_seconds == 3.5
