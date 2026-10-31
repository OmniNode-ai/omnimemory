# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The lifecycle handlers' timeouts are injected from the node contract (OMN-21623).

The database query timeout, the two circuit-breaker recovery timeouts and the
projection-reader timeout are declared in the node contract's ``descriptor`` with
the ``${env.VAR:default}`` overlay convention, expanded by
``ModelLifecycleTimeoutSettings.from_node_contract`` and injected into the archive,
expire and tick handlers; no literal stays in a handler.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from omnibase_core.container import ModelONEXContainer
from pydantic import ValidationError

from omnimemory.models.config.model_lifecycle_timeout_settings import (
    ModelLifecycleTimeoutSettings,
)
from omnimemory.nodes.node_memory_lifecycle_orchestrator.handlers import (
    HandlerMemoryArchive,
    HandlerMemoryExpire,
    HandlerMemoryTick,
)

pytestmark = pytest.mark.unit

_ENV_BY_FIELD = {
    "db_query_timeout_seconds": "OMNIMEMORY_LIFECYCLE_DB_QUERY_TIMEOUT_SECONDS",
    "db_circuit_breaker_recovery_seconds": "OMNIMEMORY_LIFECYCLE_DB_CB_RECOVERY_SECONDS",
    "projection_reader_timeout_seconds": (
        "OMNIMEMORY_LIFECYCLE_PROJECTION_READER_TIMEOUT_SECONDS"
    ),
    "tick_circuit_breaker_recovery_seconds": "OMNIMEMORY_LIFECYCLE_TICK_CB_RECOVERY_SECONDS",
}

_INJECTED = ModelLifecycleTimeoutSettings(
    db_query_timeout_seconds=1.5,
    db_circuit_breaker_recovery_seconds=2.5,
    projection_reader_timeout_seconds=3.5,
    tick_circuit_breaker_recovery_seconds=4.5,
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _ENV_BY_FIELD.values():
        monkeypatch.delenv(name, raising=False)


def test_contract_defaults_resolve() -> None:
    settings = ModelLifecycleTimeoutSettings.from_node_contract()
    assert settings.db_query_timeout_seconds == 30.0
    assert settings.db_circuit_breaker_recovery_seconds == 60.0
    assert settings.projection_reader_timeout_seconds == 10.0
    assert settings.tick_circuit_breaker_recovery_seconds == 30.0


def test_env_overlay_replaces_every_value(monkeypatch: pytest.MonkeyPatch) -> None:
    for index, name in enumerate(_ENV_BY_FIELD.values(), start=1):
        monkeypatch.setenv(name, str(index * 11))
    settings = ModelLifecycleTimeoutSettings.from_node_contract()
    assert settings.db_query_timeout_seconds == 11.0
    assert settings.db_circuit_breaker_recovery_seconds == 22.0
    assert settings.projection_reader_timeout_seconds == 33.0
    assert settings.tick_circuit_breaker_recovery_seconds == 44.0


@pytest.mark.parametrize("name", sorted(_ENV_BY_FIELD.values()))
@pytest.mark.parametrize("bad", ["abc", "0", "-3", "nan", "inf"])
def test_bad_timeout_overlay_fails_to_load(
    name: str, bad: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(name, bad)
    with pytest.raises(ValidationError):
        ModelLifecycleTimeoutSettings.from_node_contract()


def test_settings_model_has_no_defaults() -> None:
    with pytest.raises(ValidationError):
        ModelLifecycleTimeoutSettings.model_validate({})


def test_archive_handler_uses_injected_settings() -> None:
    handler = HandlerMemoryArchive(ModelONEXContainer(), settings=_INJECTED)
    assert handler._settings is _INJECTED


@pytest.mark.asyncio
async def test_archive_handler_reports_injected_query_timeout() -> None:
    handler = HandlerMemoryArchive(ModelONEXContainer(), settings=_INJECTED)
    metadata = await handler.describe()
    assert metadata.query_timeout_seconds == 1.5
    assert metadata.circuit_breaker_config.recovery_timeout == 2.5


@pytest.mark.asyncio
async def test_archive_handler_breaker_uses_injected_recovery() -> None:
    handler = HandlerMemoryArchive(ModelONEXContainer(), settings=_INJECTED)
    await handler.initialize()
    assert handler._db_circuit_breaker is not None
    assert handler._db_circuit_breaker.recovery_timeout == 2.5


@pytest.mark.asyncio
async def test_expire_handler_breaker_uses_injected_recovery() -> None:
    handler = HandlerMemoryExpire(ModelONEXContainer(), settings=_INJECTED)
    await handler.initialize(db_pool=MagicMock())
    assert handler._circuit_breaker.recovery_timeout == 2.5


@pytest.mark.asyncio
async def test_tick_handler_breaker_uses_injected_recovery() -> None:
    handler = HandlerMemoryTick(ModelONEXContainer(), settings=_INJECTED)
    await handler.initialize()
    assert handler._circuit_breaker is not None
    assert handler._circuit_breaker.recovery_timeout == 4.5
    assert handler._settings.projection_reader_timeout_seconds == 3.5


def test_handlers_without_settings_resolve_the_contract() -> None:
    container = ModelONEXContainer()
    assert HandlerMemoryArchive(container)._settings.db_query_timeout_seconds == 30.0
    assert HandlerMemoryExpire(container)._settings.db_query_timeout_seconds == 30.0
    assert (
        HandlerMemoryTick(container)._settings.projection_reader_timeout_seconds == 10.0
    )
