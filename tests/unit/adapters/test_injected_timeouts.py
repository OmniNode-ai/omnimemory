# SPDX-FileCopyrightText: 2026 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Timeouts and backoffs reach resilience code by injection, never by a literal default."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable, Coroutine
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from omnimemory.adapters.adapter_concurrency import (
    CircuitBreaker,
    CircuitBreakerOpenError,
    ConnectionPool,
    LockRequest,
    PriorityLock,
    get_priority_lock,
    with_circuit_breaker,
    with_priority_lock,
    with_retry,
)
from omnimemory.adapters.adapter_health_manager import (
    HealthManager,
    create_postgresql_health_check,
)
from omnimemory.adapters.adapter_retry_utils import retry_decorator
from omnimemory.handlers.handler_intent import HandlerIntent
from omnimemory.protocols.error_models import (
    ERROR_CATEGORIES,
    ModelErrorCategoryInfo,
    _load_error_categories,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("callable_", "parameter"),
    [
        (CircuitBreaker.__init__, "failure_threshold"),
        (CircuitBreaker.__init__, "recovery_timeout"),
        (ConnectionPool.__init__, "timeout"),
        (PriorityLock.__init__, "poll_interval_seconds"),
        (get_priority_lock, "poll_interval_seconds"),
        (with_priority_lock, "poll_interval_seconds"),
        (with_retry, "delay"),
        (with_retry, "backoff_multiplier"),
        (HealthManager.__init__, "default_timeout"),
        (create_postgresql_health_check, "connect_timeout"),
    ],
)
def test_parameter_has_no_default(
    callable_: Callable[..., object], parameter: str
) -> None:
    signature = inspect.signature(callable_)
    assert signature.parameters[parameter].default is inspect.Parameter.empty


def test_with_circuit_breaker_takes_only_an_injected_breaker() -> None:
    assert list(inspect.signature(with_circuit_breaker).parameters) == [
        "circuit_breaker"
    ]


@pytest.mark.asyncio
async def test_with_circuit_breaker_guards_with_the_injected_breaker() -> None:
    breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=7.5)
    failing = AsyncMock(side_effect=RuntimeError("down"))
    decorated = with_circuit_breaker(breaker)(failing)
    with pytest.raises(RuntimeError):
        await decorated()
    with pytest.raises(CircuitBreakerOpenError):
        await decorated()
    assert breaker.recovery_timeout == 7.5


@pytest.mark.asyncio
async def test_priority_lock_polls_at_the_injected_interval() -> None:
    seen: list[float] = []

    async def stop(interval: float) -> None:
        seen.append(interval)
        raise TimeoutError

    for interval in (0.25, 0.5):
        lock = PriorityLock("polled", poll_interval_seconds=interval)
        waiting = LockRequest()
        lock._queue.append(waiting)
        lock._current_holder = LockRequest()
        with patch("asyncio.sleep", stop), pytest.raises(TimeoutError):
            await lock._wait_for_turn(waiting)
    assert seen == [0.25, 0.5]


def test_retry_decorator_leaves_delays_to_the_retry_config() -> None:
    parameters = inspect.signature(retry_decorator).parameters
    assert parameters["base_delay_ms"].default is None
    assert parameters["max_delay_ms"].default is None
    retry_decorator(max_attempts=2)  # the ModelRetryConfig delays apply


def test_retry_decorator_passes_explicit_delays_through() -> None:
    with pytest.raises(ValidationError):
        retry_decorator(base_delay_ms=1)


@pytest.mark.asyncio
async def test_postgresql_health_check_uses_injected_connect_timeout() -> None:
    connection = AsyncMock()
    with (
        patch("asyncpg.connect", AsyncMock(return_value=connection)) as connect,
        patch("omnimemory.adapters.adapter_health_manager.HealthCheckConfig"),
        patch("omnimemory.adapters.adapter_health_manager.HealthCheckResult"),
    ):
        check = await create_postgresql_health_check(
            "postgresql://db.invalid/x", connect_timeout=2.5
        )
        await check()
    assert connect.await_args is not None
    assert connect.await_args.kwargs["timeout"] == 2.5


@pytest.mark.asyncio
async def test_handler_intent_shutdown_uses_its_initialized_timeout() -> None:
    handler = HandlerIntent(MagicMock())
    adapter = AsyncMock()
    with patch(
        "omnimemory.handlers.handler_intent.AdapterIntentGraph", return_value=adapter
    ):
        await handler.initialize(
            connection_uri="bolt://db.invalid:1", options={"timeout_seconds": 12.5}
        )
    seen: list[float | None] = []
    real_wait_for = asyncio.wait_for

    async def spy(
        awaitable: Coroutine[object, object, None], timeout: float | None = None
    ) -> None:
        seen.append(timeout)
        await real_wait_for(awaitable, timeout)

    with patch("omnimemory.handlers.handler_intent.asyncio.wait_for", spy):
        await handler.shutdown()
    assert seen == [12.5]


def test_error_category_backoff_comes_from_contract() -> None:
    backoffs = {
        name: info.default_backoff_factor for name, info in ERROR_CATEGORIES.items()
    }
    assert backoffs == {
        "VALIDATION": 1.0,
        "STORAGE": 2.0,
        "RETRIEVAL": 1.5,
        "PROCESSING": 2.0,
        "COORDINATION": 1.5,
        "SYSTEM": 1.0,
    }


def test_error_category_backoff_overlay_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OMNIMEMORY_ERROR_BACKOFF_STORAGE", "4.5")
    assert _load_error_categories()["STORAGE"].default_backoff_factor == 4.5


@pytest.mark.parametrize("value", ["broken", ""])
def test_error_category_backoff_invalid_overlay_fails_to_load(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("OMNIMEMORY_ERROR_BACKOFF_SYSTEM", value)
    with pytest.raises(ValidationError):
        _load_error_categories()


def test_error_category_info_has_no_defaults() -> None:
    for name in ("default_retry_count", "default_backoff_factor"):
        assert ModelErrorCategoryInfo.model_fields[name].is_required()
