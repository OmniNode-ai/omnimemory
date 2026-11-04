# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Retry counts live in their config models; the previous effective value is each default."""

from __future__ import annotations

import pytest

import omnimemory.adapters.adapter_retry_utils as retry_module
from omnimemory.adapters.adapter_concurrency import AsyncConnectionPool, ConnectionPool
from omnimemory.adapters.adapter_retry_utils import retry_decorator
from omnimemory.enums import EnumFileProcessingStatus
from omnimemory.models.foundation.model_migration_progress import (
    FileProcessingInfo,
    MigrationProgressMetrics,
    MigrationProgressTracker,
)
from omnimemory.models.utils.model_concurrency import ModelConnectionPoolConfig
from omnimemory.models.utils.model_retry_config import ModelRetryConfig

_PREVIOUS_DEFAULT = 3


async def _create_connection() -> object:
    return object()


async def _acquire_with_invalid_connections(retry_attempts: int | None) -> str:
    config_kwargs: dict[str, object] = {"name": "p"}
    if retry_attempts is not None:
        config_kwargs["retry_attempts"] = retry_attempts
    pool = AsyncConnectionPool(
        ModelConnectionPoolConfig.model_validate(config_kwargs),
        create_connection=_create_connection,
        validate_connection=lambda conn: False,
    )
    try:
        with pytest.raises(RuntimeError) as excinfo:
            async with pool.acquire():
                pass
        return str(excinfo.value)
    finally:
        await pool.close()


@pytest.mark.unit
class TestAsyncConnectionPoolRetries:
    @pytest.mark.asyncio
    async def test_default_keeps_three_attempts(self) -> None:
        message = await _acquire_with_invalid_connections(None)
        assert f"after {_PREVIOUS_DEFAULT} attempts" in message

    @pytest.mark.asyncio
    async def test_attempts_follow_the_pool_config(self) -> None:
        message = await _acquire_with_invalid_connections(5)
        assert "after 5 attempts" in message


@pytest.mark.unit
class TestSimpleConnectionPoolRetries:
    def test_default_keeps_three_attempts(self) -> None:
        assert ModelConnectionPoolConfig.default_retry_attempts() == _PREVIOUS_DEFAULT

    @pytest.mark.asyncio
    async def test_acquire_without_retry_argument_still_yields_a_connection(
        self,
    ) -> None:
        pool = ConnectionPool(max_size=1, timeout=1.0)
        async with pool.acquire() as connection:
            assert connection is not None


@pytest.mark.unit
class TestRetryDecoratorDefault:
    @pytest.mark.asyncio
    async def test_omitted_attempts_use_the_retry_config_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = 0

        @retry_decorator(retryable_exceptions=["RuntimeError"], base_delay_ms=100)
        async def always_fails() -> None:
            nonlocal calls
            calls += 1
            raise RuntimeError("boom")

        async def no_sleep(_: float) -> None:
            return None

        # Zero out the waiting; the count is what is under test.
        monkeypatch.setattr(retry_module.asyncio, "sleep", no_sleep)
        with pytest.raises(RuntimeError):
            await always_fails()
        assert calls == ModelRetryConfig().max_attempts == _PREVIOUS_DEFAULT


@pytest.mark.unit
class TestMigrationRetryDefault:
    def _tracker(self, **overrides: object) -> MigrationProgressTracker:
        return MigrationProgressTracker.model_validate(
            {
                "name": "m",
                "metrics": MigrationProgressMetrics(total_files=2),
                **overrides,
            }
        )

    def _failed(self, retry_count: int) -> FileProcessingInfo:
        return FileProcessingInfo(
            file_path="f",
            status=EnumFileProcessingStatus.FAILED,
            retry_count=retry_count,
        )

    def test_default_limit_is_three(self) -> None:
        assert self._tracker().max_file_retries == _PREVIOUS_DEFAULT

    def test_failed_file_at_the_default_limit_is_not_retried(self) -> None:
        tracker = self._tracker(files=[self._failed(2), self._failed(3)])
        retried = tracker.retry_failed_files()
        assert [info.retry_count for info in retried] == [3]

    def test_limit_follows_the_tracker_field(self) -> None:
        tracker = self._tracker(
            max_file_retries=1, files=[self._failed(0), self._failed(1)]
        )
        assert len(tracker.retry_failed_files()) == 1

    def test_explicit_argument_overrides_the_field(self) -> None:
        tracker = self._tracker(files=[self._failed(4)])
        assert len(tracker.retry_failed_files(max_retries=5)) == 1
