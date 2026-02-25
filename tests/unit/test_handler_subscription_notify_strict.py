# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Unit tests for HandlerSubscription.notify() strict mode parameter.

These tests verify that the strict parameter correctly controls error
propagation behavior when event bus publish operations fail.

OMN-1499: Add strict mode option to notify() for Kafka error handling.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from omnimemory.handlers.handler_subscription import HandlerSubscription


def _make_event(topic: str) -> Any:
    """Build a minimal ModelNotificationEvent for the given topic."""
    from omnimemory.models.subscription import (
        ModelNotificationEvent,
        ModelNotificationEventPayload,
    )

    return ModelNotificationEvent(
        event_id=str(uuid4()),
        topic=topic,
        payload=ModelNotificationEventPayload(
            entity_type="test",
            entity_id="item_1",
            action="created",
        ),
    )


@pytest.mark.unit
class TestNotifyStrictMode:
    """Tests for the strict parameter on HandlerSubscription.notify()."""

    def _build_handler_with_mocks(
        self,
        *,
        subscriber_ids: list[str],
        publish_side_effect: Exception | None = None,
    ) -> tuple[HandlerSubscription, AsyncMock]:
        """Return a HandlerSubscription whose internals are fully mocked.

        The handler is pre-initialized via _ensure_initialized to avoid
        needing real infrastructure connections.
        """
        handler = HandlerSubscription.__new__(HandlerSubscription)

        mock_publisher = AsyncMock()
        if publish_side_effect is not None:
            mock_publisher.publish.side_effect = publish_side_effect
        else:
            mock_publisher.publish.return_value = None

        mock_config = MagicMock()
        mock_config.kafka_notification_topic = "omnimemory.notifications"

        # Patch _ensure_initialized to return stable mocks
        handler._ensure_initialized = MagicMock(  # type: ignore[method-assign]
            return_value=(None, None, mock_publisher, mock_config)
        )

        # Patch _get_subscribers_for_topic to return the given IDs
        handler._get_subscribers_for_topic = AsyncMock(  # type: ignore[method-assign]
            return_value=subscriber_ids
        )

        # Patch _increment_metric to a no-op
        handler._increment_metric = AsyncMock()  # type: ignore[method-assign]

        return handler, mock_publisher

    # ------------------------------------------------------------------
    # Default behavior (strict=True)
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_strict_true_propagates_publish_exception(self) -> None:
        """strict=True (default) raises when publish fails."""
        topic = "memory.item.created"
        handler, _ = self._build_handler_with_mocks(
            subscriber_ids=["agent_a"],
            publish_side_effect=RuntimeError("Kafka unavailable"),
        )

        with pytest.raises(RuntimeError, match="Kafka unavailable"):
            await handler.notify(topic, _make_event(topic))

    @pytest.mark.asyncio
    async def test_strict_default_is_true(self) -> None:
        """Calling notify() without strict= behaves as strict=True."""
        topic = "memory.item.created"
        handler, _ = self._build_handler_with_mocks(
            subscriber_ids=["agent_a"],
            publish_side_effect=ConnectionError("broker gone"),
        )

        with pytest.raises(ConnectionError):
            await handler.notify(topic, _make_event(topic))

    @pytest.mark.asyncio
    async def test_strict_true_success_returns_subscriber_count(self) -> None:
        """strict=True returns subscriber count on success."""
        topic = "memory.item.created"
        handler, _ = self._build_handler_with_mocks(subscriber_ids=["a", "b", "c"])

        count = await handler.notify(topic, _make_event(topic), strict=True)

        assert count == 3

    # ------------------------------------------------------------------
    # Fire-and-forget mode (strict=False)
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_strict_false_suppresses_publish_exception(self) -> None:
        """strict=False does not raise when publish fails."""
        topic = "memory.item.created"
        handler, _ = self._build_handler_with_mocks(
            subscriber_ids=["agent_a"],
            publish_side_effect=RuntimeError("Kafka unavailable"),
        )

        # Should not raise
        count = await handler.notify(topic, _make_event(topic), strict=False)

        assert count == 0

    @pytest.mark.asyncio
    async def test_strict_false_returns_zero_on_failure(self) -> None:
        """strict=False returns 0 when publish raises, regardless of subscriber count."""
        topic = "memory.item.updated"
        handler, _ = self._build_handler_with_mocks(
            subscriber_ids=["a", "b", "c", "d"],
            publish_side_effect=OSError("network error"),
        )

        count = await handler.notify(topic, _make_event(topic), strict=False)

        assert count == 0

    @pytest.mark.asyncio
    async def test_strict_false_success_returns_subscriber_count(self) -> None:
        """strict=False still returns subscriber count when publish succeeds."""
        topic = "memory.item.deleted"
        handler, _ = self._build_handler_with_mocks(subscriber_ids=["x", "y"])

        count = await handler.notify(topic, _make_event(topic), strict=False)

        assert count == 2

    @pytest.mark.asyncio
    async def test_strict_false_logs_warning_on_failure(self) -> None:
        """strict=False logs a warning (not silently discards the error)."""
        topic = "memory.item.created"
        handler, _ = self._build_handler_with_mocks(
            subscriber_ids=["agent_a"],
            publish_side_effect=RuntimeError("broker timeout"),
        )

        with patch("omnimemory.handlers.handler_subscription.logger") as mock_logger:
            await handler.notify(topic, _make_event(topic), strict=False)
            mock_logger.warning.assert_called_once()

    # ------------------------------------------------------------------
    # Invariants that hold regardless of strict flag
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_topic_mismatch_always_raises(self) -> None:
        """ValueError for topic mismatch is not affected by the strict flag."""
        topic = "memory.item.created"
        event = _make_event("memory.item.updated")  # different topic
        handler, _ = self._build_handler_with_mocks(subscriber_ids=["a"])

        for strict_val in (True, False):
            with pytest.raises(ValueError, match="Event topic mismatch"):
                await handler.notify(topic, event, strict=strict_val)

    @pytest.mark.asyncio
    async def test_no_subscribers_returns_zero_for_both_modes(self) -> None:
        """Zero subscribers short-circuits before publish for both strict modes."""
        topic = "memory.item.created"
        handler, mock_publisher = self._build_handler_with_mocks(subscriber_ids=[])

        for strict_val in (True, False):
            count = await handler.notify(topic, _make_event(topic), strict=strict_val)
            assert count == 0

        # publish was never called
        mock_publisher.publish.assert_not_called()

    @pytest.mark.asyncio
    async def test_strict_is_keyword_only(self) -> None:
        """strict must be passed as a keyword argument (enforced by *)."""
        import inspect

        sig = inspect.signature(HandlerSubscription.notify)
        params = sig.parameters
        assert "strict" in params
        assert params["strict"].kind == inspect.Parameter.KEYWORD_ONLY
        assert params["strict"].default is True
