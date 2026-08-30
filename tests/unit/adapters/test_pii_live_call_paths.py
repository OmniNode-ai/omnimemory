# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""End-to-end tests for the two live callers of ``PIIDetector`` (OMN-17236).

Path 1 -- ``ModelIntentStorageRequest.validate_user_context_pii``
    A ``@model_validator(mode="after")`` that calls
    ``detect_pii(user_context, sensitivity_level="medium")`` and **raises**
    ``ValueError`` when PII is reported. Because the model is ``frozen``,
    this fires at construction: a storage request carrying PII never
    exists as an object, so nothing downstream can persist it.

Path 2 -- ``HandlerIntentStorageAdapter._store_intent``
    A redact-before-persist arm that recomputes detection on
    ``request.user_context``.

The Path 2 tests below pin a real finding (OMN-17236 D4): the arm computes
``sanitized_context`` and then never passes it to
``AdapterIntentGraph.store_intent``, whose signature accepts only
``session_id``, ``intent_data`` and ``correlation_id``. Combined with the
Path 1 validator -- which refuses any request whose ``user_context``
contains PII before the adapter is reached -- the redaction arm is doubly
unreachable, while still emitting a ``pii_redacted_before_storage`` log
line. This lane pins the behavior and does not change persistence
semantics.
"""

from __future__ import annotations

from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from omnibase_core.models.intelligence import (
    ModelIntentClassificationOutput,
    ModelIntentStorageResult,
)
from pydantic import ValidationError

from omnimemory.nodes.node_intent_storage_effect.adapters.adapter_intent_storage import (
    HandlerIntentStorageAdapter,
)
from omnimemory.nodes.node_intent_storage_effect.models import (
    ModelIntentStorageRequest,
)

pytestmark = pytest.mark.unit


def _intent_data() -> ModelIntentClassificationOutput:
    return ModelIntentClassificationOutput(
        success=True,
        confidence=0.92,
        keywords=["error", "fix"],
        secondary_intents=[],
    )


def _request(user_context: str) -> ModelIntentStorageRequest:
    return ModelIntentStorageRequest(
        operation="store",
        session_id="session_123",
        intent_data=_intent_data(),
        user_context=user_context,
    )


# ---------------------------------------------------------------------------
# Path 1 -- request-model validator
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestStorageRequestRefusesPII:
    """A storage request carrying PII must not be constructible."""

    @pytest.mark.parametrize(
        ("user_context", "expected_type"),
        [
            pytest.param("reach me at alice@example.com", "email", id="email"),
            pytest.param("call 555-123-4567", "phone", id="phone"),
            pytest.param("ssn 123-45-6789", "ssn", id="ssn"),
            pytest.param("card 4111111111111111", "credit_card", id="credit-card"),
            pytest.param("from 8.8.8.8", "ip_address", id="ipv4"),
            pytest.param("sk-" + "a" * 40, "api_key", id="api-key"),
        ],
    )
    def test_pii_in_user_context_is_refused(
        self, user_context: str, expected_type: str
    ) -> None:
        with pytest.raises(ValidationError) as exc_info:
            _request(user_context)
        message = str(exc_info.value)
        assert "user_context contains PII" in message
        assert expected_type in message

    def test_refusal_names_every_detected_type(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            _request("alice@example.com called from 8.8.8.8")
        message = str(exc_info.value)
        assert "email" in message
        assert "ip_address" in message

    def test_validator_message_does_not_echo_the_pii(self) -> None:
        """The validator's own message names types only, never the value."""
        with pytest.raises(ValidationError) as exc_info:
            _request("reach me at alice@example.com")
        message = exc_info.value.errors()[0]["msg"]
        assert "email" in message
        assert "alice@example.com" not in message

    def test_rendered_validation_error_echoes_the_pii(self) -> None:
        """OMN-17236 D5, residual: pydantic re-leaks the value on render.

        The validator's message is clean, but pydantic renders the rejected
        ``input_value`` into ``str(ValidationError)``. Any caller that logs
        or returns the stringified error re-emits the exact PII the
        validator just refused. Pinned here so the leak is visible; closing
        it needs a caller-side or ``ValidationError``-handling change,
        which is out of scope for this lane.
        """
        with pytest.raises(ValidationError) as exc_info:
            _request("reach me at alice@example.com")
        assert "alice@example.com" in str(exc_info.value)


@pytest.mark.unit
class TestStorageRequestAcceptsCleanContext:
    """Non-PII context must construct -- a false positive is a refusal."""

    @pytest.mark.parametrize(
        "user_context",
        [
            pytest.param("", id="empty-default"),
            pytest.param("debugging a failing test", id="prose"),
            pytest.param("OMN-17236 follow-up", id="ticket-ref"),
            pytest.param("at 2026-08-30T12:34:56Z", id="iso-timestamp"),
            pytest.param("commit 751b5cb1c", id="git-short-sha"),
            pytest.param("sha da39a3ee5e6b4b0d3255bfef95601890afd80709", id="git-sha1"),
            pytest.param("upgraded to version 1.2.3", id="semver"),
        ],
    )
    def test_clean_context_is_accepted(self, user_context: str) -> None:
        assert _request(user_context).user_context == user_context

    def test_correlation_uuid_in_context_is_accepted(self) -> None:
        """OMN-17236 D1 regression: a UUID must not read as a phone number.

        Before the PHONE boundary guard, the digit run inside a UUID
        matched the PHONE pattern, so any request whose ``user_context``
        mentioned a run/session UUID was refused outright.
        """
        context = f"replaying run {uuid4()}"
        assert _request(context).user_context == context

    def test_placeholder_ssn_in_context_is_accepted(self) -> None:
        """OMN-17236 D2 regression: an all-zero placeholder is not an SSN."""
        assert _request("redacted upstream as 000-00-0000").user_context

    def test_user_context_is_omittable(self) -> None:
        request = ModelIntentStorageRequest(
            operation="store",
            session_id="session_123",
            intent_data=_intent_data(),
        )
        assert request.user_context == ""


# ---------------------------------------------------------------------------
# Path 2 -- redact-before-persist arm
# ---------------------------------------------------------------------------


@pytest.fixture
def adapter_with_mock_graph() -> tuple[HandlerIntentStorageAdapter, AsyncMock]:
    handler = HandlerIntentStorageAdapter()
    graph = AsyncMock()
    graph.store_intent.return_value = ModelIntentStorageResult(
        success=True, intent_id=uuid4(), created=True
    )
    handler._adapter = graph
    return handler, graph


@pytest.mark.unit
class TestRedactBeforePersistArm:
    """Pin what the adapter's PII arm actually does before persisting."""

    async def test_clean_context_persists_and_calls_the_graph_adapter(
        self, adapter_with_mock_graph: tuple[HandlerIntentStorageAdapter, AsyncMock]
    ) -> None:
        handler, graph = adapter_with_mock_graph
        response = await handler._store_intent(_request("debugging a failing test"))
        assert response.status == "success"
        graph.store_intent.assert_awaited_once()

    async def test_user_context_never_reaches_the_graph_adapter(
        self, adapter_with_mock_graph: tuple[HandlerIntentStorageAdapter, AsyncMock]
    ) -> None:
        """OMN-17236 D4: ``sanitized_context`` is computed and discarded.

        ``AdapterIntentGraph.store_intent`` takes only ``session_id``,
        ``intent_data`` and ``correlation_id``. Neither the raw nor the
        sanitized ``user_context`` is passed, so the redaction arm has no
        effect on what is stored. Pinned, not fixed, in this lane.
        """
        handler, graph = adapter_with_mock_graph
        context = "debugging a failing test"
        await handler._store_intent(_request(context))

        kwargs = graph.store_intent.await_args.kwargs
        assert set(kwargs) == {"session_id", "intent_data", "correlation_id"}
        assert context not in str(kwargs)

    async def test_pii_bearing_request_cannot_reach_this_arm(self) -> None:
        """The validator refuses first, so the arm is unreachable with PII."""
        with pytest.raises(ValidationError):
            _request("reach me at alice@example.com")

    async def test_supplied_correlation_id_is_forwarded(
        self, adapter_with_mock_graph: tuple[HandlerIntentStorageAdapter, AsyncMock]
    ) -> None:
        handler, graph = adapter_with_mock_graph
        correlation_id = uuid4()
        request = ModelIntentStorageRequest(
            operation="store",
            session_id="session_123",
            intent_data=_intent_data(),
            correlation_id=correlation_id,
            user_context="clean context",
        )
        await handler._store_intent(request)
        assert graph.store_intent.await_args.kwargs["correlation_id"] == str(
            correlation_id
        )

    async def test_missing_session_id_is_rejected_before_persisting(
        self, adapter_with_mock_graph: tuple[HandlerIntentStorageAdapter, AsyncMock]
    ) -> None:
        handler, graph = adapter_with_mock_graph
        request = ModelIntentStorageRequest.model_construct(
            operation="store",
            session_id=None,
            intent_data=_intent_data(),
            user_context="",
        )
        with pytest.raises(ValueError, match="session_id is required"):
            await handler._store_intent(request)
        graph.store_intent.assert_not_awaited()

    async def test_uninitialized_adapter_is_rejected(self) -> None:
        handler = HandlerIntentStorageAdapter()
        with pytest.raises(ValueError, match="Adapter not initialized"):
            await handler._store_intent(_request("clean context"))
