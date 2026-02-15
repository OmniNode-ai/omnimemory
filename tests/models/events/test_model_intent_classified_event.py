# SPDX-FileCopyrightText: 2025-present OmniNode-ai <info@omninode.ai>
# SPDX-License-Identifier: MIT
"""Tests for ModelIntentClassifiedEvent JSON deserialization.

These tests validate that the model correctly handles JSON-deserialized payloads
from Kafka, where UUIDs and datetimes arrive as strings rather than native types.
"""

import json

import pytest

from omnimemory.models.events.model_intent_classified_event import (
    ModelIntentClassifiedEvent,
)


@pytest.mark.unit
class TestModelIntentClassifiedEventJsonDeserialization:
    """Tests for JSON deserialization behavior (the Kafka use case)."""

    def test_validates_json_string_payload(self) -> None:
        """Ensure model works with json.loads() output (strings for UUID/datetime).

        This is the critical test for Kafka message handling. JSON.loads() returns
        strings for UUID and datetime fields, and the model must coerce them
        to the proper types without strict=True blocking the conversion.
        """
        raw_json = """
        {
          "event_type": "IntentClassified",
          "session_id": "sess-123",
          "correlation_id": "550e8400-e29b-41d4-a716-446655440000",
          "intent_category": "debugging",
          "confidence": 0.85,
          "timestamp": "2026-01-27T15:30:00+00:00"
        }
        """
        data = json.loads(raw_json)
        event = ModelIntentClassifiedEvent.model_validate(data)

        assert str(event.correlation_id) == "550e8400-e29b-41d4-a716-446655440000"
        assert event.session_id == "sess-123"
        assert event.intent_category == "debugging"
        assert event.confidence == 0.85
        assert event.event_type == "IntentClassified"

    def test_ignores_unknown_fields(self) -> None:
        """Ensure extra='ignore' allows forward compatibility with new upstream fields.

        The omniintelligence service may add new fields in future versions.
        With extra='ignore', omnimemory will continue to function without
        requiring immediate updates when upstream adds fields.
        """
        data = {
            "event_type": "IntentClassified",
            "session_id": "sess-123",
            "correlation_id": "550e8400-e29b-41d4-a716-446655440000",
            "intent_category": "debugging",
            "confidence": 0.85,
            "timestamp": "2026-01-27T15:30:00+00:00",
            "new_field_from_future": "should be ignored",
            "another_unknown_field": {"nested": "data"},
        }
        event = ModelIntentClassifiedEvent.model_validate(data)

        assert event.session_id == "sess-123"
        assert not hasattr(event, "new_field_from_future")
        assert not hasattr(event, "another_unknown_field")

    def test_validates_with_keywords_field(self) -> None:
        """Ensure optional keywords field is handled correctly."""
        data = {
            "event_type": "IntentClassified",
            "session_id": "sess-456",
            "correlation_id": "550e8400-e29b-41d4-a716-446655440001",
            "intent_category": "code_generation",
            "confidence": 0.92,
            "timestamp": "2026-01-27T16:00:00+00:00",
            "keywords": ["python", "fastapi", "async"],
        }
        event = ModelIntentClassifiedEvent.model_validate(data)

        assert event.keywords == ("python", "fastapi", "async")

    def test_keywords_defaults_to_empty_tuple(self) -> None:
        """Ensure keywords field defaults to empty tuple when not provided."""
        data = {
            "event_type": "IntentClassified",
            "session_id": "sess-789",
            "correlation_id": "550e8400-e29b-41d4-a716-446655440002",
            "intent_category": "testing",
            "confidence": 0.75,
            "timestamp": "2026-01-27T17:00:00+00:00",
        }
        event = ModelIntentClassifiedEvent.model_validate(data)

        assert event.keywords == ()


@pytest.mark.unit
class TestModelIntentClassifiedEventValidation:
    """Tests for validation constraints."""

    def test_rejects_invalid_confidence_above_1(self) -> None:
        """Ensure confidence > 1.0 is rejected."""
        data = {
            "event_type": "IntentClassified",
            "session_id": "sess-123",
            "correlation_id": "550e8400-e29b-41d4-a716-446655440000",
            "intent_category": "debugging",
            "confidence": 1.5,
            "timestamp": "2026-01-27T15:30:00+00:00",
        }
        with pytest.raises(ValueError):
            ModelIntentClassifiedEvent.model_validate(data)

    def test_rejects_invalid_confidence_below_0(self) -> None:
        """Ensure confidence < 0.0 is rejected."""
        data = {
            "event_type": "IntentClassified",
            "session_id": "sess-123",
            "correlation_id": "550e8400-e29b-41d4-a716-446655440000",
            "intent_category": "debugging",
            "confidence": -0.1,
            "timestamp": "2026-01-27T15:30:00+00:00",
        }
        with pytest.raises(ValueError):
            ModelIntentClassifiedEvent.model_validate(data)

    def test_rejects_empty_session_id(self) -> None:
        """Ensure empty session_id is rejected."""
        data = {
            "event_type": "IntentClassified",
            "session_id": "",
            "correlation_id": "550e8400-e29b-41d4-a716-446655440000",
            "intent_category": "debugging",
            "confidence": 0.85,
            "timestamp": "2026-01-27T15:30:00+00:00",
        }
        with pytest.raises(ValueError):
            ModelIntentClassifiedEvent.model_validate(data)

    def test_rejects_invalid_uuid_format(self) -> None:
        """Ensure malformed UUID is rejected."""
        data = {
            "event_type": "IntentClassified",
            "session_id": "sess-123",
            "correlation_id": "not-a-valid-uuid",
            "intent_category": "debugging",
            "confidence": 0.85,
            "timestamp": "2026-01-27T15:30:00+00:00",
        }
        with pytest.raises(ValueError):
            ModelIntentClassifiedEvent.model_validate(data)
