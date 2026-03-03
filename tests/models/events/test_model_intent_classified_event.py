# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Tests for ModelIntentClassifiedEvent JSON deserialization.

These tests validate that the model correctly handles JSON-deserialized payloads
from Kafka, where UUIDs and datetimes arrive as strings rather than native types.
"""

import json

import pytest
from omnibase_core.enums.intelligence import EnumIntentClass

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
          "emitted_at": "2026-01-27T15:30:00+00:00"
        }
        """
        data = json.loads(raw_json)
        event = ModelIntentClassifiedEvent.model_validate(data)

        assert str(event.correlation_id) == "550e8400-e29b-41d4-a716-446655440000"
        assert event.session_id == "sess-123"
        # "debugging" maps to EnumIntentClass.BUGFIX via _INTENT_CLASS_ALIASES (OMN-3248)
        assert event.intent_class == EnumIntentClass.BUGFIX
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
            "emitted_at": "2026-01-27T15:30:00+00:00",
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
            "emitted_at": "2026-01-27T16:00:00+00:00",
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
            "emitted_at": "2026-01-27T17:00:00+00:00",
        }
        event = ModelIntentClassifiedEvent.model_validate(data)

        assert event.keywords == ()

    def test_accepts_null_correlation_id(self) -> None:
        """Ensure null correlation_id is accepted (OMN-2841).

        omniintelligence publishes ModelPatternLifecycleEvent with
        correlation_id: UUID | None. This consumer must not fail when
        the upstream omits correlation_id or sends null.
        """
        data = {
            "event_type": "IntentClassified",
            "session_id": "sess-null-corr",
            "correlation_id": None,
            "intent_category": "debugging",
            "confidence": 0.9,
            "emitted_at": "2026-01-27T18:00:00+00:00",
        }
        event = ModelIntentClassifiedEvent.model_validate(data)

        assert event.correlation_id is None
        assert event.session_id == "sess-null-corr"

    def test_accepts_absent_correlation_id(self) -> None:
        """Ensure missing correlation_id defaults to None (OMN-2841).

        If the upstream event omits the field entirely, deserialization
        must succeed with correlation_id defaulting to None.
        """
        data = {
            "event_type": "IntentClassified",
            "session_id": "sess-no-corr",
            "intent_category": "debugging",
            "confidence": 0.88,
            "emitted_at": "2026-01-27T18:30:00+00:00",
        }
        event = ModelIntentClassifiedEvent.model_validate(data)

        assert event.correlation_id is None
        assert event.session_id == "sess-no-corr"


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
            "emitted_at": "2026-01-27T15:30:00+00:00",
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
            "emitted_at": "2026-01-27T15:30:00+00:00",
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
            "emitted_at": "2026-01-27T15:30:00+00:00",
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
            "emitted_at": "2026-01-27T15:30:00+00:00",
        }
        with pytest.raises(ValueError):
            ModelIntentClassifiedEvent.model_validate(data)


@pytest.mark.unit
class TestModelIntentClassifiedEventFieldNormalization:
    """Tests for OMN-3248: intent_class / intent_category field split and normalization."""

    _BASE: dict[str, object] = {
        "event_type": "IntentClassified",
        "session_id": "sess-omn-3248",
        "correlation_id": "550e8400-e29b-41d4-a716-446655440000",
        "confidence": 0.85,
        "emitted_at": "2026-01-27T15:30:00+00:00",
    }

    def _payload(self, **extra: object) -> dict[str, object]:
        return {**self._BASE, **extra}

    def test_parses_canonical_intent_class(self) -> None:
        """New wire format: ``intent_class`` present → parsed directly."""
        m = ModelIntentClassifiedEvent.model_validate(
            self._payload(intent_class="feature")
        )
        assert m.intent_class == EnumIntentClass.FEATURE

    def test_parses_legacy_intent_category(self) -> None:
        """Legacy wire format: ``intent_category`` present → normalized to intent_class."""
        m = ModelIntentClassifiedEvent.model_validate(
            self._payload(intent_category="feature")
        )
        assert m.intent_class == EnumIntentClass.FEATURE

    def test_both_fields_canonical_wins(self) -> None:
        """When both fields present, ``intent_class`` wins over ``intent_category``."""
        m = ModelIntentClassifiedEvent.model_validate(
            self._payload(intent_class="feature", intent_category="bugfix")
        )
        assert m.intent_class == EnumIntentClass.FEATURE

    def test_alias_normalization(self) -> None:
        """Legacy alias ``feat`` maps to ``EnumIntentClass.FEATURE``."""
        m = ModelIntentClassifiedEvent.model_validate(
            self._payload(intent_category="feat")
        )
        assert m.intent_class == EnumIntentClass.FEATURE
