"""Tests for event mapper utility."""

from datetime import datetime, timezone
from uuid import UUID

import pytest

from omnimemory.models.events import ModelIntentClassifiedEvent
from omnimemory.nodes.intent_event_consumer_effect.utils import (
    map_event_to_storage_request,
)


@pytest.mark.unit
class TestMapEventToStorageRequest:
    """Tests for map_event_to_storage_request function."""

    def test_maps_basic_event_correctly(self) -> None:
        """Test mapping a basic event without keywords."""
        event = ModelIntentClassifiedEvent(
            event_type="IntentClassified",
            session_id="session_123",
            correlation_id=UUID("550e8400-e29b-41d4-a716-446655440000"),
            intent_category="debugging",
            confidence=0.85,
            timestamp=datetime.now(timezone.utc),
        )

        result = map_event_to_storage_request(event)

        assert result.operation == "store"
        assert result.session_id == "session_123"
        assert result.correlation_id == UUID("550e8400-e29b-41d4-a716-446655440000")
        assert result.intent_data is not None
        assert result.intent_data.intent_category == "debugging"
        assert result.intent_data.confidence == 0.85
        assert isinstance(result.intent_data.keywords, list)
        assert result.intent_data.keywords == []

    def test_maps_event_with_keywords_forward_compat(self) -> None:
        """Test mapping event with keywords (OMN-1626 forward compatibility)."""
        event = ModelIntentClassifiedEvent(
            event_type="IntentClassified",
            session_id="session_456",
            correlation_id=UUID("660e8400-e29b-41d4-a716-446655440001"),
            intent_category="code_review",
            confidence=0.92,
            keywords=["pull", "request", "review"],
            timestamp=datetime.now(timezone.utc),
        )

        result = map_event_to_storage_request(event)

        assert result.intent_data is not None
        assert isinstance(result.intent_data.keywords, list)
        assert result.intent_data.keywords == ["pull", "request", "review"]

    def test_preserves_all_fields(self) -> None:
        """Test that all event fields are preserved in mapping."""
        correlation_id = UUID("770e8400-e29b-41d4-a716-446655440002")
        event = ModelIntentClassifiedEvent(
            event_type="IntentClassified",
            session_id="session_789",
            correlation_id=correlation_id,
            intent_category="code_generation",  # Valid EnumIntentCategory value
            confidence=0.77,
            keywords=["add", "feature"],
            timestamp=datetime.now(timezone.utc),
        )

        result = map_event_to_storage_request(event)

        # Verify all mapped fields
        assert result.session_id == event.session_id
        assert result.correlation_id == event.correlation_id
        assert result.intent_data is not None
        # intent_category is now an enum, compare with .value
        assert result.intent_data.intent_category.value == event.intent_category
        assert result.intent_data.confidence == event.confidence
        assert result.intent_data.keywords == list(event.keywords)

    def test_maps_empty_keywords_correctly(self) -> None:
        """Test mapping event with explicit empty keywords list."""
        event = ModelIntentClassifiedEvent(
            event_type="IntentClassified",
            session_id="session_empty",
            correlation_id=UUID("880e8400-e29b-41d4-a716-446655440003"),
            intent_category="explanation",
            confidence=0.65,
            keywords=[],
            timestamp=datetime.now(timezone.utc),
        )

        result = map_event_to_storage_request(event)

        assert result.intent_data is not None
        assert result.intent_data.keywords == []
        assert isinstance(result.intent_data.keywords, list)


@pytest.mark.unit
class TestModelIntentClassifiedEvent:
    """Tests for the ModelIntentClassifiedEvent model."""

    def test_keywords_defaults_to_empty_tuple(self) -> None:
        """Test that keywords field defaults to empty tuple."""
        event = ModelIntentClassifiedEvent(
            session_id="session_test",
            correlation_id=UUID("880e8400-e29b-41d4-a716-446655440003"),
            intent_category="testing",
            confidence=0.5,
            timestamp=datetime.now(timezone.utc),
        )

        assert event.keywords == ()
        assert isinstance(event.keywords, tuple)

    def test_validates_confidence_range_too_high(self) -> None:
        """Test that confidence must be <= 1.0."""
        with pytest.raises(ValueError):
            ModelIntentClassifiedEvent(
                session_id="session_test",
                correlation_id=UUID("990e8400-e29b-41d4-a716-446655440004"),
                intent_category="testing",
                confidence=1.5,  # Invalid - > 1.0
                timestamp=datetime.now(timezone.utc),
            )

    def test_validates_confidence_range_too_low(self) -> None:
        """Test that confidence must be >= 0.0."""
        with pytest.raises(ValueError):
            ModelIntentClassifiedEvent(
                session_id="session_test",
                correlation_id=UUID("990e8400-e29b-41d4-a716-446655440005"),
                intent_category="testing",
                confidence=-0.1,  # Invalid - < 0.0
                timestamp=datetime.now(timezone.utc),
            )

    def test_validates_session_id_not_empty(self) -> None:
        """Test that session_id cannot be empty string."""
        with pytest.raises(ValueError):
            ModelIntentClassifiedEvent(
                session_id="",  # Invalid - empty string
                correlation_id=UUID("aa0e8400-e29b-41d4-a716-446655440005"),
                intent_category="testing",
                confidence=0.5,
                timestamp=datetime.now(timezone.utc),
            )

    def test_validates_intent_category_not_empty(self) -> None:
        """Test that intent_category cannot be empty string."""
        with pytest.raises(ValueError):
            ModelIntentClassifiedEvent(
                session_id="session_test",
                correlation_id=UUID("bb0e8400-e29b-41d4-a716-446655440006"),
                intent_category="",  # Invalid - empty string
                confidence=0.5,
                timestamp=datetime.now(timezone.utc),
            )

    def test_event_type_defaults_to_intent_classified(self) -> None:
        """Test that event_type has correct default value."""
        event = ModelIntentClassifiedEvent(
            session_id="session_default",
            correlation_id=UUID("cc0e8400-e29b-41d4-a716-446655440007"),
            intent_category="testing",
            confidence=0.5,
            timestamp=datetime.now(timezone.utc),
        )

        assert event.event_type == "IntentClassified"

    def test_accepts_valid_confidence_boundary_values(self) -> None:
        """Test that confidence accepts boundary values 0.0 and 1.0."""
        # Test 0.0
        event_zero = ModelIntentClassifiedEvent(
            session_id="session_zero",
            correlation_id=UUID("dd0e8400-e29b-41d4-a716-446655440008"),
            intent_category="testing",
            confidence=0.0,
            timestamp=datetime.now(timezone.utc),
        )
        assert event_zero.confidence == 0.0

        # Test 1.0
        event_one = ModelIntentClassifiedEvent(
            session_id="session_one",
            correlation_id=UUID("ee0e8400-e29b-41d4-a716-446655440009"),
            intent_category="testing",
            confidence=1.0,
            timestamp=datetime.now(timezone.utc),
        )
        assert event_one.confidence == 1.0
