# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Tests for frozen boundary model enforcement (OMN-2219).

Architecture handshake rule #5: all boundary-crossing models (events, intents,
actions, envelopes, projections) MUST use frozen=True with extra="forbid"
(or extra="ignore" for consumer models that need forward compatibility).

These tests verify:
1. Frozen models reject field mutation after construction
2. Boundary models reject unknown fields (or ignore them for consumers)
3. All event/boundary models have the correct ConfigDict settings
"""

from datetime import datetime, timezone
from uuid import uuid4

import pytest
from omnibase_core.models.primitives import ModelSemVer
from pydantic import ValidationError

from omnimemory.models.events.model_intent_classified_event import (
    ModelIntentClassifiedEvent,
)
from omnimemory.models.foundation.model_audit_metadata import (
    AuditEventDetails,
    PerformanceAuditDetails,
    ResourceUsageMetadata,
    SecurityAuditDetails,
)
from omnimemory.models.foundation.model_typed_collections import (
    ModelEventCollection,
    ModelEventData,
)
from omnimemory.models.subscription.model_notification_event import (
    ModelNotificationEvent,
)
from omnimemory.models.subscription.model_notification_event_payload import (
    ModelNotificationEventPayload,
)
from omnimemory.models.utils.model_audit import (
    AuditEventType,
    AuditSeverity,
    ModelAuditEvent,
)


@pytest.mark.unit
class TestModelNotificationEventFrozen:
    """Verify ModelNotificationEvent is frozen and rejects mutation."""

    def _make_event(self) -> ModelNotificationEvent:
        return ModelNotificationEvent(
            event_id="evt-001",
            topic="memory.item.created",
            payload=ModelNotificationEventPayload(
                entity_type="item",
                entity_id="item-001",
                action="created",
            ),
        )

    def test_rejects_field_mutation(self) -> None:
        event = self._make_event()
        with pytest.raises(ValidationError):
            event.event_id = "evt-002"  # type: ignore[misc]

    def test_rejects_payload_mutation(self) -> None:
        event = self._make_event()
        with pytest.raises(ValidationError):
            event.topic = "memory.item.updated"  # type: ignore[misc]

    def test_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            ModelNotificationEvent(
                event_id="evt-001",
                topic="memory.item.created",
                payload=ModelNotificationEventPayload(
                    entity_type="item",
                    entity_id="item-001",
                    action="created",
                ),
                unknown_field="should fail",  # type: ignore[call-arg]
            )

    def test_construction_succeeds(self) -> None:
        event = self._make_event()
        assert event.event_id == "evt-001"
        assert event.topic == "memory.item.created"
        assert event.payload.entity_type == "item"


@pytest.mark.unit
class TestModelNotificationEventPayloadFrozen:
    """Verify ModelNotificationEventPayload is frozen."""

    def test_rejects_field_mutation(self) -> None:
        payload = ModelNotificationEventPayload(
            entity_type="item",
            entity_id="item-001",
            action="created",
        )
        with pytest.raises(ValidationError):
            payload.action = "updated"  # type: ignore[misc]

    def test_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            ModelNotificationEventPayload(
                entity_type="item",
                entity_id="item-001",
                action="created",
                rogue_field="nope",  # type: ignore[call-arg]
            )


@pytest.mark.unit
class TestModelIntentClassifiedEventFrozen:
    """Verify ModelIntentClassifiedEvent is frozen with extra=ignore."""

    def _make_event(self) -> ModelIntentClassifiedEvent:
        return ModelIntentClassifiedEvent(
            event_type="IntentClassified",
            session_id="sess-001",
            correlation_id=uuid4(),
            intent_category="debugging",
            confidence=0.85,
            timestamp=datetime.now(timezone.utc),
        )

    def test_rejects_field_mutation(self) -> None:
        event = self._make_event()
        with pytest.raises(ValidationError):
            event.confidence = 0.99  # type: ignore[misc]

    def test_rejects_session_id_mutation(self) -> None:
        event = self._make_event()
        with pytest.raises(ValidationError):
            event.session_id = "sess-002"  # type: ignore[misc]

    def test_ignores_unknown_fields_for_forward_compat(self) -> None:
        """Consumer model uses extra='ignore' for forward compatibility."""
        event = ModelIntentClassifiedEvent(
            event_type="IntentClassified",
            session_id="sess-001",
            correlation_id=uuid4(),
            intent_category="debugging",
            confidence=0.85,
            timestamp=datetime.now(timezone.utc),
            future_field="from upstream v2",  # type: ignore[call-arg]
        )
        assert not hasattr(event, "future_field")

    def test_construction_succeeds(self) -> None:
        event = self._make_event()
        assert event.session_id == "sess-001"
        assert event.intent_category == "debugging"


@pytest.mark.unit
class TestModelAuditEventFrozen:
    """Verify ModelAuditEvent is frozen."""

    def _make_event(self) -> ModelAuditEvent:
        return ModelAuditEvent(
            event_id="audit-001",
            timestamp=datetime.now(timezone.utc),
            event_type=AuditEventType.MEMORY_STORE,
            severity=AuditSeverity.LOW,
            operation="memory_store",
            component="memory_manager",
            message="Memory store succeeded",
        )

    def test_rejects_field_mutation(self) -> None:
        event = self._make_event()
        with pytest.raises(ValidationError):
            event.message = "changed"  # type: ignore[misc]

    def test_rejects_severity_mutation(self) -> None:
        event = self._make_event()
        with pytest.raises(ValidationError):
            event.severity = AuditSeverity.CRITICAL  # type: ignore[misc]

    def test_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            ModelAuditEvent(
                event_id="audit-001",
                timestamp=datetime.now(timezone.utc),
                event_type=AuditEventType.MEMORY_STORE,
                severity=AuditSeverity.LOW,
                operation="memory_store",
                component="memory_manager",
                message="test",
                rogue="nope",  # type: ignore[call-arg]
            )

    def test_construction_succeeds(self) -> None:
        event = self._make_event()
        assert event.event_id == "audit-001"
        assert event.event_type == AuditEventType.MEMORY_STORE


@pytest.mark.unit
class TestAuditMetadataModelsFrozen:
    """Verify audit metadata sub-models are frozen."""

    def test_audit_event_details_rejects_mutation(self) -> None:
        details = AuditEventDetails(operation_type="memory_store")
        with pytest.raises(ValidationError):
            details.operation_type = "changed"  # type: ignore[misc]

    def test_resource_usage_metadata_rejects_mutation(self) -> None:
        usage = ResourceUsageMetadata(cpu_usage_percent=50.0)
        with pytest.raises(ValidationError):
            usage.cpu_usage_percent = 99.0  # type: ignore[misc]

    def test_security_audit_details_rejects_mutation(self) -> None:
        security = SecurityAuditDetails(permission_granted=True)
        with pytest.raises(ValidationError):
            security.permission_granted = False  # type: ignore[misc]

    def test_performance_audit_details_rejects_mutation(self) -> None:
        perf = PerformanceAuditDetails(operation_latency_ms=10.0)
        with pytest.raises(ValidationError):
            perf.operation_latency_ms = 999.0  # type: ignore[misc]


@pytest.mark.unit
class TestModelEventDataFrozen:
    """Verify ModelEventData is frozen."""

    def _make_event_data(self) -> ModelEventData:
        return ModelEventData(
            event_type="creation",
            timestamp="2026-01-01T00:00:00Z",
            source="test",
            message="Test event",
        )

    def test_rejects_field_mutation(self) -> None:
        data = self._make_event_data()
        with pytest.raises(ValidationError):
            data.message = "changed"  # type: ignore[misc]

    def test_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            ModelEventData(
                event_type="creation",
                timestamp="2026-01-01T00:00:00Z",
                source="test",
                message="Test event",
                rogue="nope",  # type: ignore[call-arg]
            )

    def test_construction_succeeds(self) -> None:
        data = self._make_event_data()
        assert data.event_type == "creation"
        assert data.source == "test"


@pytest.mark.unit
class TestModelEventCollectionMutable:
    """Verify ModelEventCollection remains mutable (builder pattern)."""

    def test_add_event_works(self) -> None:
        """Collection is a builder, not a boundary model -- stays mutable."""
        collection = ModelEventCollection()
        collection.add_event(
            event_type="creation",
            timestamp="2026-01-01T00:00:00Z",
            source="test",
            message="Test event",
        )
        assert len(collection.events) == 1
        assert collection.events[0].event_type == "creation"

    def test_multiple_events(self) -> None:
        collection = ModelEventCollection()
        collection.add_event(
            event_type="creation",
            timestamp="2026-01-01T00:00:00Z",
            source="test",
            message="First",
        )
        collection.add_event(
            event_type="update",
            timestamp="2026-01-01T01:00:00Z",
            source="test",
            message="Second",
        )
        assert len(collection.events) == 2

    def test_config_not_frozen(self) -> None:
        """Guard-rail: ModelEventCollection must NOT be frozen (it's a builder)."""
        assert ModelEventCollection.model_config.get("frozen") is not True

    def test_individual_events_in_collection_are_frozen(self) -> None:
        """Even though the collection is mutable, the events inside are frozen."""
        collection = ModelEventCollection()
        collection.add_event(
            event_type="creation",
            timestamp="2026-01-01T00:00:00Z",
            source="test",
            message="Test event",
        )
        with pytest.raises(ValidationError):
            collection.events[0].message = "changed"  # type: ignore[misc]


@pytest.mark.unit
class TestModelNotificationEventVersionSemVer:
    """Verify ModelNotificationEvent.version is ModelSemVer with round-trip fidelity."""

    def test_version_is_model_semver(self) -> None:
        """Assert the version field is an instance of ModelSemVer."""
        event = ModelNotificationEvent(
            event_id="evt-ver-001",
            topic="memory.item.created",
            payload=ModelNotificationEventPayload(
                entity_type="item",
                entity_id="item-001",
                action="created",
            ),
        )
        assert isinstance(event.version, ModelSemVer)

    def test_version_round_trip_serialization(self) -> None:
        """Serialize to JSON, verify version structure, round-trip back.

        Uses model_dump_json / model_validate_json because
        ModelNotificationEvent has strict=True, which rejects coerced
        types (e.g. datetime from string) through model_validate(dict).
        """
        event = ModelNotificationEvent(
            event_id="evt-ver-002",
            topic="memory.item.updated",
            payload=ModelNotificationEventPayload(
                entity_type="item",
                entity_id="item-002",
                action="updated",
            ),
            version=ModelSemVer(major=2, minor=3, patch=4),
        )

        # Verify the dict representation has the expected semver structure
        serialized = event.model_dump(mode="json")
        assert isinstance(serialized["version"], dict)
        assert serialized["version"]["major"] == 2
        assert serialized["version"]["minor"] == 3
        assert serialized["version"]["patch"] == 4

        # Round-trip through JSON string to honour strict=True on the model
        json_str = event.model_dump_json()
        restored = ModelNotificationEvent.model_validate_json(json_str)
        assert isinstance(restored.version, ModelSemVer)
        assert restored.version.major == 2
        assert restored.version.minor == 3
        assert restored.version.patch == 4


@pytest.mark.unit
class TestFrozenBoundaryModelConfigGuardRails:
    """Guard-rail: introspect model_config to assert frozen=True on boundary models.

    This prevents accidental regression where frozen=True is removed from a
    boundary-crossing model's ConfigDict.  Mirrors the inverse test
    ``TestModelEventCollectionMutable.test_config_not_frozen`` which guards
    that the builder model stays mutable.
    """

    def test_model_intent_classified_event_config_frozen(self) -> None:
        assert ModelIntentClassifiedEvent.model_config.get("frozen") is True

    def test_audit_event_details_config_frozen(self) -> None:
        assert AuditEventDetails.model_config.get("frozen") is True

    def test_model_event_data_config_frozen(self) -> None:
        assert ModelEventData.model_config.get("frozen") is True

    def test_model_notification_event_config_frozen(self) -> None:
        assert ModelNotificationEvent.model_config.get("frozen") is True

    def test_model_notification_event_payload_config_frozen(self) -> None:
        assert ModelNotificationEventPayload.model_config.get("frozen") is True

    def test_model_audit_event_config_frozen(self) -> None:
        assert ModelAuditEvent.model_config.get("frozen") is True

    def test_resource_usage_metadata_config_frozen(self) -> None:
        assert ResourceUsageMetadata.model_config.get("frozen") is True

    def test_security_audit_details_config_frozen(self) -> None:
        assert SecurityAuditDetails.model_config.get("frozen") is True

    def test_performance_audit_details_config_frozen(self) -> None:
        assert PerformanceAuditDetails.model_config.get("frozen") is True


@pytest.mark.unit
class TestFrozenModelMutableContainerLimitation:
    """Document known Pydantic v2 limitation: frozen prevents field reassignment but not container mutation.

    These tests exist to *document* the behaviour, not endorse it.
    See inline comments on dict fields in the model modules for the rationale.
    """

    def test_frozen_model_dict_contents_remain_mutable(self) -> None:
        """Dict values inside a frozen model can still be mutated in-place."""
        event = ModelNotificationEvent(
            event_id="evt-mut-001",
            topic="memory.item.created",
            payload=ModelNotificationEventPayload(
                entity_type="item",
                entity_id="item-001",
                action="created",
            ),
            metadata={"key": "value"},
        )
        # frozen=True prevents field reassignment
        with pytest.raises(ValidationError):
            event.metadata = {"new": "dict"}  # type: ignore[misc]
        # But dict contents are still mutable (known Pydantic v2 limitation)
        assert event.metadata is not None
        event.metadata["key"] = "mutated"
        assert event.metadata["key"] == "mutated"

    def test_frozen_model_tuple_contents_are_immutable(self) -> None:
        """Tuple fields (e.g. security_scan_results) are truly immutable."""
        details = SecurityAuditDetails(
            security_scan_results=("clean", "no_issues"),
        )
        with pytest.raises(ValidationError):
            details.security_scan_results = ("replaced",)  # type: ignore[misc]
        # Tuple contents cannot be mutated -- this is the safer pattern
        assert details.security_scan_results == ("clean", "no_issues")
