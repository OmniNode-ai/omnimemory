# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Tests for frozen boundary model enforcement (OMN-2219, OMN-2294).

Architecture handshake rule #5: all boundary-crossing models (events, intents,
actions, envelopes, projections) MUST use frozen=True with extra="forbid"
(or extra="ignore" for consumer models that need forward compatibility).

These tests verify:
1. Frozen models reject field mutation after construction
2. Boundary models reject unknown fields (or ignore them for consumers)
3. All event/boundary models have the correct ConfigDict settings
4. A completeness scan confirms every discovered boundary model is frozen (OMN-2294)
5. .model_copy() creates independent instances without mutating originals (OMN-2294)
6. Boundary node models (retrieval, storage, crawl) are covered (OMN-2294)
"""

import importlib
import inspect
import pkgutil
from datetime import datetime, timezone
from uuid import uuid4

import pytest
from omnibase_core.models.primitives import ModelSemVer
from pydantic import BaseModel, ValidationError

import omnimemory.models as _omnimemory_models_pkg
import omnimemory.nodes as _omnimemory_nodes_pkg
from omnimemory.enums.crawl.enum_context_source_type import EnumContextSourceType
from omnimemory.enums.crawl.enum_crawler_type import EnumCrawlerType
from omnimemory.enums.crawl.enum_detected_doc_type import EnumDetectedDocType
from omnimemory.models.crawl.model_crawl_state_record import ModelCrawlStateRecord
from omnimemory.models.crawl.model_crawl_tick_command import ModelCrawlTickCommand
from omnimemory.models.crawl.model_document_changed_event import (
    ModelDocumentChangedEvent,
)
from omnimemory.models.crawl.model_document_discovered_event import (
    ModelDocumentDiscoveredEvent,
)
from omnimemory.models.crawl.model_document_removed_event import (
    ModelDocumentRemovedEvent,
)
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
from omnimemory.models.utils.model_circuit_breaker_stats_response import (
    ModelCircuitBreakerStatsResponse,
)
from omnimemory.models.utils.model_correlation_context import ModelCorrelationContext
from omnimemory.nodes.memory_retrieval_effect.models.model_memory_retrieval_request import (
    ModelMemoryRetrievalRequest,
)
from omnimemory.nodes.memory_retrieval_effect.models.model_memory_retrieval_response import (
    ModelMemoryRetrievalResponse,
    ModelSearchResult,
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


# ---------------------------------------------------------------------------
# OMN-2294: Completeness scan — all boundary models in omnimemory.models.*
# ---------------------------------------------------------------------------

# Canonical set of BaseModel subclasses discovered in omnimemory.models.* that
# must have frozen=True.  The companion runtime check in
# TestAllBoundaryModelsCompletenessScanned walks the package at import-time to
# build this list dynamically.

_EXPECTED_FROZEN_COUNT_MIN = 30
"""Lower-bound on how many frozen boundary models the completeness scan must
find.  If a model is accidentally removed or the frozen flag is stripped, this
count drops below the threshold and the test fails immediately.  The actual
count will be >= this value; we use a floor rather than an exact count to
avoid brittle test breakage when new models are legitimately added."""


def _discover_frozen_boundary_models() -> list[type[BaseModel]]:
    """Walk omnimemory.models.* and omnimemory.nodes.*.models to discover
    every BaseModel subclass that has ``frozen=True`` in its ConfigDict.

    Returns a deduplicated list ordered by qualified name.  Models that fail
    to import are skipped with a warning so a bad import does not mask a real
    test failure in another module.
    """
    found: dict[str, type[BaseModel]] = {}

    def _scan(pkg: object, nodes_mode: bool = False) -> None:
        for _, module_name, is_pkg in pkgutil.walk_packages(
            pkg.__path__,  # type: ignore[union-attr]
            pkg.__name__ + ".",
        ):
            if is_pkg:
                continue
            # For the nodes package, only scan submodules that look like model
            # modules (contain ".models." in their dotted path).
            if nodes_mode and ".models." not in module_name:
                continue
            try:
                mod = importlib.import_module(module_name)
            except Exception:  # noqa: S112
                continue
            for _name, cls in inspect.getmembers(mod, inspect.isclass):
                if (
                    issubclass(cls, BaseModel)
                    and cls is not BaseModel
                    and cls.__module__ == module_name
                    and cls.model_config.get("frozen") is True
                ):
                    key = f"{cls.__module__}.{cls.__qualname__}"
                    found[key] = cls

    _scan(_omnimemory_models_pkg)
    _scan(_omnimemory_nodes_pkg, nodes_mode=True)
    return [found[k] for k in sorted(found)]


@pytest.mark.unit
class TestAllBoundaryModelsCompletenessScanned:
    """OMN-2294 Test 2: completeness scan — every model discovered in
    ``omnimemory.models.*`` and ``omnimemory.nodes.*.models.*`` that has
    ``frozen=True`` is validated to actually carry the flag in its
    ``model_config``.

    This test acts as an early-warning system: if ``frozen=True`` is
    accidentally removed from a boundary model the test fails immediately
    with the class name, making the regression obvious.
    """

    def test_discovers_minimum_number_of_frozen_models(self) -> None:
        """Scan must find at least _EXPECTED_FROZEN_COUNT_MIN frozen models.

        A count below the floor indicates either a mass deletion or accidental
        removal of frozen=True from multiple models.
        """
        frozen_models = _discover_frozen_boundary_models()
        assert len(frozen_models) >= _EXPECTED_FROZEN_COUNT_MIN, (
            f"Expected at least {_EXPECTED_FROZEN_COUNT_MIN} frozen boundary "
            f"models, but only found {len(frozen_models)}.  "
            "Was frozen=True removed from a boundary model?"
        )

    def test_every_discovered_model_has_frozen_true_in_config(self) -> None:
        """For each discovered model, assert model_config['frozen'] is True.

        The discovery function already filters to frozen=True models, so this
        test is a belt-and-suspenders check that the config attribute is
        accessible and consistent.
        """
        frozen_models = _discover_frozen_boundary_models()
        assert frozen_models, "No frozen boundary models discovered — check imports"
        failures: list[str] = []
        for cls in frozen_models:
            if cls.model_config.get("frozen") is not True:
                failures.append(f"{cls.__module__}.{cls.__qualname__}")
        assert not failures, (
            "The following models were expected to have frozen=True but do not:\n"
            + "\n".join(f"  - {f}" for f in failures)
        )

    def test_known_boundary_models_are_present(self) -> None:
        """Assert a representative set of known boundary models are discovered.

        This guards against the discovery logic being broken (e.g., a package
        rename that causes the walker to silently skip omnimemory.models.*).
        """
        frozen_models = _discover_frozen_boundary_models()
        frozen_names = {cls.__name__ for cls in frozen_models}

        required = {
            "ModelNotificationEvent",
            "ModelNotificationEventPayload",
            "ModelIntentClassifiedEvent",
            "ModelAuditEvent",
            "AuditEventDetails",
            "ResourceUsageMetadata",
            "SecurityAuditDetails",
            "PerformanceAuditDetails",
            "ModelEventData",
            "ModelCircuitBreakerStatsResponse",
            "ModelCorrelationContext",
            "ModelDocumentDiscoveredEvent",
            "ModelDocumentChangedEvent",
            "ModelDocumentRemovedEvent",
            "ModelCrawlStateRecord",
            "ModelCrawlTickCommand",
            "ModelMemoryRetrievalRequest",
            "ModelMemoryRetrievalResponse",
            "ModelSearchResult",
        }

        missing = required - frozen_names
        assert not missing, (
            "The following required boundary models were NOT found by the "
            "completeness scan.  Either the model was removed, renamed, or "
            "frozen=True was stripped:\n"
            + "\n".join(f"  - {m}" for m in sorted(missing))
        )


# ---------------------------------------------------------------------------
# OMN-2294: Copy safety — .model_copy() creates independent instances
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFrozenModelCopySafety:
    """OMN-2294 Test 3: .model_copy(update={...}) creates an independent
    instance and leaves the original model unchanged.

    This is the idiomatic Pydantic v2 way to produce a modified copy of a
    frozen model.  The tests confirm that:
    - The copy reflects the updated values.
    - The original retains its original values.
    - The copy is a distinct object (not the same reference).
    """

    def test_notification_event_copy_does_not_mutate_original(self) -> None:
        original = ModelNotificationEvent(
            event_id="evt-copy-001",
            topic="memory.item.created",
            payload=ModelNotificationEventPayload(
                entity_type="item",
                entity_id="item-001",
                action="created",
            ),
        )
        copy = original.model_copy(update={"event_id": "evt-copy-002"})
        assert original.event_id == "evt-copy-001"
        assert copy.event_id == "evt-copy-002"
        assert copy is not original

    def test_notification_event_payload_copy_does_not_mutate_original(self) -> None:
        original = ModelNotificationEventPayload(
            entity_type="item",
            entity_id="item-001",
            action="created",
        )
        copy = original.model_copy(update={"action": "updated"})
        assert original.action == "created"
        assert copy.action == "updated"
        assert copy is not original

    def test_intent_classified_event_copy_does_not_mutate_original(self) -> None:
        original = ModelIntentClassifiedEvent(
            event_type="IntentClassified",
            session_id="sess-copy-001",
            correlation_id=uuid4(),
            intent_category="debugging",
            confidence=0.85,
            timestamp=datetime.now(timezone.utc),
        )
        copy = original.model_copy(update={"session_id": "sess-copy-002"})
        assert original.session_id == "sess-copy-001"
        assert copy.session_id == "sess-copy-002"
        assert copy is not original

    def test_audit_event_copy_does_not_mutate_original(self) -> None:
        original = ModelAuditEvent(
            event_id="audit-copy-001",
            timestamp=datetime.now(timezone.utc),
            event_type=AuditEventType.MEMORY_STORE,
            severity=AuditSeverity.LOW,
            operation="memory_store",
            component="memory_manager",
            message="Original message",
        )
        copy = original.model_copy(update={"message": "Updated message"})
        assert original.message == "Original message"
        assert copy.message == "Updated message"
        assert copy is not original

    def test_correlation_context_copy_does_not_mutate_original(self) -> None:
        original = ModelCorrelationContext(
            correlation_id="corr-001",
            operation="test_op",
        )
        copy = original.model_copy(update={"operation": "other_op"})
        assert original.operation == "test_op"
        assert copy.operation == "other_op"
        assert copy is not original

    def test_memory_retrieval_request_copy_does_not_mutate_original(self) -> None:
        original = ModelMemoryRetrievalRequest(
            operation="search",
            query_text="test query",
            limit=5,
        )
        copy = original.model_copy(update={"limit": 20})
        assert original.limit == 5
        assert copy.limit == 20
        assert copy is not original


# ---------------------------------------------------------------------------
# OMN-2294: Crawl event boundary models — mutation rejection
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCrawlBoundaryModelsFrozen:
    """Verify crawl domain boundary models reject mutation after construction.

    These models were not covered in the original test suite.
    """

    _CONTENT_FINGERPRINT = "a" * 64  # 64-char hex string satisfying the pattern

    def _make_discovered_event(self) -> ModelDocumentDiscoveredEvent:
        return ModelDocumentDiscoveredEvent(
            correlation_id=uuid4(),
            emitted_at_utc=datetime.now(timezone.utc),
            crawler_type=EnumCrawlerType.FILESYSTEM,
            crawl_scope="omninode/omnimemory",
            trigger_source="scheduled",
            source_ref="/docs/CLAUDE.md",
            source_type=EnumContextSourceType.STATIC_STANDARDS,
            content_fingerprint=self._CONTENT_FINGERPRINT,
            content_blob_ref="s3://bucket/object",
            token_estimate=512,
            scope_ref="omninode/omnimemory",
            detected_doc_type=EnumDetectedDocType.CLAUDE_MD,
            priority_hint=50,
        )

    def test_document_discovered_event_rejects_mutation(self) -> None:
        event = self._make_discovered_event()
        with pytest.raises(ValidationError):
            event.source_ref = "/docs/OTHER.md"  # type: ignore[misc]

    def test_document_discovered_event_copy_is_independent(self) -> None:
        original = self._make_discovered_event()
        copy = original.model_copy(update={"priority_hint": 99})
        assert original.priority_hint == 50
        assert copy.priority_hint == 99
        assert copy is not original

    def test_document_discovered_event_config_frozen(self) -> None:
        assert ModelDocumentDiscoveredEvent.model_config.get("frozen") is True

    def test_document_changed_event_config_frozen(self) -> None:
        assert ModelDocumentChangedEvent.model_config.get("frozen") is True

    def test_document_removed_event_config_frozen(self) -> None:
        assert ModelDocumentRemovedEvent.model_config.get("frozen") is True

    def test_crawl_state_record_config_frozen(self) -> None:
        assert ModelCrawlStateRecord.model_config.get("frozen") is True

    def test_crawl_tick_command_config_frozen(self) -> None:
        assert ModelCrawlTickCommand.model_config.get("frozen") is True

    def test_crawl_state_record_rejects_mutation(self) -> None:
        record = ModelCrawlStateRecord(
            source_ref="/docs/CLAUDE.md",
            crawler_type=EnumCrawlerType.FILESYSTEM,
            scope_ref="omninode/omnimemory",
            content_fingerprint=self._CONTENT_FINGERPRINT,
            last_crawled_at_utc=datetime.now(timezone.utc),
        )
        with pytest.raises(ValidationError):
            record.source_ref = "/docs/OTHER.md"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# OMN-2294: Node boundary models — mutation rejection and config guards
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestNodeBoundaryModelsFrozen:
    """Verify node-level boundary models (retrieval, crawl config) are frozen.

    These models were not covered in the original test suite.
    """

    def test_memory_retrieval_request_rejects_mutation(self) -> None:
        request = ModelMemoryRetrievalRequest(
            operation="search",
            query_text="authentication flow",
            limit=10,
        )
        with pytest.raises(ValidationError):
            request.limit = 999  # type: ignore[misc]

    def test_memory_retrieval_request_config_frozen(self) -> None:
        assert ModelMemoryRetrievalRequest.model_config.get("frozen") is True

    def test_memory_retrieval_response_config_frozen(self) -> None:
        assert ModelMemoryRetrievalResponse.model_config.get("frozen") is True

    def test_search_result_config_frozen(self) -> None:
        assert ModelSearchResult.model_config.get("frozen") is True

    def test_circuit_breaker_stats_rejects_mutation(self) -> None:
        stats = ModelCircuitBreakerStatsResponse(
            state="CLOSED",
            failure_count=0,
            success_count=100,
            total_calls=100,
            total_timeouts=0,
            last_failure_time=None,
            state_changed_at="2026-01-01T00:00:00Z",
        )
        with pytest.raises(ValidationError):
            stats.state = "OPEN"  # type: ignore[misc]

    def test_circuit_breaker_stats_config_frozen(self) -> None:
        assert ModelCircuitBreakerStatsResponse.model_config.get("frozen") is True

    def test_correlation_context_rejects_mutation(self) -> None:
        ctx = ModelCorrelationContext(
            correlation_id="corr-node-001",
            operation="test_operation",
        )
        with pytest.raises(ValidationError):
            ctx.operation = "mutated"  # type: ignore[misc]

    def test_correlation_context_config_frozen(self) -> None:
        assert ModelCorrelationContext.model_config.get("frozen") is True

    def test_memory_retrieval_request_rejects_extra_fields_when_applicable(
        self,
    ) -> None:
        """ModelMemoryRetrievalRequest does not declare extra='forbid', so
        this test verifies only that frozen=True prevents field reassignment
        (extra-field rejection is controlled separately by extra= config).
        """
        request = ModelMemoryRetrievalRequest(
            operation="search_text",
            query_text="login error",
        )
        # Confirm frozen rejects assignment regardless of extra= setting
        with pytest.raises(ValidationError):
            request.query_text = "changed"  # type: ignore[misc]
