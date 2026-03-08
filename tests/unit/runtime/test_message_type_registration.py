# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Unit tests for memory message type registration.

Validates:
    - All 10 memory message types are registered
    - Registration happens correctly with proper categories and domains
    - Registry is queryable after freeze via has_message_type() and get_entry()
    - Registration after freeze() raises ModelOnexError
    - validate_startup() returns no errors for a clean registry

Related:
    - OMN-2217: Phase 6 -- Wire model registration & entry point declaration
    - OMN-937: Central Message Type Registry implementation
"""

from __future__ import annotations

import pytest
from omnibase_core.models.errors import ModelOnexError
from omnibase_infra.enums import EnumMessageCategory
from omnibase_infra.runtime.registry import RegistryMessageType

from omnimemory.runtime.message_type_registration import (
    EXPECTED_MESSAGE_TYPE_COUNT,
    MEMORY_DOMAIN,
    get_registration_metrics,
    is_registry_ready,
    register_memory_message_types,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def registry() -> RegistryMessageType:
    """Create a fresh, unfrozen RegistryMessageType."""
    return RegistryMessageType()


@pytest.fixture
def frozen_registry(registry: RegistryMessageType) -> RegistryMessageType:
    """Create a registry with all memory types registered and frozen."""
    register_memory_message_types(registry)
    registry.freeze()
    return registry


# =============================================================================
# Registration Count
# =============================================================================


@pytest.mark.unit
class TestRegistrationCount:
    """Verify correct number of message types registered."""

    def test_returns_expected_registered_types(
        self, registry: RegistryMessageType
    ) -> None:
        """register_memory_message_types returns expected number of type names."""
        registered = register_memory_message_types(registry)
        assert len(registered) == EXPECTED_MESSAGE_TYPE_COUNT

    def test_registry_has_expected_entries(
        self, frozen_registry: RegistryMessageType
    ) -> None:
        """Frozen registry contains exactly the expected number of entries."""
        assert frozen_registry.entry_count == EXPECTED_MESSAGE_TYPE_COUNT

    def test_expected_count_constant_matches(self) -> None:
        """EXPECTED_MESSAGE_TYPE_COUNT constant is 10."""
        assert EXPECTED_MESSAGE_TYPE_COUNT == 10


# =============================================================================
# has_message_type Queries
# =============================================================================


@pytest.mark.unit
class TestHasMessageType:
    """Verify all registered types are discoverable via has_message_type."""

    ALL_MESSAGE_TYPES = [
        # Consumed Kafka event
        "ModelIntentClassifiedEvent",
        # Intent storage effect
        "ModelIntentStorageRequest",
        "ModelIntentStorageResponse",
        # Memory retrieval effect
        "ModelMemoryRetrievalRequest",
        "ModelMemoryRetrievalResponse",
        # Memory storage effect
        "ModelMemoryStorageRequest",
        "ModelMemoryStorageResponse",
        # Agent coordinator orchestrator
        "ModelAgentCoordinatorRequest",
        "ModelAgentCoordinatorResponse",
        "ModelNotificationEvent",
    ]

    @pytest.mark.parametrize("message_type", ALL_MESSAGE_TYPES)
    def test_has_message_type(
        self, frozen_registry: RegistryMessageType, message_type: str
    ) -> None:
        """Each registered type is discoverable via has_message_type."""
        assert frozen_registry.has_message_type(message_type), (
            f"Message type '{message_type}' should be registered"
        )

    def test_unregistered_type_not_found(
        self, frozen_registry: RegistryMessageType
    ) -> None:
        """Unregistered type returns False."""
        assert not frozen_registry.has_message_type("NonExistentType")


# =============================================================================
# get_entry Queries
# =============================================================================


@pytest.mark.unit
class TestGetEntry:
    """Verify entry details are correct for all registered types."""

    def test_all_entries_have_memory_domain(
        self, frozen_registry: RegistryMessageType
    ) -> None:
        """Every registered entry has owning_domain='memory'."""
        for msg_type in TestHasMessageType.ALL_MESSAGE_TYPES:
            entry = frozen_registry.get_entry(msg_type)
            assert entry is not None, f"Entry for {msg_type} should exist"
            assert entry.domain_constraint.owning_domain == MEMORY_DOMAIN

    def test_all_entries_are_enabled(
        self, frozen_registry: RegistryMessageType
    ) -> None:
        """Every registered entry is enabled."""
        for msg_type in TestHasMessageType.ALL_MESSAGE_TYPES:
            entry = frozen_registry.get_entry(msg_type)
            assert entry is not None
            assert entry.enabled is True

    def test_all_entries_have_descriptions(
        self, frozen_registry: RegistryMessageType
    ) -> None:
        """Every registered entry has a non-empty description."""
        for msg_type in TestHasMessageType.ALL_MESSAGE_TYPES:
            entry = frozen_registry.get_entry(msg_type)
            assert entry is not None
            assert entry.description is not None
            assert len(entry.description) > 0


# =============================================================================
# Category Validation
# =============================================================================


@pytest.mark.unit
class TestCategoryAssignment:
    """Verify correct EnumMessageCategory assignment per type."""

    EVENT_TYPES = [
        "ModelIntentClassifiedEvent",
        "ModelIntentStorageResponse",
        "ModelMemoryRetrievalResponse",
        "ModelMemoryStorageResponse",
        "ModelAgentCoordinatorResponse",
        "ModelNotificationEvent",
    ]

    COMMAND_TYPES = [
        "ModelIntentStorageRequest",
        "ModelMemoryRetrievalRequest",
        "ModelMemoryStorageRequest",
        "ModelAgentCoordinatorRequest",
    ]

    @pytest.mark.parametrize("message_type", EVENT_TYPES)
    def test_event_category(
        self, frozen_registry: RegistryMessageType, message_type: str
    ) -> None:
        """Event types have EVENT allowed category."""
        entry = frozen_registry.get_entry(message_type)
        assert entry is not None
        assert EnumMessageCategory.EVENT in entry.allowed_categories

    @pytest.mark.parametrize("message_type", COMMAND_TYPES)
    def test_command_category(
        self, frozen_registry: RegistryMessageType, message_type: str
    ) -> None:
        """Command types have COMMAND allowed category."""
        entry = frozen_registry.get_entry(message_type)
        assert entry is not None
        assert EnumMessageCategory.COMMAND in entry.allowed_categories


# =============================================================================
# Handler ID Validation
# =============================================================================


@pytest.mark.unit
class TestHandlerIds:
    """Verify handler_id assignments match node directory names."""

    HANDLER_MAP = {
        "ModelIntentClassifiedEvent": ("node_intent_event_consumer_effect",),
        "ModelIntentStorageRequest": ("node_intent_storage_effect",),
        "ModelIntentStorageResponse": ("node_intent_storage_effect",),
        "ModelMemoryRetrievalRequest": ("node_memory_retrieval_effect",),
        "ModelMemoryRetrievalResponse": ("node_memory_retrieval_effect",),
        "ModelMemoryStorageRequest": ("node_memory_storage_effect",),
        "ModelMemoryStorageResponse": ("node_memory_storage_effect",),
        "ModelAgentCoordinatorRequest": ("node_agent_coordinator_orchestrator",),
        "ModelAgentCoordinatorResponse": ("node_agent_coordinator_orchestrator",),
        "ModelNotificationEvent": ("node_agent_coordinator_orchestrator",),
    }

    @pytest.mark.parametrize(
        ("message_type", "expected_handlers"),
        list(HANDLER_MAP.items()),
    )
    def test_handler_id(
        self,
        frozen_registry: RegistryMessageType,
        message_type: str,
        expected_handlers: tuple[str, ...],
    ) -> None:
        """Handler IDs match the expected node directory names."""
        entry = frozen_registry.get_entry(message_type)
        assert entry is not None
        assert set(entry.handler_ids) == set(expected_handlers), (
            f"Expected handlers {expected_handlers} for '{message_type}', "
            f"got {entry.handler_ids}"
        )


# =============================================================================
# Freeze Enforcement
# =============================================================================


@pytest.mark.unit
class TestFreezeEnforcement:
    """Verify registration after freeze raises ModelOnexError."""

    def test_registration_after_freeze_raises(
        self, frozen_registry: RegistryMessageType
    ) -> None:
        """Calling register_simple after freeze raises ModelOnexError."""
        with pytest.raises(ModelOnexError):
            frozen_registry.register_simple(
                message_type="ShouldFail",
                handler_id="test-handler",
                category=EnumMessageCategory.EVENT,
                domain="test",
            )

    def test_register_function_after_freeze_raises(
        self, frozen_registry: RegistryMessageType
    ) -> None:
        """Calling register_memory_message_types on frozen registry raises."""
        with pytest.raises(ModelOnexError):
            register_memory_message_types(frozen_registry)


# =============================================================================
# Startup Validation
# =============================================================================


@pytest.mark.unit
class TestStartupValidation:
    """Verify validate_startup() on a clean memory registry."""

    def test_no_validation_errors(self, frozen_registry: RegistryMessageType) -> None:
        """validate_startup() returns empty list (no errors)."""
        errors = frozen_registry.validate_startup()
        assert errors == [], f"Expected no validation errors, got: {errors}"

    def test_validation_with_available_handlers(
        self, frozen_registry: RegistryMessageType
    ) -> None:
        """validate_startup with all handler IDs available returns no errors."""
        all_handler_ids = {
            "node_intent_event_consumer_effect",
            "node_intent_storage_effect",
            "node_memory_retrieval_effect",
            "node_memory_storage_effect",
            "node_agent_coordinator_orchestrator",
        }
        errors = frozen_registry.validate_startup(
            available_handler_ids=all_handler_ids,
        )
        assert errors == [], f"Expected no validation errors, got: {errors}"

    def test_validation_missing_handler_reports_error(
        self, frozen_registry: RegistryMessageType
    ) -> None:
        """validate_startup reports missing handlers when subset provided."""
        partial_handlers = {"node_intent_event_consumer_effect"}
        errors = frozen_registry.validate_startup(
            available_handler_ids=partial_handlers,
        )
        assert len(errors) > 0, "Expected validation errors for missing handlers"


# =============================================================================
# Registry Properties
# =============================================================================


@pytest.mark.unit
class TestRegistryProperties:
    """Verify registry metadata after memory registration."""

    def test_is_frozen(self, frozen_registry: RegistryMessageType) -> None:
        """Registry is frozen after registration."""
        assert frozen_registry.is_frozen

    def test_handler_count(self, frozen_registry: RegistryMessageType) -> None:
        """Registry tracks the correct number of unique handlers."""
        # 5 unique handler IDs across all 10 registrations
        assert frozen_registry.handler_count == 5

    def test_domain_count(self, frozen_registry: RegistryMessageType) -> None:
        """Registry tracks exactly 1 domain (memory)."""
        assert frozen_registry.domain_count == 1

    def test_memory_domain_constant(self) -> None:
        """MEMORY_DOMAIN is 'memory'."""
        assert MEMORY_DOMAIN == "memory"


# =============================================================================
# Observability: Readiness & Metrics
# =============================================================================


@pytest.mark.unit
class TestRegistrationObservability:
    """Verify readiness flag and metric counters after registration."""

    def test_is_ready_after_success(self, registry: RegistryMessageType) -> None:
        """is_registry_ready() returns True after successful registration."""
        register_memory_message_types(registry)
        assert is_registry_ready() is True

    def test_metrics_after_success(self, registry: RegistryMessageType) -> None:
        """get_registration_metrics() reports correct counts after success."""
        register_memory_message_types(registry)
        metrics = get_registration_metrics()
        assert metrics["registered_count"] == EXPECTED_MESSAGE_TYPE_COUNT
        assert metrics["expected_count"] == EXPECTED_MESSAGE_TYPE_COUNT
        # failure_count is reset at the start of each call, so a successful
        # registration always reports zero failures.
        assert metrics["failure_count"] == 0

    def test_readiness_false_after_failure(
        self, frozen_registry: RegistryMessageType
    ) -> None:
        """is_registry_ready() returns False after a failed registration attempt."""
        with pytest.raises(ModelOnexError):
            register_memory_message_types(frozen_registry)
        assert is_registry_ready() is False

    def test_failure_count_increments(
        self, frozen_registry: RegistryMessageType
    ) -> None:
        """Failure counter is 1 after a single failed registration call."""
        with pytest.raises(ModelOnexError):
            register_memory_message_types(frozen_registry)
        after = get_registration_metrics()["failure_count"]
        assert after == 1
