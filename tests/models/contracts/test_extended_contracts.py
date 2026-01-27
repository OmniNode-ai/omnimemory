# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for extended contract models with handler_routing support.

Tests validate that extended contract models properly support handler_routing
configuration using ModelHandlerRoutingSubcontract from omnibase_core.

These models are temporary workarounds until OMN-1588 adds handler_routing
to the base contracts in omnibase_core.

Coverage:
- MixinHandlerRouting provides handler_routing field correctly
- All 4 extended contract models accept handler_routing
- extra="ignore" config allows unknown fields
- Handler routing subcontract structure is validated
"""

from __future__ import annotations

import pytest
from omnibase_core.enums.enum_execution_shape import EnumMessageCategory
from omnibase_core.models.contracts.model_algorithm_config import ModelAlgorithmConfig
from omnibase_core.models.contracts.model_algorithm_factor_config import (
    ModelAlgorithmFactorConfig,
)
from omnibase_core.models.contracts.model_io_operation_config import (
    ModelIOOperationConfig,
)
from omnibase_core.models.contracts.model_performance_requirements import (
    ModelPerformanceRequirements,
)
from omnibase_core.models.contracts.subcontracts import (
    ModelHandlerRoutingEntry,
    ModelHandlerRoutingSubcontract,
)
from omnibase_core.models.primitives.model_semver import ModelSemVer
from pydantic.fields import FieldInfo

from omnimemory.models.contracts import (
    MixinHandlerRouting,
    ModelContractComputeExtended,
    ModelContractEffectExtended,
    ModelContractOrchestratorExtended,
    ModelContractReducerExtended,
)


def create_handler_routing_subcontract(
    *,
    routing_key: str,
    handler_key: str,
    routing_strategy: str = "payload_type_match",
    message_category: EnumMessageCategory = EnumMessageCategory.COMMAND,
    priority: int = 1,
    version_major: int = 1,
    default_handler: str | None = None,
    output_events: list[str] | None = None,
    additional_handlers: list[ModelHandlerRoutingEntry] | None = None,
) -> ModelHandlerRoutingSubcontract:
    """Factory function to create handler routing subcontracts with customizable parameters.

    This eliminates duplication across test class fixtures while allowing each test
    to specify the exact values needed for its assertions.

    Args:
        routing_key: Primary handler routing key (e.g., "memory.store")
        handler_key: Primary handler key (e.g., "storage_handler")
        routing_strategy: Routing strategy type (default: "payload_type_match")
        message_category: Message category enum (default: COMMAND)
        priority: Handler priority (default: 1)
        version_major: Major version number (default: 1)
        default_handler: Optional default handler name
        output_events: Optional list of output event names
        additional_handlers: Optional list of additional handler entries

    Returns:
        Configured ModelHandlerRoutingSubcontract instance
    """
    # Build handler entry kwargs, only including output_events if provided
    handler_kwargs: dict[str, object] = {
        "routing_key": routing_key,
        "handler_key": handler_key,
        "message_category": message_category,
        "priority": priority,
    }
    if output_events is not None:
        handler_kwargs["output_events"] = output_events

    primary_handler = ModelHandlerRoutingEntry(**handler_kwargs)  # type: ignore[arg-type]

    handlers = [primary_handler]
    if additional_handlers:
        handlers.extend(additional_handlers)

    return ModelHandlerRoutingSubcontract(
        version=ModelSemVer(major=version_major, minor=0, patch=0),
        routing_strategy=routing_strategy,
        handlers=handlers,
        default_handler=default_handler,
    )


class TestMixinHandlerRouting:
    """Tests for MixinHandlerRouting mixin class."""

    def test_mixin_has_handler_routing_annotation(self) -> None:
        """Verify MixinHandlerRouting defines handler_routing annotation."""
        annotations = MixinHandlerRouting.__annotations__
        assert "handler_routing" in annotations
        # The annotation should be Optional[ModelHandlerRoutingSubcontract]
        assert annotations["handler_routing"] is not None

    def test_mixin_handler_routing_field_info(self) -> None:
        """Verify handler_routing is a Pydantic Field with default None."""
        field = MixinHandlerRouting.handler_routing
        assert isinstance(field, FieldInfo)
        assert field.default is None
        assert "Handler routing" in (field.description or "")


class TestModelContractEffectExtended:
    """Tests for ModelContractEffectExtended with handler_routing."""

    @pytest.fixture
    def minimal_effect_data(self) -> dict[str, object]:
        """Provide minimal valid data for Effect contract."""
        return {
            "name": "test_effect",
            "contract_version": ModelSemVer(major=0, minor=1, patch=0),
            "description": "Test effect contract",
            "node_type": "EFFECT",
            "input_model": "TestInput",
            "output_model": "TestOutput",
            "io_operations": [
                ModelIOOperationConfig(operation_type="read"),
                ModelIOOperationConfig(operation_type="write"),
            ],
        }

    @pytest.fixture
    def handler_routing_subcontract(self) -> ModelHandlerRoutingSubcontract:
        """Provide a valid handler routing subcontract for Effect tests."""
        return create_handler_routing_subcontract(
            routing_key="memory.store",
            handler_key="storage_handler",
            routing_strategy="payload_type_match",
            message_category=EnumMessageCategory.COMMAND,
            priority=1,
            default_handler="fallback_handler",
        )

    def test_effect_extended_without_handler_routing(
        self, minimal_effect_data: dict[str, object]
    ) -> None:
        """Verify Effect extended contract works without handler_routing."""
        contract = ModelContractEffectExtended(**minimal_effect_data)
        assert contract.handler_routing is None
        assert contract.name == "test_effect"

    def test_effect_extended_with_handler_routing(
        self,
        minimal_effect_data: dict[str, object],
        handler_routing_subcontract: ModelHandlerRoutingSubcontract,
    ) -> None:
        """Verify Effect extended contract accepts handler_routing."""
        minimal_effect_data["handler_routing"] = handler_routing_subcontract
        contract = ModelContractEffectExtended(**minimal_effect_data)

        assert contract.handler_routing is not None
        assert contract.handler_routing.routing_strategy == "payload_type_match"
        assert len(contract.handler_routing.handlers) == 1
        assert contract.handler_routing.handlers[0].routing_key == "memory.store"

    def test_effect_extended_extra_fields_ignored(
        self, minimal_effect_data: dict[str, object]
    ) -> None:
        """Verify extra='ignore' allows unknown fields."""
        minimal_effect_data["unknown_field"] = "should_be_ignored"
        minimal_effect_data["another_extra"] = {"nested": "value"}

        # Should not raise ValidationError
        contract = ModelContractEffectExtended(**minimal_effect_data)
        assert contract.name == "test_effect"
        # Extra fields should not be accessible
        assert not hasattr(contract, "unknown_field")
        assert not hasattr(contract, "another_extra")


class TestModelContractComputeExtended:
    """Tests for ModelContractComputeExtended with handler_routing."""

    @pytest.fixture
    def minimal_compute_data(self) -> dict[str, object]:
        """Provide minimal valid data for Compute contract."""
        return {
            "name": "test_compute",
            "contract_version": ModelSemVer(major=0, minor=1, patch=0),
            "description": "Test compute contract",
            "node_type": "COMPUTE",
            "input_model": "ComputeInput",
            "output_model": "ComputeOutput",
            "algorithm": ModelAlgorithmConfig(
                algorithm_type="similarity_cosine",
                factors={
                    "cosine": ModelAlgorithmFactorConfig(
                        weight=1.0,
                        calculation_method="cosine_similarity",
                    ),
                },
            ),
            "performance": ModelPerformanceRequirements(
                single_operation_max_ms=1000,
            ),
        }

    @pytest.fixture
    def handler_routing_subcontract(self) -> ModelHandlerRoutingSubcontract:
        """Provide a valid handler routing subcontract for Compute tests."""
        return create_handler_routing_subcontract(
            routing_key="compute.similarity",
            handler_key="similarity_handler",
            routing_strategy="operation_match",
            message_category=EnumMessageCategory.INTENT,
            priority=2,
        )

    def test_compute_extended_without_handler_routing(
        self, minimal_compute_data: dict[str, object]
    ) -> None:
        """Verify Compute extended contract works without handler_routing."""
        contract = ModelContractComputeExtended(**minimal_compute_data)
        assert contract.handler_routing is None
        assert contract.name == "test_compute"
        assert contract.algorithm.algorithm_type == "similarity_cosine"

    def test_compute_extended_with_handler_routing(
        self,
        minimal_compute_data: dict[str, object],
        handler_routing_subcontract: ModelHandlerRoutingSubcontract,
    ) -> None:
        """Verify Compute extended contract accepts handler_routing."""
        minimal_compute_data["handler_routing"] = handler_routing_subcontract
        contract = ModelContractComputeExtended(**minimal_compute_data)

        assert contract.handler_routing is not None
        assert contract.handler_routing.routing_strategy == "operation_match"
        assert contract.handler_routing.default_handler is None

    def test_compute_extended_extra_fields_ignored(
        self, minimal_compute_data: dict[str, object]
    ) -> None:
        """Verify extra='ignore' allows unknown fields."""
        minimal_compute_data["future_onex_field"] = "v5_feature"

        contract = ModelContractComputeExtended(**minimal_compute_data)
        assert contract.name == "test_compute"
        assert not hasattr(contract, "future_onex_field")


class TestModelContractReducerExtended:
    """Tests for ModelContractReducerExtended with handler_routing."""

    @pytest.fixture
    def minimal_reducer_data(self) -> dict[str, object]:
        """Provide minimal valid data for Reducer contract."""
        return {
            "name": "test_reducer",
            "contract_version": ModelSemVer(major=0, minor=1, patch=0),
            "description": "Test reducer contract",
            "input_model": "ReducerInput",
            "output_model": "ReducerOutput",
        }

    @pytest.fixture
    def handler_routing_subcontract(self) -> ModelHandlerRoutingSubcontract:
        """Provide a valid handler routing subcontract for Reducer tests."""
        return create_handler_routing_subcontract(
            routing_key="reduce.consolidate",
            handler_key="consolidation_handler",
            routing_strategy="topic_pattern",
            message_category=EnumMessageCategory.EVENT,
            priority=3,
            output_events=["memory.consolidated"],
            default_handler="passthrough_handler",
        )

    def test_reducer_extended_without_handler_routing(
        self, minimal_reducer_data: dict[str, object]
    ) -> None:
        """Verify Reducer extended contract works without handler_routing."""
        contract = ModelContractReducerExtended(**minimal_reducer_data)
        assert contract.handler_routing is None
        assert contract.name == "test_reducer"

    def test_reducer_extended_with_handler_routing(
        self,
        minimal_reducer_data: dict[str, object],
        handler_routing_subcontract: ModelHandlerRoutingSubcontract,
    ) -> None:
        """Verify Reducer extended contract accepts handler_routing."""
        minimal_reducer_data["handler_routing"] = handler_routing_subcontract
        contract = ModelContractReducerExtended(**minimal_reducer_data)

        assert contract.handler_routing is not None
        assert contract.handler_routing.routing_strategy == "topic_pattern"
        assert contract.handler_routing.handlers[0].output_events == [
            "memory.consolidated"
        ]

    def test_reducer_extended_extra_fields_ignored(
        self, minimal_reducer_data: dict[str, object]
    ) -> None:
        """Verify extra='ignore' allows unknown fields."""
        minimal_reducer_data["experimental_flag"] = True
        minimal_reducer_data["deprecated_field"] = "old_value"

        contract = ModelContractReducerExtended(**minimal_reducer_data)
        assert contract.name == "test_reducer"


class TestModelContractOrchestratorExtended:
    """Tests for ModelContractOrchestratorExtended with handler_routing."""

    @pytest.fixture
    def minimal_orchestrator_data(self) -> dict[str, object]:
        """Provide minimal valid data for Orchestrator contract."""
        return {
            "name": "test_orchestrator",
            "contract_version": ModelSemVer(major=0, minor=1, patch=0),
            "description": "Test orchestrator contract",
            "input_model": "OrchestratorInput",
            "output_model": "OrchestratorOutput",
            "performance": ModelPerformanceRequirements(
                single_operation_max_ms=5000,
            ),
        }

    @pytest.fixture
    def handler_routing_subcontract(self) -> ModelHandlerRoutingSubcontract:
        """Provide a valid handler routing subcontract for Orchestrator tests."""
        return create_handler_routing_subcontract(
            routing_key="orchestrate.lifecycle",
            handler_key="lifecycle_handler",
            routing_strategy="payload_type_match",
            message_category=EnumMessageCategory.COMMAND,
            priority=1,
            version_major=2,
            default_handler="workflow_default_handler",
            additional_handlers=[
                ModelHandlerRoutingEntry(
                    routing_key="orchestrate.archive",
                    handler_key="archive_handler",
                    message_category=EnumMessageCategory.EVENT,
                    priority=2,
                ),
            ],
        )

    def test_orchestrator_extended_without_handler_routing(
        self, minimal_orchestrator_data: dict[str, object]
    ) -> None:
        """Verify Orchestrator extended contract works without handler_routing."""
        contract = ModelContractOrchestratorExtended(**minimal_orchestrator_data)
        assert contract.handler_routing is None
        assert contract.name == "test_orchestrator"

    def test_orchestrator_extended_with_handler_routing(
        self,
        minimal_orchestrator_data: dict[str, object],
        handler_routing_subcontract: ModelHandlerRoutingSubcontract,
    ) -> None:
        """Verify Orchestrator extended contract accepts handler_routing."""
        minimal_orchestrator_data["handler_routing"] = handler_routing_subcontract
        contract = ModelContractOrchestratorExtended(**minimal_orchestrator_data)

        assert contract.handler_routing is not None
        assert contract.handler_routing.routing_strategy == "payload_type_match"
        assert len(contract.handler_routing.handlers) == 2
        assert contract.handler_routing.version.major == 2

    def test_orchestrator_extended_extra_fields_ignored(
        self, minimal_orchestrator_data: dict[str, object]
    ) -> None:
        """Verify extra='ignore' allows unknown fields."""
        minimal_orchestrator_data["orchestration_hints"] = {"parallel": True}

        contract = ModelContractOrchestratorExtended(**minimal_orchestrator_data)
        assert contract.name == "test_orchestrator"


class TestHandlerRoutingSubcontractStructure:
    """Tests for ModelHandlerRoutingSubcontract structure validation."""

    def test_handler_routing_subcontract_with_default_handler(self) -> None:
        """Verify handler routing subcontract with default_handler (no handlers list)."""
        subcontract = ModelHandlerRoutingSubcontract(
            version=ModelSemVer(major=1, minor=0, patch=0),
            routing_strategy="payload_type_match",
            handlers=[],
            default_handler="fallback",
        )
        assert subcontract.routing_strategy == "payload_type_match"
        assert subcontract.handlers == []
        assert subcontract.default_handler == "fallback"

    def test_handler_routing_subcontract_with_handlers(self) -> None:
        """Verify handler routing subcontract with handlers (no default)."""
        subcontract = ModelHandlerRoutingSubcontract(
            version=ModelSemVer(major=1, minor=0, patch=0),
            routing_strategy="operation_match",
            handlers=[
                ModelHandlerRoutingEntry(
                    routing_key="test.route",
                    handler_key="test_handler",
                )
            ],
        )
        assert subcontract.routing_strategy == "operation_match"
        assert len(subcontract.handlers) == 1
        assert subcontract.default_handler is None

    def test_handler_routing_entry_all_fields(self) -> None:
        """Verify handler routing entry with all optional fields."""
        entry = ModelHandlerRoutingEntry(
            routing_key="test.route",
            handler_key="test_handler",
            message_category=EnumMessageCategory.COMMAND,
            priority=5,
            output_events=["test.completed", "test.logged"],
        )
        assert entry.routing_key == "test.route"
        assert entry.handler_key == "test_handler"
        assert entry.message_category == EnumMessageCategory.COMMAND
        assert entry.priority == 5
        assert entry.output_events == ["test.completed", "test.logged"]

    def test_handler_routing_entry_minimal(self) -> None:
        """Verify handler routing entry with minimal required fields."""
        entry = ModelHandlerRoutingEntry(
            routing_key="minimal.route",
            handler_key="minimal_handler",
        )
        assert entry.routing_key == "minimal.route"
        assert entry.handler_key == "minimal_handler"


class TestExtendedContractsModelConfig:
    """Tests verifying model_config settings on extended contracts."""

    def test_effect_extended_model_config(self) -> None:
        """Verify Effect extended has correct model_config."""
        config = ModelContractEffectExtended.model_config
        assert config.get("extra") == "ignore"
        assert config.get("validate_assignment") is True

    def test_compute_extended_model_config(self) -> None:
        """Verify Compute extended has correct model_config."""
        config = ModelContractComputeExtended.model_config
        assert config.get("extra") == "ignore"
        assert config.get("validate_assignment") is True

    def test_reducer_extended_model_config(self) -> None:
        """Verify Reducer extended has correct model_config."""
        config = ModelContractReducerExtended.model_config
        assert config.get("extra") == "ignore"
        assert config.get("validate_assignment") is True

    def test_orchestrator_extended_model_config(self) -> None:
        """Verify Orchestrator extended has correct model_config."""
        config = ModelContractOrchestratorExtended.model_config
        assert config.get("extra") == "ignore"
        assert config.get("validate_assignment") is True


class TestExtendedContractsInheritance:
    """Tests verifying correct inheritance hierarchy."""

    def test_effect_extended_inherits_from_base(self) -> None:
        """Verify Effect extended inherits from base and mixin."""
        from omnibase_core.models.contracts import ModelContractEffect

        assert issubclass(ModelContractEffectExtended, ModelContractEffect)
        assert issubclass(ModelContractEffectExtended, MixinHandlerRouting)

    def test_compute_extended_inherits_from_base(self) -> None:
        """Verify Compute extended inherits from base and mixin."""
        from omnibase_core.models.contracts import ModelContractCompute

        assert issubclass(ModelContractComputeExtended, ModelContractCompute)
        assert issubclass(ModelContractComputeExtended, MixinHandlerRouting)

    def test_reducer_extended_inherits_from_base(self) -> None:
        """Verify Reducer extended inherits from base and mixin."""
        from omnibase_core.models.contracts import ModelContractReducer

        assert issubclass(ModelContractReducerExtended, ModelContractReducer)
        assert issubclass(ModelContractReducerExtended, MixinHandlerRouting)

    def test_orchestrator_extended_inherits_from_base(self) -> None:
        """Verify Orchestrator extended inherits from base and mixin."""
        from omnibase_core.models.contracts import ModelContractOrchestrator

        assert issubclass(ModelContractOrchestratorExtended, ModelContractOrchestrator)
        assert issubclass(ModelContractOrchestratorExtended, MixinHandlerRouting)
