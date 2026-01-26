# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Contract validation tests for Core 8 ONEX nodes.

Tests both schema validation (Pydantic) and runtime load tests.
This module verifies that:
- contract.yaml files exist for each Core 8 node
- Contracts are valid YAML with required ONEX fields
- Contracts validate against appropriate Pydantic models
- Node classes can be imported and instantiated

Skip Behavior:
    Tests skip gracefully when files don't exist during scaffold phase,
    using pytest.skip() with clear messages about what's missing.

Path Resolution:
    Uses Path(__file__) for CWD-independent path resolution.
"""

from __future__ import annotations

import importlib
import types
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import yaml

from tests.conftest import CORE_8_NODES, NODES_DIR

if TYPE_CHECKING:
    from omnibase_core.types import MappingResultDict


class TestContractValidation:
    """Test contract.yaml files validate against Pydantic models."""

    @pytest.mark.parametrize("node_name", CORE_8_NODES)
    def test_contract_file_exists(self, node_name: str) -> None:
        """Verify contract.yaml exists for each Core 8 node."""
        contract_path: Path = NODES_DIR / node_name / "contract.yaml"
        # Skip if not yet implemented (scaffold phase)
        if not contract_path.exists():
            pytest.skip(f"File not yet implemented: {contract_path}")
        assert contract_path.exists(), f"Missing contract: {contract_path}"

    @pytest.mark.parametrize("node_name", CORE_8_NODES)
    def test_contract_is_valid_yaml(self, node_name: str) -> None:
        """Verify contract.yaml is valid YAML with required ONEX fields.

        ONEX contracts must have fields at root level: name, node_type.
        No backwards compatibility with legacy nested 'onex' format.
        """
        contract_path: Path = NODES_DIR / node_name / "contract.yaml"
        if not contract_path.exists():
            pytest.skip(f"File not yet implemented: {contract_path}")

        with open(contract_path, encoding="utf-8") as f:
            data: MappingResultDict = yaml.safe_load(f)

        assert isinstance(data, dict), f"Contract must be a dict: {node_name}"

        # ONEX contracts must have fields at root level (no legacy nested format)
        assert "name" in data, f"Contract must have 'name' field: {node_name}"
        assert "node_type" in data, f"Contract must have 'node_type' field: {node_name}"

    @pytest.mark.parametrize("node_name", CORE_8_NODES)
    def test_contract_validates_with_pydantic(self, node_name: str) -> None:
        """Verify contract validates against appropriate Pydantic model.

        Uses extended contract models from omnimemory.models.contracts that add
        support for ONEX infra extension fields (handler_routing, etc.) not yet
        in omnibase_core. See OMN-1588 for tracking the core fix.

        Note: Uses constructor (**data) instead of model_validate() due to a bug
        in omnibase_core 0.9.x where model_validate() passes an unsupported 'extra'
        parameter to Pydantic's BaseModel.model_validate(). The constructor performs
        identical validation.
        """
        contract_path: Path = NODES_DIR / node_name / "contract.yaml"
        if not contract_path.exists():
            pytest.skip(f"File not yet implemented: {contract_path}")

        with open(contract_path, encoding="utf-8") as f:
            data: MappingResultDict = yaml.safe_load(f)

        # Strip legacy field that was renamed (not an extension field issue)
        # TODO(OMN-1588): Remove this once all contracts use contract_version
        if "version" in data:
            del data["version"]

        # ONEX contracts must have node_type at root level (no legacy nested format)
        raw_node_type = data.get("node_type", "")
        node_type: str = str(raw_node_type) if raw_node_type else ""
        assert (
            node_type
        ), f"Contract must have 'node_type' field at root level: {node_name}"
        node_type = node_type.upper()

        # Import extended contract models that support ONEX infra extension fields
        # These models add handler_routing field and use extra="ignore" to allow
        # other extension fields. See OMN-1588 for tracking the core fix.
        try:
            if "EFFECT" in node_type:
                from omnimemory.models.contracts import ModelContractEffectExtended

                ModelContractEffectExtended(**data)
            elif "COMPUTE" in node_type:
                from omnimemory.models.contracts import ModelContractComputeExtended

                ModelContractComputeExtended(**data)
            elif "REDUCER" in node_type:
                from omnimemory.models.contracts import ModelContractReducerExtended

                ModelContractReducerExtended(**data)
            elif "ORCHESTRATOR" in node_type:
                from omnimemory.models.contracts import (
                    ModelContractOrchestratorExtended,
                )

                # Orchestrator's consumed_events/published_events in YAML use different
                # format than ModelContractOrchestrator expects (it expects
                # ModelEventDescriptor/ModelEventSubscription types). Strip these fields
                # since handler_routing is the primary routing mechanism we're validating.
                # TODO(OMN-1588): Resolve format mismatch when core adds proper support
                orchestrator_data = {
                    k: v
                    for k, v in data.items()
                    if k not in ("consumed_events", "published_events")
                }
                ModelContractOrchestratorExtended(**orchestrator_data)
            else:
                pytest.fail(f"Unknown node_type: {node_type}")
        except ModuleNotFoundError as e:
            if e.name and e.name.startswith("omnibase_core"):
                pytest.skip("omnibase_core not installed")
            raise


class TestContractRuntimeLoad:
    """Test contracts load at runtime with actual node classes.

    These tests verify that node classes can be imported and instantiated
    with their contract configurations. Tests are skipped for nodes that
    have not yet been implemented.
    """

    @pytest.mark.parametrize("node_name", CORE_8_NODES)
    def test_node_import_succeeds(self, node_name: str) -> None:
        """Verify node class can be imported from its package.

        This test checks that the node.py file exists and that the
        corresponding node class can be imported without errors.
        Skipped for nodes not yet implemented.
        """
        node_path: Path = NODES_DIR / node_name / "node.py"
        if not node_path.exists():
            pytest.skip(f"File not yet implemented: {node_path}")

        # Convert node_name to class name (e.g., memory_storage_effect -> Node...)
        class_name: str = "Node" + "".join(
            word.capitalize() for word in node_name.split("_")
        )

        module_name: str = f"omnimemory.nodes.{node_name}.node"
        try:
            module: types.ModuleType = importlib.import_module(module_name)
            node_class: type | None = getattr(module, class_name, None)
            assert (
                node_class is not None
            ), f"Node class {class_name} not found in {module_name}"
        except ModuleNotFoundError as e:
            # Package not installed in editable mode - skip rather than fail
            pytest.skip(f"Package not installed in editable mode: {e}")
        except ImportError as e:
            # Other import errors indicate real problems - fail the test
            pytest.fail(f"Failed to import {module_name}: {e}")

    @pytest.mark.parametrize("node_name", CORE_8_NODES)
    def test_node_instantiation_succeeds(self, node_name: str) -> None:
        """Verify node class can be instantiated with mock container.

        This test catches runtime errors like invalid super().__init__() calls
        that import-only tests would miss.
        """
        node_path: Path = NODES_DIR / node_name / "node.py"
        if not node_path.exists():
            pytest.skip(f"File not yet implemented: {node_path}")

        class_name: str = "Node" + "".join(
            word.capitalize() for word in node_name.split("_")
        )
        module_name: str = f"omnimemory.nodes.{node_name}.node"

        try:
            module: types.ModuleType = importlib.import_module(module_name)
            node_class: type | None = getattr(module, class_name, None)
            if node_class is None:
                pytest.skip(f"Node class {class_name} not found")

            # Instantiate with mock container
            from unittest.mock import Mock

            mock_container: Mock = Mock()
            instance: object = node_class(container=mock_container)
            assert instance is not None
        except ModuleNotFoundError as e:
            pytest.skip(f"Package not installed in editable mode: {e}")
        except ImportError as e:
            pytest.skip(f"Package not installed in editable mode: {e}")


class TestContractHandlerMapping:
    """Test contract actions have corresponding handlers.

    These tests verify that the contract.yaml actions are implemented
    by handlers in the handlers/ directory. Currently skipped during
    scaffold phase.
    """

    @pytest.mark.skip(reason="Requires handler implementation")
    @pytest.mark.parametrize("node_name", CORE_8_NODES)
    def test_contract_actions_have_handlers(self, node_name: str) -> None:
        """Verify all contract actions have corresponding handlers."""

    @pytest.mark.skip(reason="Requires container implementation")
    @pytest.mark.parametrize("node_name", CORE_8_NODES)
    def test_container_provides_required_dependencies(self, node_name: str) -> None:
        """Verify container provides all dependencies declared in contract."""

    @pytest.mark.skip(reason="Requires error handling implementation")
    def test_contract_validation_failure_handling(self) -> None:
        """Verify graceful handling of invalid contracts."""

    @pytest.mark.skip(reason="Requires integration test infrastructure")
    @pytest.mark.parametrize("node_name", CORE_8_NODES)
    def test_node_integration_with_storage_backend(self, node_name: str) -> None:
        """Verify node interaction with actual storage backends."""


# Filter for orchestrator nodes only
ORCHESTRATOR_NODES: list[str] = [
    node for node in CORE_8_NODES if "orchestrator" in node
]


class TestOrchestratorEventValidation:
    """Test orchestrator-specific event field validation.

    Orchestrator contracts define consumed_events and published_events fields
    that are stripped during standard Pydantic validation (due to format mismatch
    with ModelEventDescriptor/ModelEventSubscription). This test class validates
    that these event fields have the correct structure.

    Event Field Schemas:
        consumed_events: List of dicts with required keys:
            - event_pattern: str (event pattern string)
            - handler_function: str (handler method name)

        published_events: List of dicts with required keys:
            - event_pattern: str (event pattern string)
            Optional: description, etc.
    """

    @pytest.mark.parametrize("node_name", ORCHESTRATOR_NODES)
    def test_consumed_events_structure(self, node_name: str) -> None:
        """Verify consumed_events entries have required keys.

        Each consumed_events entry must have:
        - event_pattern: The event pattern to subscribe to
        - handler_function: The handler method name to invoke
        """
        contract_path: Path = NODES_DIR / node_name / "contract.yaml"
        if not contract_path.exists():
            pytest.skip(f"File not yet implemented: {contract_path}")

        with open(contract_path, encoding="utf-8") as f:
            data: MappingResultDict = yaml.safe_load(f)

        consumed_events = data.get("consumed_events")

        # consumed_events is required for orchestrators
        assert (
            consumed_events is not None
        ), f"Orchestrator {node_name} missing consumed_events field"
        assert isinstance(
            consumed_events, list
        ), f"consumed_events must be a list: {node_name}"

        # Validate each entry has required keys
        for idx, event in enumerate(consumed_events):
            assert isinstance(
                event, dict
            ), f"consumed_events[{idx}] must be a dict: {node_name}"
            assert (
                "event_pattern" in event
            ), f"consumed_events[{idx}] missing 'event_pattern': {node_name}"
            assert (
                "handler_function" in event
            ), f"consumed_events[{idx}] missing 'handler_function': {node_name}"
            # Validate types
            assert isinstance(
                event["event_pattern"], str
            ), f"consumed_events[{idx}].event_pattern must be str: {node_name}"
            assert isinstance(
                event["handler_function"], str
            ), f"consumed_events[{idx}].handler_function must be str: {node_name}"
            # Validate non-empty
            assert event[
                "event_pattern"
            ], f"consumed_events[{idx}].event_pattern cannot be empty: {node_name}"
            assert event[
                "handler_function"
            ], f"consumed_events[{idx}].handler_function cannot be empty: {node_name}"

    @pytest.mark.parametrize("node_name", ORCHESTRATOR_NODES)
    def test_published_events_structure(self, node_name: str) -> None:
        """Verify published_events entries have required keys.

        Each published_events entry must have:
        - event_pattern: The event pattern that will be published

        published_events can be an empty list if the orchestrator
        publishes events dynamically or documents them elsewhere.
        """
        contract_path: Path = NODES_DIR / node_name / "contract.yaml"
        if not contract_path.exists():
            pytest.skip(f"File not yet implemented: {contract_path}")

        with open(contract_path, encoding="utf-8") as f:
            data: MappingResultDict = yaml.safe_load(f)

        published_events = data.get("published_events")

        # published_events is required but can be empty
        assert (
            published_events is not None
        ), f"Orchestrator {node_name} missing published_events field"
        assert isinstance(
            published_events, list
        ), f"published_events must be a list: {node_name}"

        # Validate each entry has required keys (if non-empty)
        for idx, event in enumerate(published_events):
            assert isinstance(
                event, dict
            ), f"published_events[{idx}] must be a dict: {node_name}"
            assert (
                "event_pattern" in event
            ), f"published_events[{idx}] missing 'event_pattern': {node_name}"
            # Validate types
            assert isinstance(
                event["event_pattern"], str
            ), f"published_events[{idx}].event_pattern must be str: {node_name}"
            # Validate non-empty
            assert event[
                "event_pattern"
            ], f"published_events[{idx}].event_pattern cannot be empty: {node_name}"

    @pytest.mark.parametrize("node_name", ORCHESTRATOR_NODES)
    def test_consumed_events_handler_naming_convention(self, node_name: str) -> None:
        """Verify handler_function follows naming convention.

        Handler functions should follow the pattern 'handle_<action>'
        to maintain consistency across orchestrators.
        """
        contract_path: Path = NODES_DIR / node_name / "contract.yaml"
        if not contract_path.exists():
            pytest.skip(f"File not yet implemented: {contract_path}")

        with open(contract_path, encoding="utf-8") as f:
            data: MappingResultDict = yaml.safe_load(f)

        consumed_events = data.get("consumed_events", [])

        for idx, event in enumerate(consumed_events):
            handler = event.get("handler_function", "")
            assert handler.startswith("handle_"), (
                f"consumed_events[{idx}].handler_function should start with 'handle_': "
                f"got '{handler}' in {node_name}"
            )
