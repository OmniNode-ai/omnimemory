# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Contract enforcement tests - ONEX declarative node validation.

Ensures all ONEX nodes follow the FULLY DECLARATIVE pattern:
- contract.yaml defines node type, inputs, outputs, handlers
- handlers/ directory contains business logic
- No node.py class needed (contracts ARE the nodes)

This module validates that node directories contain proper contracts
and handler registrations, catching violations at test time.

Valid Node Types:
    The following node_type values are accepted in contract.yaml files:
    - effect: External I/O operations (APIs, DB, files)
    - compute: Pure transforms and algorithms
    - reducer: Aggregation and persistence operations
    - orchestrator: Workflow coordination
    - effect_generic: Generic effect node (EnumNodeType suffix variant)
    - compute_generic: Generic compute node (EnumNodeType suffix variant)
    - reducer_generic: Generic reducer node (EnumNodeType suffix variant)
    - orchestrator_generic: Generic orchestrator node (EnumNodeType suffix variant)
    - runtime_host_generic: Runtime host infrastructure node

AST Enforcement:
    If node.py files exist (legacy pattern), validates they properly
    call super().__init__(container) to ensure proper initialization.

Skip Behavior:
    Tests skip gracefully when contracts don't exist during scaffold phase,
    using pytest.skip() with clear messages about what's missing.

Path Resolution:
    Uses Path(__file__) for CWD-independent path resolution via conftest.py.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import NamedTuple

import pytest
import yaml

from tests.conftest import CORE_8_NODES, NODES_DIR

# Valid ONEX node types (includes both simple and _GENERIC forms)
# The _GENERIC suffix is used by EnumNodeType in omnibase_core
# The validation lowercases the input, so these must be lowercase
VALID_NODE_TYPES: frozenset[str] = frozenset(
    {
        "effect",
        "compute",
        "reducer",
        "orchestrator",
        # _GENERIC suffix variants (from EnumNodeType)
        "effect_generic",
        "compute_generic",
        "reducer_generic",
        "orchestrator_generic",
        "runtime_host_generic",
    }
)


class ContractValidationResult(NamedTuple):
    """Result of contract.yaml validation."""

    valid: bool
    error: str | None = None


class SuperInitValidationResult(NamedTuple):
    """Result of super().__init__(container) pattern validation."""

    valid: bool
    error: str | None = None
    class_name: str | None = None


def validate_super_init_pattern(node_py_path: Path) -> SuperInitValidationResult:
    """Validate node.py contains proper super().__init__(container) call.

    Uses AST parsing to check that:
    1. The file contains at least one class
    2. Each class with an __init__ method calls super().__init__(container)

    This is a defensive check for legacy node.py files that shouldn't exist
    in the fully declarative pattern, but if they do, they must follow
    proper initialization patterns.

    Args:
        node_py_path: Path to node.py file

    Returns:
        SuperInitValidationResult with validation status
    """
    if not node_py_path.exists():
        return SuperInitValidationResult(True, None, None)

    try:
        source_code = node_py_path.read_text()
        tree = ast.parse(source_code)
    except SyntaxError as e:
        return SuperInitValidationResult(
            False, f"Syntax error in {node_py_path}: {e}", None
        )

    # Find all class definitions
    classes_with_init: list[tuple[str, bool]] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            class_name = node.name
            has_init = False
            has_super_init_container = False

            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                    has_init = True
                    # Check for super().__init__(container) or self.container
                    for stmt in ast.walk(item):
                        if isinstance(stmt, ast.Call):
                            # Check for super().__init__(...) pattern
                            if (
                                isinstance(stmt.func, ast.Attribute)
                                and stmt.func.attr == "__init__"
                                and isinstance(stmt.func.value, ast.Call)
                                and isinstance(stmt.func.value.func, ast.Name)
                                and stmt.func.value.func.id == "super"
                            ):
                                # Check if container is passed as argument
                                for arg in stmt.args:
                                    if (
                                        isinstance(arg, ast.Name)
                                        and arg.id == "container"
                                    ):
                                        has_super_init_container = True
                                        break
                                    # Also accept self.container
                                    if (
                                        isinstance(arg, ast.Attribute)
                                        and isinstance(arg.value, ast.Name)
                                        and arg.value.id == "self"
                                        and arg.attr == "container"
                                    ):
                                        has_super_init_container = True
                                        break

            if has_init:
                classes_with_init.append((class_name, has_super_init_container))

    # Validate results
    for class_name, has_super_init in classes_with_init:
        if not has_super_init:
            return SuperInitValidationResult(
                False,
                f"Class '{class_name}' missing super().__init__(container) call",
                class_name,
            )

    return SuperInitValidationResult(True, None, None)


def validate_contract(contract_path: Path) -> ContractValidationResult:
    """Validate ONEX contract follows required structure.

    Supports two formats:
    1. Flat format: name and node_type at root level (preferred)
    2. Nested format: name and node_type under 'onex' key

    Rules:
    1. Must be valid YAML
    2. Must specify valid node_type (effect, compute, reducer, orchestrator)
    3. Must have name field
    """
    if not contract_path.exists():
        return ContractValidationResult(False, f"Contract not found: {contract_path}")

    try:
        with open(contract_path, encoding="utf-8") as f:
            data: dict = yaml.safe_load(f)
    except yaml.YAMLError as e:
        return ContractValidationResult(False, f"Invalid YAML: {e}")

    if not isinstance(data, dict):
        return ContractValidationResult(False, "Contract must be a YAML mapping")

    # Support both flat format (preferred) and nested 'onex' format
    if "onex" in data and isinstance(data["onex"], dict):
        contract_data: dict = data["onex"]
    else:
        contract_data = data

    # Check for required fields
    if "node_type" not in contract_data:
        return ContractValidationResult(False, "Missing 'node_type' in contract")

    node_type: str = str(contract_data["node_type"]).lower()
    if node_type not in VALID_NODE_TYPES:
        return ContractValidationResult(
            False,
            f"Invalid node_type '{node_type}', must be one of valid types",
        )

    if "name" not in contract_data:
        return ContractValidationResult(False, "Missing 'name' in contract")

    return ContractValidationResult(True)


class TestContractEnforcement:
    """Test that all nodes have valid ONEX contracts.

    These tests verify that node directories contain proper contract.yaml
    files that define the node declaratively. No node.py files are needed -
    the contract IS the node definition.

    ONEX Declarative Pattern:
    - contract.yaml defines: node_type, name, inputs, outputs, handlers
    - handlers/ contains business logic implementations
    - Runtime instantiates nodes from contracts, not classes
    """

    @pytest.mark.parametrize("node_name", CORE_8_NODES)
    def test_contract_exists(self, node_name: str) -> None:
        """Verify contract.yaml exists for each Core 8 node.

        Skipped for nodes not yet implemented during scaffold phase.
        """
        contract_path: Path = NODES_DIR / node_name / "contract.yaml"
        # Skip if not yet implemented
        if not contract_path.exists():
            pytest.skip(f"Contract pending implementation: {contract_path}")
        assert contract_path.exists()

    @pytest.mark.parametrize("node_name", CORE_8_NODES)
    def test_contract_is_valid(self, node_name: str) -> None:
        """Verify contract.yaml follows ONEX schema.

        Validates that the contract contains required fields:
        - onex.node_type (effect, compute, reducer, orchestrator)
        - onex.name

        Skipped for nodes not yet implemented.
        """
        contract_path: Path = NODES_DIR / node_name / "contract.yaml"
        if not contract_path.exists():
            pytest.skip(f"Contract pending implementation: {contract_path}")

        result: ContractValidationResult = validate_contract(contract_path)
        assert result.valid, f"Contract {node_name} failed validation: {result.error}"

    @pytest.mark.parametrize("node_name", CORE_8_NODES)
    def test_no_node_py_exists(self, node_name: str) -> None:
        """Verify no node.py files exist (fully declarative pattern).

        ONEX nodes are defined by contracts, not Python classes.
        The presence of a node.py file indicates legacy architecture.
        """
        node_py_path: Path = NODES_DIR / node_name / "node.py"
        assert not node_py_path.exists(), (
            f"node.py should not exist for {node_name} - "
            "use contract.yaml for declarative node definition"
        )

    @pytest.mark.parametrize("node_name", CORE_8_NODES)
    def test_node_py_has_proper_super_init_if_exists(self, node_name: str) -> None:
        """Verify node.py has proper super().__init__(container) if it exists.

        This is a defensive check for legacy node.py files. In the fully
        declarative pattern, node.py files should not exist. However, if
        they do exist (legacy or transition), they must properly call
        super().__init__(container) to ensure proper initialization.

        Skipped when node.py doesn't exist (expected in declarative pattern).
        """
        node_py_path: Path = NODES_DIR / node_name / "node.py"
        if not node_py_path.exists():
            pytest.skip(f"node.py not yet created: {node_py_path}")

        result: SuperInitValidationResult = validate_super_init_pattern(node_py_path)
        assert result.valid, (
            f"node.py for {node_name} failed super().__init__(container) check: {result.error}"
        )

    def test_validate_contract_accepts_flat_format(self, tmp_path: Path) -> None:
        """Test that validator accepts flat format (no 'onex' wrapper)."""
        good_contract: Path = tmp_path / "flat_contract.yaml"
        good_contract.write_text(
            """
name: test_node
node_type: effect
version: {major: 1, minor: 0, patch: 0}
"""
        )
        result: ContractValidationResult = validate_contract(good_contract)
        assert result.valid, f"Flat format should be valid: {result.error}"

    def test_validate_contract_rejects_invalid_node_type(self, tmp_path: Path) -> None:
        """Test that validator rejects invalid node_type values."""
        bad_contract: Path = tmp_path / "bad_contract.yaml"
        bad_contract.write_text(
            """
onex:
  name: test
  node_type: invalid_type
"""
        )
        result: ContractValidationResult = validate_contract(bad_contract)
        assert not result.valid
        assert result.error is not None
        assert "invalid" in result.error.lower() or "node_type" in result.error.lower()

    def test_validate_contract_rejects_missing_name(self, tmp_path: Path) -> None:
        """Test that validator rejects contracts without name field."""
        bad_contract: Path = tmp_path / "bad_contract.yaml"
        bad_contract.write_text(
            """
onex:
  node_type: effect
"""
        )
        result: ContractValidationResult = validate_contract(bad_contract)
        assert not result.valid
        assert result.error is not None
        assert "name" in result.error.lower()

    def test_validate_contract_accepts_valid_contract(self, tmp_path: Path) -> None:
        """Test that validator accepts properly formed contracts."""
        good_contract: Path = tmp_path / "good_contract.yaml"
        good_contract.write_text(
            """
onex:
  name: memory_storage
  node_type: effect
  version: 1.0.0
  handlers:
    - handler_db
    - handler_redis
"""
        )
        result: ContractValidationResult = validate_contract(good_contract)
        assert result.valid, f"Should be valid: {result.error}"

    @pytest.mark.parametrize(
        "node_type",
        [
            # Core 4-node architecture types
            "effect",
            "compute",
            "reducer",
            "orchestrator",
            # _GENERIC suffix variants (from EnumNodeType)
            "effect_generic",
            "compute_generic",
            "reducer_generic",
            "orchestrator_generic",
            # Runtime host type
            "runtime_host_generic",
        ],
    )
    def test_validate_contract_accepts_all_node_types(
        self, tmp_path: Path, node_type: str
    ) -> None:
        """Test that validator accepts all valid node types.

        Validates all 9 valid node types defined in VALID_NODE_TYPES:
        - Core types: effect, compute, reducer, orchestrator
        - Generic variants: effect_generic, compute_generic, reducer_generic, orchestrator_generic
        - Runtime host: runtime_host_generic
        """
        contract: Path = tmp_path / f"{node_type}_contract.yaml"
        contract.write_text(
            f"""
onex:
  name: test_{node_type}
  node_type: {node_type}
"""
        )
        result: ContractValidationResult = validate_contract(contract)
        assert result.valid, f"node_type '{node_type}' should be valid: {result.error}"

    def test_runtime_host_generic_is_valid_node_type(self, tmp_path: Path) -> None:
        """Explicitly verify runtime_host_generic is accepted as valid node type.

        This test ensures the runtime_host_generic type (used for runtime host
        infrastructure nodes) is properly included in VALID_NODE_TYPES and
        passes contract validation.
        """
        contract: Path = tmp_path / "runtime_host_contract.yaml"
        contract.write_text(
            """
onex:
  name: runtime_host_node
  node_type: runtime_host_generic
  description: Runtime host infrastructure node for ONEX
"""
        )
        result: ContractValidationResult = validate_contract(contract)
        assert result.valid, f"runtime_host_generic should be valid: {result.error}"

    def test_valid_node_types_frozenset_completeness(self) -> None:
        """Verify VALID_NODE_TYPES contains all expected node types.

        This test documents and enforces the complete set of valid node types
        that should be accepted by the contract validator.
        """
        expected_types = {
            # Core 4-node architecture
            "effect",
            "compute",
            "reducer",
            "orchestrator",
            # Generic variants
            "effect_generic",
            "compute_generic",
            "reducer_generic",
            "orchestrator_generic",
            # Runtime host
            "runtime_host_generic",
        }
        assert expected_types == VALID_NODE_TYPES, (
            f"VALID_NODE_TYPES mismatch.\n"
            f"Expected: {sorted(expected_types)}\n"
            f"Actual: {sorted(VALID_NODE_TYPES)}\n"
            f"Missing: {sorted(expected_types - VALID_NODE_TYPES)}\n"
            f"Extra: {sorted(VALID_NODE_TYPES - expected_types)}"
        )

    def test_validate_contract_rejects_invalid_yaml(self, tmp_path: Path) -> None:
        """Test that validator rejects malformed YAML."""
        bad_contract: Path = tmp_path / "bad_yaml.yaml"
        bad_contract.write_text(
            """
onex:
  name: test
  node_type: effect
  invalid: [unclosed bracket
"""
        )
        result: ContractValidationResult = validate_contract(bad_contract)
        assert not result.valid
        assert result.error is not None
        assert "yaml" in result.error.lower() or "invalid" in result.error.lower()


class TestSuperInitValidation:
    """Unit tests for super().__init__(container) AST validation."""

    def test_validates_nonexistent_file_as_valid(self, tmp_path: Path) -> None:
        """Test that nonexistent files are considered valid (no violation)."""
        nonexistent: Path = tmp_path / "does_not_exist.py"
        result: SuperInitValidationResult = validate_super_init_pattern(nonexistent)
        assert result.valid

    def test_validates_proper_super_init_pattern(self, tmp_path: Path) -> None:
        """Test that proper super().__init__(container) passes validation."""
        node_py: Path = tmp_path / "node.py"
        node_py.write_text(
            """
class NodeExample:
    def __init__(self, container):
        super().__init__(container)
        self.container = container
"""
        )
        result: SuperInitValidationResult = validate_super_init_pattern(node_py)
        assert result.valid, f"Should be valid: {result.error}"

    def test_validates_self_container_pattern(self, tmp_path: Path) -> None:
        """Test that super().__init__(self.container) also passes."""
        node_py: Path = tmp_path / "node.py"
        node_py.write_text(
            """
class NodeExample:
    def __init__(self, container):
        self.container = container
        super().__init__(self.container)
"""
        )
        result: SuperInitValidationResult = validate_super_init_pattern(node_py)
        assert result.valid, f"Should be valid: {result.error}"

    def test_rejects_missing_super_init(self, tmp_path: Path) -> None:
        """Test that missing super().__init__() call is rejected."""
        node_py: Path = tmp_path / "node.py"
        node_py.write_text(
            """
class NodeExample:
    def __init__(self, container):
        self.container = container
"""
        )
        result: SuperInitValidationResult = validate_super_init_pattern(node_py)
        assert not result.valid
        assert result.error is not None
        assert "super().__init__(container)" in result.error

    def test_rejects_super_init_without_container(self, tmp_path: Path) -> None:
        """Test that super().__init__() without container arg is rejected."""
        node_py: Path = tmp_path / "node.py"
        node_py.write_text(
            """
class NodeExample:
    def __init__(self, container):
        super().__init__()
        self.container = container
"""
        )
        result: SuperInitValidationResult = validate_super_init_pattern(node_py)
        assert not result.valid
        assert result.error is not None
        assert "super().__init__(container)" in result.error

    def test_validates_class_without_init(self, tmp_path: Path) -> None:
        """Test that classes without __init__ are considered valid."""
        node_py: Path = tmp_path / "node.py"
        node_py.write_text(
            """
class NodeExample:
    def process(self, data):
        return data
"""
        )
        result: SuperInitValidationResult = validate_super_init_pattern(node_py)
        assert result.valid

    def test_rejects_syntax_errors(self, tmp_path: Path) -> None:
        """Test that syntax errors are properly reported."""
        node_py: Path = tmp_path / "node.py"
        node_py.write_text(
            """
class NodeExample:
    def __init__(self
        # Missing closing paren
"""
        )
        result: SuperInitValidationResult = validate_super_init_pattern(node_py)
        assert not result.valid
        assert "Syntax error" in str(result.error)

    def test_identifies_violating_class(self, tmp_path: Path) -> None:
        """Test that the violating class name is reported."""
        node_py: Path = tmp_path / "node.py"
        node_py.write_text(
            """
class NodeBadExample:
    def __init__(self, container):
        self.container = container
"""
        )
        result: SuperInitValidationResult = validate_super_init_pattern(node_py)
        assert not result.valid
        assert result.class_name == "NodeBadExample"
