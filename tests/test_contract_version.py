# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Contract version field validation tests (OMN-1436).

Tests that YAML contracts use the new `contract_version` field structure
after migration from the legacy root-level `version` field.

This test module verifies:
- contract_version field exists in each contract
- contract_version has correct structure (major, minor, patch integers)
- No legacy root-level 'version' field exists
- name field matches expected node name
- node_type field exists

PR Reference: #19 - YAML contract version migration
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

# Import shared constants from conftest
from tests.conftest import NODES_DIR

if TYPE_CHECKING:
    from omnibase_core.types import MappingResultDict

# Contracts that were migrated in PR #19 (OMN-1436)
# NOTE: This list tracks specific contracts from the migration. For comprehensive
# contract validation, use get_all_contracts() which dynamically discovers all contracts.
MIGRATED_CONTRACTS: list[str] = [
    "memory_retrieval_effect",
    "memory_storage_effect",
    "similarity_compute",
]


def get_all_contracts(nodes_dir: Path | None = None) -> list[str]:
    """Discover all contracts in the nodes directory dynamically.

    Scans the specified nodes directory (or NODES_DIR if not provided) for all
    subdirectories containing a contract.yaml file and returns the list of node
    names (parent directory names).

    Args:
        nodes_dir: Optional path to the nodes directory. Defaults to NODES_DIR.

    Returns:
        list[str]: List of node names that have contract.yaml files.

    Example:
        >>> contracts = get_all_contracts()
        >>> print(contracts)
        ['memory_retrieval_effect', 'memory_storage_effect', 'similarity_compute']
    """
    base = nodes_dir if nodes_dir is not None else NODES_DIR
    if not base.exists():
        return []

    return sorted(
        d.name for d in base.iterdir() if d.is_dir() and (d / "contract.yaml").exists()
    )


# Discover all contracts once at module load for parametrized tests
ALL_DISCOVERED_CONTRACTS: list[str] = get_all_contracts()


def _assert_valid_contract_version(contract_version: object, node_name: str) -> None:
    """Assert contract_version has valid structure with major, minor, patch.

    Validates that the contract_version field:
    - Is not None
    - Is a dict
    - Contains major, minor, patch fields
    - All version fields are non-negative integers

    Args:
        contract_version: The contract_version value from the YAML contract.
        node_name: Name of the node for error messages.

    Raises:
        AssertionError: If any validation fails.
    """
    assert contract_version is not None, f"contract_version field is None: {node_name}"
    assert isinstance(
        contract_version, dict
    ), f"contract_version must be a dict with major/minor/patch: {node_name}"

    for field in ("major", "minor", "patch"):
        assert (
            field in contract_version
        ), f"contract_version missing '{field}' field: {node_name}"
        value: object = contract_version[field]
        assert isinstance(value, int), (
            f"contract_version.{field} must be an integer, "
            f"got {type(value).__name__}: {node_name}"
        )
        assert value >= 0, f"contract_version.{field} must be non-negative: {node_name}"
        assert (
            value < 10000
        ), f"contract_version.{field} seems unreasonably large ({value}): {node_name}"


class TestContractVersionField:
    """Test contract fields for migrated contracts.

    Tests file existence, name matching, node_type values, and nested version
    preservation for contracts in MIGRATED_CONTRACTS. For comprehensive
    contract_version validation (existence, structure, no legacy field),
    see TestAllContractsDiscovery which covers ALL discovered contracts.
    """

    @pytest.mark.parametrize("node_name", MIGRATED_CONTRACTS, ids=str)
    def test_contract_file_exists(self, node_name: str, nodes_dir: Path) -> None:
        """Verify contract.yaml exists for each migrated node."""
        contract_path: Path = nodes_dir / node_name / "contract.yaml"
        assert contract_path.exists(), f"Missing contract: {contract_path}"

    @pytest.mark.parametrize("node_name", MIGRATED_CONTRACTS, ids=str)
    def test_name_field_matches_node(
        self, contract_data: MappingResultDict, node_name: str
    ) -> None:
        """Verify name field matches expected node name."""
        assert "name" in contract_data, f"Contract must have 'name' field: {node_name}"
        assert contract_data["name"] == node_name, (
            f"Contract name mismatch: expected '{node_name}', "
            f"got '{contract_data['name']}'"
        )

    @pytest.mark.parametrize("node_name", MIGRATED_CONTRACTS, ids=str)
    def test_node_type_field_exists(
        self, contract_data: MappingResultDict, node_name: str
    ) -> None:
        """Verify node_type field exists."""
        assert (
            "node_type" in contract_data
        ), f"Contract must have 'node_type' field: {node_name}"
        node_type: object = contract_data["node_type"]
        assert isinstance(node_type, str), f"node_type must be a string: {node_name}"
        assert node_type in (
            "EFFECT",
            "COMPUTE",
            "REDUCER",
            "ORCHESTRATOR",
        ), f"node_type must be a valid ONEX type, got '{node_type}': {node_name}"

    @pytest.mark.parametrize(
        ("node_name", "expected_type"),
        [
            ("memory_retrieval_effect", "EFFECT"),
            ("memory_storage_effect", "EFFECT"),
            ("similarity_compute", "COMPUTE"),
        ],
        ids=["memory_retrieval_effect", "memory_storage_effect", "similarity_compute"],
    )
    def test_node_type_values(
        self, contract_data: MappingResultDict, node_name: str, expected_type: str
    ) -> None:
        """Verify each contract has the expected node_type."""
        assert contract_data.get("node_type") == expected_type, (
            f"Expected node_type '{expected_type}' for {node_name}, "
            f"got '{contract_data.get('node_type')}'"
        )

    @pytest.mark.parametrize("node_name", MIGRATED_CONTRACTS, ids=str)
    def test_nested_version_fields_preserved(
        self, contract_data: MappingResultDict, node_name: str
    ) -> None:
        """Verify nested version fields (tool_specification, event_type) are preserved.

        The migration only affects root-level 'version' field. Nested version
        fields in tool_specification or event_type are intentionally kept as
        they serve different purposes.

        Version can be either:
        - A string (legacy format, e.g., "1.0.0")
        - A dict with major/minor/patch structure (YAML contract format)
        """
        # Check tool_specification.version if present (should be preserved)
        tool_spec: object = contract_data.get("tool_specification")
        if isinstance(tool_spec, dict) and "version" in tool_spec:
            version = tool_spec["version"]
            # Version can be either a string or a structured dict
            assert isinstance(
                version, str | dict
            ), f"tool_specification.version should be a string or dict: {node_name}"
            if isinstance(version, dict):
                _assert_valid_contract_version(
                    version, f"{node_name}/tool_specification"
                )

        # Check event_type.version if present (should be preserved)
        event_type: object = contract_data.get("event_type")
        if isinstance(event_type, dict) and "version" in event_type:
            version = event_type["version"]
            # Version can be either a string or a structured dict
            assert isinstance(
                version, str | dict
            ), f"event_type.version should be a string or dict: {node_name}"
            if isinstance(version, dict):
                _assert_valid_contract_version(version, f"{node_name}/event_type")


class TestAllContractsDiscovery:
    """Test all dynamically discovered contracts have valid contract_version field.

    This test class uses get_all_contracts() to automatically discover all
    contracts in the nodes directory and validates they conform to the
    contract_version structure. This ensures future contracts are validated
    without requiring manual updates to test lists.
    """

    def test_get_all_contracts_behavior(self) -> None:
        """Verify get_all_contracts returns expected list with migrated contracts."""
        contracts = get_all_contracts()
        assert isinstance(contracts, list), "get_all_contracts must return a list"
        for node_name in MIGRATED_CONTRACTS:
            assert node_name in contracts, (
                f"get_all_contracts() should find '{node_name}' "
                f"but got: {contracts}"
            )

    @pytest.mark.skipif(
        not ALL_DISCOVERED_CONTRACTS,
        reason="No contracts discovered in nodes directory",
    )
    @pytest.mark.parametrize("node_name", ALL_DISCOVERED_CONTRACTS, ids=str)
    def test_discovered_contract_has_contract_version(
        self, contract_data: MappingResultDict, node_name: str
    ) -> None:
        """Verify all discovered contracts have contract_version field.

        This test automatically validates any new contracts added to the
        nodes directory, ensuring they follow the contract_version standard.
        """
        assert (
            "contract_version" in contract_data
        ), f"Contract must have 'contract_version' field: {node_name}"

    @pytest.mark.skipif(
        not ALL_DISCOVERED_CONTRACTS,
        reason="No contracts discovered in nodes directory",
    )
    @pytest.mark.parametrize("node_name", ALL_DISCOVERED_CONTRACTS, ids=str)
    def test_discovered_contract_version_structure(
        self, contract_data: MappingResultDict, node_name: str
    ) -> None:
        """Verify all discovered contracts have valid contract_version structure.

        The contract_version field must be a dict with:
        - major: int (non-negative)
        - minor: int (non-negative)
        - patch: int (non-negative)
        """
        contract_version: object | None = contract_data.get("contract_version")
        _assert_valid_contract_version(contract_version, node_name)

    @pytest.mark.skipif(
        not ALL_DISCOVERED_CONTRACTS,
        reason="No contracts discovered in nodes directory",
    )
    @pytest.mark.parametrize("node_name", ALL_DISCOVERED_CONTRACTS, ids=str)
    def test_discovered_contract_no_legacy_version(
        self, contract_data: MappingResultDict, node_name: str
    ) -> None:
        """Verify discovered contracts do not have legacy root-level version field."""
        assert "version" not in contract_data, (
            f"Contract has legacy 'version' field - "
            f"should use 'contract_version': {node_name}"
        )

    def test_get_all_contracts_nonexistent_dir(self, tmp_path: Path) -> None:
        """Verify get_all_contracts returns empty list for nonexistent dir."""
        result = get_all_contracts(tmp_path / "nonexistent")
        assert result == []
