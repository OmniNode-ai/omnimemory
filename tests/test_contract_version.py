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
    from tests.shared_types import YamlData

# Contracts that were migrated in PR #19 (OMN-1436)
# NOTE: This list tracks specific contracts from the migration. For comprehensive
# contract validation, use get_all_contracts() which dynamically discovers all contracts.
MIGRATED_CONTRACTS: list[str] = [
    "memory_retrieval_effect",
    "memory_storage_effect",
    "similarity_compute",
]


def get_all_contracts() -> list[str]:
    """Discover all contracts in the nodes directory dynamically.

    Scans NODES_DIR for all subdirectories containing a contract.yaml file
    and returns the list of node names (parent directory names).

    Returns:
        list[str]: List of node names that have contract.yaml files.

    Example:
        >>> contracts = get_all_contracts()
        >>> print(contracts)
        ['memory_retrieval_effect', 'memory_storage_effect', 'similarity_compute']
    """
    if not NODES_DIR.exists():
        return []

    contracts: list[str] = []
    for node_dir in NODES_DIR.iterdir():
        if node_dir.is_dir():
            contract_path = node_dir / "contract.yaml"
            if contract_path.exists():
                contracts.append(node_dir.name)

    return sorted(contracts)


class TestContractVersionField:
    """Test contract_version field exists and has correct structure."""

    @pytest.mark.parametrize("node_name", MIGRATED_CONTRACTS, ids=str)
    def test_contract_file_exists(self, node_name: str, nodes_dir: Path) -> None:
        """Verify contract.yaml exists for each migrated node."""
        contract_path: Path = nodes_dir / node_name / "contract.yaml"
        assert contract_path.exists(), f"Missing contract: {contract_path}"

    @pytest.mark.parametrize("node_name", MIGRATED_CONTRACTS, ids=str)
    def test_contract_version_field_exists(
        self, contract_data: YamlData, node_name: str
    ) -> None:
        """Verify contract_version field exists (not legacy version field).

        After OMN-1436 migration, contracts must use contract_version
        instead of root-level version field.
        """
        assert (
            "contract_version" in contract_data
        ), f"Contract must have 'contract_version' field (not 'version'): {node_name}"

    @pytest.mark.parametrize("node_name", MIGRATED_CONTRACTS, ids=str)
    def test_contract_version_structure(
        self, contract_data: YamlData, node_name: str
    ) -> None:
        """Verify contract_version has major, minor, patch structure.

        The contract_version field must be a dict with:
        - major: int
        - minor: int
        - patch: int
        """
        contract_version: object | None = contract_data.get("contract_version")
        assert (
            contract_version is not None
        ), f"contract_version field is None: {node_name}"
        assert isinstance(
            contract_version, dict
        ), f"contract_version must be a dict with major/minor/patch: {node_name}"

        # Verify required version components
        required_fields: list[str] = ["major", "minor", "patch"]
        for field in required_fields:
            assert (
                field in contract_version
            ), f"contract_version missing '{field}' field: {node_name}"
            value: object = contract_version[field]
            assert isinstance(value, int), (
                f"contract_version.{field} must be an integer, "
                f"got {type(value).__name__}: {node_name}"
            )
            assert (
                value >= 0
            ), f"contract_version.{field} must be non-negative: {node_name}"

    @pytest.mark.parametrize("node_name", MIGRATED_CONTRACTS, ids=str)
    def test_no_legacy_version_field(
        self, contract_data: YamlData, node_name: str
    ) -> None:
        """Verify no legacy root-level 'version' field exists.

        After migration, contracts should not have a root-level 'version'
        field. The version is now in 'contract_version'.

        Note: tool_specification.version is still valid and separate.
        """
        assert "version" not in contract_data, (
            f"Contract has legacy root-level 'version' field - "
            f"should be 'contract_version': {node_name}"
        )

    @pytest.mark.parametrize("node_name", MIGRATED_CONTRACTS, ids=str)
    def test_name_field_matches_node(
        self, contract_data: YamlData, node_name: str
    ) -> None:
        """Verify name field matches expected node name."""
        assert "name" in contract_data, f"Contract must have 'name' field: {node_name}"
        assert contract_data["name"] == node_name, (
            f"Contract name mismatch: expected '{node_name}', "
            f"got '{contract_data['name']}'"
        )

    @pytest.mark.parametrize("node_name", MIGRATED_CONTRACTS, ids=str)
    def test_node_type_field_exists(
        self, contract_data: YamlData, node_name: str
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


class TestContractVersionValues:
    """Test specific contract_version values for migrated contracts."""

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
        self, contract_data: YamlData, node_name: str, expected_type: str
    ) -> None:
        """Verify each contract has the expected node_type."""
        assert contract_data.get("node_type") == expected_type, (
            f"Expected node_type '{expected_type}' for {node_name}, "
            f"got '{contract_data.get('node_type')}'"
        )

    @pytest.mark.parametrize("node_name", MIGRATED_CONTRACTS, ids=str)
    def test_contract_version_is_0_1_0(
        self, contract_data: YamlData, node_name: str
    ) -> None:
        """Verify all migrated contracts have version 0.1.0.

        All three contracts in this migration have contract_version 0.1.0.
        """
        contract_version: object = contract_data.get("contract_version", {})
        if isinstance(contract_version, dict):
            major: object | None = contract_version.get("major")
            minor: object | None = contract_version.get("minor")
            patch: object | None = contract_version.get("patch")

            assert (major, minor, patch) == (0, 1, 0), (
                f"Expected contract_version 0.1.0, "
                f"got {major}.{minor}.{patch}: {node_name}"
            )


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

    @pytest.mark.parametrize(
        "node_name",
        get_all_contracts(),
        ids=lambda x: x,
    )
    def test_discovered_contract_has_contract_version(
        self, contract_data: YamlData, node_name: str
    ) -> None:
        """Verify all discovered contracts have contract_version field.

        This test automatically validates any new contracts added to the
        nodes directory, ensuring they follow the contract_version standard.
        """
        assert (
            "contract_version" in contract_data
        ), f"Contract must have 'contract_version' field: {node_name}"

    @pytest.mark.parametrize(
        "node_name",
        get_all_contracts(),
        ids=lambda x: x,
    )
    def test_discovered_contract_version_structure(
        self, contract_data: YamlData, node_name: str
    ) -> None:
        """Verify all discovered contracts have valid contract_version structure.

        The contract_version field must be a dict with:
        - major: int (non-negative)
        - minor: int (non-negative)
        - patch: int (non-negative)
        """
        contract_version: object | None = contract_data.get("contract_version")
        assert (
            contract_version is not None
        ), f"contract_version field is None: {node_name}"
        assert isinstance(
            contract_version, dict
        ), f"contract_version must be a dict: {node_name}"

        for field in ("major", "minor", "patch"):
            assert (
                field in contract_version
            ), f"contract_version missing '{field}': {node_name}"
            value: object = contract_version[field]
            assert isinstance(value, int), (
                f"contract_version.{field} must be int, "
                f"got {type(value).__name__}: {node_name}"
            )
            assert (
                value >= 0
            ), f"contract_version.{field} must be non-negative: {node_name}"

    @pytest.mark.parametrize(
        "node_name",
        get_all_contracts(),
        ids=lambda x: x,
    )
    def test_discovered_contract_no_legacy_version(
        self, contract_data: YamlData, node_name: str
    ) -> None:
        """Verify discovered contracts do not have legacy root-level version field."""
        assert "version" not in contract_data, (
            f"Contract has legacy 'version' field - "
            f"should use 'contract_version': {node_name}"
        )
