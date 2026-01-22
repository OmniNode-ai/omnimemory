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

import pytest
import yaml

# Node directory path - use Path(__file__) for CWD independence
NODES_DIR: Path = Path(__file__).parent.parent / "src" / "omnimemory" / "nodes"

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


# Type alias for YAML data
YamlData = dict[str, object]


class TestContractVersionField:
    """Test contract_version field exists and has correct structure."""

    @pytest.mark.parametrize("node_name", MIGRATED_CONTRACTS)
    def test_contract_file_exists(self, node_name: str) -> None:
        """Verify contract.yaml exists for each migrated node."""
        contract_path: Path = NODES_DIR / node_name / "contract.yaml"
        assert contract_path.exists(), f"Missing contract: {contract_path}"

    @pytest.mark.parametrize("node_name", MIGRATED_CONTRACTS)
    def test_contract_version_field_exists(self, node_name: str) -> None:
        """Verify contract_version field exists (not legacy version field).

        After OMN-1436 migration, contracts must use contract_version
        instead of root-level version field.
        """
        contract_path: Path = NODES_DIR / node_name / "contract.yaml"
        if not contract_path.exists():
            pytest.skip(f"File not yet implemented: {contract_path}")

        with open(contract_path) as f:
            data: YamlData = yaml.safe_load(f)

        assert isinstance(data, dict), f"Contract must be a dict: {node_name}"
        assert (
            "contract_version" in data
        ), f"Contract must have 'contract_version' field (not 'version'): {node_name}"

    @pytest.mark.parametrize("node_name", MIGRATED_CONTRACTS)
    def test_contract_version_structure(self, node_name: str) -> None:
        """Verify contract_version has major, minor, patch structure.

        The contract_version field must be a dict with:
        - major: int
        - minor: int
        - patch: int
        """
        contract_path: Path = NODES_DIR / node_name / "contract.yaml"
        if not contract_path.exists():
            pytest.skip(f"File not yet implemented: {contract_path}")

        with open(contract_path) as f:
            data: YamlData = yaml.safe_load(f)

        contract_version: object | None = data.get("contract_version")
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

    @pytest.mark.parametrize("node_name", MIGRATED_CONTRACTS)
    def test_no_legacy_version_field(self, node_name: str) -> None:
        """Verify no legacy root-level 'version' field exists.

        After migration, contracts should not have a root-level 'version'
        field. The version is now in 'contract_version'.

        Note: tool_specification.version is still valid and separate.
        """
        contract_path: Path = NODES_DIR / node_name / "contract.yaml"
        if not contract_path.exists():
            pytest.skip(f"File not yet implemented: {contract_path}")

        with open(contract_path) as f:
            data: YamlData = yaml.safe_load(f)

        assert "version" not in data, (
            f"Contract has legacy root-level 'version' field - "
            f"should be 'contract_version': {node_name}"
        )

    @pytest.mark.parametrize("node_name", MIGRATED_CONTRACTS)
    def test_name_field_matches_node(self, node_name: str) -> None:
        """Verify name field matches expected node name."""
        contract_path: Path = NODES_DIR / node_name / "contract.yaml"
        if not contract_path.exists():
            pytest.skip(f"File not yet implemented: {contract_path}")

        with open(contract_path) as f:
            data: YamlData = yaml.safe_load(f)

        assert "name" in data, f"Contract must have 'name' field: {node_name}"
        assert data["name"] == node_name, (
            f"Contract name mismatch: expected '{node_name}', " f"got '{data['name']}'"
        )

    @pytest.mark.parametrize("node_name", MIGRATED_CONTRACTS)
    def test_node_type_field_exists(self, node_name: str) -> None:
        """Verify node_type field exists."""
        contract_path: Path = NODES_DIR / node_name / "contract.yaml"
        if not contract_path.exists():
            pytest.skip(f"File not yet implemented: {contract_path}")

        with open(contract_path) as f:
            data: YamlData = yaml.safe_load(f)

        assert "node_type" in data, f"Contract must have 'node_type' field: {node_name}"
        node_type: object = data["node_type"]
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
    )
    def test_node_type_values(self, node_name: str, expected_type: str) -> None:
        """Verify each contract has the expected node_type."""
        contract_path: Path = NODES_DIR / node_name / "contract.yaml"
        if not contract_path.exists():
            pytest.skip(f"File not yet implemented: {contract_path}")

        with open(contract_path) as f:
            data: YamlData = yaml.safe_load(f)

        assert data.get("node_type") == expected_type, (
            f"Expected node_type '{expected_type}' for {node_name}, "
            f"got '{data.get('node_type')}'"
        )

    @pytest.mark.parametrize("node_name", MIGRATED_CONTRACTS)
    def test_contract_version_is_0_1_0(self, node_name: str) -> None:
        """Verify all migrated contracts have version 0.1.0.

        All three contracts in this migration have contract_version 0.1.0.
        """
        contract_path: Path = NODES_DIR / node_name / "contract.yaml"
        if not contract_path.exists():
            pytest.skip(f"File not yet implemented: {contract_path}")

        with open(contract_path) as f:
            data: YamlData = yaml.safe_load(f)

        contract_version: object = data.get("contract_version", {})
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

    def test_get_all_contracts_returns_list(self) -> None:
        """Verify get_all_contracts returns a list."""
        contracts = get_all_contracts()
        assert isinstance(contracts, list), "get_all_contracts must return a list"

    def test_get_all_contracts_finds_existing(self) -> None:
        """Verify get_all_contracts finds at least the migrated contracts."""
        contracts = get_all_contracts()
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
    def test_discovered_contract_has_contract_version(self, node_name: str) -> None:
        """Verify all discovered contracts have contract_version field.

        This test automatically validates any new contracts added to the
        nodes directory, ensuring they follow the contract_version standard.
        """
        contract_path: Path = NODES_DIR / node_name / "contract.yaml"
        assert contract_path.exists(), f"Contract file missing: {contract_path}"

        with open(contract_path) as f:
            data: YamlData = yaml.safe_load(f)

        assert isinstance(data, dict), f"Contract must be a dict: {node_name}"
        assert (
            "contract_version" in data
        ), f"Contract must have 'contract_version' field: {node_name}"

    @pytest.mark.parametrize(
        "node_name",
        get_all_contracts(),
        ids=lambda x: x,
    )
    def test_discovered_contract_version_structure(self, node_name: str) -> None:
        """Verify all discovered contracts have valid contract_version structure.

        The contract_version field must be a dict with:
        - major: int (non-negative)
        - minor: int (non-negative)
        - patch: int (non-negative)
        """
        contract_path: Path = NODES_DIR / node_name / "contract.yaml"
        if not contract_path.exists():
            pytest.skip(f"Contract not found: {contract_path}")

        with open(contract_path) as f:
            data: YamlData = yaml.safe_load(f)

        contract_version: object | None = data.get("contract_version")
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
    def test_discovered_contract_no_legacy_version(self, node_name: str) -> None:
        """Verify discovered contracts do not have legacy root-level version field."""
        contract_path: Path = NODES_DIR / node_name / "contract.yaml"
        if not contract_path.exists():
            pytest.skip(f"Contract not found: {contract_path}")

        with open(contract_path) as f:
            data: YamlData = yaml.safe_load(f)

        assert "version" not in data, (
            f"Contract has legacy 'version' field - "
            f"should use 'contract_version': {node_name}"
        )
