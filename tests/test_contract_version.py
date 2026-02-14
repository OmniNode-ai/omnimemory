# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Contract version field validation tests (OMN-1436).

Tests that YAML contracts use the new `contract_version` and `node_version`
field structures after migration from the legacy root-level `version` field.

This test module verifies:
- contract_version field exists in each contract
- node_version field exists in each contract
- Both version fields have correct structure (major, minor, patch integers)
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

# Sanity check upper bound for SemVer version components (major, minor, patch).
# Real-world SemVer versions rarely exceed 4 digits; this catches typos and corrupt data.
MAX_REASONABLE_VERSION_COMPONENT: int = 10000

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


# Discover all contracts once at module load for parametrized tests.
#
# DESIGN DECISION: Module-level discovery is intentional and acceptable here because:
# 1. pytest.mark.parametrize requires the list at collection time (before fixtures run)
# 2. Contract files are static YAML - they don't change during test execution
# 3. Collection-time discovery matches pytest's parametrize model correctly
#
# For tests that need fresh runtime discovery (e.g., validating this constant hasn't
# gone stale, or testing with dynamically created contracts), use the `all_contracts`
# fixture which calls get_all_contracts() at test runtime.
#
# The test_module_constant_matches_runtime_discovery() test ensures this constant
# stays synchronized with runtime discovery - if contracts are added/removed between
# collection and execution (rare but possible in CI), that test will catch it.
ALL_DISCOVERED_CONTRACTS: list[str] = get_all_contracts()


@pytest.fixture
def all_contracts() -> list[str]:
    """Fixture providing fresh contract discovery at test runtime.

    Use this fixture when you need contracts discovered at test execution time
    rather than collection time. This is useful for:
    - Tests that validate the module-level constant hasn't gone stale
    - Tests that create/modify contracts during the test session
    - Integration tests that need to verify discovery behavior

    Returns:
        list[str]: Freshly discovered list of node names with contract.yaml files.

    Example:
        def test_new_contract_discovered(all_contracts: list[str]) -> None:
            # all_contracts is discovered fresh when this test runs
            assert "my_new_contract" in all_contracts
    """
    return get_all_contracts()


def _assert_valid_version_structure(
    version: object, field_name: str, node_name: str
) -> None:
    """Assert version field has valid structure with major, minor, patch.

    Validates that a version field (contract_version or node_version):
    - Is not None
    - Is a dict
    - Contains major, minor, patch fields
    - All version fields are non-negative integers
    - All version fields are within reasonable bounds

    Uses pytest.fail() for better error reporting and diff output in test failures.

    Args:
        version: The version value from the YAML contract.
        field_name: Name of the version field (e.g., "contract_version", "node_version").
        node_name: Name of the node for error messages.

    Raises:
        pytest.fail.Exception: If any validation fails.
    """
    if version is None:
        pytest.fail(f"{field_name} field is None: {node_name}")
    if not isinstance(version, dict):
        pytest.fail(
            f"{field_name} must be a dict with major/minor/patch, "
            f"got {type(version).__name__}: {node_name}"
        )

    # Type narrowing: version is now known to be a dict
    version_dict: dict[str, object] = version

    for field in ("major", "minor", "patch"):
        if field not in version_dict:
            pytest.fail(f"{field_name} missing '{field}' field: {node_name}")
        value: object = version_dict[field]
        if not isinstance(value, int):
            pytest.fail(
                f"{field_name}.{field} must be an integer, "
                f"got {type(value).__name__}: {node_name}"
            )
        if value < 0:
            pytest.fail(
                f"{field_name}.{field} must be non-negative (got {value}): {node_name}"
            )
        if value >= MAX_REASONABLE_VERSION_COMPONENT:
            pytest.fail(
                f"{field_name}.{field} seems unreasonably large ({value}): {node_name}"
            )


class TestContractVersionField:
    """Test migration-specific contract fields for MIGRATED_CONTRACTS.

    This class validates migration-specific concerns:
    - File existence for specific migrated contracts
    - Name field matches node directory name
    - node_type field exists and has expected values per contract
    - Nested version fields (tool_specification.version, event_type.version) preserved
    - Exact version values (0.1.0) set correctly during migration

    For comprehensive validation of contract_version and node_version fields
    (existence, structure, no legacy field) across ALL contracts, see
    TestAllContractsDiscovery which dynamically discovers and validates
    every contract in the nodes directory.
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
        """Verify nested version fields are valid IF present (if-present-then-valid).

        This test uses "if present, then valid" semantics intentionally:
        - Contracts WITHOUT tool_specification.version or event_type.version PASS
        - Contracts WITH these nested fields must have valid version structure
        - The migration does NOT add nested version fields; it only migrates root version

        This behavior is correct because:
        1. Not all contracts have tool_specification or event_type sections
        2. The migration only transforms root-level 'version' to 'contract_version'
        3. Nested version fields serve different purposes (API versioning, event schema)
           and are preserved unchanged if they exist

        When present, version can be either:
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
                _assert_valid_version_structure(
                    version, "tool_specification.version", node_name
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
                _assert_valid_version_structure(
                    version, "event_type.version", node_name
                )

    @pytest.mark.migration
    @pytest.mark.parametrize(
        ("node_name", "expected_version"),
        [
            ("memory_retrieval_effect", {"major": 0, "minor": 2, "patch": 0}),
            ("memory_storage_effect", {"major": 0, "minor": 2, "patch": 0}),
            ("similarity_compute", {"major": 0, "minor": 1, "patch": 0}),
        ],
        ids=["memory_retrieval_effect", "memory_storage_effect", "similarity_compute"],
    )
    def test_contract_version_values(
        self,
        contract_data: MappingResultDict,
        node_name: str,
        expected_version: dict[str, int],
    ) -> None:
        """Verify contract_version has expected values for each migrated contract.

        This test validates the actual version values, not just the structure.
        Contracts migrated in PR #19 (OMN-1436) started at 0.1.0. Some were
        bumped to 0.2.0 when event_bus subcontracts were added (OMN-1746).

        This ensures:
        - Migration set consistent initial versions across all contracts
        - Version bumps from subsequent PRs are tracked correctly
        - No accidental version drift

        Note:
            This test has the @pytest.mark.migration marker because it asserts
            exact version values that will change when versions are bumped.
            Post-release, skip with: pytest -m "not migration"
        """
        assert contract_data.get("contract_version") == expected_version, (
            f"Expected contract_version {expected_version} for {node_name}, "
            f"got {contract_data.get('contract_version')}"
        )

    @pytest.mark.migration
    @pytest.mark.parametrize("node_name", MIGRATED_CONTRACTS, ids=str)
    def test_node_version_values(
        self, contract_data: MappingResultDict, node_name: str
    ) -> None:
        """Verify node_version has expected initial values after migration.

        All contracts should have node_version 0.1.0 as their starting point.

        Note:
            This test has the @pytest.mark.migration marker because it asserts
            exact version values that will change when versions are bumped.
            Post-release, skip with: pytest -m "not migration"
        """
        expected: dict[str, int] = {"major": 0, "minor": 1, "patch": 0}
        assert contract_data.get("node_version") == expected, (
            f"Expected node_version {expected} for {node_name}, "
            f"got {contract_data.get('node_version')}"
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
        _assert_valid_version_structure(contract_version, "contract_version", node_name)

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

    @pytest.mark.skipif(
        not ALL_DISCOVERED_CONTRACTS,
        reason="No contracts discovered in nodes directory",
    )
    @pytest.mark.parametrize("node_name", ALL_DISCOVERED_CONTRACTS, ids=str)
    def test_discovered_contract_has_node_version(
        self, contract_data: MappingResultDict, node_name: str
    ) -> None:
        """Verify all discovered contracts have node_version field.

        ONEX contracts require both contract_version and node_version fields.
        """
        assert (
            "node_version" in contract_data
        ), f"Contract must have 'node_version' field: {node_name}"

    @pytest.mark.skipif(
        not ALL_DISCOVERED_CONTRACTS,
        reason="No contracts discovered in nodes directory",
    )
    @pytest.mark.parametrize("node_name", ALL_DISCOVERED_CONTRACTS, ids=str)
    def test_discovered_node_version_structure(
        self, contract_data: MappingResultDict, node_name: str
    ) -> None:
        """Verify all discovered contracts have valid node_version structure.

        The node_version field must be a dict with:
        - major: int (non-negative)
        - minor: int (non-negative)
        - patch: int (non-negative)
        """
        node_version: object | None = contract_data.get("node_version")
        _assert_valid_version_structure(node_version, "node_version", node_name)

    def test_get_all_contracts_nonexistent_dir(self, tmp_path: Path) -> None:
        """Verify get_all_contracts returns empty list for nonexistent dir."""
        result = get_all_contracts(tmp_path / "nonexistent")
        assert result == []

    def test_module_constant_matches_runtime_discovery(
        self, all_contracts: list[str]
    ) -> None:
        """Verify ALL_DISCOVERED_CONTRACTS matches fresh runtime discovery.

        This test catches staleness issues where:
        - Contracts were added after module import but before test execution
        - Contracts were removed after module import but before test execution
        - Collection-time state diverged from execution-time state (rare in CI)

        The test validates that our module-level constant (used for parametrize)
        remains synchronized with runtime discovery. If this test fails:
        1. Check if contracts were added/removed during the test session
        2. Verify NODES_DIR path is consistent between collection and execution
        3. Consider if test isolation is affecting filesystem state

        Note: In normal operation, this should always pass because contract files
        are static YAML that don't change during test runs. Failure indicates
        either a test environment issue or unexpected filesystem changes.
        """
        assert all_contracts == ALL_DISCOVERED_CONTRACTS, (
            f"Module-level ALL_DISCOVERED_CONTRACTS doesn't match runtime discovery.\n"
            f"Runtime discovery: {all_contracts}\n"
            f"Module constant: {ALL_DISCOVERED_CONTRACTS}\n"
            f"Missing from constant: {set(all_contracts) - set(ALL_DISCOVERED_CONTRACTS)}\n"
            f"Extra in constant: {set(ALL_DISCOVERED_CONTRACTS) - set(all_contracts)}"
        )
