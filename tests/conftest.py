# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Pytest configuration and fixtures for OmniMemory tests.

Test Organization
-----------------
Tests are organized by functional area at the top level of the tests/ directory:

- test_contract_validation.py: Contract YAML validation and Pydantic model tests
- test_node_enforcement.py: AST-based declarative pattern enforcement
- test_node_imports.py: Node package structure and import verification
- test_foundation.py: Core model and container tests
- test_concurrency.py: Connection pool, circuit breaker, retry patterns
- test_performance.py: Performance benchmarks and constraints
- test_health_manager.py: Health monitoring and circuit breaker integration
- test_resource_manager.py: Resource lifecycle and pool management

Integration tests in tests/integration/:

- test_handler_subscription.py: HandlerSubscription integration tests (OMN-1393)
- test_node_agent_coordinator.py: NodeAgentCoordinatorOrchestrator tests (OMN-1393)

Note: Node-related tests (imports, enforcement, contracts) are kept at the top
level rather than in a nested tests/nodes/ directory because they validate
core infrastructure patterns that affect all nodes uniformly.

Path Resolution
---------------
All test files use `Path(__file__).parent` for path resolution, ensuring tests
work correctly regardless of the current working directory. Never use `os.getcwd()`
or relative paths in tests.

Skip Handling
-------------
Tests that depend on files that may not exist during scaffold phase use
`pytest.skip()` with clear messages indicating what file is missing.
This allows the test suite to pass while providing visibility into
what implementations are pending.

Integration Test Environment Variables
--------------------------------------
- TEST_DB_DSN: PostgreSQL connection string for subscription tests
- TEST_VALKEY_HOST: Valkey hostname (default: localhost)
- TEST_VALKEY_PORT: Valkey port (default: 6379)
"""

from __future__ import annotations

import os
import socket
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import yaml

if TYPE_CHECKING:
    from omnibase_core.types import MappingResultDict

# Core 8 node names - shared across node-related test modules
CORE_8_NODES: list[str] = [
    "memory_storage_effect",
    "memory_retrieval_effect",
    "semantic_analyzer_compute",
    "similarity_compute",
    "memory_consolidator_reducer",
    "statistics_reducer",
    "memory_lifecycle_orchestrator",
    "agent_coordinator_orchestrator",
]

# Node directory path - use Path(__file__) for CWD independence
NODES_DIR: Path = Path(__file__).parent.parent / "src" / "omnimemory" / "nodes"


@pytest.fixture
def nodes_dir() -> Path:
    """Provide the nodes directory path.

    Returns:
        Path to src/omnimemory/nodes/ directory
    """
    return NODES_DIR


@pytest.fixture
def core_8_nodes() -> list[str]:
    """Provide the list of Core 8 node names.

    Returns:
        List of Core 8 node directory names
    """
    return CORE_8_NODES.copy()


@pytest.fixture
def implemented_nodes(nodes_dir: Path) -> list[str]:
    """Provide list of nodes that have contract.yaml implemented.

    In the fully declarative ONEX pattern, nodes are defined by contracts
    not Python classes. A node is considered "implemented" when it has
    a valid contract.yaml file.

    Aliases: nodes_with_contracts (deprecated, use this fixture instead)

    Returns:
        List of node names with existing contract.yaml files
    """
    return [
        node_name
        for node_name in CORE_8_NODES
        if (nodes_dir / node_name / "contract.yaml").exists()
    ]


# Alias for backwards compatibility - prefer implemented_nodes
nodes_with_contracts = implemented_nodes


@pytest.fixture
def contract_data(request: pytest.FixtureRequest, nodes_dir: Path) -> MappingResultDict:
    """Load and parse contract.yaml for a parametrized node name.

    This fixture reduces redundant YAML file reads across tests by loading
    the contract data once per test. Use with pytest.mark.parametrize to
    specify the node_name.

    Usage:
        @pytest.mark.parametrize("node_name", ["memory_storage_effect", "similarity_compute"])
        def test_something(self, contract_data: MappingResultDict, node_name: str) -> None:
            # contract_data contains the parsed YAML
            assert "name" in contract_data

    When to Use:
        - Parametrized tests that iterate over multiple contracts
        - Tests that need the same contract data in multiple assertions
        - Test classes that validate contract structure across nodes

    When NOT to Use:
        - Single-contract tests: Load YAML directly with yaml.safe_load()
        - Non-parametrized tests: Fixture requires @pytest.mark.parametrize
        - Tests that only need one specific contract: Direct loading is simpler

    Note:
        The fixture accesses node_name from request.node.callspec.params,
        so tests MUST have a 'node_name' parameter defined via parametrize.

    Args:
        request: Pytest fixture request object for accessing test parameters
        nodes_dir: Path to the nodes directory (injected fixture)

    Returns:
        Parsed YAML data as a dictionary

    Raises:
        pytest.skip: If the contract file does not exist
        ValueError: If the test is not parametrized with 'node_name'
    """
    # Extract node_name from parametrize - provide helpful error if missing
    if not hasattr(request.node, "callspec"):
        raise ValueError(
            f"contract_data fixture requires @pytest.mark.parametrize decorator.\n\n"
            f"Test '{request.node.name}' is not parametrized.\n\n"
            f"Required usage:\n"
            f"    @pytest.mark.parametrize('node_name', ['memory_storage_effect'])\n"
            f"    def test_example(self, contract_data: MappingResultDict, node_name: str) -> None:\n"
            f"        assert 'name' in contract_data\n\n"
            f"For multiple nodes:\n"
            f"    @pytest.mark.parametrize('node_name', CORE_8_NODES)\n"
            f"    def test_all_nodes(self, contract_data: MappingResultDict, node_name: str) -> None:\n"
            f"        ...\n\n"
            f"See fixture docstring for complete documentation."
        )

    if "node_name" not in request.node.callspec.params:
        available_params = list(request.node.callspec.params.keys())
        raise ValueError(
            f"contract_data fixture requires 'node_name' parameter in @pytest.mark.parametrize.\n\n"
            f"Test '{request.node.name}' has parametrize but missing 'node_name'.\n"
            f"Available parameters: {available_params}\n\n"
            f"Required pattern:\n"
            f"    @pytest.mark.parametrize('node_name', [...])\n"
            f"    def test_example(self, contract_data, node_name: str) -> None:\n\n"
            f"If you have multiple parametrize decorators, ensure one includes 'node_name'."
        )

    node_name: str = request.node.callspec.params["node_name"]
    contract_path: Path = nodes_dir / node_name / "contract.yaml"

    if not contract_path.exists():
        pytest.skip(f"Contract file not yet implemented: {contract_path}")

    with open(contract_path) as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        pytest.fail(
            f"Contract must be a dict, got {type(data).__name__}: {contract_path}"
        )
    return data


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers for test categorization."""
    config.addinivalue_line(
        "markers",
        "node: Tests related to ONEX node implementations",
    )
    config.addinivalue_line(
        "markers",
        "contract: Tests related to contract.yaml validation",
    )
    config.addinivalue_line(
        "markers",
        "enforcement: Tests for AST-based declarative pattern enforcement",
    )
    config.addinivalue_line(
        "markers",
        "scaffold: Tests that may skip during scaffold phase when files don't exist",
    )
    config.addinivalue_line(
        "markers",
        "config: Tests for configuration models and settings",
    )
    config.addinivalue_line(
        "markers",
        "bootstrap: Tests for bootstrap initialization",
    )
    config.addinivalue_line(
        "markers",
        "secrets: Tests for secrets provider",
    )
    config.addinivalue_line(
        "markers",
        "benchmark: Performance benchmark tests",
    )
    config.addinivalue_line(
        "markers",
        "integration: marks tests as integration tests (may require external services)",
    )
    config.addinivalue_line(
        "markers",
        "memgraph: marks tests as requiring Memgraph database",
    )
    config.addinivalue_line(
        "markers",
        "migration: tests specific to migration validation (may be skipped post-release)",
    )
    config.addinivalue_line(
        "markers",
        "subscription: marks tests as subscription system tests (OMN-1393)",
    )
    config.addinivalue_line(
        "markers",
        "orchestrator: marks tests as orchestrator node tests (OMN-1393)",
    )


# =============================================================================
# Integration Test Configuration
# =============================================================================

# Default connection settings for integration tests
DEFAULT_DB_DSN = "postgresql://postgres:password@localhost:5432/omnimemory_test"
DEFAULT_VALKEY_HOST = "localhost"
DEFAULT_VALKEY_PORT = 6379


def get_test_db_dsn() -> str:
    """Get PostgreSQL DSN from environment or default.

    Returns:
        PostgreSQL connection string for tests.
    """
    return os.environ.get("TEST_DB_DSN", DEFAULT_DB_DSN)


def get_test_valkey_host() -> str:
    """Get Valkey host from environment or default.

    Returns:
        Valkey hostname.
    """
    return os.environ.get("TEST_VALKEY_HOST", DEFAULT_VALKEY_HOST)


def get_test_valkey_port() -> int:
    """Get Valkey port from environment or default.

    Returns:
        Valkey port number.
    """
    return int(os.environ.get("TEST_VALKEY_PORT", str(DEFAULT_VALKEY_PORT)))


def is_port_available(host: str, port: int, timeout: float = 1.0) -> bool:
    """Check if a port is available and responding.

    Args:
        host: The hostname to check.
        port: The port number to check.
        timeout: Connection timeout in seconds.

    Returns:
        True if port is open and responding, False otherwise.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(timeout)
            result = sock.connect_ex((host, port))
            return result == 0
    except OSError:
        return False


def check_services_available() -> bool:
    """Check if PostgreSQL and Valkey are available.

    Returns:
        True if both services are reachable, False otherwise.
    """
    # Check PostgreSQL (extract host/port from DSN)
    dsn = get_test_db_dsn()
    # Simple parsing - assumes format postgresql://user:pass@host:port/db
    try:
        if "@" in dsn and ":" in dsn.split("@")[1]:
            host_port = dsn.split("@")[1].split("/")[0]
            if ":" in host_port:
                pg_host, pg_port = host_port.rsplit(":", 1)
                if not is_port_available(pg_host, int(pg_port)):
                    return False
    except (IndexError, ValueError):
        return False

    # Check Valkey
    valkey_host = get_test_valkey_host()
    valkey_port = get_test_valkey_port()
    return is_port_available(valkey_host, valkey_port)


# =============================================================================
# Integration Test Fixtures
# =============================================================================


@pytest.fixture
def test_db_dsn() -> str:
    """Provide PostgreSQL DSN for tests.

    Returns:
        PostgreSQL connection string.
    """
    return get_test_db_dsn()


@pytest.fixture
def test_valkey_host() -> str:
    """Provide Valkey host for tests.

    Returns:
        Valkey hostname.
    """
    return get_test_valkey_host()


@pytest.fixture
def test_valkey_port() -> int:
    """Provide Valkey port for tests.

    Returns:
        Valkey port number.
    """
    return get_test_valkey_port()


@pytest.fixture
def services_available() -> bool:
    """Check if required services are available.

    Returns:
        True if PostgreSQL and Valkey are available.
    """
    return check_services_available()
