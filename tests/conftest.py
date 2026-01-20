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
"""

from __future__ import annotations

from pathlib import Path

import pytest

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
