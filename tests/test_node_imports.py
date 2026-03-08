# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Node structure tests - verify Core 8 nodes follow ONEX patterns.

These tests ensure the node package structure is correct:
- The nodes package can be imported
- Each node subpackage exists and can be imported
- Directory structure follows ONEX declarative patterns
- Contract-based architecture (no node.py files)

Skip Behavior:
    Tests skip gracefully when optional files don't exist during scaffold phase.

Path Resolution:
    Uses Path(__file__) for CWD-independent path resolution via conftest.py.
"""

from __future__ import annotations

import importlib
import types
from pathlib import Path

import pytest

from tests.conftest import CORE_8_NODES, NODES_DIR


class TestNodeImports:
    """Test that all node packages can be imported."""

    def test_nodes_package_import_succeeds(self) -> None:
        """Verify the nodes package can be imported."""
        try:
            import omnimemory.nodes

            assert omnimemory.nodes is not None
        except ImportError as e:
            pytest.skip(f"Package not installed in editable mode: {e}")

    @pytest.mark.parametrize("node_name", CORE_8_NODES)
    def test_node_package_import_succeeds(self, node_name: str) -> None:
        """Verify each node package can be imported."""
        module_name: str = f"omnimemory.nodes.{node_name}"
        try:
            module: types.ModuleType = importlib.import_module(module_name)
            assert module is not None
        except ImportError as e:
            # Expected during scaffold phase
            pytest.skip(f"Node package not yet fully implemented: {e}")

    def test_nodes_package_exports_core_nodes(self) -> None:
        """Verify __all__ in nodes package lists all Core 8 nodes.

        The nodes package __init__.py should export all Core 8 node
        names in its __all__ list for proper package discoverability.
        """
        try:
            from omnimemory.nodes import __all__ as nodes_all

            for node_name in CORE_8_NODES:
                # Only check nodes whose directories exist
                node_dir: Path = NODES_DIR / node_name
                if not node_dir.exists():
                    continue
                assert node_name in nodes_all, f"Missing from __all__: {node_name}"
        except ImportError:
            pytest.skip("nodes package not properly configured")


class TestNodeStructure:
    """Test node directory structure is correct.

    Validates that the scaffold for each Core 8 node follows the
    required ONEX directory structure including:
    - Node directory exists
    - __init__.py exists in each node directory
    - No local handlers directories (handlers from omnibase_infra)

    Note: node.py enforcement tests are in test_node_enforcement.py
    to consolidate all declarative pattern validation in one place.
    """

    # Nodes excluded from the no-local-handlers check with reasons:
    # - COMPUTE nodes: Have node-specific pure computation handlers (no I/O),
    #   which are fundamentally different from infrastructure handlers
    # - Nodes with mock handlers: Temporary development/testing implementations
    #   until omnibase_infra handlers are available
    # - ORCHESTRATOR nodes with domain-specific handlers: Memory lifecycle handlers
    #   contain domain logic that doesn't belong in omnibase_infra
    NODES_WITH_ALLOWED_HANDLERS: set[str] = {
        # COMPUTE node: Pure math computation handler for vector similarity.
        # Not an infrastructure handler - performs no I/O operations.
        "node_similarity_compute",
        # COMPUTE node: Pure semantic analysis computation handler.
        # Delegates I/O to provider protocols, handler contains only pure logic.
        "node_semantic_analyzer_compute",
        # EFFECT node with MOCK handlers: Temporary mock implementations for
        # development/testing. Will be removed when omnibase_infra is integrated.
        # TODO: Remove from exclusion list when migrating to real handlers.
        "node_memory_retrieval_effect",
        # ORCHESTRATOR node with domain-specific lifecycle handlers:
        # - handler_memory_tick: TTL evaluation and event emission
        # - handler_memory_expire: ACTIVE->EXPIRED with optimistic locking
        # - handler_memory_archive: Archive to cold storage with gzip compression
        # These contain memory-domain logic (not generic infrastructure).
        "node_memory_lifecycle_orchestrator",
    }

    @pytest.mark.parametrize("node_name", CORE_8_NODES)
    def test_node_directory_exists(self, node_name: str) -> None:
        """Verify node directory exists for each Core 8 node.

        This is a fundamental scaffold requirement - all 8 node directories
        must exist even before implementation begins.
        """
        node_dir: Path = NODES_DIR / node_name
        assert node_dir.exists(), f"Missing node directory: {node_dir}"
        assert node_dir.is_dir(), f"Not a directory: {node_dir}"

    @pytest.mark.parametrize("node_name", CORE_8_NODES)
    def test_node_init_exists(self, node_name: str) -> None:
        """Verify __init__.py exists in each node directory.

        The __init__.py file is required to make each node directory
        a proper Python package that can be imported.
        """
        node_dir: Path = NODES_DIR / node_name
        if not node_dir.exists():
            pytest.skip(f"Directory not yet created: {node_dir}")
        init_path: Path = node_dir / "__init__.py"
        assert init_path.exists(), f"Missing __init__.py: {init_path}"

    # NOTE: test_no_node_py_exists moved to test_node_enforcement.py
    # to keep all declarative pattern enforcement tests in one place.

    @pytest.mark.parametrize("node_name", CORE_8_NODES)
    def test_no_local_handlers_dir(self, node_name: str) -> None:
        """Verify EFFECT/ORCHESTRATOR nodes do NOT have local handlers directories.

        Infrastructure handlers (DB, Qdrant, etc.) are reused from omnibase_infra,
        not duplicated locally. Contracts reference handlers by import path:
        - EFFECT nodes: omnibase_infra.handlers.handler_db, handler_qdrant, etc.
        - ORCHESTRATOR nodes: omnibase_infra.nodes...handlers.*

        Excluded from this check:
        - COMPUTE nodes: Have node-specific pure computation handlers (no I/O)
        - Nodes with mock handlers: Temporary development implementations
        """
        # Skip nodes that legitimately have local handlers
        if node_name in self.NODES_WITH_ALLOWED_HANDLERS:
            pytest.skip(
                f"{node_name} has allowed local handlers "
                "(see NODES_WITH_ALLOWED_HANDLERS for reason)"
            )

        node_dir: Path = NODES_DIR / node_name
        if not node_dir.exists():
            pytest.skip(f"Directory not yet created: {node_dir}")
        handlers_dir: Path = node_dir / "handlers"
        assert not handlers_dir.exists(), (
            f"handlers/ should not exist in {node_name} - "
            "reuse handlers from omnibase_infra instead"
        )
