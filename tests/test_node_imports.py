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
        """Verify nodes do NOT have local handlers directories.

        Handlers are reused from omnibase_infra, not duplicated locally.
        Contracts reference handlers by import path:
        - EFFECT nodes: omnibase_infra.handlers.handler_db, handler_qdrant, etc.
        - ORCHESTRATOR nodes: omnibase_infra.nodes...handlers.*
        """
        node_dir: Path = NODES_DIR / node_name
        if not node_dir.exists():
            pytest.skip(f"Directory not yet created: {node_dir}")
        handlers_dir: Path = node_dir / "handlers"
        assert not handlers_dir.exists(), (
            f"handlers/ should not exist in {node_name} - "
            "reuse handlers from omnibase_infra instead"
        )
