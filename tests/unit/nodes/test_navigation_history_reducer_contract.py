# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Test that node_navigation_history_reducer has a valid contract.yaml.

Related:
    - OMN-7150: Add contract.yaml for node_navigation_history_reducer
"""

from __future__ import annotations

import importlib.resources

import pytest
import yaml


@pytest.mark.unit
class TestNavigationHistoryReducerContract:
    def test_contract_yaml_exists(self) -> None:
        """contract.yaml must exist in the node package."""
        package = "omnimemory.nodes.node_navigation_history_reducer"
        pkg_files = importlib.resources.files(package)
        contract_file = pkg_files.joinpath("contract.yaml")
        content = contract_file.read_text(encoding="utf-8")
        contract = yaml.safe_load(content)
        assert isinstance(contract, dict)
        assert contract["name"] == "node_navigation_history_reducer"

    def test_contract_has_handler_config(self) -> None:
        """contract.yaml must declare handler_config with class and module."""
        package = "omnimemory.nodes.node_navigation_history_reducer"
        pkg_files = importlib.resources.files(package)
        contract_file = pkg_files.joinpath("contract.yaml")
        content = contract_file.read_text(encoding="utf-8")
        contract = yaml.safe_load(content)
        handler_config = contract.get("handler_config")
        assert handler_config is not None
        assert handler_config["handler_class"] == "HandlerNavigationHistoryReducer"
        assert (
            handler_config["handler_module"]
            == "omnimemory.nodes.node_navigation_history_reducer.handlers.handler_navigation_history_reducer"
        )
