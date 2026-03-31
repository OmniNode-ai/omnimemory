# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Test that standalone handlers have valid contract YAML files.

Standalone handlers are those not under a ``nodes/`` directory —
they're utility handlers, adapters, or runtime components that still
need contract declarations for contract-driven discovery.

Related:
    - OMN-7151: Add contract YAML for standalone omnimemory handlers
"""

from __future__ import annotations

import importlib.resources

import pytest
import yaml

# (package, contract_filename, expected_class, expected_module)
_STANDALONE_HANDLER_CONTRACTS: list[tuple[str, str, str, str]] = [
    (
        "omnimemory.handlers",
        "handler_subscription_contract.yaml",
        "HandlerSubscription",
        "omnimemory.handlers.handler_subscription",
    ),
    (
        "omnimemory.handlers.adapters",
        "adapter_intent_graph_contract.yaml",
        "AdapterIntentGraph",
        "omnimemory.handlers.adapters.adapter_intent_graph",
    ),
    (
        "omnimemory.runtime",
        "handler_lifecycle_contract.yaml",
        "HandlerMemoryLifecycle",
        "omnimemory.runtime.handler_lifecycle",
    ),
]


@pytest.mark.unit
class TestStandaloneHandlerContracts:
    @pytest.mark.parametrize(
        ("package", "contract_filename", "expected_class", "expected_module"),
        _STANDALONE_HANDLER_CONTRACTS,
        ids=["subscription", "adapter_intent_graph", "lifecycle"],
    )
    def test_contract_exists_and_valid(
        self,
        package: str,
        contract_filename: str,
        expected_class: str,
        expected_module: str,
    ) -> None:
        """Each standalone handler must have a contract YAML with handler_config."""
        pkg_files = importlib.resources.files(package)
        contract_file = pkg_files.joinpath(contract_filename)
        content = contract_file.read_text(encoding="utf-8")
        contract = yaml.safe_load(content)
        assert isinstance(contract, dict)

        handler_config = contract.get("handler_config")
        assert handler_config is not None, (
            f"Missing handler_config in {contract_filename}"
        )
        assert handler_config["handler_class"] == expected_class
        assert handler_config["handler_module"] == expected_module
