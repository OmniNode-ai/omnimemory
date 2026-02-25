# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Integration tests for entry point discovery.

Validates that PluginMemory is discoverable via ``importlib.metadata.entry_points()``
using the ``onex.domain_plugins`` group.  This requires the package to be installed
(``poetry install``) so that the entry point metadata is written.

Related:
    - OMN-2217: Phase 6 -- Wire model registration & entry point declaration
"""

from __future__ import annotations

from importlib.metadata import entry_points

import pytest
from omnibase_infra.runtime.protocol_domain_plugin import ProtocolDomainPlugin

from omnimemory.runtime.plugin import PluginMemory


@pytest.mark.integration
class TestEntryPointDiscovery:
    """Validate that PluginMemory is discoverable via entry_points."""

    def test_entry_point_discoverable(self) -> None:
        """Entry point 'memory' must exist in onex.domain_plugins group."""
        eps = entry_points(group="onex.domain_plugins")
        names = [ep.name for ep in eps]
        assert "memory" in names, (
            f"'memory' not found in onex.domain_plugins entry points. Found: {names}"
        )

    def test_entry_point_loads_plugin_class(self) -> None:
        """Loading the entry point must return the PluginMemory class."""
        eps = entry_points(group="onex.domain_plugins")
        matches = [ep for ep in eps if ep.name == "memory"]
        assert matches, "No 'memory' entry point found"
        loaded = matches[0].load()
        assert loaded is PluginMemory, f"Expected PluginMemory class, got {loaded!r}"

    def test_entry_point_plugin_satisfies_protocol(self) -> None:
        """Instantiating the loaded class must satisfy ProtocolDomainPlugin."""
        eps = entry_points(group="onex.domain_plugins")
        matches = [ep for ep in eps if ep.name == "memory"]
        assert matches, "No 'memory' entry point found"
        cls = matches[0].load()
        plugin = cls()
        assert isinstance(plugin, ProtocolDomainPlugin), (
            f"Instance of {cls.__name__} does not satisfy ProtocolDomainPlugin"
        )
        assert plugin.plugin_id == "memory"
