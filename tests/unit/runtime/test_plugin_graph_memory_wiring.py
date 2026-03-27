# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Unit tests for graph memory adapter wiring in PluginMemory (OMN-6578).

Validates:
    - AdapterGraphMemory is constructed with correct config when env vars are set
    - initialize() is called with the correct bolt:// URI
    - Adapter is stored on the plugin instance as _graph_memory_adapter
    - Dispatch engine registers the graph memory handler when adapter is provided
    - Graph memory handler is NOT registered when env vars are absent
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from omnimemory.runtime.plugin import PluginMemory

from .conftest import StubConfig, StubEventBus


def _make_config() -> StubConfig:
    """Create a minimal config for wire_dispatchers."""
    return StubConfig(event_bus=StubEventBus(), correlation_id=uuid4())


def _patch_introspection() -> patch:
    """Patch publish_memory_introspection to avoid real Kafka calls."""
    return patch(
        "omnimemory.runtime.introspection.publish_memory_introspection",
        new_callable=AsyncMock,
        return_value=MagicMock(registered_nodes=[], proxies=[]),
    )


def _patch_dispatch_factory() -> patch:
    """Patch the dispatch engine factory."""
    return patch(
        "omnimemory.runtime.dispatch_handlers.create_memory_dispatch_engine",
    )


def _patch_contract_topics() -> patch:
    """Patch contract topic collection."""
    return patch(
        "omnimemory.runtime.contract_topics.collect_publish_topics_for_dispatch",
        return_value={},
    )


@pytest.mark.unit
class TestGraphMemoryWiring:
    """Tests for AdapterGraphMemory initialization in wire_dispatchers."""

    @pytest.mark.asyncio
    async def test_graph_memory_adapter_initialized_when_env_set(self) -> None:
        """AdapterGraphMemory is constructed and initialized when OMNIMEMORY_MEMGRAPH_HOST is set."""
        plugin = PluginMemory()
        config = _make_config()

        mock_adapter_instance = MagicMock()
        mock_adapter_instance.initialize = AsyncMock()

        mock_intent_adapter = MagicMock()
        mock_intent_adapter.initialize = AsyncMock()

        env_vars = {
            "OMNIMEMORY_MEMGRAPH_HOST": "test-memgraph",
            "OMNIMEMORY_MEMGRAPH_PORT": "7687",
        }

        with (
            patch.dict("os.environ", env_vars, clear=False),
            patch(
                "omnimemory.runtime.plugin._probe_tcp_reachable",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch(
                "omnimemory.handlers.adapters.adapter_graph_memory.AdapterGraphMemory",
                return_value=mock_adapter_instance,
            ) as mock_adapter_cls,
            patch(
                "omnimemory.models.adapters.model_graph_memory_config.ModelGraphMemoryConfig",
            ) as mock_config_cls,
            patch(
                "omnimemory.handlers.adapters.adapter_intent_graph.AdapterIntentGraph",
                return_value=mock_intent_adapter,
            ),
            patch(
                "omnimemory.handlers.adapters.models.model_adapter_intent_graph_config.ModelAdapterIntentGraphConfig",
            ),
            _patch_dispatch_factory() as mock_factory,
            _patch_contract_topics(),
            _patch_introspection(),
        ):
            mock_factory.return_value = MagicMock(route_count=7, handler_count=5)

            result = await plugin.wire_dispatchers(config)

            # Adapter was constructed
            mock_config_cls.assert_called_once()
            mock_adapter_cls.assert_called_once()

            # initialize() called with correct bolt URI
            mock_adapter_instance.initialize.assert_called_once_with(
                connection_uri="bolt://test-memgraph:7687", auth=None
            )

            # Adapter stored on plugin
            assert plugin._graph_memory_adapter is mock_adapter_instance

            # Factory called with graph_memory_adapter
            mock_factory.assert_called_once()
            call_kwargs = mock_factory.call_args[1]
            assert call_kwargs["graph_memory_adapter"] is mock_adapter_instance

            assert result.success

    @pytest.mark.asyncio
    async def test_graph_memory_adapter_not_initialized_when_env_absent(
        self,
    ) -> None:
        """No graph adapter when OMNIMEMORY_MEMGRAPH_HOST is not set."""
        plugin = PluginMemory()
        config = _make_config()

        # Ensure OMNIMEMORY_MEMGRAPH_HOST is empty/absent
        env_patch = {"OMNIMEMORY_MEMGRAPH_HOST": ""}

        with (
            patch.dict("os.environ", env_patch, clear=False),
            _patch_dispatch_factory() as mock_factory,
            _patch_contract_topics(),
            _patch_introspection(),
        ):
            mock_factory.return_value = MagicMock(route_count=6, handler_count=4)

            result = await plugin.wire_dispatchers(config)

            # Factory called WITHOUT graph_memory_adapter
            call_kwargs = mock_factory.call_args[1]
            assert call_kwargs["graph_memory_adapter"] is None

            # Plugin has no adapter stored
            assert plugin._graph_memory_adapter is None

            assert result.success

    @pytest.mark.asyncio
    async def test_graph_memory_adapter_not_registered_when_unreachable(
        self,
    ) -> None:
        """Adapter is not registered when Memgraph is unreachable."""
        plugin = PluginMemory()
        config = _make_config()

        mock_adapter_instance = MagicMock()
        mock_adapter_instance.initialize = AsyncMock()

        env_vars = {
            "OMNIMEMORY_MEMGRAPH_HOST": "test-memgraph",
            "OMNIMEMORY_MEMGRAPH_PORT": "7687",
        }

        with (
            patch.dict("os.environ", env_vars, clear=False),
            patch(
                "omnimemory.runtime.plugin._probe_tcp_reachable",
                new_callable=AsyncMock,
                return_value=False,
            ),
            patch(
                "omnimemory.handlers.adapters.adapter_graph_memory.AdapterGraphMemory",
                return_value=mock_adapter_instance,
            ),
            patch(
                "omnimemory.models.adapters.model_graph_memory_config.ModelGraphMemoryConfig",
            ),
            _patch_dispatch_factory() as mock_factory,
            _patch_contract_topics(),
            _patch_introspection(),
        ):
            mock_factory.return_value = MagicMock(route_count=7, handler_count=5)

            result = await plugin.wire_dispatchers(config)

            # initialize() NOT called since unreachable
            mock_adapter_instance.initialize.assert_not_called()

            # Adapter NOT stored — uninitialized adapter must not be registered
            assert (
                not hasattr(plugin, "_graph_memory_adapter")
                or plugin._graph_memory_adapter is None
            )

            # Factory called with None for graph_memory_adapter
            mock_factory.assert_called_once()
            call_kwargs = mock_factory.call_args.kwargs
            assert call_kwargs.get("graph_memory_adapter") is None

            assert result.success

    @pytest.mark.asyncio
    async def test_custom_memgraph_port(self) -> None:
        """Custom OMNIMEMORY_MEMGRAPH_PORT is used in bolt URI."""
        plugin = PluginMemory()
        config = _make_config()

        mock_adapter_instance = MagicMock()
        mock_adapter_instance.initialize = AsyncMock()

        env_vars = {
            "OMNIMEMORY_MEMGRAPH_HOST": "custom-host",
            "OMNIMEMORY_MEMGRAPH_PORT": "9999",
        }

        with (
            patch.dict("os.environ", env_vars, clear=False),
            patch(
                "omnimemory.runtime.plugin._probe_tcp_reachable",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch(
                "omnimemory.handlers.adapters.adapter_graph_memory.AdapterGraphMemory",
                return_value=mock_adapter_instance,
            ),
            patch(
                "omnimemory.models.adapters.model_graph_memory_config.ModelGraphMemoryConfig",
            ),
            _patch_dispatch_factory() as mock_factory,
            _patch_contract_topics(),
            _patch_introspection(),
        ):
            mock_factory.return_value = MagicMock(route_count=7, handler_count=5)

            await plugin.wire_dispatchers(config)

            mock_adapter_instance.initialize.assert_called_once_with(
                connection_uri="bolt://custom-host:9999", auth=None
            )
