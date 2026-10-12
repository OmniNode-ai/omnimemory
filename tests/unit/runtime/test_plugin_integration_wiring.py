# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for integration chain wiring in PluginMemory.

Covers:
    - OMN-6579: AdapterIntentGraph initialization when Memgraph is available
    - OMN-6580: Graph adapters registered in dispatch engine
    - OMN-6583: HandlerNavigationHistoryReducer construction from env vars
    - OMN-6585: HandlerSemanticCompute construction with container
    - OMN-6588: HandlerMemoryLifecycle replaces no-op lifecycle handler
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from omnimemory.runtime.handler_lifecycle import HandlerMemoryLifecycle
from omnimemory.runtime.plugin import PluginMemory

from .conftest import StubConfig, StubContainer, StubEventBus


def _make_config() -> StubConfig:
    """Create a minimal config for wire_dispatchers."""
    return StubConfig(
        event_bus=StubEventBus(),
        correlation_id=uuid4(),
        container=StubContainer(),
    )


def _patch_introspection() -> patch:
    return patch(
        "omnimemory.runtime.introspection.publish_memory_introspection",
        new_callable=AsyncMock,
        return_value=MagicMock(registered_nodes=[], proxies=[]),
    )


def _patch_dispatch_factory() -> patch:
    return patch(
        "omnimemory.runtime.dispatch_handlers.create_memory_dispatch_engine",
    )


def _patch_contract_topics() -> patch:
    return patch(
        "omnimemory.runtime.contract_topics.collect_publish_topics_for_dispatch",
        return_value={},
    )


@pytest.mark.unit
class TestIntentGraphWiring:
    """Tests for AdapterIntentGraph initialization (OMN-6579)."""

    @pytest.mark.asyncio
    async def test_intent_graph_adapter_initialized_when_memgraph_available(
        self,
    ) -> None:
        """AdapterIntentGraph is initialized alongside AdapterGraphMemory."""
        plugin = PluginMemory()
        config = _make_config()

        mock_graph_adapter = MagicMock()
        mock_graph_adapter.initialize = AsyncMock()

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
                return_value=mock_graph_adapter,
            ),
            patch(
                "omnimemory.models.adapters.model_graph_memory_config.ModelGraphMemoryConfig",
            ),
            patch(
                "omnimemory.handlers.adapters.adapter_intent_graph.AdapterIntentGraph",
                return_value=mock_intent_adapter,
            ) as mock_intent_cls,
            patch(
                "omnimemory.handlers.adapters.models.model_adapter_intent_graph_config.ModelAdapterIntentGraphConfig",
            ),
            _patch_dispatch_factory() as mock_factory,
            _patch_contract_topics(),
            _patch_introspection(),
        ):
            mock_factory.return_value = MagicMock(route_count=10, handler_count=8)

            result = await plugin.wire_dispatchers(config)

            # Intent graph adapter was constructed and initialized
            mock_intent_cls.assert_called_once()
            mock_intent_adapter.initialize.assert_called_once_with(
                connection_uri="bolt://test-memgraph:7687", auth=None
            )

            # Stored on plugin
            assert plugin._intent_graph_adapter is mock_intent_adapter

            # Factory called with intent_graph_adapter
            call_kwargs = mock_factory.call_args[1]
            assert call_kwargs["intent_graph_adapter"] is mock_intent_adapter

            assert result.success

    @pytest.mark.asyncio
    async def test_intent_graph_not_initialized_when_graph_memory_unavailable(
        self,
    ) -> None:
        """Intent graph adapter is not created when Memgraph is unreachable."""
        plugin = PluginMemory()
        config = _make_config()

        mock_graph_adapter = MagicMock()
        mock_graph_adapter.initialize = AsyncMock()

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
                return_value=mock_graph_adapter,
            ),
            patch(
                "omnimemory.models.adapters.model_graph_memory_config.ModelGraphMemoryConfig",
            ),
            _patch_dispatch_factory() as mock_factory,
            _patch_contract_topics(),
            _patch_introspection(),
        ):
            mock_factory.return_value = MagicMock(route_count=6, handler_count=4)

            result = await plugin.wire_dispatchers(config)

            # Factory called with None for intent_graph_adapter
            call_kwargs = mock_factory.call_args[1]
            assert call_kwargs["intent_graph_adapter"] is None

            assert result.success


@pytest.mark.unit
class TestNavigationHistoryWiring:
    """Tests for HandlerNavigationHistoryReducer wiring (OMN-6583)."""

    @pytest.mark.asyncio
    async def test_navigation_handler_constructed_when_pg_dsn_set(self) -> None:
        """HandlerNavigationHistoryReducer is constructed when OMNIMEMORY_PG_DSN is set."""
        plugin = PluginMemory()
        config = _make_config()

        mock_nav_handler = MagicMock()

        env_vars = {
            "OMNIMEMORY_PG_DSN": "postgresql://test:5432/omnimemory",
            "QDRANT_HOST": "test-qdrant",
            "QDRANT_PORT": "6333",
            "LLM_EMBEDDING_URL": "http://test-embedding:8100",
            "OMNIMEMORY_EMBEDDING_MODEL": "test-model",
        }

        with (
            patch.dict("os.environ", env_vars, clear=False),
            patch(
                "omnimemory.nodes.node_navigation_history_reducer.handlers.handler_navigation_history_reducer.HandlerNavigationHistoryReducer",
                return_value=mock_nav_handler,
            ) as mock_nav_cls,
            _patch_dispatch_factory() as mock_factory,
            _patch_contract_topics(),
            _patch_introspection(),
        ):
            mock_factory.return_value = MagicMock(route_count=10, handler_count=8)

            result = await plugin.wire_dispatchers(config)

            # Handler was constructed with correct params
            mock_nav_cls.assert_called_once_with(
                writer=None,
                pg_dsn="postgresql://test:5432/omnimemory",
                qdrant_host="test-qdrant",
                qdrant_port=6333,
                embedding_url="http://test-embedding:8100",
                embedding_model="test-model",
            )

            # Factory called with navigation handler
            call_kwargs = mock_factory.call_args[1]
            assert call_kwargs["navigation_history_handler"] is mock_nav_handler

            assert result.success

    @pytest.mark.asyncio
    async def test_unset_embedding_url_ships_no_lab_default(
        self, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """OMN-20936: no LLM_EMBEDDING_URL means no embedding endpoint, said loudly."""
        plugin = PluginMemory()
        config = _make_config()

        monkeypatch.setenv("OMNIMEMORY_PG_DSN", "postgresql://test:5432/omnimemory")
        monkeypatch.setenv("QDRANT_HOST", "test-qdrant")
        monkeypatch.setenv("QDRANT_PORT", "6333")
        monkeypatch.delenv("LLM_EMBEDDING_URL", raising=False)
        monkeypatch.delenv("ONEX_SKILL_OVERLAY_ROOTS", raising=False)

        with (
            patch(
                "omnimemory.nodes.node_navigation_history_reducer.handlers.handler_navigation_history_reducer.HandlerNavigationHistoryReducer",
                return_value=MagicMock(),
            ) as mock_nav_cls,
            _patch_dispatch_factory() as mock_factory,
            _patch_contract_topics(),
            _patch_introspection(),
            caplog.at_level("WARNING"),
        ):
            mock_factory.return_value = MagicMock(route_count=10, handler_count=8)

            result = await plugin.wire_dispatchers(config)

            assert mock_nav_cls.call_args.kwargs["embedding_url"] == ""
            assert any(
                "LLM_EMBEDDING_URL" in record.getMessage() for record in caplog.records
            )
            assert result.success

    @pytest.mark.asyncio
    async def test_navigation_handler_not_constructed_when_pg_dsn_empty(
        self,
    ) -> None:
        """No navigation handler when OMNIMEMORY_PG_DSN is empty."""
        plugin = PluginMemory()
        config = _make_config()

        env_vars = {"OMNIMEMORY_PG_DSN": ""}

        with (
            patch.dict("os.environ", env_vars, clear=False),
            _patch_dispatch_factory() as mock_factory,
            _patch_contract_topics(),
            _patch_introspection(),
        ):
            mock_factory.return_value = MagicMock(route_count=6, handler_count=4)

            result = await plugin.wire_dispatchers(config)

            call_kwargs = mock_factory.call_args[1]
            assert call_kwargs["navigation_history_handler"] is None

            assert result.success


@pytest.mark.unit
class TestSemanticComputeWiring:
    """Tests for HandlerSemanticCompute wiring (OMN-6585)."""

    @pytest.mark.asyncio
    async def test_semantic_handler_constructed_when_container_available(
        self,
    ) -> None:
        """HandlerSemanticCompute is constructed when config.container is set."""
        plugin = PluginMemory()
        config = _make_config()

        mock_semantic = MagicMock()

        with (
            patch(
                "omnimemory.nodes.node_semantic_analyzer_compute.handlers.handler_semantic_compute.HandlerSemanticCompute",
                return_value=mock_semantic,
            ) as mock_cls,
            _patch_dispatch_factory() as mock_factory,
            _patch_contract_topics(),
            _patch_introspection(),
        ):
            mock_factory.return_value = MagicMock(route_count=10, handler_count=8)

            result = await plugin.wire_dispatchers(config)

            mock_cls.assert_called_once_with(container=config.container)

            call_kwargs = mock_factory.call_args[1]
            assert call_kwargs["semantic_compute_handler"] is mock_semantic

            assert result.success

    @pytest.mark.asyncio
    async def test_semantic_handler_not_constructed_when_no_container(
        self,
    ) -> None:
        """No semantic handler when config.container is None."""
        plugin = PluginMemory()
        config = StubConfig(
            event_bus=StubEventBus(),
            correlation_id=uuid4(),
            container=None,
        )

        with (
            _patch_dispatch_factory() as mock_factory,
            _patch_contract_topics(),
            _patch_introspection(),
        ):
            mock_factory.return_value = MagicMock(route_count=6, handler_count=4)

            result = await plugin.wire_dispatchers(config)

            call_kwargs = mock_factory.call_args[1]
            assert call_kwargs["semantic_compute_handler"] is None

            assert result.success


@pytest.mark.unit
class TestHandlerMemoryLifecycle:
    """Tests for HandlerMemoryLifecycle (OMN-6588)."""

    @pytest.mark.asyncio
    async def test_startup_reports_components(self) -> None:
        """handle_startup returns status dict for all provided components."""
        lifecycle = HandlerMemoryLifecycle(
            graph_memory_adapter=MagicMock(),
            intent_graph_adapter=MagicMock(),
            navigation_handler=MagicMock(),
            semantic_handler=MagicMock(),
        )

        statuses = await lifecycle.handle_startup()

        assert statuses == {
            "graph_memory": "initialized",
            "intent_graph": "initialized",
            "navigation_history": "initialized",
            "semantic_compute": "initialized",
        }
        assert lifecycle.is_started()

    @pytest.mark.asyncio
    async def test_startup_with_no_components(self) -> None:
        """handle_startup returns empty dict when no components provided."""
        lifecycle = HandlerMemoryLifecycle()

        statuses = await lifecycle.handle_startup()

        assert statuses == {}
        assert lifecycle.is_started()

    @pytest.mark.asyncio
    async def test_shutdown(self) -> None:
        """handle_shutdown sets started to False."""
        lifecycle = HandlerMemoryLifecycle(
            graph_memory_adapter=MagicMock(),
        )
        await lifecycle.handle_startup()
        assert lifecycle.is_started()

        await lifecycle.handle_shutdown()
        assert not lifecycle.is_started()

    def test_component_count(self) -> None:
        """component_count reflects number of non-None components."""
        lifecycle = HandlerMemoryLifecycle(
            graph_memory_adapter=MagicMock(),
            navigation_handler=MagicMock(),
        )
        assert lifecycle.component_count == 2

    def test_component_count_zero(self) -> None:
        """component_count is 0 when no components provided."""
        lifecycle = HandlerMemoryLifecycle()
        assert lifecycle.component_count == 0


@pytest.mark.unit
class TestExpireHandlerWiring:
    """PluginMemory builds the expire handler from OMNIMEMORY_DB_URL.

    Without a wired HandlerMemoryExpire an expire-memory command is answered
    with lifecycle-transition-failed; with one it expires the memory.
    """

    DSN = "postgresql://tester@db.invalid:5432/omnimemory"

    @pytest.mark.asyncio
    async def test_expire_handler_built_on_the_pool_when_db_url_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        plugin = PluginMemory()
        config = _make_config()
        pool = MagicMock()
        pool.close = AsyncMock()
        monkeypatch.setenv("OMNIMEMORY_DB_URL", self.DSN)

        with (
            patch(
                "asyncpg.create_pool", new_callable=AsyncMock, return_value=pool
            ) as create_pool,
            _patch_dispatch_factory() as mock_factory,
            _patch_contract_topics(),
            _patch_introspection(),
        ):
            mock_factory.return_value = MagicMock(route_count=10, handler_count=8)

            result = await plugin.wire_dispatchers(config)

            assert result.success
            assert create_pool.await_args is not None
            assert create_pool.await_args.kwargs["dsn"] == self.DSN
            expire_handler = mock_factory.call_args[1]["expire_handler"]
            assert expire_handler is not None
            assert expire_handler.initialized

    @pytest.mark.asyncio
    async def test_no_expire_handler_when_db_url_unset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        plugin = PluginMemory()
        config = _make_config()
        monkeypatch.delenv("OMNIMEMORY_DB_URL", raising=False)

        with (
            patch("asyncpg.create_pool", new_callable=AsyncMock) as create_pool,
            _patch_dispatch_factory() as mock_factory,
            _patch_contract_topics(),
            _patch_introspection(),
        ):
            mock_factory.return_value = MagicMock(route_count=6, handler_count=5)

            result = await plugin.wire_dispatchers(config)

            assert result.success
            create_pool.assert_not_awaited()
            assert mock_factory.call_args[1]["expire_handler"] is None

    @pytest.mark.asyncio
    async def test_unreachable_database_leaves_no_expire_handler_and_wires(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        plugin = PluginMemory()
        config = _make_config()
        monkeypatch.setenv("OMNIMEMORY_DB_URL", self.DSN)

        with (
            patch(
                "asyncpg.create_pool",
                new_callable=AsyncMock,
                side_effect=OSError("connection refused"),
            ),
            _patch_dispatch_factory() as mock_factory,
            _patch_contract_topics(),
            _patch_introspection(),
        ):
            mock_factory.return_value = MagicMock(route_count=6, handler_count=5)

            result = await plugin.wire_dispatchers(config)

            assert result.success
            assert mock_factory.call_args[1]["expire_handler"] is None
        assert "expire-memory commands will be answered" in caplog.text

    @pytest.mark.asyncio
    async def test_shutdown_closes_the_pool(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        plugin = PluginMemory()
        config = _make_config()
        pool = MagicMock()
        pool.close = AsyncMock()
        monkeypatch.setenv("OMNIMEMORY_DB_URL", self.DSN)

        with (
            patch("asyncpg.create_pool", new_callable=AsyncMock, return_value=pool),
            _patch_dispatch_factory() as mock_factory,
            _patch_contract_topics(),
            _patch_introspection(),
        ):
            mock_factory.return_value = MagicMock(route_count=10, handler_count=8)
            await plugin.wire_dispatchers(config)

            await plugin.shutdown(config)

        pool.close.assert_awaited_once()
