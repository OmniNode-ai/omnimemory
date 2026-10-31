# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Binding site for the navigation history handlers (OMN-21623).

The one place that builds the navigation history reducer and its writer. The
plugin, the node shell and the reducer take them from these binders instead of
constructing handler classes by hand.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omnimemory.models.config.model_navigation_client_settings import (
        ModelNavigationClientSettings,
    )
    from omnimemory.nodes.node_navigation_history_reducer.handlers import (
        HandlerNavigationHistoryReducer,
        HandlerNavigationHistoryWriter,
    )

__all__ = [
    "bind_configured_navigation_history_reducer",
    "bind_navigation_history_reducer",
    "bind_navigation_history_writer",
]


def bind_navigation_history_writer(
    *,
    pg_dsn: str,
    qdrant_host: str | None,
    qdrant_port: int | None,
    embedding_url: str,
    settings: ModelNavigationClientSettings,
) -> HandlerNavigationHistoryWriter:
    """Build the navigation history writer from explicit connection values."""
    from omnimemory.nodes.node_navigation_history_reducer.handlers.handler_navigation_history_reducer import (
        HandlerNavigationHistoryWriter,
    )

    return HandlerNavigationHistoryWriter(
        pg_dsn=pg_dsn,
        qdrant_host=qdrant_host,
        qdrant_port=qdrant_port,
        embedding_url=embedding_url,
        settings=settings,
    )


def bind_navigation_history_reducer() -> HandlerNavigationHistoryReducer:
    """Build the navigation history reducer with the handler's own defaults."""
    from omnimemory.nodes.node_navigation_history_reducer.handlers.handler_navigation_history_reducer import (
        HandlerNavigationHistoryReducer,
    )

    return HandlerNavigationHistoryReducer()


def bind_configured_navigation_history_reducer(
    *,
    pg_dsn: str,
    qdrant_host: str | None,
    qdrant_port: int | None,
    embedding_url: str,
) -> HandlerNavigationHistoryReducer:
    """Build the navigation history reducer from explicit connection values."""
    from omnimemory.nodes.node_navigation_history_reducer.handlers.handler_navigation_history_reducer import (
        HandlerNavigationHistoryReducer,
    )

    return HandlerNavigationHistoryReducer(
        writer=None,
        pg_dsn=pg_dsn,
        qdrant_host=qdrant_host,
        qdrant_port=qdrant_port,
        embedding_url=embedding_url,
    )
