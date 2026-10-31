# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The navigation reducer's model name and client timeouts are injected (OMN-21623).

The embedding model and the three client timeouts of the navigation-history
handler are declared in the node contract's ``descriptor`` with the
``${env.VAR:default}`` overlay convention, expanded by the binding site
(``ModelNavigationClientSettings.from_node_contract``, called by ``runtime.plugin``) and injected into the handler as
``ModelNavigationClientSettings``; no literal stays in the handler.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from omnimemory.models.config.model_navigation_client_settings import (
    ModelNavigationClientSettings,
)
from omnimemory.nodes.node_navigation_history_reducer.handlers import (
    handler_navigation_history_reducer as module,
)

pytestmark = pytest.mark.unit

_ENV_NAMES = (
    "OMNIMEMORY_EMBEDDING_MODEL",
    "NAVIGATION_PG_COMMAND_TIMEOUT_SECONDS",
    "NAVIGATION_QDRANT_TIMEOUT_SECONDS",
    "NAVIGATION_EMBEDDING_HTTP_TIMEOUT_SECONDS",
)

_VALID = {
    "embedding_model": "m",
    "pg_command_timeout_seconds": 1.0,
    "qdrant_timeout_seconds": 1,
    "embedding_http_timeout_seconds": 1.0,
}


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _ENV_NAMES:
        monkeypatch.delenv(name, raising=False)


def test_contract_defaults_resolve() -> None:
    settings = ModelNavigationClientSettings.from_node_contract()
    assert settings.embedding_model == "default"
    assert settings.pg_command_timeout_seconds == 30.0
    assert settings.qdrant_timeout_seconds == 30
    assert settings.embedding_http_timeout_seconds == 30.0


def test_env_overlay_replaces_every_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OMNIMEMORY_EMBEDDING_MODEL", "overlay-embed-v2")
    monkeypatch.setenv("NAVIGATION_PG_COMMAND_TIMEOUT_SECONDS", "12")
    monkeypatch.setenv("NAVIGATION_QDRANT_TIMEOUT_SECONDS", "7")
    monkeypatch.setenv("NAVIGATION_EMBEDDING_HTTP_TIMEOUT_SECONDS", "45.5")
    settings = ModelNavigationClientSettings.from_node_contract()
    assert settings.embedding_model == "overlay-embed-v2"
    assert settings.pg_command_timeout_seconds == 12.0
    assert settings.qdrant_timeout_seconds == 7
    assert settings.embedding_http_timeout_seconds == 45.5


@pytest.mark.parametrize(
    "name",
    [
        "NAVIGATION_PG_COMMAND_TIMEOUT_SECONDS",
        "NAVIGATION_QDRANT_TIMEOUT_SECONDS",
        "NAVIGATION_EMBEDDING_HTTP_TIMEOUT_SECONDS",
    ],
)
@pytest.mark.parametrize("bad", ["abc", "0", "-3", "nan", "inf"])
def test_bad_timeout_overlay_fails_to_load(
    name: str, bad: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(name, bad)
    with pytest.raises(ValidationError):
        ModelNavigationClientSettings.from_node_contract()


def test_fractional_qdrant_timeout_fails_to_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NAVIGATION_QDRANT_TIMEOUT_SECONDS", "7.5")
    with pytest.raises(ValidationError):
        ModelNavigationClientSettings.from_node_contract()


@pytest.mark.parametrize("missing", sorted(_VALID))
def test_settings_model_has_no_defaults(missing: str) -> None:
    values = {key: value for key, value in _VALID.items() if key != missing}
    with pytest.raises(ValidationError):
        ModelNavigationClientSettings.model_validate(values)


def test_reducer_without_settings_resolves_the_node_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OMNIMEMORY_EMBEDDING_MODEL", "overlay-embed-v4")
    handler = module.HandlerNavigationHistoryReducer()
    writer = handler._writer
    assert isinstance(writer, module.HandlerNavigationHistoryWriter)
    assert writer._settings.embedding_model == "overlay-embed-v4"


@pytest.mark.asyncio
async def test_writer_clients_receive_injected_settings() -> None:
    settings = ModelNavigationClientSettings(
        embedding_model="injected-embed",
        pg_command_timeout_seconds=11.0,
        qdrant_timeout_seconds=13,
        embedding_http_timeout_seconds=17.0,
    )
    writer = module.HandlerNavigationHistoryWriter(
        qdrant_host="qdrant.test", qdrant_port=6333, settings=settings
    )

    with patch("asyncpg.create_pool", new=AsyncMock(return_value=MagicMock())) as pool:
        await writer._get_pg_pool()
    assert pool.call_args.kwargs["command_timeout"] == 11.0

    with patch.object(module, "AsyncQdrantClient") as qdrant:
        await writer._get_qdrant_client()
    assert qdrant.call_args.kwargs["timeout"] == 13

    response = MagicMock()
    response.json.return_value = {"data": [{"embedding": [0.5]}]}
    client = MagicMock()
    client.post = AsyncMock(return_value=response)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    with patch("httpx.AsyncClient", return_value=client) as http:
        await writer._embed_text("hello")
    assert http.call_args.kwargs["timeout"] == 17.0
    assert client.post.call_args.kwargs["json"]["model"] == "injected-embed"
