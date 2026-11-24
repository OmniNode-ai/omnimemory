# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Where the memory search stores are, loaded from the environment (OMN-26659)."""

from __future__ import annotations

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = ["ModelRetrievalStoreSettings"]


class ModelRetrievalStoreSettings(BaseSettings):
    """Where the memory search stores are, loaded from environment.

    The retrieval handler is built only from stores these settings describe.
    A field left unset means that store is not configured, and a search that
    needs it is refused by name; nothing is substituted for it.

    Environment variables (no prefix, the names the deployments already set):
        QDRANT_HOST: Qdrant host for the vector store
        QDRANT_PORT: Qdrant port (the Qdrant config model supplies its default)
        LLM_EMBEDDING_URL: Embedding server the vector store embeds queries with
    """

    model_config = SettingsConfigDict(
        env_prefix="",
        extra="ignore",
        populate_by_name=True,
    )

    qdrant_host: str | None = Field(
        default=None,
        validation_alias=AliasChoices("QDRANT_HOST", "qdrant_host"),
        description="Qdrant host for the vector store; unset means not configured",
    )
    qdrant_port: int | None = Field(
        default=None,
        ge=1,
        le=65535,
        validation_alias=AliasChoices("QDRANT_PORT", "qdrant_port"),
        description="Qdrant port; unset leaves the Qdrant config model's default",
    )
    embedding_server_url: str | None = Field(
        default=None,
        validation_alias=AliasChoices("LLM_EMBEDDING_URL", "embedding_server_url"),
        description="Embedding server URL for the vector store; unset means not configured",
    )
