# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Unit tests for ModelHandlerQdrantConfig and updated ModelHandlerMemoryRetrievalConfig.

Tests for OMN-4474.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from omnimemory.nodes.node_memory_retrieval_effect.models import (
    ModelHandlerMemoryRetrievalConfig,
    ModelHandlerQdrantConfig,
)


@pytest.mark.unit
def test_defaults_with_required_url() -> None:
    """ModelHandlerQdrantConfig has correct defaults when embedding_server_url provided."""
    cfg = ModelHandlerQdrantConfig(embedding_server_url="http://localhost:8100")
    assert cfg.qdrant_host == "localhost"
    assert cfg.qdrant_port == 6333
    assert cfg.collection_name == "omnimemory_documents"
    assert cfg.vector_size == 4096
    assert cfg.max_chunk_chars == 2000


@pytest.mark.unit
def test_requires_embedding_server_url() -> None:
    """ModelHandlerQdrantConfig raises ValidationError when embedding_server_url missing."""
    with pytest.raises(ValidationError):
        ModelHandlerQdrantConfig()  # type: ignore[call-arg]


@pytest.mark.unit
def test_rejects_invalid_url() -> None:
    """ModelHandlerQdrantConfig raises ValidationError for non-HTTP(S) URL."""
    with pytest.raises(ValidationError):
        ModelHandlerQdrantConfig(embedding_server_url="not-a-url")


@pytest.mark.unit
def test_retrieval_config_requires_qdrant_config_when_not_stub() -> None:
    """ModelHandlerMemoryRetrievalConfig raises ValidationError when use_stub_handlers=False without qdrant_config."""
    with pytest.raises(ValidationError):
        ModelHandlerMemoryRetrievalConfig(
            use_stub_handlers=False
        )  # missing qdrant_config


@pytest.mark.unit
def test_retrieval_config_stub_mode_default() -> None:
    """ModelHandlerMemoryRetrievalConfig defaults to stub mode with no qdrant_config required."""
    cfg = ModelHandlerMemoryRetrievalConfig()
    assert cfg.use_stub_handlers is True
    assert cfg.qdrant_config is None


@pytest.mark.unit
def test_retrieval_config_production_mode_with_qdrant_config() -> None:
    """ModelHandlerMemoryRetrievalConfig accepts use_stub_handlers=False when qdrant_config provided."""
    qdrant_cfg = ModelHandlerQdrantConfig(embedding_server_url="http://localhost:8100")
    cfg = ModelHandlerMemoryRetrievalConfig(
        use_stub_handlers=False,
        qdrant_config=qdrant_cfg,
    )
    assert cfg.use_stub_handlers is False
    assert cfg.qdrant_config is not None
    assert cfg.qdrant_config.embedding_server_url == "http://localhost:8100"


@pytest.mark.unit
def test_qdrant_config_port_bounds() -> None:
    """ModelHandlerQdrantConfig rejects out-of-range ports."""
    with pytest.raises(ValidationError):
        ModelHandlerQdrantConfig(
            embedding_server_url="http://localhost:8100", qdrant_port=0
        )
    with pytest.raises(ValidationError):
        ModelHandlerQdrantConfig(
            embedding_server_url="http://localhost:8100", qdrant_port=65536
        )
