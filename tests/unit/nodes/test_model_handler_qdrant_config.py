# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Unit tests for ModelHandlerQdrantConfig.

Tests for OMN-4474.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from omnimemory.nodes.node_memory_retrieval_effect.models import (
    ModelHandlerQdrantConfig,
)


@pytest.mark.unit
def test_defaults_with_required_fields() -> None:
    """ModelHandlerQdrantConfig has correct defaults when required fields provided."""
    cfg = ModelHandlerQdrantConfig(
        qdrant_host="qdrant.test",
        embedding_server_url="http://embed.test:8100",
    )
    assert cfg.qdrant_host == "qdrant.test"
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
def test_qdrant_config_port_bounds() -> None:
    """ModelHandlerQdrantConfig rejects out-of-range ports."""
    with pytest.raises(ValidationError):
        ModelHandlerQdrantConfig(
            embedding_server_url="http://svc.example.test:8100", qdrant_port=0
        )
    with pytest.raises(ValidationError):
        ModelHandlerQdrantConfig(
            embedding_server_url="http://svc.example.test:8100", qdrant_port=65536
        )
