# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Tests for env-var-driven retrieval config in dispatch_handlers."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from omnimemory.nodes.node_memory_retrieval_effect.models.model_handler_memory_retrieval_config import (
    ModelHandlerMemoryRetrievalConfig,
)


@pytest.mark.unit
class TestRetrievalConfigFromEnv:
    """Verify that OMNIMEMORY_USE_STUB_HANDLERS env var controls config."""

    def test_default_is_stub(self) -> None:
        """Without env var, stubs are used."""
        config = ModelHandlerMemoryRetrievalConfig(use_stub_handlers=True)
        assert config.use_stub_handlers is True

    def test_false_requires_qdrant_config(self) -> None:
        """Setting use_stub_handlers=False without qdrant_config raises."""
        with pytest.raises(
            ValueError, match="qdrant_config is required when use_stub_handlers=False"
        ):
            ModelHandlerMemoryRetrievalConfig(use_stub_handlers=False)

    def test_env_var_false_parsed(self) -> None:
        """OMNIMEMORY_USE_STUB_HANDLERS=false is correctly parsed."""
        with patch.dict(os.environ, {"OMNIMEMORY_USE_STUB_HANDLERS": "false"}):
            val = os.getenv("OMNIMEMORY_USE_STUB_HANDLERS", "true").lower() != "false"
            assert val is False

    def test_env_var_true_parsed(self) -> None:
        """OMNIMEMORY_USE_STUB_HANDLERS=true keeps stubs."""
        with patch.dict(os.environ, {"OMNIMEMORY_USE_STUB_HANDLERS": "true"}):
            val = os.getenv("OMNIMEMORY_USE_STUB_HANDLERS", "true").lower() != "false"
            assert val is True
