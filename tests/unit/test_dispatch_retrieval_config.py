# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Tests for env-var-driven retrieval config in dispatch_handlers.

These tests exercise ``build_retrieval_config_from_env`` — the same code path
used by the runtime dispatch handler — to verify that
``OMNIMEMORY_USE_STUB_HANDLERS``, ``QDRANT_HOST``, ``QDRANT_PORT``, and
``LLM_EMBEDDING_URL`` are resolved correctly.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from omnimemory.runtime.dispatch_handlers import build_retrieval_config_from_env


@pytest.mark.unit
class TestRetrievalConfigFromEnv:
    """Verify that OMNIMEMORY_USE_STUB_HANDLERS env var controls config."""

    def test_default_is_stub(self) -> None:
        """Without env var, stubs are used."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove any pre-existing value so the default ("true") applies
            os.environ.pop("OMNIMEMORY_USE_STUB_HANDLERS", None)
            config = build_retrieval_config_from_env()
        assert config.use_stub_handlers is True
        assert config.qdrant_config is None

    def test_env_var_false_disables_stubs(self) -> None:
        """OMNIMEMORY_USE_STUB_HANDLERS=false wires real Qdrant config."""
        env = {
            "OMNIMEMORY_USE_STUB_HANDLERS": "false",
            "QDRANT_HOST": "qdrant.example.com",
            "QDRANT_PORT": "6334",
            "LLM_EMBEDDING_URL": "http://embed:8100",
        }
        with patch.dict(os.environ, env, clear=True):
            config = build_retrieval_config_from_env()
        assert config.use_stub_handlers is False
        assert config.qdrant_config is not None
        assert config.qdrant_config.qdrant_host == "qdrant.example.com"
        assert config.qdrant_config.qdrant_port == 6334
        assert config.qdrant_config.embedding_server_url == "http://embed:8100"

    def test_env_var_true_keeps_stubs(self) -> None:
        """OMNIMEMORY_USE_STUB_HANDLERS=true explicitly keeps stubs."""
        with patch.dict(
            os.environ, {"OMNIMEMORY_USE_STUB_HANDLERS": "true"}, clear=True
        ):
            config = build_retrieval_config_from_env()
        assert config.use_stub_handlers is True
        assert config.qdrant_config is None

    def test_whitespace_padded_false_disables_stubs(self) -> None:
        """Whitespace around 'false' is stripped before comparison."""
        env = {
            "OMNIMEMORY_USE_STUB_HANDLERS": "  false  ",
            "QDRANT_HOST": "localhost",
            "QDRANT_PORT": "6333",
            "LLM_EMBEDDING_URL": "http://localhost:8100",
        }
        with patch.dict(os.environ, env, clear=True):
            config = build_retrieval_config_from_env()
        assert config.use_stub_handlers is False
        assert config.qdrant_config is not None

    def test_false_without_qdrant_env_raises_keyerror(self) -> None:
        """When stubs disabled but QDRANT_* unset, KeyError is raised (no fallback defaults)."""
        with patch.dict(
            os.environ, {"OMNIMEMORY_USE_STUB_HANDLERS": "false"}, clear=True
        ):
            with pytest.raises(KeyError):
                build_retrieval_config_from_env()
