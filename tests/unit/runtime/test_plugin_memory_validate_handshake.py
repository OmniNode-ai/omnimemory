# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for PluginMemory.validate_handshake() Memgraph TCP probe.

Validates the B1 check: that validate_handshake() correctly probes Memgraph
TCP reachability before allowing consumers to subscribe to Kafka topics.
"""

import pytest

from omnimemory.runtime.plugin import PluginMemory


@pytest.mark.unit
@pytest.mark.asyncio
async def test_validate_handshake_fails_when_memgraph_unreachable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """validate_handshake must return failed result when Memgraph TCP probe fails."""
    monkeypatch.setenv("OMNIMEMORY_ENABLED", "true")
    monkeypatch.setenv("OMNIMEMORY_MEMGRAPH_HOST", "127.0.0.1")
    monkeypatch.setenv("OMNIMEMORY_MEMGRAPH_PORT", "19999")  # nothing listens here

    plugin = PluginMemory()
    result = await plugin.validate_handshake(config=None)

    assert result.passed is False
    failure_text = result.error_message or ""
    assert "Memgraph" in failure_text
    assert "19999" in failure_text


@pytest.mark.unit
@pytest.mark.asyncio
async def test_validate_handshake_skips_probe_when_plugin_inactive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If OMNIMEMORY_ENABLED is not set, validate_handshake default-passes without probing."""
    monkeypatch.delenv("OMNIMEMORY_ENABLED", raising=False)

    plugin = PluginMemory()
    result = await plugin.validate_handshake(config=None)

    assert result.passed is True
