# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Resolution-equivalence tests for the retrieval-node Qdrant endpoint descriptor.

OMN-13562 / OMN-13556 Wave-1 endpoint→overlay migration. Proves the
overlay-resolved ``descriptor.qdrant_host`` / ``descriptor.qdrant_port`` return
exactly the value the old direct ``os.environ["QDRANT_HOST"]`` /
``int(os.environ["QDRANT_PORT"])`` reads returned for the same env, across
dev / stability / prod lane values, and that the host resolution fails closed
when the var is unset (no silent ``localhost`` fallback).
"""

from __future__ import annotations

import os

import pytest

from omnimemory.nodes.node_memory_retrieval_effect.contract_descriptor import (
    contract_qdrant_host,
    contract_qdrant_port,
)

pytestmark = pytest.mark.unit


# Representative per-lane QDRANT_HOST values (the same shape an operator overlay /
# the per-lane service env supplies). Dev, stability-test, and prod each point at
# a distinct host; the overlay must resolve each identically to a raw env read.
_LANE_HOSTS = [
    "localhost",  # dev
    "qdrant.stability-test.svc",  # stability-test
    "qdrant.prod.svc",  # prod
]


@pytest.mark.parametrize("host", _LANE_HOSTS)
def test_host_overlay_resolution_equals_direct_env_read(
    host: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Overlay descriptor resolves the same host the old env read produced."""
    monkeypatch.setenv("QDRANT_HOST", host)

    # The value the pre-migration code read directly.
    direct = os.environ["QDRANT_HOST"]
    # The value the migrated overlay seam resolves.
    resolved = contract_qdrant_host()

    assert resolved == direct == host


@pytest.mark.parametrize("port", ["6333", "16333", "26333"])
def test_port_overlay_resolution_equals_direct_env_read(
    port: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Overlay descriptor resolves the same port the old env read produced."""
    monkeypatch.setenv("QDRANT_PORT", port)

    direct = int(os.environ["QDRANT_PORT"])
    resolved = contract_qdrant_port()

    assert resolved == direct == int(port)


def test_port_defaults_to_6333_when_env_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset QDRANT_PORT resolves via the inline ``${env.QDRANT_PORT:6333}`` default."""
    monkeypatch.delenv("QDRANT_PORT", raising=False)

    assert contract_qdrant_port() == 6333


def test_host_fails_closed_when_env_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset QDRANT_HOST raises rather than defaulting to localhost."""
    monkeypatch.delenv("QDRANT_HOST", raising=False)

    with pytest.raises(ValueError, match=r"descriptor\.qdrant_host resolved empty"):
        contract_qdrant_host()


def test_host_fails_closed_when_env_blank(monkeypatch: pytest.MonkeyPatch) -> None:
    """Whitespace-only QDRANT_HOST is treated as unset and fails closed."""
    monkeypatch.setenv("QDRANT_HOST", "   ")

    with pytest.raises(ValueError, match=r"descriptor\.qdrant_host resolved empty"):
        contract_qdrant_host()
