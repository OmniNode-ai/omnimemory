# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Regression test: OMN-13701.

Ensures omnimemory does not locally shadow node_intent_event_consumer_effect,
which is canonically owned by omnimarket.

The trigger: both packages carried UUID 6fc349b1-34f0-4760-b547-6e8bebf0c9c0,
causing a dual-consumer race on onex.evt.omniintelligence.intent-classified.v1.

Uses filesystem-level scanning to avoid triggering the heavy omnimemory
model-init chain during collection.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_OMNIMEMORY_NODES_SRC = (
    Path(__file__).parent.parent.parent  # repo root of the worktree
    / "src"
    / "omnimemory"
    / "nodes"
)


@pytest.mark.unit
class TestNoLocalNodeDuplicate:
    """omnimemory must not define node_intent_event_consumer_effect locally.

    Per node-consolidation doctrine the canonical owner is omnimarket.
    A local copy creates a dual Kafka consumer on the same consumer group,
    which is a non-deterministic race (OMN-13701).
    """

    def test_node_intent_event_consumer_effect_not_in_omnimemory_nodes(
        self,
    ) -> None:
        """src/omnimemory/nodes/ must not contain a local intent_event_consumer_effect.

        This node is the canonical owner of omnimarket. Its presence in
        omnimemory creates two consumers on the same Kafka consumer group for
        the intent-classified topic, causing a non-deterministic race where
        events are split between two competing handlers with no ordering
        guarantee (OMN-13701).
        """
        local_node_dir = _OMNIMEMORY_NODES_SRC / "node_intent_event_consumer_effect"
        assert not local_node_dir.exists(), (
            f"node_intent_event_consumer_effect must not exist in omnimemory "
            f"(found: {local_node_dir}). "
            "Canonical owner is omnimarket. "
            "Remove the local directory and redirect runtime wiring to "
            "omnimarket.nodes.node_intent_event_consumer_effect (OMN-13701)."
        )
