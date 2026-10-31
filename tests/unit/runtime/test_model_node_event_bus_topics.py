# SPDX-FileCopyrightText: 2026 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A node contract's event_bus section is the one place its topics are written."""

from __future__ import annotations

import pytest

from omnimemory.models.config.model_node_event_bus_topics import (
    ModelNodeEventBusTopics,
)
from omnimemory.runtime.contract_topics import collect_subscribe_topics_from_contracts

LIFECYCLE = "omnimemory.nodes.node_memory_lifecycle_orchestrator"


@pytest.mark.unit
class TestModelNodeEventBusTopics:
    def test_resolves_a_subscribe_topic_by_event_name(self) -> None:
        topics = ModelNodeEventBusTopics.from_node_contract(LIFECYCLE)

        assert (
            topics.subscribed("runtime-tick") == "onex.cmd.omnimemory.runtime-tick.v1"
        )

    def test_resolves_a_publish_topic_by_event_name(self) -> None:
        topics = ModelNodeEventBusTopics.from_node_contract(
            "omnimemory.nodes.node_kreuzberg_parse_effect"
        )

        assert (
            topics.published("document-indexed")
            == "onex.evt.omnimemory.document-indexed.v1"
        )

    @pytest.mark.parametrize(
        ("package", "event_name", "expected"),
        [
            (
                "omnimemory.nodes.node_navigation_history_reducer",
                "navigation-history-session",
                "onex.cmd.omnimemory.navigation-history-session.v1",
            ),
            (
                "omnimemory.nodes.node_semantic_analyzer_compute",
                "semantic-analysis",
                "onex.cmd.omnimemory.semantic-analysis.v1",
            ),
        ],
    )
    def test_dispatch_only_nodes_declare_their_command_topic(
        self, package: str, event_name: str, expected: str
    ) -> None:
        topics = ModelNodeEventBusTopics.from_node_contract(package)

        assert topics.dispatched(event_name) == expected

    def test_dispatch_only_nodes_are_not_in_the_plugin_subscription_set(self) -> None:
        subscribed = collect_subscribe_topics_from_contracts()

        assert "onex.cmd.omnimemory.navigation-history-session.v1" not in subscribed
        assert "onex.cmd.omnimemory.semantic-analysis.v1" not in subscribed

    def test_undeclared_event_name_is_refused(self) -> None:
        topics = ModelNodeEventBusTopics.from_node_contract(LIFECYCLE)

        with pytest.raises(ValueError, match="expected exactly 1"):
            topics.subscribed("no-such-event")

    def test_a_publish_topic_is_not_found_among_subscribe_topics(self) -> None:
        topics = ModelNodeEventBusTopics.from_node_contract(LIFECYCLE)

        with pytest.raises(ValueError, match="subscribe_topics"):
            topics.subscribed("memory-expired")
