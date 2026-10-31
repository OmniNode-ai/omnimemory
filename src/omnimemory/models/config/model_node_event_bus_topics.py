# SPDX-FileCopyrightText: 2026 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Topics a node contract declares for the bus and for the dispatch engine."""

from __future__ import annotations

import importlib.resources

import yaml
from pydantic import BaseModel, ConfigDict, Field


class ModelNodeEventBusTopics(BaseModel):
    """The subscribe, publish and dispatch topics one node contract declares.

    The contract is the only place a topic string is written; code asks for a
    topic by its event name, the segment before the version
    (``runtime-tick`` for ``onex.cmd.omnimemory.runtime-tick.v1``).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    package: str = Field(..., description="Node package that holds the contract.yaml")
    subscribe_topics: tuple[str, ...] = Field(
        ..., description="Topics the contract declares under event_bus.subscribe_topics"
    )
    publish_topics: tuple[str, ...] = Field(
        ..., description="Topics the contract declares under event_bus.publish_topics"
    )
    dispatch_topics: tuple[str, ...] = Field(
        ...,
        description=(
            "Topics the contract declares under descriptor.dispatch_topics: "
            "routed to the node's handler by the dispatch engine, never subscribed"
        ),
    )

    @classmethod
    def from_node_contract(cls, package: str) -> ModelNodeEventBusTopics:
        """Read the node package's contract.yaml; a missing section declares no topics."""
        contract = yaml.safe_load(
            importlib.resources.files(package)
            .joinpath("contract.yaml")
            .read_text(encoding="utf-8")
        )
        if not isinstance(contract, dict):
            raise ValueError(f"{package} contract is not a mapping")
        event_bus = contract.get("event_bus") or {}
        descriptor = contract.get("descriptor") or {}
        return cls(
            package=package,
            subscribe_topics=tuple(event_bus.get("subscribe_topics") or ()),
            publish_topics=tuple(event_bus.get("publish_topics") or ()),
            dispatch_topics=tuple(descriptor.get("dispatch_topics") or ()),
        )

    def subscribed(self, event_name: str) -> str:
        """The one subscribe topic declared for the event name."""
        return self._only(self.subscribe_topics, "subscribe_topics", event_name)

    def published(self, event_name: str) -> str:
        """The one publish topic declared for the event name."""
        return self._only(self.publish_topics, "publish_topics", event_name)

    def dispatched(self, event_name: str) -> str:
        """The one dispatch topic declared for the event name."""
        return self._only(self.dispatch_topics, "dispatch_topics", event_name)

    def _only(self, topics: tuple[str, ...], field: str, event_name: str) -> str:
        matches = [t for t in topics if t.rsplit(".", 2)[-2] == event_name]
        if len(matches) != 1:
            raise ValueError(
                f"{self.package} contract declares {len(matches)} {field} for "
                f"event '{event_name}', expected exactly 1"
            )
        return matches[0]


__all__ = ["ModelNodeEventBusTopics"]
