# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Canonical Kafka topic registry for omnimemory events and commands.

Holds only the topics that no omnimemory node contract declares yet: the two
graph command topics served by adapters with no node, and the intent-classified
event owned by omnimarket. Every other omnimemory topic is declared in its node
contract's ``event_bus`` section and read with
``omnimemory.runtime.contract_topics.contract_topic``.

All topic names follow ONEX canonical format:
``onex.{kind}.{producer}.{event-name}.v{n}``

ONEX Compliance:
    - Topic names are immutable StrEnum values.
    - All `emitted_at` fields in envelope models must be injected by callers.
    - No datetime.now() defaults permitted.

Reference: OMN-8605, OMN-8633
"""

from __future__ import annotations

from enum import Enum, unique

from omnibase_core.utils.util_str_enum_base import UtilStrValueHelper


@unique
class EnumMemoryCommandTopic(UtilStrValueHelper, str, Enum):
    """Canonical Kafka topic names for omnimemory commands with no node contract.

    All topics use ``onex.cmd.omnimemory.*`` as the consumer namespace
    (producer/consumer: ``omnimemory``, kind: ``cmd``).
    """

    GRAPH_MEMORY_QUERY = "onex.cmd.omnimemory.graph-memory-query.v1"
    """Graph memory query/mutation command (OMN-6578)."""

    INTENT_GRAPH_QUERY = "onex.cmd.omnimemory.intent-graph-query.v1"
    """Intent graph query/mutation command (OMN-6579)."""


@unique
class EnumMemoryEventTopic(UtilStrValueHelper, str, Enum):
    """Canonical Kafka topic names for events omnimemory consumes with no node contract."""

    INTENT_CLASSIFIED = "onex.evt.omniintelligence.intent-classified.v1"
    """Intent classified event produced by omniintelligence."""


__all__ = ["EnumMemoryCommandTopic", "EnumMemoryEventTopic"]
