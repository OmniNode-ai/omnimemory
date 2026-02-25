# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Intent Event Consumer Effect Node.

Consumes intent-classified.v1 events from omniintelligence
and persists them to Memgraph using HandlerIntentStorageAdapter.
"""

from omnimemory.nodes.intent_event_consumer_effect.handler_intent_event_consumer import (
    HandlerIntentEventConsumer,
)
from omnimemory.nodes.intent_event_consumer_effect.models import (
    ModelIntentEventConsumerConfig,
    ModelIntentEventConsumerHealth,
)

__all__ = [
    "HandlerIntentEventConsumer",
    "ModelIntentEventConsumerConfig",
    "ModelIntentEventConsumerHealth",
]
