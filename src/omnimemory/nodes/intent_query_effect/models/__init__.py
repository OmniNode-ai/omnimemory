# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Models for the intent_query_effect node.

This module exports configuration and event models for the intent query effect node.

Exports:
    ModelIntentRecordPayload: Payload model for intent records in events.
    ModelHandlerIntentQueryConfig: Handler configuration model.
    ModelIntentQueryRequestedEvent: Request event for intent queries.
    ModelIntentQueryResponseEvent: Response event with query results.

.. versionadded:: 0.1.0
    Initial implementation for OMN-1504.
"""

from omnimemory.nodes.intent_query_effect.models.model_handler_intent_query_config import (
    ModelHandlerIntentQueryConfig,
)
from omnimemory.nodes.intent_query_effect.models.model_intent_query_events import (
    ModelIntentQueryRequestedEvent,
    ModelIntentQueryResponseEvent,
    ModelIntentRecordPayload,
)

__all__ = [
    "ModelIntentRecordPayload",
    "ModelHandlerIntentQueryConfig",
    "ModelIntentQueryRequestedEvent",
    "ModelIntentQueryResponseEvent",
]
