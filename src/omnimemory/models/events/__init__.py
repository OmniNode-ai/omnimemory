"""Event models for Kafka message processing.

This module contains Pydantic models for:
- Incoming events from omniintelligence (intent classification)

Note: Outgoing events (ModelIntentStoredEvent) are imported from omnibase_core
to avoid contract duplication. See omnibase_core.models.events.
"""

from .model_intent_classified_event import ModelIntentClassifiedEvent

__all__ = [
    "ModelIntentClassifiedEvent",
]
