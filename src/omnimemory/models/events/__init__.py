"""Event models for Kafka message processing.

This module contains Pydantic models for:
- Incoming events from omniintelligence (intent classification)
- Outgoing events for storage confirmations and failures
"""

from .model_intent_classified_event import ModelIntentClassifiedEvent
from .model_intent_store_failed_event import ModelIntentStoreFailedEvent
from .model_intent_stored_event import ModelIntentStoredEvent

__all__ = [
    "ModelIntentClassifiedEvent",
    "ModelIntentStoredEvent",
    "ModelIntentStoreFailedEvent",
]
