"""Event models for Kafka message processing.

This module contains Pydantic models for:
- Incoming events from omniintelligence (intent classification)
- Document ingestion pipeline events (OMN-2426)

Note: Outgoing events (ModelIntentStoredEvent) are imported from omnibase_core
to avoid contract duplication. See omnibase_core.models.events.
"""

from .model_crawl_tick_command import ModelCrawlTickCommand
from .model_document_changed_event import ModelDocumentChangedEvent
from .model_document_discovered_event import ModelDocumentDiscoveredEvent
from .model_document_removed_event import ModelDocumentRemovedEvent
from .model_intent_classified_event import ModelIntentClassifiedEvent

__all__ = [
    "ModelCrawlTickCommand",
    "ModelDocumentChangedEvent",
    "ModelDocumentDiscoveredEvent",
    "ModelDocumentRemovedEvent",
    "ModelIntentClassifiedEvent",
]
