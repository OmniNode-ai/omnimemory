# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Event models for Kafka message processing.

- Incoming events from omniintelligence (intent classification)
- Crawl tick commands for the document ingestion pipeline (OMN-2426)

Note: Document pipeline events (ModelDocumentChangedEvent,
ModelDocumentDiscoveredEvent, ModelDocumentRemovedEvent) are in
omnimemory.models.crawl to avoid duplication.

Note: Outgoing events (ModelIntentStoredEvent) are imported from omnibase_core
to avoid contract duplication. See omnibase_core.models.events.
"""

from .model_crawl_tick_command import ModelCrawlTickCommand
from .model_intent_classified_event import ModelIntentClassifiedEvent

__all__ = [
    "ModelCrawlTickCommand",
    "ModelIntentClassifiedEvent",
]
