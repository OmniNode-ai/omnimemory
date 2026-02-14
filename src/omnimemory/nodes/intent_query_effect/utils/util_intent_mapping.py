# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Utilities for mapping between core intent models and event payloads.

This module provides mapping functions to convert between
ModelIntentRecord and the event payload models used for Kafka event
transmission.

The key difference between models:
    - ModelIntentRecord.session_ref is optional (str | None, local domain model)
    - ModelIntentRecordPayload.session_ref is required (str, for event transmission)
    - ModelIntentRecord.intent_category is str
    - ModelIntentRecordPayload.intent_category is str
    - ModelIntentRecord.created_at_utc -> ModelIntentRecordPayload.created_at

Example::

    from datetime import datetime, timezone
    from uuid import uuid4

    from omnimemory.handlers.adapters.models import ModelIntentRecord
    from omnimemory.nodes.intent_query_effect.utils import map_to_intent_payload

    record = ModelIntentRecord(
        intent_id=uuid4(),
        session_ref="session_abc123",
        intent_category="debugging",
        confidence=0.9,
        keywords=["error", "fix"],
        created_at_utc=datetime.now(tz=timezone.utc),
    )

    payload = map_to_intent_payload(record)
    # payload is ready for event transmission

.. versionadded:: 0.1.0
    Initial implementation for OMN-1504.

.. versionchanged:: 0.2.0
    Updated to use omnibase_core.models.intelligence.ModelIntentRecord
    instead of local domain model (omnibase-core 0.13.1).

.. versionchanged:: 0.3.0
    Uses ModelIntentRecordPayload from omnibase-core 0.17.
    ModelIntentRecord now imported from local domain model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from omnibase_core.models.events import ModelIntentRecordPayload

if TYPE_CHECKING:
    from omnimemory.handlers.adapters.models import ModelIntentRecord

__all__ = ["map_intent_records", "map_to_intent_payload"]


def map_to_intent_payload(record: ModelIntentRecord) -> ModelIntentRecordPayload:
    """Convert a ModelIntentRecord to ModelIntentRecordPayload.

    Maps from the core intent model to the event payload model
    for transmission in Kafka events.

    Args:
        record: The intent record from AdapterIntentGraph (omnimemory domain model).

    Returns:
        ModelIntentRecordPayload suitable for event transmission.

    Note:
        Field mappings:
            - ModelIntentRecord.session_ref (str | None) -> ModelIntentRecordPayload.session_ref (str, defaults to "")
            - ModelIntentRecord.intent_category (str) -> ModelIntentRecordPayload.intent_category (str)
            - ModelIntentRecord.created_at_utc -> ModelIntentRecordPayload.created_at
    """
    # ModelIntentRecord.intent_category is typed as str, so use directly.
    return ModelIntentRecordPayload(
        intent_id=record.intent_id,
        session_ref=record.session_ref or "",
        intent_category=record.intent_category,
        confidence=record.confidence,
        keywords=record.keywords,
        created_at=record.created_at_utc,
    )


def map_intent_records(
    records: list[ModelIntentRecord],
) -> list[ModelIntentRecordPayload]:
    """Convert a list of ModelIntentRecord to ModelIntentRecordPayload.

    Convenience function for bulk conversion of intent records.

    Args:
        records: List of intent records from omnimemory.

    Returns:
        List of payload models for event transmission.
    """
    return [map_to_intent_payload(record) for record in records]
