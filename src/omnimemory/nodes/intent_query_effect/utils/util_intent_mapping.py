# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Utilities for mapping between core intent models and event payloads.

This module provides mapping functions to convert between core
ModelIntentRecord and the event payload models used for Kafka event
transmission.

The key difference between models:
    - ModelIntentRecord.session_id is required (from omnibase_core)
    - ModelIntentRecordPayload.session_ref is required (for event transmission)
    - ModelIntentRecord.intent_category is EnumIntentCategory (enum)
    - ModelIntentRecordPayload.intent_category is str

Example::

    from omnibase_core.models.intelligence import ModelIntentRecord
    from omnibase_core.enums.intelligence import EnumIntentCategory
    from omnimemory.nodes.intent_query_effect.utils import map_to_intent_payload

    record = ModelIntentRecord(
        intent_id=uuid4(),
        session_id="session_abc123",
        intent_category=EnumIntentCategory.DEBUGGING,
        confidence=0.9,
        keywords=["error", "fix"],
    )

    payload = map_to_intent_payload(record)
    # payload is ready for event transmission

.. versionadded:: 0.1.0
    Initial implementation for OMN-1504.

.. versionchanged:: 0.2.0
    Updated to use omnibase_core.models.intelligence.ModelIntentRecord
    instead of local domain model (omnibase-core 0.13.1).
"""

from __future__ import annotations

from datetime import UTC, datetime

from omnibase_core.models.events import ModelIntentRecordPayload
from omnibase_core.models.intelligence import ModelIntentRecord  # noqa: TC002

__all__ = ["map_intent_records", "map_to_intent_payload"]


def map_to_intent_payload(record: ModelIntentRecord) -> ModelIntentRecordPayload:
    """Convert a ModelIntentRecord to ModelIntentRecordPayload.

    Maps from the core intent model to the event payload model
    for transmission in Kafka events.

    Args:
        record: The intent record from AdapterIntentGraph (omnibase_core model).

    Returns:
        ModelIntentRecordPayload suitable for event transmission.

    Note:
        Field mappings:
            - ModelIntentRecord.session_id -> ModelIntentRecordPayload.session_ref
            - ModelIntentRecord.intent_category (enum) -> str value
            - ModelIntentRecord.created_at -> ModelIntentRecordPayload.created_at
    """
    return ModelIntentRecordPayload(
        intent_id=record.intent_id,
        session_ref=record.session_id,
        intent_category=record.intent_category.value,
        confidence=record.confidence,
        keywords=record.keywords,
        created_at=record.created_at or datetime.now(UTC),
    )


def map_intent_records(
    records: list[ModelIntentRecord],
) -> list[ModelIntentRecordPayload]:
    """Convert a list of ModelIntentRecord to ModelIntentRecordPayload.

    Convenience function for bulk conversion of intent records.

    Args:
        records: List of intent records from omnibase_core.

    Returns:
        List of payload models for event transmission.
    """
    return [map_to_intent_payload(record) for record in records]
