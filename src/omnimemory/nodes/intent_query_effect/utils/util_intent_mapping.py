# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Utilities for mapping between omnimemory and core intent models.

This module provides mapping functions to convert between internal
omnimemory intent records and the core event payload models used
for Kafka event transmission.

The key difference between models:
    - ModelIntentRecord.session_ref is optional (from graph queries)
    - IntentRecordPayload.session_ref is required (for event transmission)
    - Field name: created_at_utc (internal) -> created_at (payload)

Example::

    from omnimemory.handlers.adapters.models import ModelIntentRecord
    from omnimemory.nodes.intent_query_effect.models import map_to_intent_payload

    record = ModelIntentRecord(
        intent_id=uuid4(),
        session_ref="session_abc123",
        intent_category="debugging",
        confidence=0.9,
        keywords=["error", "fix"],
        created_at_utc=datetime.now(UTC),
    )

    payload = map_to_intent_payload(record)
    # payload is ready for event transmission

.. versionadded:: 0.1.0
    Initial implementation for OMN-1504.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from omnibase_core.models.events import IntentRecordPayload

if TYPE_CHECKING:
    from omnimemory.handlers.adapters.models import ModelIntentRecord

__all__ = ["map_intent_records", "map_to_intent_payload"]


def map_to_intent_payload(record: ModelIntentRecord) -> IntentRecordPayload:
    """Convert a ModelIntentRecord to IntentRecordPayload.

    Maps from the omnimemory internal model to the core event payload model
    for transmission in Kafka events.

    Args:
        record: The internal intent record from AdapterIntentGraph.

    Returns:
        IntentRecordPayload suitable for event transmission.

    Raises:
        ValueError: If session_ref is None (required for payload).

    Note:
        The field name differs between models:
            - ModelIntentRecord uses ``created_at_utc``
            - IntentRecordPayload uses ``created_at``
    """
    if record.session_ref is None:
        raise ValueError(
            f"Cannot map intent {record.intent_id} to payload: session_ref is required"
        )

    return IntentRecordPayload(
        intent_id=record.intent_id,
        session_ref=record.session_ref,
        intent_category=record.intent_category,
        confidence=record.confidence,
        keywords=record.keywords,
        created_at=record.created_at_utc,
    )


def map_intent_records(records: list[ModelIntentRecord]) -> list[IntentRecordPayload]:
    """Convert a list of ModelIntentRecord to IntentRecordPayload.

    Convenience function for bulk conversion of intent records.

    Args:
        records: List of internal intent records.

    Returns:
        List of payload models for event transmission.

    Raises:
        ValueError: If any record has a None session_ref.
    """
    return [map_to_intent_payload(record) for record in records]
