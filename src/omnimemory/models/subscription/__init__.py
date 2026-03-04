# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Subscription domain models for OmniMemory following ONEX standards.

for the memory change notification system.

Delivery Mechanism:
    Notifications are published to Kafka. Agents consume events directly via
    consumer groups. If webhook delivery to external systems is needed in the
    future, implement a WebhookEmitterEffect node separately.

Topic Naming Convention: memory.<entity>.<event>
Examples:
    - memory.item.created
    - memory.item.updated
    - memory.item.deleted
    - memory.collection.created
"""

from ...enums.enum_subscription_status import EnumSubscriptionStatus
from .constants import (
    TOPIC_PATTERN,
    TOPIC_PATTERN_REGEX,
)
from .model_notification_event import ModelNotificationEvent
from .model_notification_event_payload import ModelNotificationEventPayload
from .model_subscription import ModelSubscription

__all__ = [
    # Constants
    "TOPIC_PATTERN",
    "TOPIC_PATTERN_REGEX",
    # Enums
    "EnumSubscriptionStatus",
    # Models
    "ModelNotificationEvent",
    "ModelNotificationEventPayload",
    "ModelSubscription",
]
