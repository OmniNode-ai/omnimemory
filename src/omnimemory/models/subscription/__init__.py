"""
Subscription domain models for OmniMemory following ONEX standards.

This module provides models for agent subscriptions, notification delivery,
and circuit breaker state management for the memory change notification system.

Topic Naming Convention: memory.<entity>.<event>
Examples:
    - memory.item.created
    - memory.item.updated
    - memory.item.deleted
    - memory.collection.created
"""

from ...enums.enum_subscription_status import (
    EnumCircuitBreakerState,
    EnumDeliveryStatus,
    EnumSubscriptionStatus,
)
from .constants import TOPIC_PATTERN, TOPIC_PATTERN_REGEX
from .model_circuit_breaker_state import ModelCircuitBreakerState
from .model_notification_delivery_attempt import ModelNotificationDeliveryAttempt
from .model_notification_event import ModelNotificationEvent
from .model_notification_event_payload import ModelNotificationEventPayload
from .model_subscription import ModelSubscription
from .model_subscription_delivery import ModelSubscriptionDeliveryWebhook

__all__ = [
    # Constants
    "TOPIC_PATTERN",
    "TOPIC_PATTERN_REGEX",
    # Enums
    "EnumCircuitBreakerState",
    "EnumDeliveryStatus",
    "EnumSubscriptionStatus",
    # Models
    "ModelCircuitBreakerState",
    "ModelNotificationDeliveryAttempt",
    "ModelNotificationEvent",
    "ModelNotificationEventPayload",
    "ModelSubscription",
    "ModelSubscriptionDeliveryWebhook",
]
