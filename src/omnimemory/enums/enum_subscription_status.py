# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""
Subscription status enumeration following ONEX standards.
"""

from enum import Enum


class EnumSubscriptionStatus(str, Enum):
    """
    Status values for subscriptions following ONEX standards.

    Represents the current state of a subscription:
    - ACTIVE: Subscription is active and receiving notifications
    - SUSPENDED: Subscription temporarily paused
    - DELETED: Subscription has been soft-deleted
    """

    ACTIVE = "active"
    SUSPENDED = "suspended"
    DELETED = "deleted"


# Note: EnumDeliveryStatus and EnumCircuitBreakerState were removed in v0.2.0
# when webhook delivery was replaced with Kafka event bus.
# If WebhookEmitterEffect node is implemented in the future, these enums
# can be restored in that node's module.
