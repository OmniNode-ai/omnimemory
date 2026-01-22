"""
Subscription status enumeration following ONEX standards.
"""

from enum import Enum


class EnumSubscriptionStatus(str, Enum):
    """
    Status values for subscriptions following ONEX standards.

    Represents the current state of a subscription:
    - ACTIVE: Subscription is active and receiving notifications
    - SUSPENDED: Subscription temporarily paused (e.g., circuit breaker open)
    - DELETED: Subscription has been soft-deleted
    """

    ACTIVE = "active"
    SUSPENDED = "suspended"
    DELETED = "deleted"


class EnumDeliveryStatus(str, Enum):
    """
    Status values for notification delivery attempts following ONEX standards.

    Represents the current state of a delivery attempt:
    - PENDING: Delivery queued but not yet attempted
    - SUCCESS: Delivery completed successfully
    - FAILED: Delivery attempt failed
    - DLQ: Delivery moved to dead letter queue after max retries
    """

    PENDING = "pending"
    SUCCESS = "success"
    FAILED = "failed"
    DLQ = "dlq"


class EnumCircuitBreakerState(str, Enum):
    """
    Circuit breaker states following ONEX standards.

    - CLOSED: Circuit is closed, requests flow normally
    - OPEN: Circuit is open, requests are rejected
    - HALF_OPEN: Circuit is testing if the endpoint has recovered
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"
