# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Agent Coordinator Response model for cross-agent memory coordination operations.

This module defines the response envelope used by the agent_coordinator_orchestrator
node to return results from subscription and notification operations.

Example:
    >>> from omnimemory.nodes.agent_coordinator_orchestrator.models import (
    ...     ModelAgentCoordinatorResponse,
    ...     EnumAgentCoordinatorAction,
    ... )
    >>> response = ModelAgentCoordinatorResponse(
    ...     success=True,
    ...     action=EnumAgentCoordinatorAction.SUBSCRIBE,
    ...     correlation_id=request.correlation_id,
    ...     subscription=created_subscription,
    ... )

.. versionadded:: 0.1.0
    Initial implementation for OMN-1393.
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from ....models.subscription.model_notification_delivery_attempt import (
    ModelNotificationDeliveryAttempt,  # noqa: TC001 - runtime import for Pydantic field type
)
from ....models.subscription.model_subscription import (
    ModelSubscription,  # noqa: TC001 - runtime import for Pydantic field type
)
from .model_request import (
    EnumAgentCoordinatorAction,  # noqa: TC001 - runtime import for Pydantic field type
)

__all__ = ["ModelAgentCoordinatorResponse"]


class ModelAgentCoordinatorResponse(BaseModel):
    """Response envelope for agent coordinator operations.

    This model provides a consistent response structure for all agent coordinator
    operations (subscribe, unsubscribe, list_subscriptions, notify). The success
    field indicates the operation outcome, while optional fields carry
    operation-specific results.

    Attributes:
        success: Whether the action succeeded.
        action: The action that was performed.
        correlation_id: Request correlation ID for tracing.
        subscription: Created or matched subscription (for subscribe action).
        subscriptions: List of agent subscriptions (for list_subscriptions action).
        delivery_attempts: Record of all delivery attempts (for notify action).
        success_count: Number of successful deliveries (for notify action).
        failure_count: Number of failed deliveries (for notify action).
        error_message: Detailed error information when success is False.
        error_code: Machine-readable error code for programmatic handling.

    Example:
        >>> # Successful subscribe response
        >>> subscribe_response = ModelAgentCoordinatorResponse(
        ...     success=True,
        ...     action=EnumAgentCoordinatorAction.SUBSCRIBE,
        ...     correlation_id=request.correlation_id,
        ...     subscription=created_subscription,
        ... )
        >>>
        >>> # Successful notify response
        >>> notify_response = ModelAgentCoordinatorResponse(
        ...     success=True,
        ...     action=EnumAgentCoordinatorAction.NOTIFY,
        ...     correlation_id=request.correlation_id,
        ...     delivery_attempts=attempts,
        ...     success_count=5,
        ...     failure_count=1,
        ... )
        >>>
        >>> # Error response
        >>> error_response = ModelAgentCoordinatorResponse(
        ...     success=False,
        ...     action=EnumAgentCoordinatorAction.SUBSCRIBE,
        ...     correlation_id=request.correlation_id,
        ...     error_message="Duplicate subscription exists",
        ...     error_code="DUPLICATE_SUBSCRIPTION",
        ... )
    """

    model_config = ConfigDict(extra="forbid")

    success: bool = Field(
        ...,
        description="Whether the action succeeded",
    )

    action: EnumAgentCoordinatorAction = Field(
        ...,
        description="The action that was performed",
    )

    correlation_id: UUID = Field(
        ...,
        description="Request correlation ID for tracing",
    )

    # For subscribe action
    subscription: ModelSubscription | None = Field(
        default=None,
        description="Created or matched subscription (for subscribe action)",
    )

    # For list_subscriptions action
    subscriptions: list[ModelSubscription] | None = Field(
        default=None,
        description="List of agent subscriptions (for list_subscriptions action)",
    )

    # For notify action
    delivery_attempts: list[ModelNotificationDeliveryAttempt] | None = Field(
        default=None,
        description="Record of all delivery attempts (for notify action)",
    )

    success_count: int | None = Field(
        default=None,
        ge=0,
        description="Number of successful deliveries (for notify action)",
    )

    failure_count: int | None = Field(
        default=None,
        ge=0,
        description="Number of failed deliveries (for notify action)",
    )

    # Error handling
    error_message: str | None = Field(
        default=None,
        max_length=2048,
        description="Detailed error information when success is False",
    )

    error_code: str | None = Field(
        default=None,
        max_length=64,
        description="Machine-readable error code for programmatic handling",
    )
