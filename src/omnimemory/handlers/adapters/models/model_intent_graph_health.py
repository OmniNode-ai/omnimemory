"""Health status model for the Intent Graph adapter.

This module provides the Pydantic model for representing health check results
from the AdapterIntentGraph, which stores intent classification results in Memgraph.
"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["ModelIntentGraphHealth"]


class ModelIntentGraphHealth(  # omnimemory-model-exempt: adapter-internal health
    BaseModel
):
    """Health status information for the intent graph adapter.

    This model represents the health check result from AdapterIntentGraph,
    which manages intent classification data stored in Memgraph. It includes
    connectivity status, initialization state, and graph statistics.

    Attributes:
        is_healthy: Overall health status of the adapter.
        initialized: Whether the adapter has been initialized.
        handler_healthy: Health status from the underlying graph handler.
        error_message: Error details if unhealthy.
        session_count: Number of sessions stored in the intent graph.
        intent_count: Number of intents stored in the intent graph.
        last_check_timestamp: Timestamp of the last health check.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    is_healthy: bool = Field(
        ...,
        description="Overall health status of the intent graph adapter",
    )
    initialized: bool = Field(
        ...,
        description="Whether the adapter has been initialized",
    )
    handler_healthy: bool | None = Field(
        default=None,
        description="Health status from underlying graph handler (None if not checked)",
    )
    error_message: str | None = Field(
        default=None,
        description="Error details if unhealthy",
    )
    session_count: int | None = Field(
        default=None,
        ge=0,
        description="Number of sessions stored in the intent graph (None if not queried)",
    )
    intent_count: int | None = Field(
        default=None,
        ge=0,
        description="Number of intents stored in the intent graph (None if not queried)",
    )
    last_check_timestamp: datetime | None = Field(
        default=None,
        description="Timestamp of the last health check (None if not recorded)",
    )
