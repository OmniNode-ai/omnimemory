from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelIntentStoredEvent(BaseModel):
    """Published when intent is successfully stored in graph."""

    model_config = ConfigDict(extra="forbid", strict=True)

    event_type: Literal["IntentStored"] = Field(
        "IntentStored", description="Event type discriminator for message routing"
    )
    intent_id: UUID = Field(..., description="Generated intent identifier")
    session_id: str = Field(..., description="Session that had the intent")
    intent_category: str = Field(..., description="Stored intent category")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Classification confidence"
    )
    created: bool = Field(..., description="True if new node created, False if merged")
    correlation_id: UUID = Field(..., description="Correlation ID from source event")
    timestamp: datetime = Field(..., description="Storage timestamp")
    storage_latency_ms: float = Field(
        ..., ge=0, description="Storage operation latency"
    )
