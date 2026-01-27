from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelIntentStoreFailedEvent(BaseModel):
    """Published when intent storage fails (also routed to DLQ)."""

    model_config = ConfigDict(extra="forbid", strict=True)

    event_type: Literal["IntentStoreFailed"] = Field(
        "IntentStoreFailed", description="Event type discriminator for message routing"
    )
    session_id: str = Field(..., description="Session from failed event")
    intent_category: str = Field(..., description="Intent category from failed event")
    correlation_id: UUID = Field(..., description="Correlation ID from source event")
    error_type: str = Field(..., description="Error classification")
    error_message: str = Field(..., description="Error details")
    timestamp: datetime = Field(..., description="Failure timestamp")
    retry_count: int = Field(default=0, ge=0, description="Number of retry attempts")
