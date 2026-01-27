from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelIntentClassifiedEvent(BaseModel):
    """Incoming event from omniintelligence intent classifier.

    TODO(OMN-future): Consider migrating to omnibase_core.models.events
    once cross-repo event schemas are standardized. This is logically
    an omniintelligence-owned event; omnimemory is a consumer.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    event_type: Literal["IntentClassified"] = "IntentClassified"
    session_id: str = Field(..., min_length=1, description="Session identifier")
    correlation_id: UUID = Field(..., description="Correlation ID for tracing")
    intent_category: str = Field(
        ..., min_length=1, description="Classified intent category"
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Classification confidence score"
    )
    keywords: list[str] = Field(
        default_factory=list,
        description="Extracted keywords (forward-compatible with OMN-1626)",
    )
    timestamp: datetime = Field(..., description="Event timestamp")
