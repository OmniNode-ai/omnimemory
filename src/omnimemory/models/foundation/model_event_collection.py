"""
Event collection model for ONEX Foundation Architecture.

Provides strongly typed event collection replacing List[Dict[str, Any]].
"""

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field

from .model_event_data import ModelEventData


class ModelEventCollection(BaseModel):
    """Strongly typed event collection replacing List[Dict[str, Any]]."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    events: List[ModelEventData] = Field(
        default_factory=list, description="List of system events with structured data"
    )

    def add_event(
        self,
        event_type: str,
        timestamp: str,
        source: str,
        message: str,
        severity: str = "info",
        correlation_id: Optional[str] = None,
    ) -> None:
        """Add a new event to the collection."""
        event = ModelEventData(
            event_type=event_type,
            timestamp=timestamp,
            source=source,
            message=message,
            severity=severity,
            correlation_id=correlation_id,
        )
        self.events.append(event)

    def get_events_by_type(self, event_type: str) -> List[ModelEventData]:
        """Get all events of a specific type."""
        return [event for event in self.events if event.event_type == event_type]

    def get_events_by_severity(self, severity: str) -> List[ModelEventData]:
        """Get all events of a specific severity."""
        return [event for event in self.events if event.severity == severity.lower()]
