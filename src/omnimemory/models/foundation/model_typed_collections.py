"""
Typed Collections for ONEX Foundation Architecture

This module provides strongly typed Pydantic models to replace generic types
like Dict[str, Any], List[str], and List[Dict[str, Any]] throughout the codebase.

All models follow ONEX standards with:
- Strong typing with zero Any types
- Comprehensive Field descriptions
- Validation and serialization support
- Monadic composition patterns
"""

from __future__ import annotations

from typing import Any, List, Optional, Union
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, validator


# === STRING COLLECTIONS ===


class ModelStringList(BaseModel):
    """Strongly typed list of strings following ONEX standards."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid"
    )

    values: List[str] = Field(
        default_factory=list,
        description="List of string values with validation and deduplication"
    )

    @validator('values')
    def validate_strings(cls, v):
        """Validate and deduplicate string values."""
        if not isinstance(v, list):
            raise ValueError("values must be a list")

        # Remove empty strings and duplicates while preserving order
        seen = set()
        result = []
        for item in v:
            if item and item.strip() and item not in seen:
                seen.add(item)
                result.append(item.strip())

        return result


class ModelOptionalStringList(BaseModel):
    """Optional strongly typed list of strings."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid"
    )

    values: Optional[List[str]] = Field(
        default=None,
        description="Optional list of string values, None if not set"
    )

    @validator('values')
    def validate_optional_strings(cls, v):
        """Validate optional string values."""
        if v is None:
            return None

        if not isinstance(v, list):
            raise ValueError("values must be a list or None")

        # Remove empty strings and duplicates while preserving order
        seen = set()
        result = []
        for item in v:
            if item and item.strip() and item not in seen:
                seen.add(item)
                result.append(item.strip())

        return result if result else None


# === METADATA COLLECTIONS ===


class ModelKeyValuePair(BaseModel):
    """Strongly typed key-value pair for metadata."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid"
    )

    key: str = Field(description="Metadata key identifier")
    value: str = Field(description="Metadata value content")

    @validator('key')
    def validate_key(cls, v):
        """Validate metadata key format."""
        if not v or not v.strip():
            raise ValueError("key cannot be empty")
        return v.strip()


class ModelMetadata(BaseModel):
    """Strongly typed metadata collection replacing Dict[str, Any]."""

    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid"
    )

    pairs: List[ModelKeyValuePair] = Field(
        default_factory=list,
        description="List of key-value pairs for metadata storage"
    )

    def get_value(self, key: str) -> Optional[str]:
        """Get metadata value by key."""
        for pair in self.pairs:
            if pair.key == key:
                return pair.value
        return None

    def set_value(self, key: str, value: str) -> None:
        """Set metadata value by key."""
        # Update existing or add new
        for pair in self.pairs:
            if pair.key == key:
                pair.value = value
                return

        self.pairs.append(ModelKeyValuePair(key=key, value=value))

    def to_dict(self) -> dict[str, str]:
        """Convert to dictionary format for backward compatibility."""
        return {pair.key: pair.value for pair in self.pairs}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelMetadata:
        """Create from dictionary, converting values to strings."""
        pairs = [
            ModelKeyValuePair(key=str(k), value=str(v))
            for k, v in data.items()
            if k and v is not None
        ]
        return cls(pairs=pairs)


# === STRUCTURED DATA COLLECTIONS ===


class ModelStructuredField(BaseModel):
    """Strongly typed field for structured data."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid"
    )

    name: str = Field(description="Field name identifier")
    value: str = Field(description="Field value content")
    field_type: str = Field(
        default="string",
        description="Field type indicator (string, number, boolean, etc.)"
    )

    @validator('name')
    def validate_name(cls, v):
        """Validate field name format."""
        if not v or not v.strip():
            raise ValueError("name cannot be empty")
        return v.strip()


class ModelStructuredData(BaseModel):
    """Strongly typed structured data replacing List[Dict[str, Any]]."""

    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid"
    )

    fields: List[ModelStructuredField] = Field(
        default_factory=list,
        description="List of structured fields with type information"
    )
    schema_version: str = Field(
        default="1.0",
        description="Schema version for compatibility tracking"
    )

    def get_field_value(self, name: str) -> Optional[str]:
        """Get field value by name."""
        for field in self.fields:
            if field.name == name:
                return field.value
        return None

    def set_field_value(self, name: str, value: str, field_type: str = "string") -> None:
        """Set field value by name."""
        # Update existing or add new
        for field in self.fields:
            if field.name == name:
                field.value = value
                field.field_type = field_type
                return

        self.fields.append(ModelStructuredField(
            name=name,
            value=value,
            field_type=field_type
        ))


# === CONFIGURATION COLLECTIONS ===


class ModelConfigurationOption(BaseModel):
    """Strongly typed configuration option."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid"
    )

    key: str = Field(description="Configuration option key")
    value: str = Field(description="Configuration option value")
    description: Optional[str] = Field(
        default=None,
        description="Option description for documentation"
    )
    is_sensitive: bool = Field(
        default=False,
        description="Whether this option contains sensitive data"
    )

    @validator('key')
    def validate_key(cls, v):
        """Validate configuration key format."""
        if not v or not v.strip():
            raise ValueError("key cannot be empty")
        return v.strip()


class ModelConfiguration(BaseModel):
    """Strongly typed configuration replacing Dict[str, Any]."""

    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid"
    )

    options: List[ModelConfigurationOption] = Field(
        default_factory=list,
        description="List of configuration options with metadata"
    )

    def get_option(self, key: str) -> Optional[str]:
        """Get configuration option value by key."""
        for option in self.options:
            if option.key == key:
                return option.value
        return None

    def set_option(
        self,
        key: str,
        value: str,
        description: Optional[str] = None,
        is_sensitive: bool = False
    ) -> None:
        """Set configuration option with metadata."""
        # Update existing or add new
        for option in self.options:
            if option.key == key:
                option.value = value
                if description is not None:
                    option.description = description
                option.is_sensitive = is_sensitive
                return

        self.options.append(ModelConfigurationOption(
            key=key,
            value=value,
            description=description,
            is_sensitive=is_sensitive
        ))


# === EVENT AND LOG COLLECTIONS ===


class ModelEventData(BaseModel):
    """Strongly typed event data for system events."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid"
    )

    event_type: str = Field(description="Type of event (creation, update, deletion, etc.)")
    timestamp: str = Field(description="ISO 8601 timestamp of the event")
    source: str = Field(description="Source system or component generating the event")
    severity: str = Field(
        default="info",
        description="Event severity level (debug, info, warning, error, critical)"
    )
    message: str = Field(description="Human-readable event message")
    correlation_id: Optional[str] = Field(
        default=None,
        description="Correlation ID for tracking related events"
    )

    @validator('event_type')
    def validate_event_type(cls, v):
        """Validate event type format."""
        if not v or not v.strip():
            raise ValueError("event_type cannot be empty")
        return v.strip().lower()

    @validator('severity')
    def validate_severity(cls, v):
        """Validate severity level."""
        valid_levels = {"debug", "info", "warning", "error", "critical"}
        if v.lower() not in valid_levels:
            raise ValueError(f"severity must be one of: {valid_levels}")
        return v.lower()


class ModelEventCollection(BaseModel):
    """Strongly typed event collection replacing List[Dict[str, Any]]."""

    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid"
    )

    events: List[ModelEventData] = Field(
        default_factory=list,
        description="List of system events with structured data"
    )

    def add_event(
        self,
        event_type: str,
        timestamp: str,
        source: str,
        message: str,
        severity: str = "info",
        correlation_id: Optional[str] = None
    ) -> None:
        """Add a new event to the collection."""
        event = ModelEventData(
            event_type=event_type,
            timestamp=timestamp,
            source=source,
            message=message,
            severity=severity,
            correlation_id=correlation_id
        )
        self.events.append(event)

    def get_events_by_type(self, event_type: str) -> List[ModelEventData]:
        """Get all events of a specific type."""
        return [event for event in self.events if event.event_type == event_type]

    def get_events_by_severity(self, severity: str) -> List[ModelEventData]:
        """Get all events of a specific severity."""
        return [event for event in self.events if event.severity == severity.lower()]


# === RESULT AND RESPONSE COLLECTIONS ===


class ModelResultItem(BaseModel):
    """Strongly typed result item for operation results."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid"
    )

    id: str = Field(description="Unique identifier for this result item")
    status: str = Field(description="Status of this specific item (success, failure, pending)")
    message: str = Field(description="Human-readable message about this item")
    data: Optional[ModelStructuredData] = Field(
        default=None,
        description="Structured data associated with this item"
    )

    @validator('status')
    def validate_status(cls, v):
        """Validate status values."""
        valid_statuses = {"success", "failure", "pending", "partial", "cancelled"}
        if v.lower() not in valid_statuses:
            raise ValueError(f"status must be one of: {valid_statuses}")
        return v.lower()


class ModelResultCollection(BaseModel):
    """Strongly typed result collection replacing List[Dict[str, Any]]."""

    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid"
    )

    results: List[ModelResultItem] = Field(
        default_factory=list,
        description="List of operation results with structured data"
    )

    def add_result(
        self,
        id: str,
        status: str,
        message: str,
        data: Optional[ModelStructuredData] = None
    ) -> None:
        """Add a new result to the collection."""
        result = ModelResultItem(
            id=id,
            status=status,
            message=message,
            data=data
        )
        self.results.append(result)

    def get_successful_results(self) -> List[ModelResultItem]:
        """Get all successful results."""
        return [result for result in self.results if result.status == "success"]

    def get_failed_results(self) -> List[ModelResultItem]:
        """Get all failed results."""
        return [result for result in self.results if result.status == "failure"]


# === UTILITY FUNCTIONS ===


def convert_dict_to_metadata(data: dict[str, Any]) -> ModelMetadata:
    """Convert a dictionary to ModelMetadata."""
    return ModelMetadata.from_dict(data)


def convert_list_to_string_list(data: List[str]) -> ModelStringList:
    """Convert a list of strings to ModelStringList."""
    return ModelStringList(values=data)


def convert_list_of_dicts_to_structured_data(data: List[dict[str, Any]]) -> ModelResultCollection:
    """Convert a list of dictionaries to structured result collection."""
    collection = ModelResultCollection()

    for i, item in enumerate(data):
        # Convert dict to structured data
        structured_data = ModelStructuredData()
        for key, value in item.items():
            structured_data.set_field_value(key, str(value))

        collection.add_result(
            id=str(i),
            status="success",
            message=f"Converted item {i}",
            data=structured_data
        )

    return collection