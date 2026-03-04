# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Typed Collections for ONEX Foundation Architecture

like Dict[str, Any], List[str], and List[Dict[str, Any]] throughout the codebase.

All models follow ONEX standards with:
- Strong typing with zero Any types
- Comprehensive Field descriptions
- Validation and serialization support
- Monadic composition patterns
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field, field_validator

from omnimemory.models.foundation.model_semver import ModelSemVer

if TYPE_CHECKING:
    from collections.abc import Iterator

# === STRING COLLECTIONS ===


class ModelStringList(BaseModel):
    """Strongly typed list of strings following ONEX standards."""

    model_config = ConfigDict(
        str_strip_whitespace=True, validate_assignment=True, extra="forbid"
    )

    values: list[str] = Field(
        default_factory=list,
        description="List of string values with validation and deduplication",
    )

    @field_validator("values")
    @classmethod
    def validate_strings(cls, v: list[str]) -> list[str]:
        """Validate and deduplicate string values."""
        if not isinstance(v, list):
            raise ValueError("values must be a list")

        # Remove empty strings and duplicates while preserving order
        # Use O(1) set operations for efficient deduplication
        seen: set[str] = set()
        result: list[str] = []
        for item in v:
            if item:
                stripped_item = item.strip()
                if stripped_item and stripped_item not in seen:
                    seen.add(stripped_item)
                    result.append(stripped_item)

        return result

    def __contains__(self, item: str) -> bool:
        """Support 'in' operator for checking membership."""
        return item in self.values

    def __iter__(self) -> Iterator[str]:  # type: ignore[override]
        """Support iteration over values."""
        return iter(self.values)

    def __len__(self) -> int:
        """Support len() function."""
        return len(self.values)


class ModelOptionalStringList(BaseModel):
    """Optional strongly typed list of strings.

    Provides container protocol methods for consistency with ModelStringList.
    When values is None, container operations return appropriate defaults.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True, validate_assignment=True, extra="forbid"
    )

    values: list[str] | None = Field(
        default=None, description="Optional list of string values, None if not set"
    )

    @field_validator("values")
    @classmethod
    def validate_optional_strings(cls, v: list[str] | None) -> list[str] | None:
        """Validate optional string values.

        Note: An explicitly provided empty list is preserved as an empty list,
        not silently converted to None. This allows distinguishing between
        "not set" (None) and "explicitly set to empty" ([]).
        """
        if v is None:
            return None

        if not isinstance(v, list):
            raise ValueError("values must be a list or None")

        # Remove empty strings and duplicates while preserving order
        # Use O(1) set operations for efficient deduplication
        seen: set[str] = set()
        result: list[str] = []
        for item in v:
            if item:
                stripped_item = item.strip()
                if stripped_item and stripped_item not in seen:
                    seen.add(stripped_item)
                    result.append(stripped_item)

        return result

    def __contains__(self, item: str) -> bool:
        """Support 'in' operator for checking membership.

        Returns False if values is None.
        """
        if self.values is None:
            return False
        return item in self.values

    def __iter__(self) -> Iterator[str]:  # type: ignore[override]
        """Support iteration over values.

        Returns empty iterator if values is None.

        Note: We use iterate() instead of __iter__ to avoid conflict with
        Pydantic's BaseModel.__iter__ which returns field-value pairs.
        """
        if self.values is None:
            return iter([])
        return iter(self.values)

    def __len__(self) -> int:
        """Support len() function.

        Returns 0 if values is None.
        """
        if self.values is None:
            return 0
        return len(self.values)

    def __bool__(self) -> bool:
        """Support bool() function.

        Returns False if values is None or empty.
        """
        return self.values is not None and len(self.values) > 0

    def is_empty(self) -> bool:
        """Check if the list is empty or None."""
        return self.values is None or len(self.values) == 0

    def or_default(self, default: list[str]) -> list[str]:
        """Return values or a default if None.

        Args:
            default: Default list to return if values is None

        Returns:
            The values list or the provided default
        """
        if self.values is None:
            return default
        return self.values


# === METADATA COLLECTIONS ===


class ModelKeyValuePair(BaseModel):
    """Strongly typed key-value pair for metadata."""

    model_config = ConfigDict(
        str_strip_whitespace=True, validate_assignment=True, extra="forbid"
    )

    key: str = Field(description="Metadata key identifier")
    value: str = Field(description="Metadata value content")

    @field_validator("key")
    @classmethod
    def validate_key(cls, v: str) -> str:
        """Validate metadata key format."""
        if not v or not v.strip():
            raise ValueError("key cannot be empty")
        return v.strip()


class ModelMetadata(BaseModel):
    """Strongly typed metadata collection replacing Dict[str, Any]."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    pairs: list[ModelKeyValuePair] = Field(
        default_factory=list, description="List of key-value pairs for metadata storage"
    )

    def get_value(self, key: str) -> str | None:
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
    def from_dict(cls, data: dict[str, object]) -> ModelMetadata:
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
        str_strip_whitespace=True, validate_assignment=True, extra="forbid"
    )

    name: str = Field(description="Field name identifier")
    value: str = Field(description="Field value content")
    field_type: str = Field(
        default="string",
        description="Field type indicator (string, number, boolean, etc.)",
    )

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate field name format."""
        if not v or not v.strip():
            raise ValueError("name cannot be empty")
        return v.strip()


class ModelStructuredData(BaseModel):
    """Strongly typed structured data replacing List[Dict[str, Any]]."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    fields: list[ModelStructuredField] = Field(
        default_factory=list,
        description="List of structured fields with type information",
    )
    schema_version: ModelSemVer = Field(
        default_factory=lambda: ModelSemVer(major=1, minor=0, patch=0),
        description="Semantic schema version for compatibility tracking",
    )

    def get_field_value(self, name: str) -> str | None:
        """Get field value by name."""
        for field in self.fields:
            if field.name == name:
                return field.value
        return None

    def set_field_value(
        self, name: str, value: str, field_type: str = "string"
    ) -> None:
        """Set field value by name."""
        # Update existing or add new
        for field in self.fields:
            if field.name == name:
                field.value = value
                field.field_type = field_type
                return

        self.fields.append(
            ModelStructuredField(name=name, value=value, field_type=field_type)
        )


# === CONFIGURATION COLLECTIONS ===


class ModelConfigurationOption(BaseModel):
    """Strongly typed configuration option."""

    model_config = ConfigDict(
        str_strip_whitespace=True, validate_assignment=True, extra="forbid"
    )

    key: str = Field(description="Configuration option key")
    value: str = Field(description="Configuration option value")
    description: str | None = Field(
        default=None, description="Option description for documentation"
    )
    is_sensitive: bool = Field(
        default=False, description="Whether this option contains sensitive data"
    )

    @field_validator("key")
    @classmethod
    def validate_key(cls, v: str) -> str:
        """Validate configuration key format."""
        if not v or not v.strip():
            raise ValueError("key cannot be empty")
        return v.strip()


class ModelConfiguration(BaseModel):
    """Strongly typed configuration replacing Dict[str, Any]."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    options: list[ModelConfigurationOption] = Field(
        default_factory=list, description="List of configuration options with metadata"
    )

    def get_option(self, key: str) -> str | None:
        """Get configuration option value by key."""
        for option in self.options:
            if option.key == key:
                return option.value
        return None

    def set_option(
        self,
        key: str,
        value: str,
        description: str | None = None,
        is_sensitive: bool = False,
    ) -> None:
        """Set configuration option with metadata.

        Args:
            key: Configuration option key.
            value: Configuration option value.
            description: Optional description. Only updates if not None.
            is_sensitive: Whether this option contains sensitive data.
                Only updates if not None. Defaults to False for new options.
        """
        # Update existing or add new
        for option in self.options:
            if option.key == key:
                option.value = value
                if description is not None:
                    option.description = description
                option.is_sensitive = is_sensitive
                return

        # For new options, default is_sensitive to False if not specified
        self.options.append(
            ModelConfigurationOption(
                key=key,
                value=value,
                description=description,
                is_sensitive=is_sensitive,
            )
        )


# === EVENT AND LOG COLLECTIONS ===


class ModelEventData(BaseModel):
    """Strongly typed event data for system events."""

    # Defense-in-depth: validate_assignment=True is retained alongside frozen=True.
    # While frozen=True prevents normal field assignment, validate_assignment ensures
    # that Pydantic-aware mutation paths (e.g., model_copy(update=...)) also enforce
    # type validation. This is a deliberate redundancy for safety.
    model_config = ConfigDict(
        frozen=True, validate_assignment=True, str_strip_whitespace=True, extra="forbid"
    )

    event_type: str = Field(
        description="Type of event (creation, update, deletion, etc.)"
    )
    timestamp: str = Field(description="ISO 8601 timestamp of the event")
    source: str = Field(description="Source system or component generating the event")
    severity: str = Field(
        default="info",
        description="Event severity level (debug, info, warning, error, critical)",
    )
    message: str = Field(description="Human-readable event message")
    correlation_id: str | None = Field(
        default=None, description="Correlation ID for tracking related events"
    )

    @field_validator("event_type")
    @classmethod
    def validate_event_type(cls, v: str) -> str:
        """Validate event type format."""
        if not v or not v.strip():
            raise ValueError("event_type cannot be empty")
        return v.strip().lower()

    @field_validator("severity")
    @classmethod
    def validate_severity(cls, v: str) -> str:
        """Validate severity level."""
        valid_levels = {"debug", "info", "warning", "error", "critical"}
        if v.lower() not in valid_levels:
            raise ValueError(f"severity must be one of: {valid_levels}")
        return v.lower()


class ModelEventCollection(BaseModel):
    """Strongly typed event collection replacing List[Dict[str, Any]]."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    events: list[ModelEventData] = Field(
        default_factory=list, description="List of system events with structured data"
    )

    def add_event(
        self,
        event_type: str,
        timestamp: str,
        source: str,
        message: str,
        severity: str = "info",
        correlation_id: str | None = None,
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

    def get_events_by_type(self, event_type: str) -> list[ModelEventData]:
        """Get all events of a specific type."""
        return [event for event in self.events if event.event_type == event_type]

    def get_events_by_severity(self, severity: str) -> list[ModelEventData]:
        """Get all events of a specific severity."""
        return [event for event in self.events if event.severity == severity.lower()]


# === RESULT AND RESPONSE COLLECTIONS ===


class ModelResultItem(BaseModel):
    """Strongly typed result item for operation results."""

    model_config = ConfigDict(
        str_strip_whitespace=True, validate_assignment=True, extra="forbid"
    )

    id: str = Field(description="Unique identifier for this result item")
    status: str = Field(
        description="Status of this specific item (success, failure, pending)"
    )
    message: str = Field(description="Human-readable message about this item")
    data: ModelStructuredData | None = Field(
        default=None, description="Structured data associated with this item"
    )

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str) -> str:
        """Validate status values."""
        valid_statuses = {"success", "failure", "pending", "partial", "cancelled"}
        if v.lower() not in valid_statuses:
            raise ValueError(f"status must be one of: {valid_statuses}")
        return v.lower()


class ModelResultCollection(BaseModel):
    """Strongly typed result collection replacing List[Dict[str, Any]]."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    results: list[ModelResultItem] = Field(
        default_factory=list,
        description="List of operation results with structured data",
    )

    def add_result(
        self,
        id: str,
        status: str,
        message: str,
        data: ModelStructuredData | None = None,
    ) -> None:
        """Add a new result to the collection."""
        result = ModelResultItem(id=id, status=status, message=message, data=data)
        self.results.append(result)

    def get_successful_results(self) -> list[ModelResultItem]:
        """Get all successful results."""
        return [result for result in self.results if result.status == "success"]

    def get_failed_results(self) -> list[ModelResultItem]:
        """Get all failed results."""
        return [result for result in self.results if result.status == "failure"]


# === UTILITY FUNCTIONS ===


def convert_dict_to_metadata(data: dict[str, object]) -> ModelMetadata:
    """Convert a dictionary to ModelMetadata."""
    return ModelMetadata.from_dict(data)


def convert_list_to_string_list(data: list[str]) -> ModelStringList:
    """Convert a list of strings to ModelStringList."""
    return ModelStringList(values=data)


def convert_list_of_dicts_to_structured_data(
    data: list[dict[str, object]],
    default_status: str = "success",
) -> ModelResultCollection:
    """Convert a list of dictionaries to structured result collection.

    Args:
        data: List of dictionaries to convert
        default_status: Default status to use if item doesn't have a "status" field.
            Valid values are "success", "failure", "pending", "partial", "cancelled".

    Returns:
        ModelResultCollection with converted items
    """
    collection = ModelResultCollection()

    for i, item in enumerate(data):
        # Convert dict to structured data
        structured_data = ModelStructuredData()
        for key, value in item.items():
            if key != "status":  # Don't include status in structured data
                structured_data.set_field_value(key, str(value))

        # Use status from item if present, otherwise use default_status
        item_status = item.get("status")
        status = item_status if isinstance(item_status, str) else default_status

        collection.add_result(
            id=str(i),
            status=status,
            message=f"Converted item {i}",
            data=structured_data,
        )

    return collection
