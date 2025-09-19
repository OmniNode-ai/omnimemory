"""
Strongly typed metadata collection model for ONEX Foundation Architecture.

Provides strongly typed metadata collection to replace Dict[str, Any].
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

from .model_key_value_pair import ModelKeyValuePair


class ModelMetadata(BaseModel):
    """Strongly typed metadata collection replacing Dict[str, Any]."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    pairs: List[ModelKeyValuePair] = Field(
        default_factory=list, description="List of key-value pairs for metadata storage"
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
    def from_dict(
        cls, data: Dict[str, Union[str, int, float, bool]]
    ) -> "ModelMetadata":
        """Create from dictionary, converting values to strings."""
        pairs = [
            ModelKeyValuePair(key=str(k), value=str(v))
            for k, v in data.items()
            if k and v is not None
        ]
        return cls(pairs=pairs)
