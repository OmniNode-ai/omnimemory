# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Semantic entity list container model following ONEX standards.

Provides a container for lists of semantic entities with aggregate metadata
and convenience methods for filtering and accessing entities.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from ...enums.enum_semantic_entity_type import EnumSemanticEntityType  # noqa: TC001
from .model_semantic_entity import ModelSemanticEntity  # noqa: TC001


class ModelSemanticEntityList(BaseModel):
    """Container for a list of semantic entities with aggregate metadata.

    Provides convenience methods for filtering and accessing entities.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    entities: list[ModelSemanticEntity] = Field(
        default_factory=list,
        description="List of extracted semantic entities",
    )

    source_text_length: int = Field(
        ge=0,
        description="Length of the source text in characters",
    )

    extraction_model: str = Field(
        description="Name/identifier of the model used for extraction",
    )

    is_deterministic: bool = Field(
        default=True,
        description="Whether the extraction used deterministic mode",
    )

    def filter_by_type(
        self, entity_type: EnumSemanticEntityType
    ) -> list[ModelSemanticEntity]:
        """Return entities matching the specified type."""
        return [e for e in self.entities if e.entity_type == entity_type]

    def filter_by_confidence(self, min_confidence: float) -> list[ModelSemanticEntity]:
        """Return entities with confidence >= min_confidence."""
        return [e for e in self.entities if e.confidence >= min_confidence]

    @property
    def entity_count(self) -> int:
        """Return the total number of entities."""
        return len(self.entities)

    @property
    def unique_types(self) -> set[EnumSemanticEntityType]:
        """Return the set of unique entity types found."""
        return {e.entity_type for e in self.entities}
