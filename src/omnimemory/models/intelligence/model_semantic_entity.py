# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Semantic entity model following ONEX standards.

Represents a named entity extracted from natural language content with
full type information, confidence scores, and span positions.
"""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ...enums.enum_semantic_entity_type import EnumSemanticEntityType  # noqa: TC001


class ModelSemanticEntity(BaseModel):
    """A named entity extracted from text content.

    Provides strongly-typed entity representation with confidence scoring
    and character span positions for precise localization.

    Example:
        >>> entity = ModelSemanticEntity(
        ...     entity_type=EnumSemanticEntityType.PERSON,
        ...     text="Albert Einstein",
        ...     confidence=0.95,
        ...     span_start=10,
        ...     span_end=25,
        ... )
        >>> entity.entity_type.value
        'person'
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    entity_type: EnumSemanticEntityType = Field(
        description="The type/category of the extracted entity",
    )

    text: str = Field(
        min_length=1,
        description="The actual text of the entity as it appears in the source",
    )

    normalized_text: str | None = Field(
        default=None,
        description="Normalized/canonical form of the entity (e.g., 'USA' -> 'United States')",
    )

    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score for the entity extraction (0.0 to 1.0)",
    )

    span_start: int = Field(
        ge=0,
        description="Starting character position in the source text (0-indexed)",
    )

    span_end: int = Field(
        ge=0,
        description="Ending character position in the source text (exclusive)",
    )

    sentence_index: int | None = Field(
        default=None,
        ge=0,
        description="Index of the sentence containing this entity (0-indexed)",
    )

    metadata: dict[str, str] = Field(
        default_factory=dict,
        description="Additional metadata about the entity (e.g., wikidata_id, dbpedia_uri)",
    )

    @model_validator(mode="after")
    def validate_span_ordering(self) -> Self:
        """Validate span ordering and that span length matches text length."""
        if self.span_end <= self.span_start:
            raise ValueError(
                f"span_end ({self.span_end}) must be greater than span_start ({self.span_start})"
            )
        span_length = self.span_end - self.span_start
        text_length = len(self.text)
        if span_length != text_length:
            raise ValueError(
                f"span length ({span_length}) must equal text length ({text_length}); "
                f"span_start={self.span_start}, span_end={self.span_end}, text={self.text!r}"
            )
        return self

    @property
    def span_length(self) -> int:
        """Return the length of the entity span in characters."""
        return self.span_end - self.span_start

    def overlaps_with(self, other: ModelSemanticEntity) -> bool:
        """Check if this entity's span overlaps with another entity's span."""
        return not (
            self.span_end <= other.span_start or other.span_end <= self.span_start
        )

    def contains(self, other: ModelSemanticEntity) -> bool:
        """Check if this entity's span fully contains another entity's span."""
        return self.span_start <= other.span_start and self.span_end >= other.span_end
