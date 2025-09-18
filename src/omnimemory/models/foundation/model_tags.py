"""
Tags model following ONEX standards.
"""

from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


class ModelTag(BaseModel):
    """Individual tag model with metadata."""

    name: str = Field(
        description="Tag name",
        min_length=1,
        max_length=100,
    )
    category: Optional[str] = Field(
        default=None,
        description="Optional tag category for organization",
        max_length=50,
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the tag was created",
    )
    created_by: Optional[UUID] = Field(
        default=None,
        description="User who created the tag",
    )
    weight: float = Field(
        default=1.0,
        ge=0.0,
        le=10.0,
        description="Tag importance weight",
    )

    @field_validator("name")
    @classmethod
    def validate_tag_name(cls, v: str) -> str:
        """Validate tag name format."""
        # Remove whitespace and convert to lowercase
        v = v.strip().lower()

        # Check for invalid characters
        invalid_chars = set("!@#$%^&*()+={}[]|\\:\";'<>?,/`~")
        if any(char in v for char in invalid_chars):
            raise ValueError(f"Tag name contains invalid characters: {v}")

        # Replace spaces with underscores
        v = v.replace(" ", "_").replace("-", "_")

        return v


class ModelTagCollection(BaseModel):
    """Collection of tags with validation and management."""

    tags: list[ModelTag] = Field(
        default_factory=list,
        description="Collection of tags",
        max_length=100,  # Maximum 100 tags
    )
    auto_generated: bool = Field(
        default=False,
        description="Whether tags were auto-generated",
    )
    last_updated: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the tag collection was last updated",
    )

    @field_validator("tags")
    @classmethod
    def validate_unique_tags(cls, v: list["ModelTag"]) -> list["ModelTag"]:
        """Ensure tag names are unique."""
        tag_names = [tag.name for tag in v]
        if len(tag_names) != len(set(tag_names)):
            raise ValueError("Duplicate tag names are not allowed")
        return v

    def add_tag(
        self,
        name: str,
        category: Optional[str] = None,
        weight: float = 1.0,
        created_by: Optional[UUID] = None,
    ) -> None:
        """Add a new tag to the collection."""
        # Check if tag already exists
        if any(
            tag.name == name.strip().lower().replace(" ", "_").replace("-", "_")
            for tag in self.tags
        ):
            return  # Tag already exists, skip

        new_tag = ModelTag(
            name=name,
            category=category,
            weight=weight,
            created_by=created_by,
        )
        self.tags.append(new_tag)
        self.last_updated = datetime.utcnow()

    def remove_tag(self, name: str) -> bool:
        """Remove a tag by name."""
        normalized_name = name.strip().lower().replace(" ", "_").replace("-", "_")
        for i, tag in enumerate(self.tags):
            if tag.name == normalized_name:
                self.tags.pop(i)
                self.last_updated = datetime.utcnow()
                return True
        return False

    def get_tag_names(self) -> list[str]:
        """Get list of tag names."""
        return [tag.name for tag in self.tags]

    def get_tags_by_category(self, category: str) -> list[ModelTag]:
        """Get tags filtered by category."""
        return [tag for tag in self.tags if tag.category == category]

    def get_weighted_tags(self) -> list[tuple[str, float]]:
        """Get tags with their weights."""
        return [(tag.name, tag.weight) for tag in self.tags]

    @classmethod
    def from_string_list(
        cls, tag_names: list[str], created_by: Optional[UUID] = None
    ) -> "ModelTagCollection":
        """Create tag collection from legacy string list."""
        collection = cls()
        for name in tag_names:
            collection.add_tag(name, created_by=created_by)
        return collection
