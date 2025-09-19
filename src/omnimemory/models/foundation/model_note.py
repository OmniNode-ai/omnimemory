"""
Individual note model following ONEX standards.
"""

from datetime import datetime
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from ...enums import EnumSeverity


class ModelNote(BaseModel):
    """Individual note entry following ONEX standards."""

    note_id: UUID = Field(
        default_factory=uuid4,
        description="Unique identifier for this note",
    )
    content: str = Field(
        min_length=1,
        description="Content of the note",
    )
    category: str = Field(
        description="Category or type of note (e.g., 'debug', 'performance', 'user_feedback')",
    )
    severity: EnumSeverity = Field(
        default=EnumSeverity.INFO,
        description="Severity level of the note",
    )
    author: str | None = Field(
        default=None,
        description="Author or source of the note",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Tags for categorizing the note",
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the note was created",
    )
    updated_at: datetime | None = Field(
        default=None,
        description="When the note was last updated",
    )
    correlation_id: UUID | None = Field(
        default=None,
        description="Correlation ID for linking related notes",
    )
    metadata: dict[str, str] = Field(
        default_factory=dict,
        description="Additional metadata for the note",
    )
    is_system_generated: bool = Field(
        default=False,
        description="Whether this note was automatically generated",
    )
    is_archived: bool = Field(
        default=False,
        description="Whether this note is archived",
    )

    def archive(self) -> None:
        """Archive this note."""
        self.is_archived = True
        self.updated_at = datetime.utcnow()

    def update_content(self, new_content: str) -> None:
        """Update note content."""
        self.content = new_content
        self.updated_at = datetime.utcnow()

    def add_tag(self, tag: str) -> None:
        """Add a tag to this note."""
        if tag not in self.tags:
            self.tags.append(tag)
            self.updated_at = datetime.utcnow()

    def remove_tag(self, tag: str) -> None:
        """Remove a tag from this note."""
        if tag in self.tags:
            self.tags.remove(tag)
            self.updated_at = datetime.utcnow()
