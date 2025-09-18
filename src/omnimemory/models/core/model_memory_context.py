"""
Memory context model following ONEX standards.
"""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field

from ...enums.enum_node_type import EnumNodeType
from ..foundation.model_priority import ModelPriority
from ..foundation.model_tags import ModelTagCollection
from ..foundation.model_trust_score import ModelTrustScore
from ..foundation.model_user import ModelUser


class ModelMemoryContext(BaseModel):
    """Context information for memory operations following ONEX standards."""

    correlation_id: UUID = Field(
        description="Unique correlation identifier for tracing operations",
    )
    session_id: UUID | None = Field(
        default=None,
        description="Session identifier for grouping related operations",
    )
    user: ModelUser | None = Field(
        default=None,
        description="User information for authorization and personalization",
    )

    # ONEX node information
    source_node_type: EnumNodeType = Field(
        description="Type of ONEX node initiating the operation",
    )
    source_node_id: UUID = Field(
        description="Identifier of the source node",
    )

    # Operation metadata
    operation_timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp when the operation was initiated",
    )
    timeout_ms: int = Field(
        default=30000,
        description="Timeout for the operation in milliseconds",
    )
    priority: ModelPriority = Field(
        default_factory=lambda: ModelPriority.create_normal(
            "Default operation priority"
        ),
        description="Operation priority with comprehensive metadata",
    )

    # Context tags and metadata
    tags: ModelTagCollection = Field(
        default_factory=ModelTagCollection,
        description="Tags for categorizing and filtering operations with metadata",
    )
    metadata: dict[str, str] = Field(
        default_factory=dict,
        description="Additional metadata for the operation",
    )

    # Trust and validation
    trust_score: ModelTrustScore = Field(
        default_factory=lambda: ModelTrustScore.create_from_float(1.0),
        description="Trust score with time decay and comprehensive validation",
    )
    validation_required: bool = Field(
        default=False,
        description="Whether the operation requires additional validation",
    )

    # Helper methods for working with typed models

    def get_effective_user_id(self) -> UUID | None:
        """Get user ID from the user model."""
        if self.user:
            return self.user.user_id
        return None

    def get_user_display_name(self) -> str:
        """Get display name for the user."""
        if self.user:
            return self.user.display_name or self.user.username
        return "Unknown User"

    def get_effective_priority_score(self) -> float:
        """Get numeric priority score considering boosts and expiration."""
        return self.priority.get_effective_priority()

    def is_high_priority(self) -> bool:
        """Check if this context has high priority."""
        return self.priority.is_high_priority()

    def get_current_trust_score(self) -> float:
        """Get current trust score with time decay applied."""
        self.trust_score.refresh_current_score()
        return self.trust_score.current_score

    def add_context_tag(self, tag_name: str, category: str | None = None) -> bool:
        """Add a tag to the context."""
        return self.tags.add_tag(tag_name, category=category)

    def has_context_tag(self, tag_name: str) -> bool:
        """Check if context has a specific tag."""
        return self.tags.has_tag(tag_name)

    def get_tag_names(self) -> list[str]:
        """Get list of all tag names."""
        return self.tags.get_tag_names()

    @classmethod
    def create_for_user(
        cls,
        user: ModelUser,
        source_node_type: EnumNodeType,
        source_node_id: UUID,
        correlation_id: UUID | None = None,
        priority_level: str = "normal",
    ) -> "ModelMemoryContext":
        """Factory method to create context for a specific user."""
        from uuid import uuid4

        if correlation_id is None:
            correlation_id = uuid4()

        # Create appropriate priority
        if priority_level == "high":
            priority = ModelPriority.create_high("User operation", user.username)
        elif priority_level == "critical":
            priority = ModelPriority.create_critical(
                "Critical user operation", user.username
            )
        else:
            priority = ModelPriority.create_normal("User operation")

        return cls(
            correlation_id=correlation_id,
            user=user,
            source_node_type=source_node_type,
            source_node_id=source_node_id,
            priority=priority,
        )

    @classmethod
    def create_system_context(
        cls,
        source_node_type: EnumNodeType,
        source_node_id: UUID,
        correlation_id: UUID | None = None,
    ) -> "ModelMemoryContext":
        """Factory method to create system context."""
        from uuid import uuid4

        if correlation_id is None:
            correlation_id = uuid4()

        system_user = ModelUser.create_system_user()
        priority = ModelPriority.create_normal("System operation")

        context = cls(
            correlation_id=correlation_id,
            user=system_user,
            source_node_type=source_node_type,
            source_node_id=source_node_id,
            priority=priority,
        )

        # Add system tags
        context.add_context_tag("system", "source")
        context.add_context_tag("automated", "source")

        return context
