"""
Priority model following ONEX foundation patterns.
"""

from datetime import datetime

from omnibase_core.enums.enum_priority_level import EnumPriorityLevel
from pydantic import BaseModel, Field


class ModelPriority(BaseModel):
    """Priority model with level, context, and metadata."""

    level: EnumPriorityLevel = Field(
        description="Priority level using ONEX standard enum",
    )
    reason: str | None = Field(
        default=None,
        description="Reason for this priority level",
    )
    expires_at: datetime | None = Field(
        default=None,
        description="When this priority expires (for temporary high priority)",
    )
    boost_factor: float = Field(
        default=1.0,
        ge=0.1,
        le=10.0,
        description="Priority boost factor for fine-tuning (1.0 = normal)",
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When this priority was set",
    )
    created_by: str | None = Field(
        default=None,
        description="Who or what set this priority",
    )

    # Context and categorization
    category: str | None = Field(
        default=None,
        description="Priority category (e.g., 'user_request', 'system_maintenance')",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Tags for categorizing priority context",
    )

    def is_expired(self) -> bool:
        """Check if priority has expired."""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at

    def get_effective_priority(self) -> float:
        """Get effective priority value considering boost and expiration."""
        if self.is_expired():
            # If expired, fallback to normal priority
            base_priority = EnumPriorityLevel.NORMAL.get_numeric_value()
        else:
            base_priority = self.level.get_numeric_value()

        return base_priority * self.boost_factor

    def is_high_priority(self) -> bool:
        """Check if this is high priority."""
        return self.level.is_high_priority() and not self.is_expired()

    def requires_immediate_action(self) -> bool:
        """Check if this requires immediate action."""
        return self.level.requires_immediate_action() and not self.is_expired()

    def add_tag(self, tag: str) -> None:
        """Add a tag to this priority."""
        if tag not in self.tags:
            self.tags.append(tag)

    def has_tag(self, tag: str) -> bool:
        """Check if priority has a specific tag."""
        return tag in self.tags

    @classmethod
    def create_normal(cls, reason: str | None = None) -> "ModelPriority":
        """Create normal priority."""
        return cls(level=EnumPriorityLevel.NORMAL, reason=reason, category="standard")

    @classmethod
    def create_high(cls, reason: str, created_by: str | None = None) -> "ModelPriority":
        """Create high priority with reason."""
        return cls(
            level=EnumPriorityLevel.HIGH,
            reason=reason,
            created_by=created_by,
            category="high_priority",
            tags=["high", "attention_required"],
        )

    @classmethod
    def create_critical(
        cls,
        reason: str,
        created_by: str | None = None,
        expires_in_minutes: int | None = None,
    ) -> "ModelPriority":
        """Create critical priority with optional expiration."""
        expires_at = None
        if expires_in_minutes:
            from datetime import timedelta

            expires_at = datetime.utcnow() + timedelta(minutes=expires_in_minutes)

        return cls(
            level=EnumPriorityLevel.CRITICAL,
            reason=reason,
            created_by=created_by,
            expires_at=expires_at,
            category="critical",
            tags=["critical", "urgent", "immediate_action"],
        )

    @classmethod
    def create_temporary_boost(
        cls,
        base_level: EnumPriorityLevel,
        boost_factor: float,
        expires_in_minutes: int,
        reason: str,
    ) -> "ModelPriority":
        """Create temporarily boosted priority."""
        from datetime import timedelta

        return cls(
            level=base_level,
            reason=reason,
            boost_factor=boost_factor,
            expires_at=datetime.utcnow() + timedelta(minutes=expires_in_minutes),
            category="temporary_boost",
            tags=["boosted", "temporary"],
        )
