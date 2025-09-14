"""
User model following ONEX foundation patterns.
"""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class ModelUser(BaseModel):
    """User model with comprehensive identity and authorization information."""

    user_id: UUID = Field(
        description="Unique user identifier",
    )
    username: str = Field(
        description="Human-readable username",
        min_length=1,
        max_length=100,
    )
    email: str | None = Field(
        default=None,
        description="User email address for notifications and identity",
    )
    display_name: str | None = Field(
        default=None,
        description="Display name for user interface",
    )

    # Authorization and access
    roles: list[str] = Field(
        default_factory=list,
        description="User roles for authorization",
    )
    permissions: list[str] = Field(
        default_factory=list,
        description="Specific permissions granted to user",
    )

    # Metadata
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the user was created",
    )
    last_active: datetime | None = Field(
        default=None,
        description="When the user was last active",
    )
    is_active: bool = Field(
        default=True,
        description="Whether the user account is active",
    )

    # Additional attributes
    attributes: dict[str, str] = Field(
        default_factory=dict,
        description="Additional user attributes for customization",
    )

    def has_role(self, role: str) -> bool:
        """Check if user has specific role."""
        return role in self.roles

    def has_permission(self, permission: str) -> bool:
        """Check if user has specific permission."""
        return permission in self.permissions

    def add_role(self, role: str) -> None:
        """Add role to user."""
        if role not in self.roles:
            self.roles.append(role)

    def remove_role(self, role: str) -> None:
        """Remove role from user."""
        if role in self.roles:
            self.roles.remove(role)

    def update_last_active(self) -> None:
        """Update last active timestamp to now."""
        self.last_active = datetime.utcnow()

    @classmethod
    def create_system_user(cls) -> "ModelUser":
        """Create a system user for automated operations."""
        from uuid import uuid4

        return cls(
            user_id=uuid4(),
            username="system",
            display_name="System User",
            roles=["system", "admin"],
            permissions=["system.all"],
            attributes={"type": "system", "automated": "true"}
        )

    @classmethod
    def create_anonymous_user(cls) -> "ModelUser":
        """Create an anonymous user for unauthenticated operations."""
        from uuid import uuid4

        return cls(
            user_id=uuid4(),
            username="anonymous",
            display_name="Anonymous User",
            roles=["anonymous"],
            permissions=["read"],
            attributes={"type": "anonymous", "temporary": "true"}
        )