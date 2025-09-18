"""
Semantic version model following ONEX standards.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field, field_validator


class ModelSemVer(BaseModel):
    """Semantic version model following ONEX standards."""

    major: int = Field(
        ge=0,
        description="Major version number (breaking changes)",
    )
    minor: int = Field(
        ge=0,
        description="Minor version number (backward compatible features)",
    )
    patch: int = Field(
        ge=0,
        description="Patch version number (backward compatible fixes)",
    )
    pre_release: str | None = Field(
        default=None,
        description="Pre-release identifier (e.g., 'alpha.1', 'beta.2', 'rc.1')",
    )
    build_metadata: str | None = Field(
        default=None,
        description="Build metadata identifier",
    )

    @field_validator("pre_release")
    @classmethod
    def validate_pre_release(cls, v: str | None) -> str | None:
        """Validate pre-release identifier format."""
        if v is None:
            return v
        if not re.match(r"^[0-9A-Za-z-]+(\.[0-9A-Za-z-]+)*$", v):
            raise ValueError("Invalid pre-release identifier format")
        return v

    @field_validator("build_metadata")
    @classmethod
    def validate_build_metadata(cls, v: str | None) -> str | None:
        """Validate build metadata identifier format."""
        if v is None:
            return v
        if not re.match(r"^[0-9A-Za-z-]+(\.[0-9A-Za-z-]+)*$", v):
            raise ValueError("Invalid build metadata identifier format")
        return v

    def __str__(self) -> str:
        """Return string representation of semantic version."""
        version = f"{self.major}.{self.minor}.{self.patch}"
        if self.pre_release:
            version += f"-{self.pre_release}"
        if self.build_metadata:
            version += f"+{self.build_metadata}"
        return version

    def __lt__(self, other: ModelSemVer) -> bool:
        """Compare versions for less than."""
        if not isinstance(other, ModelSemVer):
            return NotImplemented

        # Compare major.minor.patch
        self_core = (self.major, self.minor, self.patch)
        other_core = (other.major, other.minor, other.patch)

        if self_core != other_core:
            return self_core < other_core

        # Handle pre-release comparison
        if self.pre_release is None and other.pre_release is None:
            return False
        if self.pre_release is None:
            return False  # 1.0.0 > 1.0.0-alpha
        if other.pre_release is None:
            return True  # 1.0.0-alpha < 1.0.0

        return self.pre_release < other.pre_release

    def __eq__(self, other: object) -> bool:
        """Compare versions for equality."""
        if not isinstance(other, ModelSemVer):
            return NotImplemented
        return (
            self.major == other.major
            and self.minor == other.minor
            and self.patch == other.patch
            and self.pre_release == other.pre_release
        )

    def __le__(self, other: ModelSemVer) -> bool:
        """Compare versions for less than or equal."""
        return self == other or self < other

    def __gt__(self, other: ModelSemVer) -> bool:
        """Compare versions for greater than."""
        return not self <= other

    def __ge__(self, other: ModelSemVer) -> bool:
        """Compare versions for greater than or equal."""
        return not self < other

    @classmethod
    def from_string(cls, version_string: str) -> ModelSemVer:
        """Create ModelSemVer from string representation."""
        # Regular expression to match semantic version
        pattern = (
            r"^(?P<major>0|[1-9]\d*)\."
            r"(?P<minor>0|[1-9]\d*)\."
            r"(?P<patch>0|[1-9]\d*)"
            r"(?:-(?P<prerelease>[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?"
            r"(?:\+(?P<buildmetadata>[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?$"
        )

        match = re.match(pattern, version_string.strip())
        if not match:
            raise ValueError(f"Invalid semantic version format: {version_string}")

        return cls(
            major=int(match.group("major")),
            minor=int(match.group("minor")),
            patch=int(match.group("patch")),
            pre_release=match.group("prerelease"),
            build_metadata=match.group("buildmetadata"),
        )

    def increment_major(self) -> ModelSemVer:
        """Create new version with incremented major version."""
        return ModelSemVer(
            major=self.major + 1,
            minor=0,
            patch=0,
        )

    def increment_minor(self) -> ModelSemVer:
        """Create new version with incremented minor version."""
        return ModelSemVer(
            major=self.major,
            minor=self.minor + 1,
            patch=0,
        )

    def increment_patch(self) -> ModelSemVer:
        """Create new version with incremented patch version."""
        return ModelSemVer(
            major=self.major,
            minor=self.minor,
            patch=self.patch + 1,
        )

    def is_stable(self) -> bool:
        """Check if this is a stable release version."""
        return self.pre_release is None

    def is_compatible_with(self, other: ModelSemVer) -> bool:
        """Check if this version is compatible with another (same major version)."""
        return self.major == other.major and self >= other
