"""
Contract version model for tracking model schema versions.

This module provides a reusable contract version field that can be added
to models requiring schema version tracking for ONEX compliance.
"""

from pydantic import BaseModel, ConfigDict, Field

from .model_semver import ModelSemVer

# Default contract version for omnimemory models
DEFAULT_CONTRACT_VERSION = ModelSemVer(major=1, minor=0, patch=0)


class ModelContractVersion(BaseModel):
    """
    Contract version tracking for ONEX models.

    Provides explicit version tracking for model schemas to support:
    - Schema evolution and migration
    - Backward compatibility checks
    - API versioning
    - Serialization/deserialization validation

    Example:
        class MyRequest(ModelContractVersion):
            # Will have contract_version field automatically
            data: str

        request = MyRequest(data="test")
        print(request.contract_version)  # ModelSemVer(major=1, minor=0, patch=0)
    """

    model_config = ConfigDict(extra="forbid", frozen=False)

    contract_version: ModelSemVer = Field(
        default=DEFAULT_CONTRACT_VERSION,
        description="Schema version for this contract as ModelSemVer",
    )

    def is_compatible_with(self, other_version: ModelSemVer) -> bool:
        """
        Check if this contract version is compatible with another.

        Compatibility is determined by major version equality (semver rules).

        Args:
            other_version: ModelSemVer to compare against

        Returns:
            True if versions are compatible (same major version)
        """
        return self.contract_version.is_compatible_with(other_version)


# Type alias for backward compatibility with omnibase_core naming
ContractVersionMixin = ModelContractVersion
