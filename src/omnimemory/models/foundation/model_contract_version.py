"""
Contract version model for tracking model schema versions.

This module provides a reusable contract version field that can be added
to models requiring schema version tracking for ONEX compliance.
"""

from pydantic import BaseModel, ConfigDict, Field

from .model_semver import ModelSemVer

# Default contract version for omnimemory models
DEFAULT_CONTRACT_VERSION = "1.0.0"


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
        print(request.contract_version)  # "1.0.0"
    """

    model_config = ConfigDict(extra="forbid", frozen=False)

    contract_version: str = Field(
        default=DEFAULT_CONTRACT_VERSION,
        description="Schema version for this contract (semver format)",
    )

    def get_semver(self) -> ModelSemVer:
        """
        Parse contract version as semantic version.

        Returns:
            ModelSemVer instance for version comparison operations
        """
        return ModelSemVer.from_string(self.contract_version)

    def is_compatible_with(self, other_version: str) -> bool:
        """
        Check if this contract version is compatible with another.

        Compatibility is determined by major version equality (semver rules).

        Args:
            other_version: Version string to compare against

        Returns:
            True if versions are compatible (same major version)
        """
        self_semver = self.get_semver()
        other_semver = ModelSemVer.from_string(other_version)
        return self_semver.is_compatible_with(other_semver)


# Type alias for backward compatibility with omnibase_core naming
ContractVersionMixin = ModelContractVersion
