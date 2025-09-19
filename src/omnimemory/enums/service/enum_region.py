"""
Deployment region enum for OmniMemory following ONEX standards.

Defines deployment regions for cloud services and geographic distribution.
"""

from enum import Enum


class EnumRegion(str, Enum):
    """Deployment regions for ONEX services."""

    # AWS Regions
    US_WEST_1 = "us-west-1"
    US_WEST_2 = "us-west-2"
    US_EAST_1 = "us-east-1"
    US_EAST_2 = "us-east-2"
    EU_WEST_1 = "eu-west-1"
    EU_WEST_2 = "eu-west-2"
    EU_CENTRAL_1 = "eu-central-1"
    AP_SOUTHEAST_1 = "ap-southeast-1"
    AP_SOUTHEAST_2 = "ap-southeast-2"
    AP_NORTHEAST_1 = "ap-northeast-1"

    # Azure Regions
    AZURE_WEST_US = "azure-west-us"
    AZURE_EAST_US = "azure-east-us"
    AZURE_NORTH_EUROPE = "azure-north-europe"
    AZURE_WEST_EUROPE = "azure-west-europe"

    # Google Cloud Regions
    GCP_US_CENTRAL1 = "gcp-us-central1"
    GCP_US_WEST1 = "gcp-us-west1"
    GCP_EUROPE_WEST1 = "gcp-europe-west1"
    GCP_ASIA_EAST1 = "gcp-asia-east1"

    # Local/Edge Regions
    LOCAL = "local"
    EDGE_US = "edge-us"
    EDGE_EU = "edge-eu"
    EDGE_ASIA = "edge-asia"

    def __str__(self) -> str:
        """Return string representation."""
        return self.value

    @classmethod
    def default(cls) -> "EnumRegion":
        """Return default region."""
        return cls.US_WEST_2

    @property
    def cloud_provider(self) -> str:
        """Get the cloud provider for this region."""
        if self.value.startswith("azure-"):
            return "Azure"
        elif self.value.startswith("gcp-"):
            return "Google Cloud"
        elif self.value.startswith("edge-") or self.value == "local":
            return "Edge/Local"
        else:
            return "AWS"

    @property
    def is_us_region(self) -> bool:
        """Check if region is in the United States."""
        us_regions = {
            self.US_WEST_1,
            self.US_WEST_2,
            self.US_EAST_1,
            self.US_EAST_2,
            self.AZURE_WEST_US,
            self.AZURE_EAST_US,
            self.GCP_US_CENTRAL1,
            self.GCP_US_WEST1,
            self.EDGE_US,
        }
        return self in us_regions

    @property
    def is_eu_region(self) -> bool:
        """Check if region is in Europe."""
        eu_regions = {
            self.EU_WEST_1,
            self.EU_WEST_2,
            self.EU_CENTRAL_1,
            self.AZURE_NORTH_EUROPE,
            self.AZURE_WEST_EUROPE,
            self.GCP_EUROPE_WEST1,
            self.EDGE_EU,
        }
        return self in eu_regions

    @property
    def supports_high_availability(self) -> bool:
        """Check if region supports high availability deployments."""
        ha_regions = {
            self.US_WEST_2,
            self.US_EAST_1,
            self.EU_WEST_1,
            self.EU_CENTRAL_1,
            self.AP_SOUTHEAST_1,
            self.AZURE_WEST_US,
            self.AZURE_EAST_US,
            self.GCP_US_CENTRAL1,
            self.GCP_EUROPE_WEST1,
        }
        return self in ha_regions
