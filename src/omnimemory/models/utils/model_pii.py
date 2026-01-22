"""
PII detection Pydantic models for OmniMemory ONEX architecture.

This module contains models for PII detection configuration and results.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

__all__ = [
    "ModelPIIDetectionResult",
    "ModelPIIDetectorConfig",
    "ModelPIIMatch",
    "ModelPIIPatternConfig",
    "PIIType",
    # Backward compatibility aliases
    "PIIDetectionResult",
    "PIIDetectorConfig",
    "PIIMatch",
    "PIIPatternConfig",
]


class PIIType(str, Enum):
    """Types of PII that can be detected.

    Note: Not all types have detection patterns implemented. See implementation
    status below:

    Implemented:
    - EMAIL: Regex-based email detection
    - PHONE: US/International phone number patterns
    - SSN: Social Security Number patterns with validation
    - CREDIT_CARD: Major card formats (Visa, Mastercard, Amex)
    - IP_ADDRESS: IPv4 and IPv6 patterns
    - API_KEY: Common API key formats (OpenAI, GitHub, Google, AWS)
    - PASSWORD_HASH: Password field detection

    TODO - Needs Implementation:
    - URL: Web URL pattern detection (requires URL validation patterns)
    - PERSON_NAME: Dictionary-based + NLP detection (requires expanded name database)
    - ADDRESS: Physical address detection (requires geocoding or NLP integration)
    """

    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    IP_ADDRESS = "ip_address"
    URL = "url"  # TODO: Implement URL detection patterns
    API_KEY = "api_key"
    PASSWORD_HASH = "password_hash"  # noqa: S105  # Not a password - PII type enum value
    PERSON_NAME = "person_name"  # TODO: Implement dictionary-based + NLP name detection
    ADDRESS = "address"  # TODO: Implement address detection with geocoding/NLP


class ModelPIIMatch(BaseModel):
    """A detected PII match in content."""

    pii_type: PIIType = Field(description="Type of PII detected")
    value: str = Field(description="The detected PII value (may be masked)")
    start_index: int = Field(description="Start position in the content")
    end_index: int = Field(description="End position in the content")
    confidence: float = Field(description="Confidence score (0.0-1.0)")
    masked_value: str = Field(description="Masked version of the detected value")


class ModelPIIDetectionResult(BaseModel):
    """Result of PII detection scan."""

    has_pii: bool = Field(description="Whether any PII was detected")
    matches: list[ModelPIIMatch] = Field(
        default_factory=list, description="List of PII matches found"
    )
    sanitized_content: str = Field(description="Content with PII masked/removed")
    pii_types_detected: set[PIIType] = Field(
        default_factory=set, description="Types of PII found"
    )
    scan_duration_ms: float = Field(
        description="Time taken for the scan in milliseconds"
    )


class ModelPIIDetectorConfig(BaseModel):
    """Configuration for PII detection with extracted magic numbers."""

    # Confidence thresholds
    high_confidence: float = Field(
        default=0.98, ge=0.0, le=1.0, description="High confidence threshold"
    )
    medium_high_confidence: float = Field(
        default=0.95, ge=0.0, le=1.0, description="Medium-high confidence threshold"
    )
    medium_confidence: float = Field(
        default=0.90, ge=0.0, le=1.0, description="Medium confidence threshold"
    )
    reduced_confidence: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="Reduced confidence for complex patterns",
    )
    low_confidence: float = Field(
        default=0.60, ge=0.0, le=1.0, description="Low confidence threshold"
    )

    # Pattern matching limits
    max_text_length: int = Field(
        default=50000, ge=1000, description="Maximum text length to analyze"
    )
    max_matches_per_type: int = Field(
        default=100, ge=1, description="Maximum matches per PII type"
    )

    # Context analysis settings
    enable_context_analysis: bool = Field(
        default=True, description="Enable context-aware detection"
    )
    context_window_size: int = Field(
        default=50, ge=10, le=200, description="Context analysis window size"
    )


class ModelPIIPatternConfig(BaseModel):
    """Strongly typed PII pattern configuration replacing Dict[str, Any]."""

    pattern: str = Field(description="Regex pattern for PII detection")
    confidence: float = Field(
        ge=0.0, le=1.0, description="Base confidence score for matches"
    )
    mask_template: str = Field(description="Template for masking detected values")


# Backward compatibility aliases
PIIMatch = ModelPIIMatch
PIIDetectionResult = ModelPIIDetectionResult
PIIDetectorConfig = ModelPIIDetectorConfig
PIIPatternConfig = ModelPIIPatternConfig
