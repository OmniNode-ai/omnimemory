"""
PII Detection utility for memory content security.

Provides comprehensive detection of Personally Identifiable Information (PII)
in memory content to ensure compliance with privacy regulations.
"""

import re
from enum import Enum
from typing import Any, Dict, List, Set

from pydantic import BaseModel, Field


class PIIType(str, Enum):
    """Types of PII that can be detected."""

    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    IP_ADDRESS = "ip_address"
    URL = "url"
    API_KEY = "api_key"
    PASSWORD_HASH = "password_hash"
    PERSON_NAME = "person_name"
    ADDRESS = "address"


class PIIMatch(BaseModel):
    """A detected PII match in content."""

    pii_type: PIIType = Field(description="Type of PII detected")
    value: str = Field(description="The detected PII value (may be masked)")
    start_index: int = Field(description="Start position in the content")
    end_index: int = Field(description="End position in the content")
    confidence: float = Field(description="Confidence score (0.0-1.0)")
    masked_value: str = Field(description="Masked version of the detected value")


class PIIDetectionResult(BaseModel):
    """Result of PII detection scan."""

    has_pii: bool = Field(description="Whether any PII was detected")
    matches: List[PIIMatch] = Field(default_factory=list, description="List of PII matches found")
    sanitized_content: str = Field(description="Content with PII masked/removed")
    pii_types_detected: Set[PIIType] = Field(default_factory=set, description="Types of PII found")
    scan_duration_ms: float = Field(description="Time taken for the scan in milliseconds")


class PIIDetector:
    """Advanced PII detection with configurable patterns and sensitivity levels."""

    def __init__(self):
        """Initialize PII detector with predefined patterns."""
        self._patterns = self._initialize_patterns()
        self._common_names = self._load_common_names()

    def _initialize_patterns(self) -> Dict[PIIType, List[Dict[str, Any]]]:
        """Initialize regex patterns for different PII types."""
        return {
            PIIType.EMAIL: [
                {
                    "pattern": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                    "confidence": 0.95,
                    "mask_template": "***@***.***"
                }
            ],
            PIIType.PHONE: [
                {
                    "pattern": r'(\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
                    "confidence": 0.85,
                    "mask_template": "***-***-****"
                },
                {
                    "pattern": r'\+\d{1,3}[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}',
                    "confidence": 0.80,
                    "mask_template": "+***-***-***"
                }
            ],
            PIIType.SSN: [
                {
                    "pattern": r'\b\d{3}-\d{2}-\d{4}\b',
                    "confidence": 0.98,
                    "mask_template": "***-**-****"
                },
                {
                    "pattern": r'\b(?!(?:000|666|9\d{2}))\d{3}(?!00)\d{2}(?!0000)\d{4}\b',  # More restrictive SSN pattern
                    "confidence": 0.75,  # Reduced false positives with better validation
                    "mask_template": "*********"
                }
            ],
            PIIType.CREDIT_CARD: [
                {
                    "pattern": r'\b4\d{15}\b|\b5[1-5]\d{14}\b|\b3[47]\d{13}\b|\b6011\d{12}\b',
                    "confidence": 0.90,
                    "mask_template": "****-****-****-****"
                }
            ],
            PIIType.IP_ADDRESS: [
                {
                    "pattern": r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
                    "confidence": 0.85,
                    "mask_template": "***.***.***.***"
                },
                {
                    "pattern": r'\b[0-9a-fA-F]{1,4}(:[0-9a-fA-F]{1,4}){7}\b',  # IPv6
                    "confidence": 0.90,
                    "mask_template": "****:****:****:****"
                }
            ],
            PIIType.API_KEY: [
                {
                    "pattern": r'[Aa]pi[_-]?[Kk]ey["\s]*[:=]["\s]*([A-Za-z0-9\-_]{16,})',
                    "confidence": 0.95,
                    "mask_template": "api_key=***REDACTED***"
                },
                {
                    "pattern": r'[Tt]oken["\s]*[:=]["\s]*([A-Za-z0-9\-_]{20,})',
                    "confidence": 0.90,
                    "mask_template": "token=***REDACTED***"
                },
                {
                    "pattern": r'sk-[A-Za-z0-9]{32,}',  # OpenAI API keys
                    "confidence": 0.98,
                    "mask_template": "sk-***REDACTED***"
                },
                {
                    "pattern": r'ghp_[A-Za-z0-9]{36}',  # GitHub personal access tokens
                    "confidence": 0.98,
                    "mask_template": "ghp_***REDACTED***"
                },
                {
                    "pattern": r'AIza[A-Za-z0-9\-_]{35}',  # Google API keys
                    "confidence": 0.98,
                    "mask_template": "AIza***REDACTED***"
                },
                {
                    "pattern": r'AWS[A-Z0-9]{16,}',  # AWS access keys
                    "confidence": 0.95,
                    "mask_template": "AWS***REDACTED***"
                }
            ],
            PIIType.PASSWORD_HASH: [
                {
                    "pattern": r'[Pp]assword["\s]*[:=]["\s]*([A-Za-z0-9\-_\$\.\/]{20,})',
                    "confidence": 0.85,
                    "mask_template": "password=***REDACTED***"
                }
            ]
        }

    def _load_common_names(self) -> Set[str]:
        """Load common first and last names for person name detection."""
        # In a production system, this would load from a more comprehensive database
        return {
            "john", "jane", "michael", "sarah", "david", "jennifer", "robert", "lisa",
            "smith", "johnson", "williams", "brown", "jones", "garcia", "miller", "davis"
        }

    def detect_pii(self, content: str, sensitivity_level: str = "medium") -> PIIDetectionResult:
        """
        Detect PII in the given content.

        Args:
            content: The content to scan for PII
            sensitivity_level: Detection sensitivity ('low', 'medium', 'high')

        Returns:
            PIIDetectionResult with all detected PII and sanitized content
        """
        import time
        start_time = time.time()

        matches: List[PIIMatch] = []
        pii_types_detected: Set[PIIType] = set()
        sanitized_content = content

        # Adjust confidence thresholds based on sensitivity
        confidence_threshold = {
            "low": 0.95,
            "medium": 0.80,
            "high": 0.70
        }.get(sensitivity_level, 0.80)

        # Scan for each PII type
        for pii_type, patterns in self._patterns.items():
            for pattern_config in patterns:
                pattern = pattern_config["pattern"]
                base_confidence = pattern_config["confidence"]
                mask_template = pattern_config["mask_template"]

                # Skip if confidence is below threshold
                if base_confidence < confidence_threshold:
                    continue

                # Find all matches
                for match in re.finditer(pattern, content, re.IGNORECASE):
                    pii_match = PIIMatch(
                        pii_type=pii_type,
                        value=match.group(0),
                        start_index=match.start(),
                        end_index=match.end(),
                        confidence=base_confidence,
                        masked_value=mask_template
                    )
                    matches.append(pii_match)
                    pii_types_detected.add(pii_type)

        # Remove duplicates and sort by position
        matches = self._deduplicate_matches(matches)
        matches.sort(key=lambda x: x.start_index)

        # Create sanitized content
        if matches:
            sanitized_content = self._sanitize_content(content, matches)

        # Calculate scan duration
        scan_duration_ms = (time.time() - start_time) * 1000

        return PIIDetectionResult(
            has_pii=len(matches) > 0,
            matches=matches,
            sanitized_content=sanitized_content,
            pii_types_detected=pii_types_detected,
            scan_duration_ms=scan_duration_ms
        )

    def _deduplicate_matches(self, matches: List[PIIMatch]) -> List[PIIMatch]:
        """Remove overlapping or duplicate matches, keeping the highest confidence ones."""
        if not matches:
            return matches

        # Sort by start position and confidence
        matches.sort(key=lambda x: (x.start_index, -x.confidence))

        deduplicated = []
        for match in matches:
            # Check if this match overlaps with any existing match
            overlap = False
            for existing in deduplicated:
                if (match.start_index < existing.end_index and
                    match.end_index > existing.start_index):
                    overlap = True
                    break

            if not overlap:
                deduplicated.append(match)

        return deduplicated

    def _sanitize_content(self, content: str, matches: List[PIIMatch]) -> str:
        """Replace PII in content with masked values."""
        # Sort matches by start position in reverse order for proper replacement
        sorted_matches = sorted(matches, key=lambda x: x.start_index, reverse=True)

        sanitized = content
        for match in sorted_matches:
            sanitized = (
                sanitized[:match.start_index] +
                match.masked_value +
                sanitized[match.end_index:]
            )

        return sanitized

    def is_content_safe(self, content: str, max_pii_count: int = 0) -> bool:
        """
        Check if content is safe for storage (contains no or minimal PII).

        Args:
            content: Content to check
            max_pii_count: Maximum number of PII items allowed (0 = none)

        Returns:
            True if content is safe, False otherwise
        """
        result = self.detect_pii(content, sensitivity_level="high")
        return len(result.matches) <= max_pii_count