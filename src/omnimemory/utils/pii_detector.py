# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""
PII Detection utility for memory content security.

Provides comprehensive detection of Personally Identifiable Information (PII)
in memory content to ensure compliance with privacy regulations.
"""

import re

from ..models.utils import (
    ModelPIIDetectionResult,
    ModelPIIDetectorConfig,
    ModelPIIMatch,
    ModelPIIPatternConfig,
    PIIType,
)

__all__ = [
    "ModelPIIDetectionResult",
    "ModelPIIDetectorConfig",
    "ModelPIIMatch",
    "ModelPIIPatternConfig",
    "PIIDetector",
    "PIIType",
]


class PIIDetector:
    """Advanced PII detection with configurable patterns and sensitivity levels."""

    def __init__(self, config: ModelPIIDetectorConfig | None = None):
        """Initialize PII detector with configurable settings."""
        self.config = config or ModelPIIDetectorConfig()
        self._patterns = self._initialize_patterns()
        self._compiled_patterns = self._compile_patterns()
        self._common_names = self._load_common_names()

    def _compile_patterns(self) -> dict[PIIType, list[re.Pattern[str]]]:
        """Pre-compile regex patterns for better performance.

        Compiling patterns once during initialization avoids repeated
        compilation overhead in detect_pii().
        """
        compiled: dict[PIIType, list[re.Pattern[str]]] = {}
        for pii_type, pattern_configs in self._patterns.items():
            compiled[pii_type] = [
                re.compile(config.pattern, re.IGNORECASE) for config in pattern_configs
            ]
        return compiled

    def _build_ssn_validation_pattern(self) -> str:
        """
        Build a readable SSN validation regex pattern.

        SSN Format: AAA-GG-SSSS where:
        - AAA (Area): Cannot be 000, 666, or 900-999
        - GG (Group): Cannot be 00
        - SSSS (Serial): Cannot be 0000

        Returns:
            Compiled regex pattern for valid SSN numbers
        """
        # Invalid area codes: 000, 666, 900-999
        invalid_areas = r"(?!(?:000|666|9\d{2}))"
        # Valid area code: 3 digits
        area_code = r"\d{3}"
        # Invalid group: 00
        invalid_group = r"(?!00)"
        # Valid group: 2 digits
        group_code = r"\d{2}"
        # Invalid serial: 0000
        invalid_serial = r"(?!0000)"
        # Valid serial: 4 digits
        serial_code = r"\d{4}"

        # Combine with word boundaries
        pattern = (
            rf"\b{invalid_areas}{area_code}{invalid_group}"
            rf"{group_code}{invalid_serial}{serial_code}\b"
        )
        return pattern

    def _initialize_patterns(  # stub-ok: pii-url-person-address-deferred
        self,
    ) -> dict[PIIType, list[ModelPIIPatternConfig]]:
        """Initialize regex patterns for different PII types using configuration.

        Note: The following PIIType values do NOT have patterns implemented:
        - PIIType.URL: TODO(OMN-5762) - Add URL validation patterns
        - PIIType.PERSON_NAME: TODO(OMN-5762) - Add dictionary-based name matching
        - PIIType.ADDRESS: TODO(OMN-5762) - Add address pattern detection

        These types are defined in PIIType for future extensibility but will
        not match any content until patterns are added.
        """
        return {
            PIIType.EMAIL: [
                ModelPIIPatternConfig(
                    pattern=r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
                    confidence=self.config.medium_high_confidence,
                    mask_template="***@***.***",
                )
            ],
            PIIType.PHONE: [
                ModelPIIPatternConfig(
                    pattern=r"(\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}",
                    confidence=self.config.medium_confidence,
                    mask_template="***-***-****",
                ),
                ModelPIIPatternConfig(
                    pattern=r"\+\d{1,3}[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}",
                    confidence=self.config.reduced_confidence,
                    mask_template="+***-***-***",
                ),
            ],
            PIIType.SSN: [
                ModelPIIPatternConfig(
                    pattern=r"\b\d{3}-\d{2}-\d{4}\b",
                    confidence=self.config.high_confidence,
                    mask_template="***-**-****",
                ),
                ModelPIIPatternConfig(
                    # Improved SSN validation: excludes invalid area codes
                    # Format: (?!invalid_areas)AAA(?!00)GG(?!0000)SSSS
                    pattern=self._build_ssn_validation_pattern(),
                    confidence=self.config.reduced_confidence,
                    mask_template="*********",
                ),
            ],
            PIIType.CREDIT_CARD: [
                ModelPIIPatternConfig(
                    # Implemented: Visa (4xxx), Mastercard (51-55xx), Amex (34xx/37xx)
                    # NOT implemented: Discover (starts with 6011, 65, or 644-649)
                    pattern=r"\b4\d{15}\b|\b5[1-5]\d{14}\b|\b3[47]\d{13}\b",
                    confidence=self.config.medium_confidence,
                    mask_template="****-****-****-****",
                )
            ],
            PIIType.IP_ADDRESS: [
                ModelPIIPatternConfig(
                    # IPv4 address pattern (e.g., 192.168.1.1)  # onex-allow-internal-ip
                    pattern=r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
                    confidence=self.config.medium_confidence,
                    mask_template="***.***.***.***",
                ),
                ModelPIIPatternConfig(
                    # IPv6 full-form only (e.g., 2001:0db8:85a3::8a2e:0370:7334)
                    # Does not match abbreviated forms (e.g., ::1, fe80::1)
                    pattern=r"\b[0-9a-fA-F]{1,4}(:[0-9a-fA-F]{1,4}){7}\b",
                    confidence=self.config.medium_confidence,
                    mask_template="****:****:****:****",
                ),
            ],
            PIIType.API_KEY: [
                ModelPIIPatternConfig(
                    pattern=r'[Aa]pi[_-]?[Kk]ey["\s]*[:=]["\s]*([A-Za-z0-9\-_]{16,})',
                    confidence=self.config.medium_high_confidence,
                    mask_template="api_key=***REDACTED***",
                ),
                ModelPIIPatternConfig(
                    pattern=r'[Tt]oken["\s]*[:=]["\s]*([A-Za-z0-9\-_]{20,})',
                    confidence=self.config.medium_confidence,
                    mask_template="token=***REDACTED***",
                ),
                ModelPIIPatternConfig(
                    pattern=r"sk-[A-Za-z0-9]{32,}",  # OpenAI API keys
                    confidence=self.config.high_confidence,
                    mask_template="sk-***REDACTED***",
                ),
                ModelPIIPatternConfig(
                    pattern=r"ghp_[A-Za-z0-9]{36}",  # GitHub personal access tokens
                    confidence=self.config.high_confidence,
                    mask_template="ghp_***REDACTED***",
                ),
                ModelPIIPatternConfig(
                    pattern=r"AIza[A-Za-z0-9\-_]{35}",  # Google API keys
                    confidence=self.config.high_confidence,
                    mask_template="AIza***REDACTED***",
                ),
                ModelPIIPatternConfig(
                    # AWS access key IDs - broad pattern may have false positives
                    # Real keys use prefixes: AKIA (IAM), ASIA (STS), AIDA (user ID)
                    # Consider stricter pattern: r"A[SK]IA[A-Z0-9]{16}" for fewer FPs
                    pattern=r"AWS[A-Z0-9]{16,}",
                    confidence=self.config.medium_high_confidence,
                    mask_template="AWS***REDACTED***",
                ),
            ],
            PIIType.PASSWORD_HASH: [
                ModelPIIPatternConfig(
                    pattern=r'[Pp]assword["\s]*[:=]["\s]*([A-Za-z0-9\-_\$\.\/]{20,})',
                    confidence=self.config.medium_confidence,
                    mask_template="password=***REDACTED***",
                )
            ],
        }

    def _load_common_names(  # stub-ok: pii-person-name-detection-deferred
        self,
    ) -> set[str]:
        """Load common first and last names for person name detection.

        TODO(OMN-5762): This name database is loaded but not actively used for detection.
        To enable PERSON_NAME detection:
        1. Add PIIType.PERSON_NAME patterns to _initialize_patterns()
        2. Or implement context-aware NLP-based name detection
        3. Consider loading from a comprehensive name database file

        In a production system, this would load from a more comprehensive database
        such as the US Census Bureau name frequency lists.
        """
        return {
            "john",
            "jane",
            "michael",
            "sarah",
            "david",
            "jennifer",
            "robert",
            "lisa",
            "smith",
            "johnson",
            "williams",
            "brown",
            "jones",
            "garcia",
            "miller",
            "davis",
        }

    def detect_pii(
        self, content: str, sensitivity_level: str = "medium"
    ) -> ModelPIIDetectionResult:
        """
        Detect PII in the given content.

        Args:
            content: The content to scan for PII
            sensitivity_level: Detection sensitivity ('low', 'medium', 'high')

        Returns:
            ModelPIIDetectionResult with all detected PII and sanitized content
        """
        import time

        start_time = time.time()

        # Check content length against configuration limit
        if len(content) > self.config.max_text_length:
            max_len = self.config.max_text_length
            msg = f"Content length {len(content)} exceeds max {max_len}"
            raise ValueError(msg)

        matches: list[ModelPIIMatch] = []
        pii_types_detected: set[PIIType] = set()
        sanitized_content = content

        # Adjust confidence thresholds based on sensitivity
        confidence_threshold = {
            "low": self.config.medium_high_confidence,  # 0.95 - stricter
            "medium": self.config.reduced_confidence,  # 0.75 - balanced
            "high": self.config.low_confidence,  # 0.60 - permissive
        }.get(sensitivity_level, self.config.reduced_confidence)

        # Scan for each PII type using pre-compiled patterns
        for pii_type, pattern_configs in self._patterns.items():
            compiled_patterns = self._compiled_patterns[pii_type]
            matches_for_type = 0
            for idx, pattern_config in enumerate(pattern_configs):
                compiled_pattern = compiled_patterns[idx]
                base_confidence = pattern_config.confidence
                mask_template = pattern_config.mask_template

                # Skip if confidence is below threshold
                if base_confidence < confidence_threshold:
                    continue

                # Find all matches with per-type limit using pre-compiled pattern
                for match in compiled_pattern.finditer(content):
                    if matches_for_type >= self.config.max_matches_per_type:
                        break  # Prevent excessive matches for any single PII type

                    pii_match = ModelPIIMatch(
                        pii_type=pii_type,
                        value=match.group(0),
                        start_index=match.start(),
                        end_index=match.end(),
                        confidence=base_confidence,
                        masked_value=mask_template,
                    )
                    matches.append(pii_match)
                    pii_types_detected.add(pii_type)
                    matches_for_type += 1

        # Remove duplicates and sort by position
        matches = self._deduplicate_matches(matches)
        matches.sort(key=lambda x: x.start_index)

        # Create sanitized content
        if matches:
            sanitized_content = self._sanitize_content(content, matches)

        # Calculate scan duration
        scan_duration_ms = (time.time() - start_time) * 1000

        return ModelPIIDetectionResult(
            has_pii=len(matches) > 0,
            matches=matches,
            sanitized_content=sanitized_content,
            pii_types_detected=pii_types_detected,
            scan_duration_ms=scan_duration_ms,
        )

    def _deduplicate_matches(self, matches: list[ModelPIIMatch]) -> list[ModelPIIMatch]:
        """Remove overlapping or duplicate matches, keeping the highest confidence ones."""
        if not matches:
            return matches

        # Sort by start position and confidence
        matches.sort(key=lambda x: (x.start_index, -x.confidence))

        deduplicated: list[ModelPIIMatch] = []
        for match in matches:
            # Check if this match overlaps with any existing match
            overlap = False
            for existing in deduplicated:
                if (
                    match.start_index < existing.end_index
                    and match.end_index > existing.start_index
                ):
                    overlap = True
                    break

            if not overlap:
                deduplicated.append(match)

        return deduplicated

    def _sanitize_content(self, content: str, matches: list[ModelPIIMatch]) -> str:
        """Replace PII in content with masked values."""
        # Sort matches by start position in reverse order for proper replacement
        sorted_matches = sorted(matches, key=lambda x: x.start_index, reverse=True)

        sanitized = content
        for match in sorted_matches:
            sanitized = (
                sanitized[: match.start_index]
                + match.masked_value
                + sanitized[match.end_index :]
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
