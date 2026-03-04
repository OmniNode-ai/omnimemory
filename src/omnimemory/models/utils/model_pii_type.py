# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""PIIType enum for OmniMemory ONEX architecture."""

from __future__ import annotations

from enum import Enum

__all__ = [
    "PIIType",
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
