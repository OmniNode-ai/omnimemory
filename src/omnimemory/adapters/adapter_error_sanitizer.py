# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Enhanced error sanitization utility for OmniMemory ONEX architecture.

disclosure while maintaining useful debugging information for developers.
"""

from __future__ import annotations

__all__ = [
    "ErrorSanitizer",
    "SanitizationLevel",
    "sanitize_dict",
    "sanitize_error",
]

import re
from enum import Enum
from re import Pattern
from typing import Any


class SanitizationLevel(Enum):
    """Levels of error sanitization."""

    MINIMAL = "minimal"  # Only remove secrets, keep most information
    STANDARD = "standard"  # Balance between security and debugging
    STRICT = "strict"  # Maximum security, minimal information
    AUDIT = "audit"  # For audit logs, remove all sensitive data


# === PRE-COMPILED REGEX PATTERNS ===
# These patterns are compiled once at module load time for optimal performance.
# This avoids repeated regex compilation on each sanitization call.

# Credential patterns - detect passwords, API keys, tokens, etc.
_CREDENTIAL_PATTERNS: list[Pattern[str]] = [
    re.compile(r'\bpassword\s*[=:]\s*[\'"]?([^\s\'"]+)', re.IGNORECASE),
    re.compile(r'\bapi[_-]?key\s*[=:]\s*[\'"]?([^\s\'"]+)', re.IGNORECASE),
    re.compile(r'\bsecret\s*[=:]\s*[\'"]?([^\s\'"]+)', re.IGNORECASE),
    re.compile(r'\btoken\s*[=:]\s*[\'"]?([^\s\'"]+)', re.IGNORECASE),
    re.compile(r'\bauth\s*[=:]\s*[\'"]?([^\s\'"]+)', re.IGNORECASE),
    re.compile(r"\bbearer\s+([^\s]+)", re.IGNORECASE),
    re.compile(r"\basic\s+([^\s]+)", re.IGNORECASE),
]

# Connection string patterns - detect database/service URLs
_CONNECTION_STRING_PATTERNS: list[Pattern[str]] = [
    re.compile(r"postgresql://[^@]+@[^/]+/[^\s]+", re.IGNORECASE),
    re.compile(r"mysql://[^@]+@[^/]+/[^\s]+", re.IGNORECASE),
    re.compile(r"mongodb://[^@]+@[^/]+/[^\s]+", re.IGNORECASE),
    re.compile(r"redis://[^@]+@[^/]+/[^\s]*", re.IGNORECASE),
]

# IP address patterns - detect IPv4 and IPv6 addresses
_IP_ADDRESS_PATTERNS: list[Pattern[str]] = [
    re.compile(r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}(?::[0-9]+)?\b"),
    re.compile(r"\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b"),
]

# File path patterns - detect system paths
_FILE_PATH_PATTERNS: list[Pattern[str]] = [
    re.compile(r"/[a-zA-Z0-9/_-]+(?:\.[a-zA-Z0-9]+)?"),
    re.compile(r"[A-Za-z]:\\\\[a-zA-Z0-9\\\\._-]+"),
]

# Personal information patterns - detect PII
_PERSONAL_INFO_PATTERNS: list[Pattern[str]] = [
    re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),  # email
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),  # SSN
    re.compile(r"\b\d{16}\b"),  # Credit card
]

# Strict mode patterns - additional sanitization for numbers and identifiers
_STRICT_NUMBER_PATTERN: Pattern[str] = re.compile(r"\d+")
_STRICT_IDENTIFIER_PATTERN: Pattern[str] = re.compile(r"\b[a-zA-Z0-9]{8,}\b")

# Pattern category mapping for easy iteration (uses pre-compiled patterns)
_SENSITIVE_PATTERNS: dict[str, list[Pattern[str]]] = {
    "credentials": _CREDENTIAL_PATTERNS,
    "connection_strings": _CONNECTION_STRING_PATTERNS,
    "ip_addresses": _IP_ADDRESS_PATTERNS,
    "file_paths": _FILE_PATH_PATTERNS,
    "personal_info": _PERSONAL_INFO_PATTERNS,
}


class ErrorSanitizer:
    """
    Enhanced error sanitizer with configurable security levels.

    Features:
    - Pattern-based sensitive data detection (pre-compiled for performance)
    - Configurable sanitization levels
    - Structured error categorization
    - Context-aware sanitization rules
    """

    def __init__(self, level: SanitizationLevel = SanitizationLevel.STANDARD):
        """Initialize sanitizer with specified security level."""
        self.level = level
        self._safe_error_types = {
            "ValueError",
            "TypeError",
            "AttributeError",
            "KeyError",
            "IndexError",
            "ImportError",
            "ModuleNotFoundError",
            "FileNotFoundError",
            "PermissionError",
            "TimeoutError",
            "ConnectionError",
            "HTTPError",
            "ValidationError",
        }

    def sanitize_error(self, error: Exception, context: str | None = None) -> str:
        """
        Sanitize error message based on security level and context.

        Args:
            error: Exception to sanitize
            context: Optional context for context-aware sanitization

        Returns:
            Sanitized error message
        """
        error_type = type(error).__name__
        error_message = str(error)

        # Apply sanitization based on level
        if self.level == SanitizationLevel.MINIMAL:
            return self._minimal_sanitize(error_message, error_type)
        elif self.level == SanitizationLevel.STANDARD:
            return self._standard_sanitize(error_message, error_type, context)
        elif self.level == SanitizationLevel.STRICT:
            return self._strict_sanitize(error_message, error_type)
        else:  # AUDIT
            return self._audit_sanitize(error_message, error_type)

    def _minimal_sanitize(self, message: str, error_type: str) -> str:
        """Minimal sanitization - only remove obvious secrets."""
        sanitized = message

        # Only sanitize credentials using pre-compiled patterns
        for pattern in _CREDENTIAL_PATTERNS:
            sanitized = pattern.sub("[REDACTED]", sanitized)

        return f"{error_type}: {sanitized}"

    def _standard_sanitize(
        self, message: str, error_type: str, context: str | None = None
    ) -> str:
        """Standard sanitization - balance security and debugging."""
        sanitized = message

        # Sanitize credentials and connection strings using pre-compiled patterns
        for pattern in _CREDENTIAL_PATTERNS:
            sanitized = pattern.sub("[REDACTED]", sanitized)
        for pattern in _CONNECTION_STRING_PATTERNS:
            sanitized = pattern.sub("[REDACTED]", sanitized)

        # Context-aware sanitization
        if context in ["health_check", "connection_pool"]:
            # Keep connection info but sanitize auth
            for pattern in _IP_ADDRESS_PATTERNS:
                sanitized = pattern.sub("[IP:REDACTED]", sanitized)
        elif context in ["audit", "security"]:
            # More aggressive sanitization for security contexts
            for pattern in _PERSONAL_INFO_PATTERNS:
                sanitized = pattern.sub("[PII:REDACTED]", sanitized)

        # Keep error type for debugging
        return f"{error_type}: {sanitized}"

    def _strict_sanitize(self, message: str, error_type: str) -> str:
        """Strict sanitization - remove most identifiable information."""
        sanitized = message

        # Sanitize all sensitive patterns using pre-compiled patterns
        for patterns in _SENSITIVE_PATTERNS.values():
            for pattern in patterns:
                sanitized = pattern.sub("[REDACTED]", sanitized)

        # Remove specific details using pre-compiled patterns
        sanitized = _STRICT_NUMBER_PATTERN.sub("[NUM]", sanitized)
        sanitized = _STRICT_IDENTIFIER_PATTERN.sub("[ID]", sanitized)

        return f"{error_type}: Connection/operation failed - [DETAILS_REDACTED]"

    def _audit_sanitize(self, message: str, error_type: str) -> str:
        """Audit-level sanitization - minimal information for compliance."""
        if error_type in self._safe_error_types:
            return f"{error_type}: Operation failed"
        else:
            return "Exception: Operation failed - details suppressed for audit"

    def sanitize_dict(
        self, data: dict[str, Any], keys_to_sanitize: set[str] | None = None
    ) -> dict[str, Any]:
        """
        Sanitize sensitive keys in dictionary data.

        Args:
            data: Dictionary to sanitize
            keys_to_sanitize: Optional set of keys to sanitize

        Returns:
            Sanitized dictionary
        """
        if keys_to_sanitize is None:
            keys_to_sanitize = {
                "password",
                "secret",
                "token",
                "key",
                "auth",
                "credential",
                "api_key",
                "access_key",
                "private_key",
                "session_id",
            }

        sanitized: dict[str, Any] = {}
        for key, value in data.items():
            if any(sensitive in key.lower() for sensitive in keys_to_sanitize):
                sanitized[key] = "[REDACTED]"
            elif isinstance(value, dict):
                # Recursive call for nested dicts
                nested_dict: dict[str, Any] = value
                sanitized[key] = self.sanitize_dict(nested_dict, keys_to_sanitize)
            elif isinstance(value, str):
                sanitized[key] = self._apply_patterns(value)
            else:
                sanitized[key] = value

        return sanitized

    def _apply_patterns(self, text: str) -> str:
        """Apply sanitization patterns to text using pre-compiled patterns."""
        sanitized = text
        for patterns in _SENSITIVE_PATTERNS.values():
            for pattern in patterns:
                sanitized = pattern.sub("[REDACTED]", sanitized)
        return sanitized

    def is_safe_error_type(self, error_type: str) -> bool:
        """Check if error type is considered safe for logging."""
        return error_type in self._safe_error_types

    def get_error_category(self, error: Exception) -> str:
        """Categorize error for appropriate handling."""
        error_type = type(error).__name__
        message = str(error).lower()

        if any(word in message for word in ["connection", "timeout", "network"]):
            return "connectivity"
        elif any(word in message for word in ["permission", "access", "auth"]):
            return "authorization"
        elif any(word in message for word in ["validation", "invalid", "format"]):
            return "validation"
        elif error_type in ["ValueError", "TypeError", "AttributeError"]:
            return "programming"
        else:
            return "system"


# Global instance for convenient access
default_sanitizer = ErrorSanitizer(SanitizationLevel.STANDARD)


def sanitize_error(
    error: Exception,
    context: str | None = None,
    level: SanitizationLevel = SanitizationLevel.STANDARD,
) -> str:
    """
    Convenient function for error sanitization.

    Args:
        error: Exception to sanitize
        context: Optional context for context-aware sanitization
        level: Sanitization level

    Returns:
        Sanitized error message
    """
    if level != SanitizationLevel.STANDARD:
        sanitizer = ErrorSanitizer(level)
    else:
        sanitizer = default_sanitizer

    return sanitizer.sanitize_error(error, context)


def sanitize_dict(
    data: dict[str, Any],
    keys_to_sanitize: set[str] | None = None,
    level: SanitizationLevel = SanitizationLevel.STANDARD,
) -> dict[str, Any]:
    """
    Convenient function for dictionary sanitization.

    Args:
        data: Dictionary to sanitize
        keys_to_sanitize: Optional set of keys to sanitize
        level: Sanitization level

    Returns:
        Sanitized dictionary
    """
    if level != SanitizationLevel.STANDARD:
        sanitizer = ErrorSanitizer(level)
    else:
        sanitizer = default_sanitizer

    return sanitizer.sanitize_dict(data, keys_to_sanitize)
