"""
Enhanced error sanitization utility for OmniMemory ONEX architecture.

This module provides comprehensive error sanitization to prevent information
disclosure while maintaining useful debugging information for developers.
"""

__all__ = ["SanitizationLevel", "ErrorSanitizer"]

import re
from enum import Enum
from typing import Dict, List, Optional, Set


class SanitizationLevel(Enum):
    """Levels of error sanitization."""

    MINIMAL = "minimal"  # Only remove secrets, keep most information
    STANDARD = "standard"  # Balance between security and debugging
    STRICT = "strict"  # Maximum security, minimal information
    AUDIT = "audit"  # For audit logs, remove all sensitive data


class ErrorSanitizer:
    """
    Enhanced error sanitizer with configurable security levels.

    Features:
    - Pattern-based sensitive data detection
    - Configurable sanitization levels
    - Structured error categorization
    - Context-aware sanitization rules
    """

    def __init__(self, level: SanitizationLevel = SanitizationLevel.STANDARD):
        """Initialize sanitizer with specified security level."""
        self.level = level
        self._sensitive_patterns = self._initialize_patterns()
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

    def _initialize_patterns(self) -> Dict[str, List[str]]:
        """Initialize regex patterns for sensitive data detection."""
        return {
            "credentials": [
                r'\bpassword\s*[=:]\s*[\'"]?([^\s\'"]+)',
                r'\bapi[_-]?key\s*[=:]\s*[\'"]?([^\s\'"]+)',
                r'\bsecret\s*[=:]\s*[\'"]?([^\s\'"]+)',
                r'\btoken\s*[=:]\s*[\'"]?([^\s\'"]+)',
                r'\bauth\s*[=:]\s*[\'"]?([^\s\'"]+)',
                r"\bbearer\s+([^\s]+)",
                r"\basic\s+([^\s]+)",
            ],
            "connection_strings": [
                r"postgresql://[^@]+@[^/]+/[^\s]+",
                r"mysql://[^@]+@[^/]+/[^\s]+",
                r"mongodb://[^@]+@[^/]+/[^\s]+",
            ],
            "ip_addresses": [
                r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}(?::[0-9]+)?\b",
                r"\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b",
            ],
            "file_paths": [
                r"/[a-zA-Z0-9/_-]+(?:\.[a-zA-Z0-9]+)?",
                r"[A-Za-z]:\\\\[a-zA-Z0-9\\\\._-]+",
            ],
            "personal_info": [
                r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # email
                r"\b\d{3}-\d{2}-\d{4}\b",  # SSN
                r"\b\d{16}\b",  # Credit card
            ],
        }

    def sanitize_error(self, error: Exception, context: Optional[str] = None) -> str:
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

        # Only sanitize credentials
        for pattern in self._sensitive_patterns["credentials"]:
            sanitized = re.sub(pattern, r"[REDACTED]", sanitized, flags=re.IGNORECASE)

        return f"{error_type}: {sanitized}"

    def _standard_sanitize(
        self, message: str, error_type: str, context: Optional[str] = None
    ) -> str:
        """Standard sanitization - balance security and debugging."""
        sanitized = message

        # Sanitize credentials and connection strings
        for category in ["credentials", "connection_strings"]:
            for pattern in self._sensitive_patterns[category]:
                sanitized = re.sub(
                    pattern, "[REDACTED]", sanitized, flags=re.IGNORECASE
                )

        # Context-aware sanitization
        if context in ["health_check", "connection_pool"]:
            # Keep connection info but sanitize auth
            for pattern in self._sensitive_patterns["ip_addresses"]:
                sanitized = re.sub(pattern, "[IP:REDACTED]", sanitized)
        elif context in ["audit", "security"]:
            # More aggressive sanitization for security contexts
            for pattern in self._sensitive_patterns["personal_info"]:
                sanitized = re.sub(pattern, "[PII:REDACTED]", sanitized)

        # Keep error type for debugging
        return f"{error_type}: {sanitized}"

    def _strict_sanitize(self, message: str, error_type: str) -> str:
        """Strict sanitization - remove most identifiable information."""
        sanitized = message

        # Sanitize all sensitive patterns
        for category, patterns in self._sensitive_patterns.items():
            for pattern in patterns:
                sanitized = re.sub(
                    pattern, "[REDACTED]", sanitized, flags=re.IGNORECASE
                )

        # Remove specific details but keep general structure
        sanitized = re.sub(r"\d+", "[NUM]", sanitized)  # Replace numbers
        sanitized = re.sub(
            r"\b[a-zA-Z0-9]{8,}\b", "[ID]", sanitized
        )  # Long identifiers

        return f"{error_type}: Connection/operation failed - [DETAILS_REDACTED]"

    def _audit_sanitize(self, message: str, error_type: str) -> str:
        """Audit-level sanitization - minimal information for compliance."""
        if error_type in self._safe_error_types:
            return f"{error_type}: Operation failed"
        else:
            return "Exception: Operation failed - details suppressed for audit"

    def sanitize_dict(
        self, data: Dict, keys_to_sanitize: Optional[Set[str]] = None
    ) -> Dict:
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

        sanitized = {}
        for key, value in data.items():
            if any(sensitive in key.lower() for sensitive in keys_to_sanitize):
                sanitized[key] = "[REDACTED]"
            elif isinstance(value, dict):
                sanitized[key] = self.sanitize_dict(value, keys_to_sanitize)
            elif isinstance(value, str):
                sanitized[key] = self._apply_patterns(value)
            else:
                sanitized[key] = value

        return sanitized

    def _apply_patterns(self, text: str) -> str:
        """Apply sanitization patterns to text."""
        sanitized = text
        for category, patterns in self._sensitive_patterns.items():
            for pattern in patterns:
                sanitized = re.sub(
                    pattern, "[REDACTED]", sanitized, flags=re.IGNORECASE
                )
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
    context: Optional[str] = None,
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
    data: Dict,
    keys_to_sanitize: Optional[Set[str]] = None,
    level: SanitizationLevel = SanitizationLevel.STANDARD,
) -> Dict:
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
