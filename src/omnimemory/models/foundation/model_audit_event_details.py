"""
ONEX-compliant typed model for audit event details.

This module provides strongly typed replacement for Dict[str, Any] patterns
in audit logging, ensuring type safety and validation.
"""

from typing import Dict, Optional

from pydantic import BaseModel, Field, field_validator

from ...enums.core.enum_operation_status import EnumOperationStatus
from ...enums.core.enum_operation_type import EnumOperationType
from ...enums.foundation.enum_resource_type import EnumResourceType


class ModelAuditEventDetails(BaseModel):
    """Strongly typed details for audit events."""

    operation_type: EnumOperationType = Field(
        description="Type of operation being audited",
    )

    resource_id: Optional[str] = Field(
        default=None,
        max_length=255,
        description="Identifier of the resource being accessed (sanitized for security)",
    )

    resource_type: Optional[EnumResourceType] = Field(
        default=None,
        description="Type of resource being accessed",
    )

    old_value: Optional[str] = Field(
        default=None, description="Previous value before change"
    )

    new_value: Optional[str] = Field(default=None, description="New value after change")

    request_parameters: Optional[Dict[str, str]] = Field(
        default=None, description="Parameters passed with the request"
    )

    response_status: Optional[EnumOperationStatus] = Field(
        default=None, description="Response status or result"
    )

    error_details: Optional[str] = Field(
        default=None, description="Error details if operation failed"
    )

    ip_address: Optional[str] = Field(
        default=None, description="IP address of the requestor"
    )

    user_agent: Optional[str] = Field(
        default=None, description="User agent string from the request"
    )

    @field_validator("ip_address")
    @classmethod
    def validate_ip_address(cls, v: Optional[str]) -> Optional[str]:
        """Validate and sanitize IP address for security."""
        if v is None:
            return v
        # Basic IP address format validation and sanitization
        if len(v) > 45:  # Max IPv6 length
            return v[:45] + "[TRUNCATED]"
        return v

    @field_validator("user_agent")
    @classmethod
    def validate_user_agent(cls, v: Optional[str]) -> Optional[str]:
        """Validate and sanitize user agent string."""
        if v is None:
            return v
        # Limit user agent length to prevent abuse
        if len(v) > 512:
            return v[:512] + "[TRUNCATED]"
        return v

    @field_validator("old_value", "new_value")
    @classmethod
    def validate_sensitive_values(cls, v: Optional[str]) -> Optional[str]:
        """Sanitize potentially sensitive values for audit logging."""
        if v is None:
            return v
        # Check for potential sensitive data patterns
        v_lower = v.lower()
        sensitive_patterns = ["password", "secret", "key", "token", "credential"]
        if any(pattern in v_lower for pattern in sensitive_patterns):
            return "[REDACTED_SENSITIVE_DATA]"
        # Limit value length for security
        if len(v) > 1000:
            return v[:1000] + "[TRUNCATED]"
        return v
