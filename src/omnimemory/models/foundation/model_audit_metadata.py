"""
ONEX-compliant typed models for audit logging metadata.

This module provides strongly typed replacements for Dict[str, Any] patterns
in audit logging, ensuring type safety and validation.
"""

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class ModelAuditEventDetails(BaseModel):
    """Strongly typed details for audit events."""

    operation_type: str = Field(
        max_length=100,
        description="Type of operation being audited (CREATE, READ, UPDATE, DELETE, etc.)",
    )

    resource_id: Optional[str] = Field(
        default=None,
        max_length=255,
        description="Identifier of the resource being accessed (sanitized for security)",
    )

    resource_type: Optional[str] = Field(
        default=None,
        max_length=50,
        description="Type of resource (memory, configuration, cache, etc.)",
    )

    old_value: Optional[str] = Field(
        default=None, description="Previous value before change"
    )

    new_value: Optional[str] = Field(default=None, description="New value after change")

    request_parameters: Optional[Dict[str, str]] = Field(
        default=None, description="Parameters passed with the request"
    )

    response_status: Optional[str] = Field(
        default=None, description="Response status code or result"
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


class ModelResourceUsageMetadata(BaseModel):
    """Strongly typed resource usage metrics."""

    cpu_usage_percent: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="CPU usage percentage during operation (0-100%)",
    )

    memory_usage_mb: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Memory usage in megabytes (positive values only)",
    )

    disk_io_bytes: Optional[int] = Field(default=None, description="Disk I/O in bytes")

    network_io_bytes: Optional[int] = Field(
        default=None, description="Network I/O in bytes"
    )

    operation_duration_ms: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Duration of operation in milliseconds (for performance monitoring)",
    )

    database_queries: Optional[int] = Field(
        default=None, description="Number of database queries performed"
    )

    cache_hits: Optional[int] = Field(default=None, description="Number of cache hits")

    cache_misses: Optional[int] = Field(
        default=None, description="Number of cache misses"
    )


class ModelSecurityAuditDetails(BaseModel):
    """Strongly typed security audit information."""

    authentication_method: Optional[str] = Field(
        default=None, description="Authentication method used"
    )

    authorization_level: Optional[str] = Field(
        default=None, description="Authorization level granted"
    )

    permission_required: Optional[str] = Field(
        default=None, description="Permission required for the operation"
    )

    permission_granted: bool = Field(
        default=False, description="Whether permission was granted"
    )

    security_scan_results: Optional[List[str]] = Field(
        default=None, description="Results of security scanning"
    )

    pii_detected: bool = Field(
        default=False, description="Whether PII was detected in the request"
    )

    data_classification: Optional[str] = Field(
        default=None, description="Classification level of data accessed"
    )

    risk_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=10.0,
        description="Calculated risk score for the operation (0-10 scale)",
    )


class ModelPerformanceAuditDetails(BaseModel):
    """Strongly typed performance audit information."""

    operation_latency_ms: float = Field(description="Operation latency in milliseconds")

    throughput_ops_per_second: Optional[float] = Field(
        default=None, description="Throughput in operations per second"
    )

    queue_depth: Optional[int] = Field(
        default=None, description="Queue depth at operation time"
    )

    connection_pool_usage: Optional[float] = Field(
        default=None, description="Connection pool usage percentage"
    )

    circuit_breaker_state: Optional[str] = Field(
        default=None, description="Circuit breaker state during operation"
    )

    retry_count: int = Field(default=0, description="Number of retries attempted")

    cache_efficiency: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Cache hit ratio (0.0-1.0 where 1.0 is 100% hit rate)",
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
