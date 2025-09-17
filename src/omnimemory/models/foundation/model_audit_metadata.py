"""
ONEX-compliant typed models for audit logging metadata.

This module provides strongly typed replacements for Dict[str, Any] patterns
in audit logging, ensuring type safety and validation.
"""

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class AuditEventDetails(BaseModel):
    """Strongly typed details for audit events."""

    operation_type: str = Field(description="Type of operation being audited")

    resource_id: Optional[str] = Field(
        default=None, description="Identifier of the resource being accessed"
    )

    resource_type: Optional[str] = Field(
        default=None, description="Type of resource (memory, configuration, etc.)"
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
        default=None, description="CPU usage percentage during operation"
    )

    memory_usage_mb: Optional[float] = Field(
        default=None, description="Memory usage in megabytes"
    )

    disk_io_bytes: Optional[int] = Field(default=None, description="Disk I/O in bytes")

    network_io_bytes: Optional[int] = Field(
        default=None, description="Network I/O in bytes"
    )

    operation_duration_ms: Optional[float] = Field(
        default=None, description="Duration of operation in milliseconds"
    )

    database_queries: Optional[int] = Field(
        default=None, description="Number of database queries performed"
    )

    cache_hits: Optional[int] = Field(default=None, description="Number of cache hits")

    cache_misses: Optional[int] = Field(
        default=None, description="Number of cache misses"
    )


class SecurityAuditDetails(BaseModel):
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
        default=None, description="Calculated risk score for the operation"
    )


class PerformanceAuditDetails(BaseModel):
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
        default=None, description="Cache hit ratio"
    )
