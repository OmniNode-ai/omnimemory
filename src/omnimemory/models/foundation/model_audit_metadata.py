# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""ONEX-compliant typed models for audit logging metadata.

in audit logging, ensuring type safety and validation.
"""

from pydantic import BaseModel, ConfigDict, Field


class AuditEventDetails(BaseModel):
    """Strongly typed details for audit events."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    operation_type: str = Field(
        default="unknown", description="Type of operation being audited"
    )

    resource_id: str | None = Field(
        default=None, description="Identifier of the resource being accessed"
    )

    resource_type: str | None = Field(
        default=None, description="Type of resource (memory, configuration, etc.)"
    )

    old_value: str | None = Field(
        default=None, description="Previous value before change"
    )

    new_value: str | None = Field(default=None, description="New value after change")

    # NOTE: dict contents remain mutable even on frozen models; this is a known
    # Pydantic v2 limitation.  frozen=True prevents field *reassignment* but not
    # in-place mutation of mutable containers.  A MappingProxyType wrapper is not
    # Pydantic-friendly, so we accept this trade-off and rely on convention.
    request_parameters: dict[str, str] | None = Field(
        default=None, description="Parameters passed with the request"
    )

    response_status: str | None = Field(
        default=None, description="Response status code or result"
    )

    error_details: str | None = Field(
        default=None, description="Error details if operation failed"
    )

    ip_address: str | None = Field(
        default=None, description="IP address of the requestor"
    )

    user_agent: str | None = Field(
        default=None, description="User agent string from the request"
    )


class ResourceUsageMetadata(BaseModel):
    """Strongly typed resource usage metrics."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    cpu_usage_percent: float | None = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="CPU usage percentage during operation",
    )

    memory_usage_mb: float | None = Field(
        default=None, ge=0.0, description="Memory usage in megabytes"
    )

    disk_io_bytes: int | None = Field(
        default=None, ge=0, description="Disk I/O in bytes"
    )

    network_io_bytes: int | None = Field(
        default=None, ge=0, description="Network I/O in bytes"
    )

    operation_duration_ms: float | None = Field(
        default=None, ge=0.0, description="Duration of operation in milliseconds"
    )

    database_queries: int | None = Field(
        default=None, ge=0, description="Number of database queries performed"
    )

    cache_hits: int | None = Field(
        default=None, ge=0, description="Number of cache hits"
    )

    cache_misses: int | None = Field(
        default=None, ge=0, description="Number of cache misses"
    )


class SecurityAuditDetails(BaseModel):
    """Strongly typed security audit information."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    authentication_method: str | None = Field(
        default=None, description="Authentication method used"
    )

    authorization_level: str | None = Field(
        default=None, description="Authorization level granted"
    )

    permission_required: str | None = Field(
        default=None, description="Permission required for the operation"
    )

    permission_granted: bool = Field(
        default=False, description="Whether permission was granted"
    )

    security_scan_results: tuple[str, ...] | None = Field(
        default=None, description="Results of security scanning"
    )

    pii_detected: bool = Field(
        default=False, description="Whether PII was detected in the request"
    )

    data_classification: str | None = Field(
        default=None, description="Classification level of data accessed"
    )

    risk_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Calculated risk score for the operation (0.0-1.0)",
    )


class PerformanceAuditDetails(BaseModel):
    """Strongly typed performance audit information."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    operation_latency_ms: float = Field(
        ge=0.0, description="Operation latency in milliseconds"
    )

    throughput_ops_per_second: float | None = Field(
        default=None, ge=0.0, description="Throughput in operations per second"
    )

    queue_depth: int | None = Field(
        default=None, ge=0, description="Queue depth at operation time"
    )

    connection_pool_usage: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Connection pool usage percentage (0.0-1.0)",
    )

    circuit_breaker_state: str | None = Field(
        default=None, description="Circuit breaker state during operation"
    )

    retry_count: int = Field(default=0, ge=0, description="Number of retries attempted")

    cache_efficiency: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Cache hit ratio (0.0-1.0)"
    )
