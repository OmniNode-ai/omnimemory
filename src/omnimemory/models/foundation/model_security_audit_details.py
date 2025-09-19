"""
ONEX-compliant typed model for security audit details.

This module provides strongly typed replacement for Dict[str, Any] patterns
in audit logging, ensuring type safety and validation.
"""

from typing import List, Optional

from pydantic import BaseModel, Field


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
