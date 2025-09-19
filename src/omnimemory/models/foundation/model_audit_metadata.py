"""
ONEX-compliant typed models for audit logging metadata.

This module provides strongly typed replacements for Dict[str, Any] patterns
in audit logging, ensuring type safety and validation.

NOTICE: This module has been refactored to follow one-model-per-file standards.
Individual models are now in separate files but re-exported here for backwards compatibility.
"""

# Re-export individual models for backwards compatibility
from .model_audit_event_details import ModelAuditEventDetails
from .model_performance_audit_details import ModelPerformanceAuditDetails
from .model_resource_usage_metadata import ModelResourceUsageMetadata
from .model_security_audit_details import ModelSecurityAuditDetails

__all__ = [
    "ModelAuditEventDetails",
    "ModelResourceUsageMetadata",
    "ModelSecurityAuditDetails",
    "ModelPerformanceAuditDetails",
]
