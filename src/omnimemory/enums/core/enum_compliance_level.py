"""
Compliance level enumeration for ONEX Foundation Architecture.

Defines standardized compliance levels for ONEX quality gates and validation.
"""

from enum import Enum


class EnumComplianceLevel(str, Enum):
    """Standardized compliance levels for ONEX operations."""

    # Standard compliance - basic ONEX requirements
    STANDARD = "standard"

    # Strict compliance - enhanced validation and quality gates
    STRICT = "strict"

    # Audit compliance - comprehensive logging and traceability
    AUDIT = "audit"

    # Development compliance - relaxed for development environments
    DEVELOPMENT = "development"

    # Production compliance - full production-grade validation
    PRODUCTION = "production"

    # Enterprise compliance - enterprise-grade security and compliance
    ENTERPRISE = "enterprise"
