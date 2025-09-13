"""
Enum for error codes following ONEX standards.
"""

from enum import Enum


class EnumErrorCode(str, Enum):
    """Error codes for the ONEX memory system."""

    # General errors
    UNKNOWN_ERROR = "unknown_error"
    INTERNAL_ERROR = "internal_error"
    CONFIGURATION_ERROR = "configuration_error"
    VALIDATION_ERROR = "validation_error"

    # Service errors
    SERVICE_UNAVAILABLE = "service_unavailable"
    SERVICE_TIMEOUT = "service_timeout"
    SERVICE_OVERLOADED = "service_overloaded"
    SERVICE_INITIALIZATION_FAILED = "service_initialization_failed"

    # Memory operation errors
    MEMORY_STORAGE_FAILED = "memory_storage_failed"
    MEMORY_RETRIEVAL_FAILED = "memory_retrieval_failed"
    MEMORY_UPDATE_FAILED = "memory_update_failed"
    MEMORY_DELETE_FAILED = "memory_delete_failed"

    # Intelligence operation errors
    ANALYSIS_FAILED = "analysis_failed"
    PATTERN_RECOGNITION_FAILED = "pattern_recognition_failed"
    SEMANTIC_PROCESSING_FAILED = "semantic_processing_failed"

    # Authentication and authorization errors
    AUTHENTICATION_FAILED = "authentication_failed"
    AUTHORIZATION_FAILED = "authorization_failed"
    ACCESS_DENIED = "access_denied"
    TOKEN_EXPIRED = "token_expired"

    # Network and connectivity errors
    CONNECTION_FAILED = "connection_failed"
    NETWORK_TIMEOUT = "network_timeout"
    DNS_RESOLUTION_FAILED = "dns_resolution_failed"

    # Resource errors
    RESOURCE_NOT_FOUND = "resource_not_found"
    RESOURCE_EXHAUSTED = "resource_exhausted"
    QUOTA_EXCEEDED = "quota_exceeded"

    # Data errors
    DATA_CORRUPTION = "data_corruption"
    SERIALIZATION_ERROR = "serialization_error"
    DESERIALIZATION_ERROR = "deserialization_error"
    SCHEMA_VALIDATION_ERROR = "schema_validation_error"