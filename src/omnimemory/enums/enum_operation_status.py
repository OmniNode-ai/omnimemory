"""
Operation status enumeration following ONEX standards.
"""

from enum import Enum


class EnumOperationStatus(str, Enum):
    """Status values for memory operations."""
    
    PENDING = "pending"
    PROCESSING = "processing"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"
    RETRY = "retry"