"""
Operation status enumeration for core operations.
"""

from enum import Enum


class EnumOperationStatus(Enum):
    """Operation status enumeration."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"
