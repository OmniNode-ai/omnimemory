"""
Operation status enumeration following ONEX standards.
"""

from enum import Enum


class EnumOperationStatus(str, Enum):
    """
    Status values for memory operations following ONEX standards.

    Represents the current state of memory operations throughout their lifecycle:
    - PENDING: Operation queued but not yet started
    - PROCESSING: Operation currently being executed
    - SUCCESS: Operation completed successfully
    - FAILED: Operation encountered an error and failed
    - CANCELLED: Operation was cancelled before completion
    - TIMEOUT: Operation exceeded time limits and was terminated
    - RETRY: Operation failed but is eligible for retry
    """

    PENDING = "pending"
    PROCESSING = "processing"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"
    RETRY = "retry"
