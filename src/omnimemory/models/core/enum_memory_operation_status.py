"""
Enum for memory operation status following ONEX standards.
"""

from __future__ import annotations

from enum import Enum


class EnumMemoryOperationStatus(str, Enum):
    """Status of memory operations in the ONEX memory system."""

    SUCCESS = "success"
    FAILURE = "failure"
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    RETRYING = "retrying"
    PARTIAL_SUCCESS = "partial_success"