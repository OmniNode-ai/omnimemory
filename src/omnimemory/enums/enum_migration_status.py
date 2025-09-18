"""
Migration status enumerations for ONEX compliance.

This module contains all migration-related enum types following ONEX standards.
"""

from enum import Enum


class MigrationStatus(Enum):
    """Migration status enumeration."""

    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class MigrationPriority(Enum):
    """Migration priority levels."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class FileProcessingStatus(Enum):
    """File processing status enumeration."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
