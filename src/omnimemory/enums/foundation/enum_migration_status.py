"""
Migration and file processing status enumerations.
"""

from enum import Enum, IntEnum


class FileProcessingStatus(Enum):
    """Status of file processing operations."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


class MigrationStatus(Enum):
    """Status of migration operations."""

    PENDING = "pending"
    NOT_STARTED = "not_started"
    INITIALIZING = "initializing"
    IN_PROGRESS = "in_progress"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


class MigrationPriority(IntEnum):
    """Priority levels for migration operations."""

    LOW = 1
    NORMAL = 2
    MEDIUM = 3
    HIGH = 4
    CRITICAL = 5
    IMMEDIATE = 6
