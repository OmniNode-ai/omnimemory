# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Migration status enumerations for ONEX compliance.

All enums inherit from (str, Enum) for proper Pydantic serialization.
"""

from enum import Enum


class EnumMigrationStatus(str, Enum):
    """
    Migration status enumeration.

    Inherits from (str, Enum) for proper Pydantic serialization,
    matching the EnumPriorityLevel pattern used throughout omnimemory.
    """

    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class EnumMigrationPriority(str, Enum):
    """
    Migration priority levels.

    Inherits from (str, Enum) for proper Pydantic serialization.
    """

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class EnumFileProcessingStatus(str, Enum):
    """
    File processing status enumeration.

    Inherits from (str, Enum) for proper Pydantic serialization.
    """

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
