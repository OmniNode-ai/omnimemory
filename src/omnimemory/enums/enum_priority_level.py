# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Priority level enumerations for ONEX compliance."""

from enum import Enum


class EnumPriorityLevel(str, Enum):
    """Priority levels for ONEX operations."""

    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"
    BACKGROUND = "background"
