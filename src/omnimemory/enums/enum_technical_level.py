# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Inferred user technical capability level for persona inference."""

from enum import Enum


class EnumTechnicalLevel(str, Enum):
    """Inferred user technical capability.

    Derived from session behavior: tool usage patterns, prompt complexity,
    error recovery rate, and vocabulary. Updated conservatively between
    sessions (requires 3+ consistent high-confidence signals to shift).
    """

    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"
