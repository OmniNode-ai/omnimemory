"""
Shared constants for subscription models following ONEX standards.

This module provides shared constants used across subscription-related models
to ensure consistency and avoid duplication.
"""

from __future__ import annotations

import re

# Topic pattern regex string: memory.<entity>.<event>
# Examples: memory.item.created, memory.item.updated, memory.item.deleted
TOPIC_PATTERN_REGEX = r"^memory\.[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*$"

# Compiled topic pattern for validation
TOPIC_PATTERN = re.compile(TOPIC_PATTERN_REGEX)

__all__ = [
    "TOPIC_PATTERN",
    "TOPIC_PATTERN_REGEX",
]
