# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""
Constants for subscription domain models following ONEX standards.

Topic Naming Convention:
    Topics follow the pattern: memory.<entity>.<event>

    Entity types:
        - item: Individual memory items
        - collection: Memory collections/groups
        - index: Search indexes

    Event types:
        - created: New entity created
        - updated: Entity modified
        - deleted: Entity removed
        - archived: Entity archived
        - promoted: Entity promoted (e.g., from cache to persistent)

Examples:
    - memory.item.created
    - memory.collection.updated
    - memory.index.deleted
"""

from __future__ import annotations

import re

# Topic pattern regex string: memory.<entity>.<event>
# Examples: memory.item.created, memory.item.updated, memory.item.deleted
TOPIC_PATTERN_REGEX = r"^memory\.[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*$"

# Compiled topic pattern for validation
TOPIC_PATTERN = re.compile(TOPIC_PATTERN_REGEX)

# Error message for invalid topic format
TOPIC_VALIDATION_ERROR = (
    "Topic must match pattern 'memory.<entity>.<event>', got: {topic}"
)

__all__ = [
    "TOPIC_PATTERN",
    "TOPIC_PATTERN_REGEX",
    "TOPIC_VALIDATION_ERROR",
]
