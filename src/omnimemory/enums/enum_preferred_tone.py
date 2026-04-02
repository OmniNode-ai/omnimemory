# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Inferred preferred interaction style for persona inference."""

from enum import Enum


class EnumPreferredTone(str, Enum):
    """Inferred preferred interaction style.

    Derived from prompt patterns and interaction history. Allowed to shift
    faster than technical_level because tone is more situational and less
    consequential if wrong.
    """

    EXPLANATORY = "explanatory"
    CONCISE = "concise"
    FORMAL = "formal"
    CASUAL = "casual"
