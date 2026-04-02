# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Memory type classification for persona derivation eligibility."""

from enum import Enum


class EnumMemoryType(str, Enum):
    """Classification of memory items for persona derivation eligibility.

    Phase A: Controls which persona fields each memory type contributes to.
    Phase B (future, OMN-3980): Will also control retention duration and sharing
    eligibility. Consent enforcement is deferred to Phase B.
    """

    FACTUAL = "factual"
    PREFERENCE = "preference"
    SENSITIVE = "sensitive"
    EPHEMERAL = "ephemeral"
