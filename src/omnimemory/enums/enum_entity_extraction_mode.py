# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Enum for entity extraction mode following ONEX standards.

Controls the trade-off between reproducibility and quality in entity extraction.
"""

from enum import Enum


class EnumEntityExtractionMode(str, Enum):
    """Mode for entity extraction operations.

    Controls the determinism vs. quality trade-off:
    - DETERMINISTIC: Same input always produces same output (reproducible)
    - BEST_EFFORT: May use non-deterministic methods for better accuracy

    For infrastructure credibility and test stability, DETERMINISTIC is the
    recommended default.

    Example:
        >>> mode = EnumEntityExtractionMode.DETERMINISTIC
        >>> mode.is_reproducible
        True
    """

    DETERMINISTIC = "deterministic"
    """Reproducible extraction with fixed parameters.

    Uses:
    - temperature=0 for LLM-based extraction
    - fixed random seed
    - strict output schema enforcement

    Trade-off: Slightly lower recall on edge cases, but results are
    reproducible and testable with golden fixtures.
    """

    BEST_EFFORT = "best_effort"
    """Non-deterministic extraction for maximum accuracy.

    Uses:
    - temperature>0 for LLM-based extraction
    - no seed constraints
    - relaxed schema matching

    Trade-off: Better accuracy on edge cases, but results may vary
    between runs.
    """

    @property
    def is_reproducible(self) -> bool:
        """Return True if this mode guarantees reproducible results."""
        return self == EnumEntityExtractionMode.DETERMINISTIC

    @property
    def default_llm_temperature(self) -> float:
        """Return the default LLM temperature for this mode."""
        if self == EnumEntityExtractionMode.DETERMINISTIC:
            return 0.0
        return 0.7

    @property
    def default_llm_seed(self) -> int | None:
        """Return the default LLM seed for this mode."""
        if self == EnumEntityExtractionMode.DETERMINISTIC:
            return 42
        return None
