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

# Circuit breaker default configuration
# These values are used by both ModelCircuitBreakerState and HandlerSubscription
DEFAULT_CIRCUIT_BREAKER_FAILURE_THRESHOLD = 5
"""Number of consecutive failures before the circuit opens."""

DEFAULT_CIRCUIT_BREAKER_SUCCESS_THRESHOLD = 3
"""Number of consecutive successes in half_open state before closing the circuit."""

DEFAULT_CIRCUIT_BREAKER_COOLDOWN_MS = 60000
"""Cooldown period in milliseconds before transitioning from open to half_open.

This is equivalent to DEFAULT_CIRCUIT_BREAKER_COOLDOWN_SECONDS (60 seconds).
Use this constant when working with millisecond-based APIs or model fields.
"""

DEFAULT_CIRCUIT_BREAKER_COOLDOWN_SECONDS = 60
"""Cooldown period in seconds before transitioning from open to half_open.

This is equivalent to DEFAULT_CIRCUIT_BREAKER_COOLDOWN_MS (60000 milliseconds).
Use this constant when working with second-based configuration or APIs.
"""

__all__ = [
    # Topic patterns
    "TOPIC_PATTERN",
    "TOPIC_PATTERN_REGEX",
    # Circuit breaker defaults
    "DEFAULT_CIRCUIT_BREAKER_FAILURE_THRESHOLD",
    "DEFAULT_CIRCUIT_BREAKER_SUCCESS_THRESHOLD",
    "DEFAULT_CIRCUIT_BREAKER_COOLDOWN_MS",
    "DEFAULT_CIRCUIT_BREAKER_COOLDOWN_SECONDS",
]
