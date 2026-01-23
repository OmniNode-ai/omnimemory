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

DEFAULT_CIRCUIT_BREAKER_COOLDOWN_MS = 30000
"""Cooldown period in milliseconds before transitioning from open to half_open."""

DEFAULT_CIRCUIT_BREAKER_COOLDOWN_SECONDS = 60
"""Cooldown period in seconds before transitioning from open to half_open.

Note: This is used by HandlerSubscription for config-level defaults.
The cooldown_ms constant (30000ms = 30s) is used by ModelCircuitBreakerState
for model-level defaults. Applications may use either based on their unit preference.
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
