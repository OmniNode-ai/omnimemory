# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""
Health status enumeration for OmniMemory ONEX architecture.

This module contains the HealthStatus enum used across health check models.
"""

from __future__ import annotations

from enum import Enum

__all__ = [
    "HealthStatus",
]


class HealthStatus(Enum):
    """Enhanced health status enumeration."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"
    TIMEOUT = "timeout"
    RATE_LIMITED = "rate_limited"
    CIRCUIT_OPEN = "circuit_open"
