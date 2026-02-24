# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Embedding client exceptions.

This module defines contract boundary exceptions for embedding operations.
These exceptions provide a clean separation between infrastructure-level
errors (connection, timeout) and domain-level errors (invalid input,
malformed response).

The exception hierarchy:
    EmbeddingClientError (base)
    ├── EmbeddingConnectionError - Server unreachable
    └── EmbeddingTimeoutError - Request timed out

All exceptions support correlation_id for distributed tracing, enabling
observability systems to correlate exceptions with specific requests.

.. versionadded:: 0.2.0
    Initial implementation for OMN-1391.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from uuid import UUID

__all__ = [
    "EmbeddingClientError",
    "EmbeddingConnectionError",
    "EmbeddingTimeoutError",
]


class EmbeddingClientError(Exception):
    """Base exception for embedding client errors.

    Attributes:
        correlation_id: Optional correlation ID for distributed tracing.
            When set, enables correlating exceptions with specific requests
            in observability systems.
    """

    __slots__ = ("correlation_id",)

    def __init__(self, message: str, correlation_id: UUID | None = None) -> None:
        """Initialize the exception.

        Args:
            message: Human-readable error message.
            correlation_id: Optional correlation ID for distributed tracing.
        """
        super().__init__(message)
        self.correlation_id = correlation_id


class EmbeddingConnectionError(EmbeddingClientError):
    """Raised when connection to embedding server fails."""


class EmbeddingTimeoutError(EmbeddingClientError):
    """Raised when embedding request times out."""
