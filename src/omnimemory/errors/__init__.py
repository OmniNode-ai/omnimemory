# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""OmniMemory exception hierarchy.

This package contains all exception classes for OmniMemory. Exceptions
are organized by domain to support clean error handling at contract
boundaries.

Available Modules:
    - embedding_errors: Exceptions for embedding client operations

Example::

    from omnimemory.errors import (
        EmbeddingClientError,
        EmbeddingConnectionError,
        EmbeddingTimeoutError,
    )

    try:
        embedding = await client.get_embedding(text)
    except EmbeddingConnectionError as e:
        logger.error("Server unreachable: %s (correlation_id=%s)", e, e.correlation_id)
    except EmbeddingTimeoutError as e:
        logger.error("Request timed out: %s (correlation_id=%s)", e, e.correlation_id)
    except EmbeddingClientError as e:
        logger.error("Embedding error: %s (correlation_id=%s)", e, e.correlation_id)

.. versionadded:: 0.2.0
    Initial implementation for OMN-1391.
"""

from omnimemory.errors.embedding_errors import (
    EmbeddingClientError,
    EmbeddingConnectionError,
    EmbeddingTimeoutError,
)

__all__ = [
    # Embedding client exceptions
    "EmbeddingClientError",
    "EmbeddingConnectionError",
    "EmbeddingTimeoutError",
]
