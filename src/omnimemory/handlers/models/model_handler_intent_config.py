# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Configuration model for HandlerIntent.

This module provides the configuration model for HandlerIntent, which manages
intent storage and query operations against a graph database. The configuration
controls connection settings, circuit breaker behavior, and query limits.

Example:
    >>> config = ModelHandlerIntentConfig(
    ...     connection_uri="bolt://localhost:7687",
    ...     timeout_seconds=60.0,
    ...     circuit_breaker_threshold=3,
    ... )
    >>> config.timeout_seconds
    60.0

    >>> # Load from environment variables
    >>> config = ModelHandlerIntentConfig.from_env()
"""

from __future__ import annotations

import os

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["ModelHandlerIntentConfig"]


class ModelHandlerIntentConfig(BaseModel):  # omnimemory-model-exempt: handler config
    """Configuration for the HandlerIntent handler.

    Controls connection settings, authentication, circuit breaker behavior,
    and query limits for intent storage and retrieval operations.

    Attributes:
        connection_uri: Graph database connection URI (e.g., "bolt://localhost:7687").
        auth_username: Optional username for database authentication.
        auth_password: Optional password for database authentication.
        timeout_seconds: Connection and operation timeout in seconds.
            Bounded to prevent both overly aggressive timeouts and
            indefinite hangs. Defaults to 30.0.
        circuit_breaker_threshold: Number of consecutive failures before
            the circuit breaker opens. Once open, operations fail fast
            without attempting the actual operation. Defaults to 5.
        circuit_breaker_reset_timeout: Seconds to wait before attempting
            to close the circuit breaker after it opens. Defaults to 60.0.
        max_intents_per_session: Maximum number of intents to return per
            session in queries. Prevents unbounded result sets. Defaults to 100.
        default_confidence_threshold: Minimum confidence score for intent
            queries. Intents below this threshold are filtered out.
            Defaults to 0.0 (no filtering).
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        json_schema_extra={
            "examples": [
                {
                    "connection_uri": "bolt://localhost:7687",
                    "auth_username": "neo4j",
                    "auth_password": "<secret>",
                    "timeout_seconds": 30.0,
                    "circuit_breaker_threshold": 5,
                    "circuit_breaker_reset_timeout": 60.0,
                    "max_intents_per_session": 100,
                    "default_confidence_threshold": 0.5,
                },
                {
                    "connection_uri": "bolt://memgraph:7687",
                    "timeout_seconds": 60.0,
                    "circuit_breaker_threshold": 3,
                    "max_intents_per_session": 50,
                },
            ]
        },
    )

    connection_uri: str = Field(
        ...,
        min_length=1,
        description=(
            "Graph database connection URI. "
            "Example: 'bolt://localhost:7687' or 'bolt://memgraph:7687'."
        ),
    )
    auth_username: str | None = Field(
        default=None,
        description=(
            "Optional username for database authentication. "
            "If provided, auth_password should also be set."
        ),
    )
    auth_password: str | None = Field(
        default=None,
        description=(
            "Optional password for database authentication. "
            "If provided, auth_username should also be set."
        ),
    )
    timeout_seconds: float = Field(
        default=30.0,
        ge=0.1,
        le=300.0,
        description=(
            "Connection and operation timeout in seconds. " "Range: 0.1-300.0 seconds."
        ),
    )
    circuit_breaker_threshold: int = Field(
        default=5,
        ge=1,
        description=(
            "Number of consecutive failures before the circuit breaker opens. "
            "Once open, operations fail fast without attempting the actual operation."
        ),
    )
    circuit_breaker_reset_timeout: float = Field(
        default=60.0,
        ge=1.0,
        le=3600.0,
        description=(
            "Seconds to wait before attempting to close the circuit breaker "
            "after it opens. Range: 1.0-3600.0 seconds."
        ),
    )
    max_intents_per_session: int = Field(
        default=100,
        ge=1,
        le=1000,
        description=(
            "Maximum number of intents to return per session in queries. "
            "Prevents unbounded result sets. Range: 1-1000."
        ),
    )
    default_confidence_threshold: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum confidence score for intent queries. "
            "Intents below this threshold are filtered out. "
            "Range: 0.0-1.0 where 0.0 means no filtering."
        ),
    )

    @classmethod
    def from_env(
        cls,
        prefix: str = "HANDLER_INTENT_",
    ) -> ModelHandlerIntentConfig:
        """Create configuration from environment variables.

        Reads configuration values from environment variables with an optional
        prefix. This factory method provides a convenient way to configure
        the handler without hardcoding values.

        Environment Variables (with default prefix HANDLER_INTENT_):
            - HANDLER_INTENT_CONNECTION_URI: Required. Database connection URI.
            - HANDLER_INTENT_AUTH_USERNAME: Optional. Authentication username.
            - HANDLER_INTENT_AUTH_PASSWORD: Optional. Authentication password.
            - HANDLER_INTENT_TIMEOUT_SECONDS: Optional. Timeout in seconds.
            - HANDLER_INTENT_CIRCUIT_BREAKER_THRESHOLD: Optional. Failure threshold.
            - HANDLER_INTENT_CIRCUIT_BREAKER_RESET_TIMEOUT: Optional. Reset timeout.
            - HANDLER_INTENT_MAX_INTENTS_PER_SESSION: Optional. Max intents per query.
            - HANDLER_INTENT_DEFAULT_CONFIDENCE_THRESHOLD: Optional. Min confidence.

        Args:
            prefix: Environment variable prefix. Defaults to "HANDLER_INTENT_".

        Returns:
            ModelHandlerIntentConfig instance populated from environment.

        Raises:
            ValueError: If required environment variables are missing or invalid.

        Example:
            >>> import os
            >>> os.environ["HANDLER_INTENT_CONNECTION_URI"] = "bolt://localhost:7687"
            >>> os.environ["HANDLER_INTENT_TIMEOUT_SECONDS"] = "45.0"
            >>> config = ModelHandlerIntentConfig.from_env()
            >>> config.connection_uri
            'bolt://localhost:7687'
            >>> config.timeout_seconds
            45.0
        """
        connection_uri = os.environ.get(f"{prefix}CONNECTION_URI")
        if not connection_uri:
            msg = f"Required environment variable {prefix}CONNECTION_URI is not set"
            raise ValueError(msg)

        auth_username = os.environ.get(f"{prefix}AUTH_USERNAME")
        auth_password = os.environ.get(f"{prefix}AUTH_PASSWORD")

        timeout_str = os.environ.get(f"{prefix}TIMEOUT_SECONDS")
        timeout_seconds = float(timeout_str) if timeout_str else 30.0

        cb_threshold_str = os.environ.get(f"{prefix}CIRCUIT_BREAKER_THRESHOLD")
        circuit_breaker_threshold = int(cb_threshold_str) if cb_threshold_str else 5

        cb_reset_str = os.environ.get(f"{prefix}CIRCUIT_BREAKER_RESET_TIMEOUT")
        circuit_breaker_reset_timeout = float(cb_reset_str) if cb_reset_str else 60.0

        max_intents_str = os.environ.get(f"{prefix}MAX_INTENTS_PER_SESSION")
        max_intents_per_session = int(max_intents_str) if max_intents_str else 100

        threshold_str = os.environ.get(f"{prefix}DEFAULT_CONFIDENCE_THRESHOLD")
        default_confidence_threshold = float(threshold_str) if threshold_str else 0.0

        return cls(
            connection_uri=connection_uri,
            auth_username=auth_username,
            auth_password=auth_password,
            timeout_seconds=timeout_seconds,
            circuit_breaker_threshold=circuit_breaker_threshold,
            circuit_breaker_reset_timeout=circuit_breaker_reset_timeout,
            max_intents_per_session=max_intents_per_session,
            default_confidence_threshold=default_confidence_threshold,
        )

    def get_auth_tuple(self) -> tuple[str, str] | None:
        """Get authentication credentials as a tuple.

        Returns:
            Tuple of (username, password) if both are set, None otherwise.

        Example:
            >>> config = ModelHandlerIntentConfig(
            ...     connection_uri="bolt://localhost:7687",
            ...     auth_username="neo4j",
            ...     auth_password="<secret>",  # noqa: S106
            ... )
            >>> config.get_auth_tuple()
            ('neo4j', '<secret>')
        """
        if self.auth_username is not None and self.auth_password is not None:
            return (self.auth_username, self.auth_password)
        return None
