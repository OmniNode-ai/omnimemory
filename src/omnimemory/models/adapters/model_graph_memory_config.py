# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Graph memory configuration model for Graph Memory adapter.

This module contains the ModelGraphMemoryConfig Pydantic model representing
the configuration for the Graph Memory adapter.

.. versionadded:: 0.1.0
    Initial implementation for OMN-1389.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

__all__ = [
    "ModelGraphMemoryConfig",
]


class ModelGraphMemoryConfig(BaseModel):
    """Configuration for the Graph Memory adapter.

    Attributes:
        max_depth: Maximum allowed traversal depth. Bounded to prevent
            expensive deep traversals. Defaults to 5.
        default_depth: Default traversal depth if not specified. Defaults to 2.
        default_limit: Default result limit. Defaults to 100.
        max_limit: Maximum allowed result limit. Defaults to 1000.
        bidirectional: Whether to traverse relationships in both directions.
            Defaults to True.
        memory_node_label: Graph label for memory nodes. Defaults to "Memory".
        timeout_seconds: Query timeout in seconds. Defaults to 30.0.
        score_filter_multiplier: Multiplier for query limit when min_score
            filtering is used. Higher values fetch more candidates but
            increase query cost. Defaults to 3.0. Range: 1.0-10.0.
        ensure_indexes: Whether to create indexes on memory_id during
            initialization. Defaults to True. Index creation is idempotent.
        max_retries: Maximum number of retry attempts for transient connection
            errors. Set to 0 to disable retries. Defaults to 3.
        retry_base_delay_seconds: Base delay (in seconds) for the first retry.
            Subsequent retries use exponential backoff:
            delay = min(base * 2^attempt + jitter, max_delay). Defaults to 1.0.
        retry_max_delay_seconds: Maximum delay cap (in seconds) for exponential
            backoff. Prevents unbounded wait times. Defaults to 30.0.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    max_depth: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Maximum allowed traversal depth",
    )
    default_depth: int = Field(
        default=2,
        ge=1,
        description="Default traversal depth",
    )
    default_limit: int = Field(
        default=100,
        ge=1,
        description="Default result limit",
    )
    max_limit: int = Field(
        default=1000,
        ge=1,
        description="Maximum allowed result limit",
    )
    bidirectional: bool = Field(
        default=True,
        description="Traverse relationships in both directions",
    )
    memory_node_label: str = Field(
        default="Memory",
        description="Graph label for memory nodes",
    )
    timeout_seconds: float = Field(
        default=30.0,
        gt=0.0,
        description="Query timeout in seconds",
    )
    score_filter_multiplier: float = Field(
        default=3.0,
        ge=1.0,
        le=10.0,
        description=(
            "Multiplier for query limit when min_score filtering is used. "
            "Higher values fetch more candidates but increase query cost. "
            "If min_score is high (>0.5), consider increasing this value."
        ),
    )
    ensure_indexes: bool = Field(
        default=True,
        description=(
            "Whether to create indexes on memory_id during initialization. "
            "Index creation is idempotent (safe to run multiple times). "
            "Set to False to skip index creation if managing indexes manually."
        ),
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description=(
            "Maximum number of retry attempts for transient connection errors "
            "(InfraConnectionError, InfraTimeoutError, InfraUnavailableError). "
            "Set to 0 to disable retries entirely. Range: 0-10."
        ),
    )
    retry_base_delay_seconds: float = Field(
        default=1.0,
        gt=0.0,
        le=60.0,
        description=(
            "Base delay in seconds for the first retry attempt. "
            "Subsequent retries use exponential backoff: "
            "delay = min(base * 2^attempt + jitter, retry_max_delay_seconds)."
        ),
    )
    retry_max_delay_seconds: float = Field(
        default=30.0,
        gt=0.0,
        description=(
            "Maximum delay cap in seconds for exponential backoff. "
            "Prevents unbounded wait times between retry attempts."
        ),
    )

    @field_validator("memory_node_label")
    @classmethod
    def validate_node_label(cls, v: str) -> str:
        """Validate memory_node_label is a valid Cypher label.

        Cypher labels must start with a letter or underscore and contain
        only letters, numbers, and underscores. This validation prevents
        potential injection issues when the label is used in queries.

        Args:
            v: The memory_node_label value to validate.

        Returns:
            The validated label if valid.

        Raises:
            ValueError: If the label does not match the required pattern.
        """
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", v):
            msg = (
                f"memory_node_label '{v}' is not a valid Cypher label. "
                "Must start with a letter or underscore, and contain only "
                "letters, numbers, and underscores."
            )
            raise ValueError(msg)
        return v

    @model_validator(mode="after")
    def validate_bounds(self) -> ModelGraphMemoryConfig:
        """Ensure default values do not exceed their maximums."""
        if self.default_depth > self.max_depth:
            msg = (
                f"default_depth ({self.default_depth}) "
                f"must be <= max_depth ({self.max_depth})"
            )
            raise ValueError(msg)
        if self.default_limit > self.max_limit:
            msg = (
                f"default_limit ({self.default_limit}) "
                f"must be <= max_limit ({self.max_limit})"
            )
            raise ValueError(msg)
        return self
