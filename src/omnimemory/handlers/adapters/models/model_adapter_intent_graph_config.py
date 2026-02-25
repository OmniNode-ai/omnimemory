# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Configuration model for the Intent Graph adapter.

This module provides the configuration model for AdapterIntentGraph,
which stores intent classification results in Memgraph. The configuration
controls timeouts, node labels, relationship types, and query limits.

Example:
    >>> config = ModelAdapterIntentGraphConfig(
    ...     timeout_seconds=60.0,
    ...     max_intents_per_session=50,
    ...     default_confidence_threshold=0.5,
    ... )
    >>> config.session_node_label
    'Session'
"""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

__all__ = ["ModelAdapterIntentGraphConfig"]

# Pattern for valid Cypher identifiers (labels and relationship types)
_CYPHER_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class ModelAdapterIntentGraphConfig(  # omnimemory-model-exempt: adapter config
    BaseModel
):
    """Configuration for the Intent Graph adapter.

    Controls connection settings, graph schema configuration, and query
    behavior for storing and retrieving intent classification results
    from Memgraph.

    Attributes:
        timeout_seconds: Connection and operation timeout in seconds.
            Bounded to prevent both overly aggressive timeouts and
            indefinite hangs. Defaults to 30.0.
        session_node_label: Graph label for Session nodes. Must be a valid
            Cypher identifier. Defaults to "Session".
        intent_node_label: Graph label for Intent nodes. Must be a valid
            Cypher identifier. Defaults to "Intent".
        relationship_type: Relationship type connecting Session to Intent
            nodes. Must be a valid Cypher identifier. Defaults to "HAD_INTENT".
        max_intents_per_session: Maximum number of intents to return per
            session in queries. Prevents unbounded result sets. Defaults to 100.
        default_confidence_threshold: Minimum confidence score for intent
            queries. Intents below this threshold are filtered out.
            Defaults to 0.0 (no filtering).
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    timeout_seconds: float = Field(
        default=30.0,
        ge=0.1,
        le=300.0,
        description=(
            "Connection and operation timeout in seconds. Range: 0.1-300.0 seconds."
        ),
    )
    session_node_label: str = Field(
        default="Session",
        description="Graph label for Session nodes",
    )
    intent_node_label: str = Field(
        default="Intent",
        description="Graph label for Intent nodes",
    )
    relationship_type: str = Field(
        default="HAD_INTENT",
        description="Relationship type connecting Session to Intent nodes",
    )
    max_intents_per_session: int = Field(
        default=100,
        ge=1,
        le=1000,
        description=(
            "Maximum number of intents to return per session in queries. "
            "Prevents unbounded result sets."
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
    auto_create_indexes: bool = Field(
        default=True,
        description=(
            "Whether to automatically create indexes during initialization. "
            "Set to False if indexes are managed externally or for testing."
        ),
    )

    @field_validator("session_node_label", "intent_node_label", "relationship_type")
    @classmethod
    def validate_cypher_identifier(cls, v: str, info: ValidationInfo) -> str:
        """Validate that the field is a valid Cypher identifier.

        Cypher identifiers (labels and relationship types) must start with
        a letter or underscore and contain only letters, numbers, and
        underscores. This validation prevents potential injection issues
        when the identifier is used in queries.

        Args:
            v: The value to validate.
            info: Pydantic validation context containing the field name.

        Returns:
            The validated value if valid.

        Raises:
            ValueError: If the value does not match the required pattern.
        """
        if not _CYPHER_IDENTIFIER_PATTERN.match(v):
            field_name = info.field_name
            msg = (
                f"{field_name} '{v}' is not a valid Cypher identifier. "
                "Must start with a letter or underscore, and contain only "
                "letters, numbers, and underscores."
            )
            raise ValueError(msg)
        return v
