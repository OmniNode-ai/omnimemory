# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Client settings the navigation-history writer is injected with (OMN-21623)."""

from __future__ import annotations

import importlib.resources

import yaml
from omnibase_infra.runtime.overlay.contract_env_ref import expand_contract_env_refs
from pydantic import BaseModel, ConfigDict, Field


class ModelNavigationClientSettings(BaseModel):
    """The embedding model and client timeouts of the navigation-history writer.

    Every field is required and has no default: the values are the node contract's
    ``descriptor`` entries (``${env.VAR:default}`` overlay values), resolved by the
    binding site and injected, so the handler source holds none.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    embedding_model: str = Field(
        ...,
        min_length=1,
        description="Model identifier sent to the embedding endpoint",
    )
    pg_command_timeout_seconds: float = Field(
        ...,
        gt=0,
        description="Command timeout of the asyncpg pool, in seconds",
    )
    qdrant_timeout_seconds: int = Field(
        ...,
        gt=0,
        description="Request timeout of the Qdrant client, in whole seconds",
    )
    embedding_http_timeout_seconds: float = Field(
        ...,
        gt=0,
        description="Timeout of the embedding HTTP call, in seconds",
    )

    @classmethod
    def from_node_contract(cls) -> ModelNavigationClientSettings:
        """Build the settings from ``descriptor`` in the navigation node's contract.yaml.

        The binding site (``runtime.plugin``) calls this once and injects the result.
        Each descriptor entry is a ``${env.VAR:default}`` overlay value expanded by
        ``expand_contract_env_refs``, the sanctioned env-reading boundary; a missing
        entry or a value that is not a positive number raises ``ValidationError``.
        """
        package = "omnimemory.nodes.node_navigation_history_reducer"
        contract = yaml.safe_load(
            importlib.resources.files(package)
            .joinpath("contract.yaml")
            .read_text(encoding="utf-8")
        )
        descriptor = contract.get("descriptor") if isinstance(contract, dict) else None
        if not isinstance(descriptor, dict):
            raise ValueError(f"{package} contract has no descriptor mapping")
        return cls.model_validate(
            {
                name: expand_contract_env_refs(str(descriptor.get(name, "")))
                for name in cls.model_fields
            }
        )


__all__ = ["ModelNavigationClientSettings"]
