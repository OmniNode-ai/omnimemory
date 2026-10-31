# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Timeouts the memory lifecycle orchestrator handlers are injected with (OMN-21623)."""

from __future__ import annotations

import importlib.resources

import yaml
from omnibase_infra.runtime.overlay.contract_env_ref import expand_contract_env_refs
from pydantic import BaseModel, ConfigDict, Field


class ModelLifecycleTimeoutSettings(BaseModel):
    """The database, circuit-breaker and projection-reader timeouts of the lifecycle handlers.

    Every field is required and has no default: the values are the node contract's
    ``descriptor`` entries (``${env.VAR:default}`` overlay values), resolved by the
    binding site and injected, so the handler source holds none.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    db_query_timeout_seconds: float = Field(
        ...,
        gt=0,
        description="Timeout of one archive or expire database query, in seconds",
    )
    db_circuit_breaker_recovery_seconds: float = Field(
        ...,
        gt=0,
        description="Seconds the archive and expire database circuit breaker waits before a recovery attempt",
    )
    projection_reader_timeout_seconds: float = Field(
        ...,
        gt=0,
        description="Timeout of one tick projection-reader call, in seconds",
    )
    tick_circuit_breaker_recovery_seconds: float = Field(
        ...,
        gt=0,
        description="Seconds the tick projection-reader circuit breaker waits before a recovery attempt",
    )

    @classmethod
    def from_node_contract(cls) -> ModelLifecycleTimeoutSettings:
        """Build the settings from ``descriptor`` in the lifecycle node's contract.yaml.

        The handlers call this when no settings are injected. Each descriptor entry
        is a ``${env.VAR:default}`` overlay value expanded by
        ``expand_contract_env_refs``, the sanctioned env-reading boundary; a missing
        entry or a value that is not a positive number raises ``ValidationError``.
        """
        package = "omnimemory.nodes.node_memory_lifecycle_orchestrator"
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


__all__ = ["ModelLifecycleTimeoutSettings"]
