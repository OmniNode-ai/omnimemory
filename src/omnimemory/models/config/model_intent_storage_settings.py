# SPDX-FileCopyrightText: 2026 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Contract-resolved circuit breaker settings for the intent storage adapter."""

from __future__ import annotations

import importlib.resources

import yaml
from omnibase_infra.runtime.overlay.contract_env_ref import expand_contract_env_refs
from pydantic import BaseModel, ConfigDict, Field


class ModelIntentStorageSettings(BaseModel):
    """Required settings injected at the intent storage handler binding boundary."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    failure_threshold: int = Field(
        ..., ge=1, description="Consecutive failures before opening the circuit"
    )
    recovery_timeout: float = Field(
        ..., ge=0, description="Seconds before trying a half-open circuit"
    )
    success_threshold: int = Field(
        ..., ge=1, description="Successful half-open calls before closing the circuit"
    )

    @classmethod
    def from_node_contract(cls) -> ModelIntentStorageSettings:
        """Resolve descriptor values and environment overlays at the binding site."""
        package = "omnimemory.nodes.node_intent_storage_effect"
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


__all__ = ["ModelIntentStorageSettings"]
