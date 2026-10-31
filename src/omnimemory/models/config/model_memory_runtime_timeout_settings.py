# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Timeouts, intervals and bounds the memory runtime and storage config are injected with (OMN-21738)."""

from __future__ import annotations

import importlib.resources

import yaml
from omnibase_infra.runtime.overlay.contract_env_ref import expand_contract_env_refs
from pydantic import BaseModel, ConfigDict, Field


class ModelMemoryRuntimeTimeoutSettings(BaseModel):
    """The probe timeout, heartbeat interval and storage timeout bounds of the memory runtime.

    Every field is required and has no default: the values are the storage effect
    node contract's ``descriptor`` entries (``${env.VAR:default}`` overlay values),
    resolved at the binding site and injected, so the code holds none.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    memgraph_probe_timeout_seconds: float = Field(
        ...,
        gt=0,
        description="Seconds a TCP reachability probe of the graph store waits for a connection",
    )
    heartbeat_interval_seconds: float = Field(
        ...,
        gt=0,
        description="Seconds between introspection heartbeats of the memory effect nodes",
    )
    min_timeout_ms: int = Field(
        ...,
        gt=0,
        description="Smallest connection or idle timeout a storage config accepts, in milliseconds",
    )
    max_timeout_ms: int = Field(
        ...,
        gt=0,
        description="Largest connection or idle timeout a storage config accepts, in milliseconds",
    )

    @classmethod
    def from_node_contract(cls) -> ModelMemoryRuntimeTimeoutSettings:
        """Build the settings from ``descriptor`` in the storage effect node's contract.yaml.

        Each descriptor entry is a ``${env.VAR:default}`` overlay value expanded by
        ``expand_contract_env_refs``, the sanctioned env-reading boundary; a missing
        entry or a value that is not a positive number raises ``ValidationError``.
        """
        package = "omnimemory.nodes.node_memory_storage_effect"
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


__all__ = ["ModelMemoryRuntimeTimeoutSettings"]
