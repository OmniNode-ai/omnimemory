# SPDX-FileCopyrightText: 2026 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Model routing the semantic analyzer node contract declares (OMN-21794)."""

from __future__ import annotations

import functools
import importlib.resources

import yaml
from omnibase_infra.runtime.overlay.contract_env_ref import expand_contract_env_refs
from pydantic import BaseModel, ConfigDict, Field, field_validator

_SEMANTIC_NODE_PACKAGE = "omnimemory.nodes.node_semantic_analyzer_compute"


class ModelSemanticModelRouting(BaseModel):
    """The embedding and LLM models the semantic analyzer contract allows.

    Every field is required and has no default: the values are the node
    contract's ``model_routing`` entries (``${env.VAR}`` overlay values), so no
    model identifier is written in source or in the repository. A model list is
    comma separated in the overlay; an unset overlay expands to the empty string
    and fails validation.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    allowed_embedding_models: tuple[str, ...] = Field(
        ..., min_length=1, description="Embedding model identifiers the contract allows"
    )
    default_embedding_model: str = Field(
        ..., min_length=1, description="Embedding model used when a request names none"
    )
    allowed_llm_models: tuple[str, ...] = Field(
        ..., min_length=1, description="LLM model identifiers the contract allows"
    )
    default_llm_model: str = Field(
        ..., min_length=1, description="LLM model used when a request names none"
    )

    @field_validator("allowed_embedding_models", "allowed_llm_models", mode="before")
    @classmethod
    def _split_overlay_list(cls, value: object) -> object:
        if isinstance(value, str):
            return tuple(item.strip() for item in value.split(",") if item.strip())
        return value

    @classmethod
    @functools.cache
    def from_node_contract(
        cls, package: str = _SEMANTIC_NODE_PACKAGE
    ) -> ModelSemanticModelRouting:
        """Read the node package's contract.yaml ``model_routing`` section.

        Each entry is expanded by ``expand_contract_env_refs``, the sanctioned
        env-reading boundary. The result is cached for the process.
        """
        contract = yaml.safe_load(
            importlib.resources.files(package)
            .joinpath("contract.yaml")
            .read_text(encoding="utf-8")
        )
        routing = contract.get("model_routing") if isinstance(contract, dict) else None
        if not isinstance(routing, dict):
            raise ValueError(f"{package} contract declares no model_routing section")
        return cls.model_validate(
            {
                name: expand_contract_env_refs(str(value))
                for name, value in routing.items()
            }
        )


__all__ = ["ModelSemanticModelRouting"]
