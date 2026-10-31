# SPDX-FileCopyrightText: 2026 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The semantic compute policy reads its models from the node contract overlay (OMN-21794)."""

from __future__ import annotations

import importlib.resources
from collections.abc import Iterator

import pydantic
import pytest
import yaml

from omnimemory.models.config import ModelSemanticComputePolicyConfig
from omnimemory.models.config.model_semantic_model_routing import (
    ModelSemanticModelRouting,
)

_NODE_PACKAGE = "omnimemory.nodes.node_semantic_analyzer_compute"
_ALLOWED_EMBEDDING = "OMNIMEMORY_SEMANTIC_ALLOWED_EMBEDDING_MODELS"
_DEFAULT_EMBEDDING = "OMNIMEMORY_SEMANTIC_DEFAULT_EMBEDDING_MODEL"
_ALLOWED_LLM = "OMNIMEMORY_SEMANTIC_ALLOWED_LLM_MODELS"
_DEFAULT_LLM = "OMNIMEMORY_SEMANTIC_DEFAULT_LLM_MODEL"


@pytest.fixture
def overlay(monkeypatch: pytest.MonkeyPatch) -> Iterator[pytest.MonkeyPatch]:
    monkeypatch.setenv(_ALLOWED_EMBEDDING, "embed-a, embed-b")
    monkeypatch.setenv(_DEFAULT_EMBEDDING, "embed-b")
    monkeypatch.setenv(_ALLOWED_LLM, "llm-a,llm-b")
    monkeypatch.setenv(_DEFAULT_LLM, "llm-a")
    ModelSemanticModelRouting.from_node_contract.cache_clear()
    yield monkeypatch
    monkeypatch.undo()
    ModelSemanticModelRouting.from_node_contract.cache_clear()


@pytest.mark.unit
class TestSemanticModelRouting:
    def test_contract_binds_every_model_to_an_overlay_reference(self) -> None:
        document = yaml.safe_load(
            importlib.resources.files(_NODE_PACKAGE)
            .joinpath("contract.yaml")
            .read_text(encoding="utf-8")
        )
        routing = document["model_routing"]
        assert set(routing) == set(ModelSemanticModelRouting.model_fields)
        assert all(str(v).startswith("${env.") for v in routing.values())

    def test_routing_comes_from_the_overlay(self, overlay: pytest.MonkeyPatch) -> None:
        routing = ModelSemanticModelRouting.from_node_contract()
        assert routing.allowed_embedding_models == ("embed-a", "embed-b")
        assert routing.default_embedding_model == "embed-b"
        assert routing.allowed_llm_models == ("llm-a", "llm-b")
        assert routing.default_llm_model == "llm-a"

    def test_policy_defaults_come_from_the_overlay(
        self, overlay: pytest.MonkeyPatch
    ) -> None:
        config = ModelSemanticComputePolicyConfig()
        assert config.allowed_embedding_models == ["embed-a", "embed-b"]
        assert config.default_embedding_model == "embed-b"
        assert config.allowed_llm_models == ["llm-a", "llm-b"]
        assert config.default_llm_model == "llm-a"

    def test_explicit_models_still_override_the_overlay(
        self, overlay: pytest.MonkeyPatch
    ) -> None:
        config = ModelSemanticComputePolicyConfig(
            allowed_llm_models=["llm-x"], default_llm_model="llm-x"
        )
        assert config.default_llm_model == "llm-x"

    def test_unset_overlay_fails_closed(self, overlay: pytest.MonkeyPatch) -> None:
        overlay.delenv(_DEFAULT_LLM)
        ModelSemanticModelRouting.from_node_contract.cache_clear()
        with pytest.raises(pydantic.ValidationError, match="default_llm_model"):
            ModelSemanticModelRouting.from_node_contract()

    def test_node_without_model_routing_is_refused(self) -> None:
        with pytest.raises(ValueError, match="declares no model_routing section"):
            ModelSemanticModelRouting.from_node_contract(
                "omnimemory.nodes.node_similarity_compute"
            )
