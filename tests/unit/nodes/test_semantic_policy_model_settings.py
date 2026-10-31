# SPDX-FileCopyrightText: 2026 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Semantic compute model names resolve from the node contract descriptor (OMN-21794)."""

from __future__ import annotations

import importlib.resources

import pytest
import yaml
from pydantic import ValidationError

from omnimemory.models.config import ModelSemanticComputePolicyConfig

_NODE = "omnimemory.nodes.node_semantic_analyzer_compute"


def _descriptor() -> dict[str, str]:
    contract = yaml.safe_load(
        importlib.resources.files(_NODE)
        .joinpath("contract.yaml")
        .read_text(encoding="utf-8")
    )
    return contract["descriptor"]


@pytest.mark.unit
class TestSemanticPolicyModelSettings:
    def test_contract_declares_every_model_value(self) -> None:
        descriptor = _descriptor()
        for name in (
            "allowed_embedding_models",
            "default_embedding_model",
            "allowed_llm_models",
            "default_llm_model",
        ):
            assert name in descriptor

    def test_descriptor_carries_no_model_id(self) -> None:
        for value in _descriptor().values():
            assert "${env." in str(value) or isinstance(value, list)

    def test_defaults_come_from_overlay(self) -> None:
        config = ModelSemanticComputePolicyConfig()
        assert config.default_embedding_model == "fixture-embed"
        assert config.default_llm_model == "fixture-llm"
        assert config.allowed_llm_models == ["fixture-llm-large", "fixture-llm"]

    def test_unset_overlay_fails_closed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for name in (
            "OMNIMEMORY_ALLOWED_EMBEDDING_MODELS",
            "OMNIMEMORY_DEFAULT_EMBEDDING_MODEL",
            "OMNIMEMORY_ALLOWED_LLM_MODELS",
            "OMNIMEMORY_DEFAULT_LLM_MODEL",
        ):
            monkeypatch.delenv(name)
        with pytest.raises(ValidationError, match="default_embedding_model"):
            ModelSemanticComputePolicyConfig()

    def test_env_overlay_replaces_descriptor_values(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OMNIMEMORY_ALLOWED_LLM_MODELS", "llm-a, llm-b")
        monkeypatch.setenv("OMNIMEMORY_DEFAULT_LLM_MODEL", "llm-b")
        monkeypatch.setenv("OMNIMEMORY_ALLOWED_EMBEDDING_MODELS", "embed-a")
        monkeypatch.setenv("OMNIMEMORY_DEFAULT_EMBEDDING_MODEL", "embed-a")
        config = ModelSemanticComputePolicyConfig()
        assert config.allowed_llm_models == ["llm-a", "llm-b"]
        assert config.default_llm_model == "llm-b"
        assert config.allowed_embedding_models == ["embed-a"]
        assert config.default_embedding_model == "embed-a"

    def test_overlay_default_outside_allowed_list_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OMNIMEMORY_DEFAULT_LLM_MODEL", "not-allowed")
        with pytest.raises(ValidationError, match="default_llm_model"):
            ModelSemanticComputePolicyConfig()
