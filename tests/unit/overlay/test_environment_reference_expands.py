# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-23140: an environment reference expands to the same value after the cut.

The memory consumers used to import the expansion from the infra runtime overlay package
(the graph's seams-verify side) and now import it from the primitive layer
(``omnibase_core.overlays.contract_env_ref``). The recorded cases below are the values the
infra copy returned for each reference form before the cut; the core function must return
the same value for each, and every consumer module must hold the core function.
"""

from __future__ import annotations

import importlib

import pytest
from omnibase_core.overlays.contract_env_ref import expand_contract_env_refs

pytestmark = pytest.mark.unit

# (reference text, environment, value recorded before the cut)
RECORDED_CASES: tuple[tuple[str, dict[str, str], str], ...] = (
    ("${env.QDRANT_HOST}", {"QDRANT_HOST": "qdrant.lane"}, "qdrant.lane"),
    ("${env.QDRANT_PORT:6333}", {}, "6333"),
    ("${env.QDRANT_PORT:6333}", {"QDRANT_PORT": "7000"}, "7000"),
    ("${env.QDRANT_HOST}", {}, ""),
    ("${env.LLM_EMBEDDING_URL:}", {}, ""),
    ("http://${env.HOST}:${env.PORT:80}/v1", {"HOST": "h1"}, "http://h1:80/v1"),
    ("no reference here", {}, "no reference here"),
)

CONSUMER_MODULES = (
    "omnimemory.models.config.model_intent_storage_settings",
    "omnimemory.models.config.model_lifecycle_timeout_settings",
    "omnimemory.models.config.model_memory_runtime_timeout_settings",
    "omnimemory.models.config.model_navigation_client_settings",
    "omnimemory.models.config.model_semantic_compute_policy_config",
    "omnimemory.nodes.node_memory_retrieval_effect.contract_descriptor",
    "omnimemory.protocols.error_models",
    "omnimemory.runtime.plugin",
)


@pytest.mark.parametrize(("reference", "environ", "expected"), RECORDED_CASES)
def test_environment_reference_expands_recorded_cases(
    reference: str,
    environ: dict[str, str],
    expected: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in ("QDRANT_HOST", "QDRANT_PORT", "LLM_EMBEDDING_URL", "HOST", "PORT"):
        monkeypatch.delenv(name, raising=False)
    for name, value in environ.items():
        monkeypatch.setenv(name, value)
    assert expand_contract_env_refs(reference) == expected


@pytest.mark.parametrize("module_name", CONSUMER_MODULES)
def test_environment_reference_expands_through_the_core_function_in_each_consumer(
    module_name: str,
) -> None:
    module = importlib.import_module(module_name)
    assert module.expand_contract_env_refs is expand_contract_env_refs


def test_environment_reference_expands_in_a_consumer_setting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One consumer end to end: the retrieval descriptor reads its host from the environment."""
    from omnimemory.nodes.node_memory_retrieval_effect import contract_descriptor

    monkeypatch.setenv("QDRANT_HOST", "qdrant.lane")
    monkeypatch.delenv("QDRANT_PORT", raising=False)
    assert (
        expand_contract_env_refs(
            str(
                contract_descriptor._load_descriptor(contract_descriptor._CONTRACT)[
                    "qdrant_host"
                ]
            )
        )
        == "qdrant.lane"
    )
