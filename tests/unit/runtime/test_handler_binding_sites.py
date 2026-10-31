# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Handlers are built only at binding sites (OMN-21623).

The contracts-and-DI rule ``handler_hand_construction`` allows a handler or node
class to be constructed in a ``container/registry_*.py`` binding site. These tests
run the core check over the package and build each binder's product.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from omnibase_core.container import ModelONEXContainer
from omnibase_core.validation.validator_contracts_and_di import ValidatorContractsAndDi

from omnimemory.handlers.container.registry_http_rest import bind_http_rest_handler
from omnimemory.handlers.container.registry_lifecycle import bind_memory_lifecycle
from omnimemory.models.config import ModelSemanticComputePolicyConfig
from omnimemory.nodes.node_memory_lifecycle_orchestrator.container.registry_memory_lifecycle import (
    bind_memory_expire,
)
from omnimemory.nodes.node_memory_retrieval_effect.container.registry_memory_retrieval import (
    bind_activation_decay,
    bind_db_mock,
    bind_graph_mock,
    bind_qdrant_mock,
    bind_rank_fusion,
)
from omnimemory.nodes.node_memory_retrieval_effect.models import (
    ModelHandlerDbMockConfig,
    ModelHandlerGraphMockConfig,
    ModelHandlerQdrantMockConfig,
)
from omnimemory.nodes.node_memory_storage_effect.container.registry_filesystem import (
    bind_filesystem_handler,
)
from omnimemory.nodes.node_navigation_history_reducer.container.registry_navigation_history import (
    bind_navigation_history_reducer,
)
from omnimemory.nodes.node_semantic_analyzer_compute.container.registry_semantic_compute import (
    bind_semantic_compute,
    bind_semantic_compute_policy,
)

_SRC = Path(__file__).resolve().parents[3] / "src" / "omnimemory"

_BOUND_SITES = (
    "adapters/adapter_observability_wrapper.py",
    "handlers/adapters/adapter_embedding_http.py",
    "nodes/node_intent_query_effect/container/registry_intent_query_effect.py",
    "nodes/node_memory_lifecycle_orchestrator/adapters/adapter_postgres_deactivate_memory.py",
    "nodes/node_memory_retrieval_effect/handlers/handler_graph_mock.py",
    "nodes/node_memory_retrieval_effect/handlers/handler_memory_retrieval.py",
    "nodes/node_memory_storage_effect/adapters/adapter_filesystem.py",
    "nodes/node_navigation_history_reducer/handlers/handler_navigation_history_reducer.py",
    "nodes/node_navigation_history_reducer/node_navigation_history_reducer.py",
    "nodes/node_semantic_analyzer_compute/handlers/handler_semantic_compute.py",
    "nodes/node_semantic_analyzer_compute/node_semantic_analyzer_compute.py",
    "runtime/plugin.py",
)


@pytest.mark.unit
def test_no_handler_is_constructed_outside_a_binding_site() -> None:
    """The core check finds no hand-constructed handler in the files bound here."""
    validator = ValidatorContractsAndDi()
    offenders = [
        f"{issue.file_path}:{issue.line_number}"
        for path in (_SRC / site for site in _BOUND_SITES)
        for issue in validator.validate_file(path).issues
        if issue.code == "handler_hand_construction"
    ]
    assert offenders == []


@pytest.mark.unit
def test_binders_build_their_handlers() -> None:
    container = ModelONEXContainer()
    assert type(bind_rank_fusion()).__name__ == "HandlerRankFusion"
    assert type(bind_activation_decay()).__name__ == "HandlerActivationDecay"
    assert (
        type(bind_qdrant_mock(ModelHandlerQdrantMockConfig())).__name__
        == "HandlerQdrantMock"
    )
    assert type(bind_db_mock(ModelHandlerDbMockConfig())).__name__ == "HandlerDbMock"
    assert (
        type(bind_graph_mock(ModelHandlerGraphMockConfig())).__name__
        == "HandlerGraphMock"
    )
    assert type(bind_memory_expire(container)).__name__ == "HandlerMemoryExpire"
    assert (
        type(
            bind_memory_lifecycle(
                graph_memory_adapter=None,
                intent_graph_adapter=None,
                navigation_handler=None,
                semantic_handler=None,
            )
        ).__name__
        == "HandlerMemoryLifecycle"
    )
    assert type(bind_semantic_compute(container)).__name__ == "HandlerSemanticCompute"
    assert (
        type(bind_semantic_compute_policy(ModelSemanticComputePolicyConfig())).__name__
        == "HandlerSemanticComputePolicy"
    )
    assert type(bind_filesystem_handler()).__name__ == "HandlerFileSystem"
    assert type(bind_http_rest_handler(container)).__name__ == "HandlerHttpRest"
    assert (
        type(bind_navigation_history_reducer()).__name__
        == "HandlerNavigationHistoryReducer"
    )
