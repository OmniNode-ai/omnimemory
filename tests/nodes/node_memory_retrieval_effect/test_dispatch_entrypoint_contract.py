# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Dispatch-entrypoint and boot-resolvability gates for node_memory_retrieval_effect.

OMN-15227 (slice A2 of OMN-15027).

WHY THIS FILE EXISTS
--------------------
``omnimarket`` declares a downstream copy of this node's contract and runs two
fail-closed gates over every ``handler_routing.handlers[]`` entry:

* ``omnimarket/src/omnimarket/validators/handler_dispatch_entrypoint.py``
  (OMN-14617) — every declared handler must expose a ``handle_async``/``handle``
  dispatch entrypoint (canonical definition-B shape, CLAUDE.md rule 7a).
* ``omnimarket/tests/test_handler_routing_boot_resolvable.py`` (OMN-13603) —
  every declared handler must be constructible by the boot resolver, i.e. have
  no required, default-less constructor parameter outside the injectable set.

Neither gate reads ``default_handler`` — both iterate ``handlers[]`` and read
``entry["handler"]["name"|"module"]``. ``omnimemory`` has **zero**
``import omnimarket`` in ``src/``/``tests/`` and ``omnimarket`` already depends
on ``omnimemory``, so importing the real gates here would create a dependency
cycle. The predicates below are therefore copied **verbatim** from those two
files, each carrying a source citation. Do not "improve" them — if they drift
from the omnimarket originals they stop describing the gate and start
describing themselves.

.. versionadded:: 0.1.0
    Initial implementation for OMN-15227.
"""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Callable
from pathlib import Path
from typing import Any, Final, Literal, get_args
from uuid import uuid4

import pytest
import yaml
from omnibase_core.enums.enum_subject_type import EnumSubjectType
from omnibase_core.models.omnimemory import (
    ModelCostLedger,
    ModelMemorySnapshot,
    ModelSubjectRef,
)
from omnibase_core.models.primitives.model_semver import ModelSemVer

from omnimemory.nodes.node_memory_retrieval_effect import (
    ModelHandlerMemoryRetrievalConfig,
    ModelMemoryRetrievalRequest,
)
from omnimemory.nodes.node_memory_retrieval_effect.handlers import (
    HandlerMemoryRetrieval,
    ModelHandlerDbMockConfig,
    ModelHandlerGraphMockConfig,
    ModelHandlerQdrantMockConfig,
)

# =============================================================================
# Verbatim gate predicates (copied — see module docstring)
# =============================================================================

# VERBATIM COPY of omnimarket/src/omnimarket/validators/handler_dispatch_entrypoint.py
# ::has_dispatch_entrypoint (OMN-14617). Kept byte-identical to the shared
# auto-wiring resolution order (``handle_async`` then ``handle``).


def has_dispatch_entrypoint(cls: type) -> bool:
    """The EXACT predicate ``_make_dispatch_callback`` uses to bind an entrypoint."""
    return callable(getattr(cls, "handle_async", None)) or callable(
        getattr(cls, "handle", None)
    )


# VERBATIM COPY of omnimarket/tests/test_handler_routing_boot_resolvable.py
# ::_KNOWN_INJECTABLE / ::_CONCRETE_PARAM_KINDS / ::_required_non_injectable_params
# (OMN-13603). The omnimarket ``_UNWIRED_DEPENDENCY_ALLOWLIST`` is empty and
# fails closed, so no allowlist is mirrored here.

_KNOWN_INJECTABLE: Final = frozenset({"event_bus", "container", "ownership_query"})

_CONCRETE_PARAM_KINDS: Final = (
    inspect.Parameter.POSITIONAL_OR_KEYWORD,
    inspect.Parameter.KEYWORD_ONLY,
)


def _required_non_injectable_params(obj: Callable[..., Any]) -> list[str]:
    try:
        sig = inspect.signature(obj)
    except (TypeError, ValueError):
        return []
    required: list[str] = []
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        if (
            param.kind in _CONCRETE_PARAM_KINDS
            and param.default is inspect.Parameter.empty
            and name not in _KNOWN_INJECTABLE
        ):
            required.append(name)
    return required


# =============================================================================
# Contract parsing
# =============================================================================

_CONTRACT_PATH: Final = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "omnimemory"
    / "nodes"
    / "node_memory_retrieval_effect"
    / "contract.yaml"
)

# The operations this node can actually be asked to perform. Derived from the
# request model's own ``Literal`` rather than hand-listed, so it cannot drift.
# ``handler_routing`` entries for any other operation are unreachable by
# dispatch: ``HandlerMemoryRetrieval.execute`` closes the match with
# ``assert_never(request.operation)``, and no caller can construct a request
# carrying an operation outside this set.
_DISPATCHABLE_OPERATIONS: Final[frozenset[str]] = frozenset(
    get_args(ModelMemoryRetrievalRequest.model_fields["operation"].annotation)
)


def _dispatchable_contract_targets() -> list[tuple[str, str, str]]:
    """Return ``(operation, module, name)`` for every dispatchable routing entry.

    Drives the real ``contract.yaml`` — not a fixture — mirroring how the
    omnimarket gates walk ``handler_routing.handlers[]``.
    """
    data = yaml.safe_load(_CONTRACT_PATH.read_text())
    routing = (data or {}).get("handler_routing") or {}
    targets: list[tuple[str, str, str]] = []
    for entry in routing.get("handlers") or []:
        handler = (entry or {}).get("handler") or {}
        name, module = handler.get("name"), handler.get("module")
        operation = str((entry or {}).get("operation") or "")
        if name and module and operation in _DISPATCHABLE_OPERATIONS:
            targets.append((operation, str(module), str(name)))
    return targets


_DISPATCHABLE_TARGETS: Final = _dispatchable_contract_targets()

# OMN-15227 seam deliverable, consumed verbatim by the downstream omnimarket
# slice (A3). The omnimarket conversion must use THESE values, not a mechanical
# ``handler_key`` -> ``handler.name`` transliteration of the old Mock names.
_SEAM_MODULE: Final = (
    "omnimemory.nodes.node_memory_retrieval_effect.handlers.handler_memory_retrieval"
)
_SEAM_NAME: Final = "HandlerMemoryRetrieval"


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
async def seeded_handler() -> tuple[HandlerMemoryRetrieval, str]:
    """An initialised handler seeded with two related snapshots.

    Returns:
        The handler and the ``snapshot_id`` of the traversal root, as a string.
    """
    handler = HandlerMemoryRetrieval(
        ModelHandlerMemoryRetrievalConfig(
            use_stub_handlers=True,
            qdrant_mock_config=ModelHandlerQdrantMockConfig(embedding_dimension=1024),
            db_config=ModelHandlerDbMockConfig(case_sensitive=False),
            graph_config=ModelHandlerGraphMockConfig(bidirectional=True),
        )
    )
    await handler.initialize()

    root = _create_snapshot("root node")
    child = _create_snapshot("child node")
    handler.seed_snapshots([root, child])
    handler.add_graph_relationship(
        str(root.snapshot_id),
        str(child.snapshot_id),
        "related_to",
    )
    return handler, str(root.snapshot_id)


def _create_snapshot(subject_text: str) -> ModelMemorySnapshot:
    """Create a unique memory snapshot for testing."""
    return ModelMemorySnapshot(
        snapshot_id=uuid4(),
        subject=ModelSubjectRef(
            subject_type=EnumSubjectType.AGENT,
            subject_id=uuid4(),
            subject_key=subject_text,
        ),
        cost_ledger=ModelCostLedger(budget_total=100.0),
        schema_version=ModelSemVer(major=1, minor=0, patch=0),
    )


# =============================================================================
# 1. Entrypoint gate against the production dispatch target
# =============================================================================


class TestDispatchEntrypoint:
    """HandlerMemoryRetrieval must satisfy the OMN-14617 entrypoint gate."""

    def test_handler_memory_retrieval_has_dispatch_entrypoint(self) -> None:
        """The production retrieval handler exposes a definition-B entrypoint."""
        assert has_dispatch_entrypoint(HandlerMemoryRetrieval), (
            "HandlerMemoryRetrieval exposes neither 'handle_async' nor 'handle'; "
            "the omnimarket dispatch-entrypoint gate (OMN-14617) cannot bind it."
        )

    def test_handle_is_definition_b_shaped(self) -> None:
        """``handle`` takes exactly one request argument and is a coroutine."""
        handle = HandlerMemoryRetrieval.handle
        assert inspect.iscoroutinefunction(handle)
        assert _required_non_injectable_params(handle) == ["request"]

    def test_handler_memory_retrieval_is_boot_resolvable(self) -> None:
        """The production retrieval handler satisfies the OMN-13603 boot gate."""
        assert _required_non_injectable_params(HandlerMemoryRetrieval) == []


# =============================================================================
# 2. Contract-target gate — the correction's proof (OMN-15227)
# =============================================================================


class TestContractDeclaredTargets:
    """Every dispatchable contract-declared handler must pass BOTH gates.

    This is the test that would have caught the OMN-15027 mis-scope: before the
    OMN-15227 repoint the contract routed ``search``/``search_text``/
    ``search_graph`` at ``HandlerQdrantMock``/``HandlerDbMock``/
    ``HandlerGraphMock``, each of which exposes only ``execute()`` and requires
    a default-less ``config`` parameter — failing both gates twice over.
    """

    def test_contract_declares_every_dispatchable_operation(self) -> None:
        """No dispatchable operation is missing a routing entry."""
        routed = {operation for operation, _, _ in _DISPATCHABLE_TARGETS}
        assert routed == _DISPATCHABLE_OPERATIONS

    @pytest.mark.parametrize(
        ("operation", "module", "name"),
        _DISPATCHABLE_TARGETS,
        ids=[f"{op}:{nm}" for op, _, nm in _DISPATCHABLE_TARGETS],
    )
    def test_declared_target_has_dispatch_entrypoint(
        self, operation: str, module: str, name: str
    ) -> None:
        """Contract-declared handler exposes ``handle_async`` or ``handle``."""
        cls = getattr(importlib.import_module(module), name)
        assert has_dispatch_entrypoint(cls), (
            f"contract.yaml routes operation '{operation}' to {module}.{name}, "
            "which exposes neither 'handle_async' nor 'handle'."
        )

    @pytest.mark.parametrize(
        ("operation", "module", "name"),
        _DISPATCHABLE_TARGETS,
        ids=[f"{op}:{nm}" for op, _, nm in _DISPATCHABLE_TARGETS],
    )
    def test_declared_target_is_boot_resolvable(
        self, operation: str, module: str, name: str
    ) -> None:
        """Contract-declared handler has no required non-injectable ctor param."""
        cls = getattr(importlib.import_module(module), name)
        offenders = _required_non_injectable_params(cls)
        assert offenders == [], (
            f"contract.yaml routes operation '{operation}' to {module}.{name}, "
            f"whose constructor requires non-injectable parameter(s) {offenders}; "
            f"the boot resolver injects only {sorted(_KNOWN_INJECTABLE)}."
        )

    def test_seam_deliverable_target_is_the_production_handler(self) -> None:
        """Pin the seam the downstream omnimarket slice (A3) must transliterate.

        The omnimarket conversion MUST use these values, NOT a mechanical
        ``handler_key`` -> ``handler.name`` transliteration of the old Mock names.
        """
        assert {(module, name) for _, module, name in _DISPATCHABLE_TARGETS} == {
            (_SEAM_MODULE, _SEAM_NAME)
        }


# =============================================================================
# 3. Behavioural equivalence: handle() == execute()
# =============================================================================


def _build_request(operation: str, snapshot_id: str) -> ModelMemoryRetrievalRequest:
    """Build a valid request for each dispatchable operation."""
    if operation == "search_graph":
        return ModelMemoryRetrievalRequest(
            operation="search_graph",
            snapshot_id=snapshot_id,
            traversal_depth=2,
        )
    return ModelMemoryRetrievalRequest(
        operation=cast_operation(operation),
        query_text="root node",
    )


def cast_operation(
    operation: str,
) -> Literal["search", "search_text", "search_graph"]:
    """Narrow a contract-sourced operation string to the request literal."""
    assert operation in _DISPATCHABLE_OPERATIONS
    return operation  # type: ignore[return-value]


class TestHandleMatchesExecute:
    """``handle()`` is an alias, not a second implementation."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("operation", sorted(_DISPATCHABLE_OPERATIONS))
    async def test_handle_equals_execute(
        self,
        operation: str,
        seeded_handler: tuple[HandlerMemoryRetrieval, str],
    ) -> None:
        """For every operation, ``handle`` returns what ``execute`` returns."""
        handler, root_id = seeded_handler
        request = _build_request(operation, root_id)

        via_execute = await handler.execute(request)
        via_handle = await handler.handle(request)

        assert via_handle.model_dump() == via_execute.model_dump()


# =============================================================================
# 4. Routing preservation through handle()
# =============================================================================


class TestRoutingPreservedThroughHandle:
    """The repoint must not change dispatch semantics."""

    @pytest.mark.asyncio
    async def test_search_graph_through_handle_reaches_graph_sub_handler(
        self,
        seeded_handler: tuple[HandlerMemoryRetrieval, str],
    ) -> None:
        """``search_graph`` via ``handle()`` still lands on the graph sub-handler.

        Asserted on the observable result: only the graph sub-handler populates
        ``result.path`` with the traversal origin.
        """
        handler, root_id = seeded_handler
        request = _build_request("search_graph", root_id)

        response = await handler.handle(request)

        assert response.status == "success"
        assert response.results, "graph traversal returned no connected snapshots"
        for result in response.results:
            assert result.path is not None
            assert root_id in result.path

    @pytest.mark.asyncio
    async def test_search_through_handle_reaches_semantic_sub_handler(
        self,
        seeded_handler: tuple[HandlerMemoryRetrieval, str],
    ) -> None:
        """``search`` via ``handle()`` still lands on the semantic sub-handler.

        Only the Qdrant sub-handler reports the embedding it searched with.
        """
        handler, root_id = seeded_handler
        response = await handler.handle(_build_request("search", root_id))

        assert response.status in ("success", "no_results")
        assert response.query_embedding_used is not None
