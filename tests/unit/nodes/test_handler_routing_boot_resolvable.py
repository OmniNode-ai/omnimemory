# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Contract guard (OMN-15235): every handler_routing target must be boot-resolvable.

Port of the omnimarket gate ``tests/test_handler_routing_boot_resolvable.py``
(OMN-13551 / OMN-13603) to omnimemory. The omnimarket copy's scan root is
``omnimarket/src/omnimarket/nodes`` and never saw omnimemory contracts, which is
why OMN-15235's ``index`` -> ``HandlerQdrant`` route sat latent instead of red.
The predicates below are the omnimarket ones verbatim, repointed at
``src/omnimemory/nodes``.

**The predicate.** At boot the runtime walks each contract's
``handler_routing.handlers`` and asks ``ServiceHandlerResolver`` to instantiate
the declared handler. The only providers available at that point are the three
known-injectable params (``event_bus``, ``container``, ``ownership_query``). A
handler whose constructor requires any *other* parameter with no default cannot
be resolved -> the resolver raises ``TypeError`` and the runtime quarantines the
handler with a boot warning. The contract then advertises an operation the
runtime can never serve: silent wiring death, green everywhere else.

**The allowlist is closed.** OMN-13603 converted the last three carve-out
handlers to the container-driven shape and emptied
``_UNWIRED_DEPENDENCY_ALLOWLIST``. It stays empty here. Padding it is the
rejected fix (OMN-15235 scope note 3) — a route to a handler that cannot be
constructed is removed or repointed, never excused.

**The unresolvable-target baseline is now empty (OMN-15268).**
``_PREEXISTING_UNIMPORTABLE_TARGETS`` froze the 11 omnimemory entries whose
declared target did not resolve to an object at all — a phantom class name, or a
flat entry with no nested ``handler: {name, module}`` for the OMN-14141 loader to
resolve. Those entries could not be ctor-checked because there was nothing to
inspect, so freezing them is what kept this gate from silently skipping them.
OMN-15268 burned all 11 down (repointed, implemented, converted to nested, or
removed with an unreachability proof — see the per-node comments in each
``contract.yaml`` and the per-entry regression tests at the bottom of this file).
The constant is retained, empty, and asserted **two** ways: by set equality
against the live scan (so a new phantom/flat entry fails immediately) and by a
closure test (so the baseline cannot be re-padded to excuse one). Both are
required — set equality alone would pass if a future change added an entry here
and a matching defect to a contract.
"""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Callable
from pathlib import Path
from typing import Any, get_args

import pytest
import yaml

# Providers the runtime can inject at boot via known-param injection
# (ServiceHandlerResolver step 4). Any other required, default-less param is
# unresolvable and quarantines the handler.
_KNOWN_INJECTABLE = frozenset({"event_bus", "container", "ownership_query"})

_CONCRETE_PARAM_KINDS = (
    inspect.Parameter.POSITIONAL_OR_KEYWORD,
    inspect.Parameter.KEYWORD_ONLY,
)

# Closed carve-out (OMN-13603): real, correctly-shaped handlers whose
# protocol-typed external dependency is not yet registered in the boot
# container. Format: "module.attr" -> reason. Empty, and it stays empty — the
# guard fails closed on any new misclassified or unwired entry.
_UNWIRED_DEPENDENCY_ALLOWLIST: dict[str, str] = {}

# Entries whose declared target does not resolve to an object, so the ctor
# predicate has nothing to inspect. EMPTY as of OMN-15268 and it stays empty:
# the 11 frozen entries were burned down at their source, not excused here.
# Format: "<node>::<operation>". Padding this is the rejected fix — a route
# whose target does not exist is repointed, implemented, converted to the nested
# schema, or removed with an unreachability proof.
_PREEXISTING_UNIMPORTABLE_TARGETS: frozenset[str] = frozenset()

# tests/unit/nodes/<this file> -> parents[3] is the repo root.
#
# Placement is load-bearing, not stylistic. omnimemory's governed test selector
# (scripts/ci/detect_test_paths.py) maps a change under
# src/omnimemory/nodes/** to the single path "tests/unit/nodes/", and contributes
# NOTHING for a changed test file outside tests/unit/ (`_resolve` only inspects
# SRC_PREFIX and TEST_UNIT_PREFIX). A future PR that adds a bad
# handler_routing entry touches src/omnimemory/nodes/<node>/contract.yaml, so at
# tests/ root this gate would be selected away on exactly the PRs it exists to
# catch — green until the dev->main promotion. Here it runs on every such PR.
# The root-level-tests-are-unselectable defect itself is OMN-15271; do not move
# this file back to tests/ root when that lands without re-checking the mapping.
_NODES_DIR = Path(__file__).parents[3] / "src" / "omnimemory" / "nodes"


def _required_non_injectable_params(obj: Callable[..., Any]) -> list[str]:
    """Return required, default-less ctor params the boot resolver cannot supply."""
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


def _route_key(node: str, entry: dict[str, Any]) -> str:
    """Return the stable "<node>::<operation>" identity of a routing entry."""
    for field in ("operation", "routing_key", "event_type"):
        value = entry.get(field)
        if isinstance(value, str) and value:
            return f"{node}::{value}"
    return f"{node}::?"


def _iter_routing_entries() -> list[tuple[str, str, dict[str, Any]]]:
    """Yield (node, route_key, entry) for every declared handler_routing entry."""
    out: list[tuple[str, str, dict[str, Any]]] = []
    for contract in sorted(_NODES_DIR.glob("*/contract.yaml")):
        node = contract.parent.name
        data = yaml.safe_load(contract.read_text(encoding="utf-8"))
        routing = (data or {}).get("handler_routing") or {}
        for entry in routing.get("handlers") or []:
            out.append((node, _route_key(node, entry), entry))
    return out


def _resolve(entry: dict[str, Any]) -> tuple[str, Callable[..., Any]] | None:
    """Resolve an entry to (fqn, object), or None if the target does not exist."""
    handler = entry.get("handler") or {}
    module = handler.get("module")
    name = handler.get("name")
    if not module or not name:
        return None
    try:
        mod = importlib.import_module(module)
    except ImportError:
        return None
    obj = getattr(mod, name, None)
    if obj is None:
        return None
    return f"{module}.{name}", obj


def _iter_declared_handlers() -> list[tuple[str, str, str, Callable[..., Any]]]:
    """Yield (node, route_key, fqn, obj) for every *resolvable* routing entry."""
    out: list[tuple[str, str, str, Callable[..., Any]]] = []
    for node, route_key, entry in _iter_routing_entries():
        resolved = _resolve(entry)
        if resolved is None:
            continue
        fqn, obj = resolved
        out.append((node, route_key, fqn, obj))
    return out


def test_all_handler_routing_targets_are_boot_resolvable() -> None:
    """Every resolvable handler_routing entry constructs from injectable params alone.

    OMN-15235 RED case: ``node_memory_retrieval_effect`` declared
    ``index`` -> ``HandlerQdrant``, whose ``__init__(self, config:
    ModelHandlerQdrantConfig)`` has no default and is not injectable, so
    ``_required_non_injectable_params`` returns ``["config"]``.
    """
    quarantines: list[str] = []
    for _node, route_key, fqn, obj in _iter_declared_handlers():
        unresolvable = _required_non_injectable_params(obj)
        if not unresolvable:
            continue
        if fqn in _UNWIRED_DEPENDENCY_ALLOWLIST:
            continue
        quarantines.append(
            f"  {route_key} :: {fqn} :: required-unresolvable={unresolvable}"
        )

    assert not quarantines, (
        f"{len(quarantines)} handler_routing entr(ies) would quarantine at boot "
        "(non-injectable required ctor params). These are not runtime-dispatched "
        "handlers — remove the misclassified entry (the node keeps its real "
        "handler) or repoint at a handler that can actually serve the operation. "
        "Do NOT add an allowlist entry:\n" + "\n".join(quarantines)
    )


def test_unwired_dependency_allowlist_is_closed() -> None:
    """The OMN-13603 carve-out is closed and stays closed.

    Guards the carve-out two ways: it must be empty, and if a future change
    reopens it, every entry must still be declared and still actually require an
    unwired dependency (a stale entry is deleted, not carried).
    """
    assert not _UNWIRED_DEPENDENCY_ALLOWLIST, (
        "_UNWIRED_DEPENDENCY_ALLOWLIST was closed under OMN-13603 and padding it "
        "is a rejected fix (OMN-15235). Fix the handler or remove the route: "
        f"{sorted(_UNWIRED_DEPENDENCY_ALLOWLIST)}"
    )
    declared = {fqn: obj for _, _, fqn, obj in _iter_declared_handlers()}
    for fqn in _UNWIRED_DEPENDENCY_ALLOWLIST:
        assert fqn in declared, (
            f"Allowlisted handler {fqn!r} is no longer declared in any "
            "handler_routing — remove it from _UNWIRED_DEPENDENCY_ALLOWLIST."
        )
        assert _required_non_injectable_params(declared[fqn]), (
            f"Allowlisted handler {fqn!r} no longer requires an unwired "
            "dependency — its DI is satisfied, so remove it from "
            "_UNWIRED_DEPENDENCY_ALLOWLIST."
        )


def test_unimportable_target_baseline_is_exact() -> None:
    """The set of unresolvable-target entries matches the frozen baseline exactly.

    Set equality, not membership: a NEW phantom/flat entry fails immediately, and
    burning one down also fails until the baseline is edited in the same PR.
    Without this the ctor gate above would silently skip these entries, and a
    silent skip is a check that does not exist.

    The baseline is empty since OMN-15268, so this now reads as "no
    handler_routing entry anywhere in omnimemory names a target that does not
    resolve to an object".
    """
    unimportable = {
        route_key
        for _, route_key, entry in _iter_routing_entries()
        if _resolve(entry) is None
    }
    assert unimportable == set(_PREEXISTING_UNIMPORTABLE_TARGETS), (
        "handler_routing entries with unresolvable targets drifted from the "
        "OMN-15268 baseline (empty).\n"
        f"  newly unresolvable: {sorted(unimportable - _PREEXISTING_UNIMPORTABLE_TARGETS)}\n"
        f"  fixed (remove from _PREEXISTING_UNIMPORTABLE_TARGETS): "
        f"{sorted(_PREEXISTING_UNIMPORTABLE_TARGETS - unimportable)}"
    )


def test_unimportable_target_baseline_is_closed() -> None:
    """The OMN-15268 burn-down baseline is empty and stays empty.

    Set equality alone is not enough: a future change could add a phantom route
    AND list it here, and the equality assert would pass. This closes that hole
    the same way ``test_unwired_dependency_allowlist_is_closed`` closes the
    ctor carve-out. There is no legitimate reason to record a route whose target
    does not exist — repoint it, implement it, convert it to the nested
    ``handler: {name, module}`` schema, or delete it with a written
    unreachability proof in the contract (the OMN-15235 precedent).
    """
    assert not _PREEXISTING_UNIMPORTABLE_TARGETS, (
        "_PREEXISTING_UNIMPORTABLE_TARGETS was driven to empty under OMN-15268 "
        "and re-padding it is a rejected fix. Fix the route at its source: "
        f"{sorted(_PREEXISTING_UNIMPORTABLE_TARGETS)}"
    )


def test_gate_scans_every_node_contract() -> None:
    """The scan root actually resolves — an empty scan would be a vacuous green.

    The omnimarket copy of this gate was green on omnimemory contracts only
    because it never read them (OMN-15235). A gate whose collector yields
    nothing passes for the wrong reason, so assert the root exists and produces
    entries.
    """
    assert _NODES_DIR.is_dir(), f"scan root does not exist: {_NODES_DIR}"
    entries = _iter_routing_entries()
    assert entries, f"no handler_routing entries collected from {_NODES_DIR}"


def test_index_operation_is_absent_from_memory_retrieval_contract() -> None:
    """OMN-15235 regression: `index` is gone from the retrieval node, both places.

    ``index`` was declared as a routing operation and in the contract's operation
    Literal, but ``ModelMemoryRetrievalRequest.operation`` — this node's declared
    ``input_model`` — never accepted it, and ``HandlerMemoryRetrieval.execute()``
    closes its match with ``assert_never(request.operation)``. Reintroducing the
    route (or the literal) reintroduces a route nothing can serve.
    """
    from omnimemory.nodes.node_memory_retrieval_effect.models import (
        ModelMemoryRetrievalRequest,
    )

    contract = _NODES_DIR / "node_memory_retrieval_effect" / "contract.yaml"
    data = yaml.safe_load(contract.read_text(encoding="utf-8"))

    routed_ops = {
        entry.get("operation")
        for entry in (data["handler_routing"].get("handlers") or [])
    }
    assert "index" not in routed_ops, (
        "node_memory_retrieval_effect routes 'index' again. No handler on this "
        "node can serve it — see the OMN-15235 note in contract.yaml."
    )

    constraint = str(data["validation_rules"]["constraint_definitions"]["operation"])
    quoted_index = {"'index'", '"index"'}
    assert not any(token in constraint for token in quoted_index), (
        "'index' is back in the operation constraint Literal, which contradicts "
        f"ModelMemoryRetrievalRequest.operation. Constraint: {constraint}"
    )

    # The authoritative seam: the DTO the runtime actually validates against.
    with pytest.raises(ValueError):
        ModelMemoryRetrievalRequest(operation="index")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# OMN-15268 per-entry burn-down regressions.
#
# The set-equality gate above proves "no unresolvable target anywhere". These
# pin the SPECIFIC disposition each of the 11 frozen entries received, so a
# future edit cannot satisfy the aggregate gate by re-introducing the defect in
# a different shape (e.g. re-appending a method name, or deleting a route whose
# handler_type is load-bearing as a transport declaration).
# ---------------------------------------------------------------------------


def _contract(node: str) -> dict[str, Any]:
    """Load one node contract.yaml as a dict."""
    path = _NODES_DIR / node / "contract.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict), f"{path} did not parse to a mapping"
    return data


def _routing(node: str) -> dict[str, Any]:
    """Return the handler_routing block of one node contract."""
    routing = _contract(node).get("handler_routing") or {}
    assert isinstance(routing, dict)
    return routing


def test_coordinator_routes_the_class_not_a_method_path() -> None:
    """OMN-15268: coordinator entries name the CLASS, never ``Class.method``.

    All four operations declared ``name: "HandlerSubscription.<method>"``. The
    OMN-14141 loader resolves ``handler.name`` with a single
    ``getattr(module, name)`` — a dotted path is an AttributeError, not a
    traversal — so every entry resolved to nothing at boot.
    """
    from omnimemory.handlers.handler_subscription import HandlerSubscription

    entries = _routing("node_agent_coordinator_orchestrator").get("handlers") or []
    assert {entry["operation"] for entry in entries} == {
        "subscribe",
        "unsubscribe",
        "list_subscriptions",
        "notify",
    }
    for entry in entries:
        name = entry["handler"]["name"]
        assert "." not in name, (
            f"handler.name {name!r} is a dotted path again. getattr() does not "
            "traverse dots — this resolves to nothing at boot."
        )
        assert name == HandlerSubscription.__name__
        assert entry["handler"]["module"] == HandlerSubscription.__module__

    # The repoint is only valid because the ctor is boot-injectable.
    assert not _required_non_injectable_params(HandlerSubscription)


def test_navigation_reducer_transport_entries_name_a_real_class() -> None:
    """OMN-15268: the transport entries resolve, and still declare their transport.

    ``qdrant``/``http`` named HandlerQdrant/HandlerHttp, which do not exist. They
    are repointed at the node's real handler rather than deleted, because
    ``handler_type`` on a routing entry is the transport-declaration surface the
    OCC imperative-contract-guard reads (``parse_contract_transports``) and
    ``metadata.transport_type`` is a scalar with no list form. Deleting them made
    the guard fail this node LIVE with "undeclared transport HTTP/QDRANT". This
    pins BOTH halves: every target resolves, and QDRANT/HTTP stay declared.
    """
    from omnimemory.nodes.node_navigation_history_reducer.handlers import (
        handler_navigation_history_reducer as nav_module,
    )
    from omnimemory.nodes.node_navigation_history_reducer.handlers.handler_navigation_history_reducer import (
        HandlerNavigationHistoryReducer,
    )
    from omnimemory.nodes.node_navigation_history_reducer.models.model_navigation_history_request import (
        ModelNavigationHistoryRequest,
    )

    contract = _contract("node_navigation_history_reducer")
    entries = (contract.get("handler_routing") or {}).get("handlers") or []
    keys = {entry.get("routing_key") for entry in entries}
    assert keys == {"qdrant", "http", "reduce"}, (
        f"node_navigation_history_reducer routing keys drifted: {sorted(keys)}"
    )

    # Half 1: no phantom target survives. The classes the old entries named do
    # not exist — re-executed here, not asserted from the ticket.
    assert not hasattr(nav_module, "HandlerQdrant")
    assert not hasattr(nav_module, "HandlerHttp")
    for entry in entries:
        assert entry["handler"]["name"] == HandlerNavigationHistoryReducer.__name__
        assert entry["handler"]["module"] == HandlerNavigationHistoryReducer.__module__

    # Half 2: the transport declarations the OCC guard reads are still present.
    declared = {contract["metadata"]["transport_type"].upper()} | {
        str(entry["handler"]["handler_type"]).upper()
        for entry in entries
        if entry["handler"].get("handler_type")
    }
    assert {"QDRANT", "HTTP"} <= declared, (
        "QDRANT/HTTP transport declarations were dropped. The handler really "
        "does open an AsyncQdrantClient and call the embedding endpoint over "
        f"HTTP, so the guard fails this node LIVE. Declared: {sorted(declared)}"
    )

    # Why the extra keys are harmless as routes: nothing can select them.
    assert "operation" not in ModelNavigationHistoryRequest.model_fields


def test_learning_retrieval_declares_no_phantom_handler() -> None:
    """OMN-15268: the route to the never-written HandlerAgentLearningRetrieval is gone.

    The module the entry named holds pure helper functions and no class. The
    route is removed rather than repointed: none of the helpers performs the
    retrieval this EFFECT node declares, and a bare function fails the boot ctor
    predicate anyway.
    """
    from omnimemory.nodes.node_agent_learning_retrieval_effect.handlers import (
        handler_agent_learning_retrieval as learning_module,
    )

    routing = _routing("node_agent_learning_retrieval_effect")
    handlers = routing.get("handlers") or []
    assert not handlers, (
        "node_agent_learning_retrieval_effect declares a handler again. There is "
        "no handler class in this node package — see the OMN-15268 note in "
        "contract.yaml. If one was implemented, update this test with it."
    )
    assert routing.get("default_handler") is None, (
        "default_handler names a handler that does not exist; it was set to null "
        "under OMN-15268."
    )
    assert not hasattr(learning_module, "HandlerAgentLearningRetrieval")


def test_persona_classify_target_is_a_real_def_b_handler() -> None:
    """OMN-15268: HandlerPersonaClassify exists and is def-B shaped.

    The contract reference was always correct; the class was missing. This pins
    all three properties the route depends on: it resolves, it constructs from
    injectable params alone, and it exposes a def-B ``handle`` over the node's
    declared input/output models.
    """
    from omnimemory.nodes.node_persona_builder_compute.handlers.handler_persona_classify import (
        HandlerPersonaClassify,
        classify_persona,
    )
    from omnimemory.nodes.node_persona_builder_compute.models import (
        ModelPersonaClassifyRequest,
        ModelPersonaClassifyResult,
    )

    entries = _routing("node_persona_builder_compute").get("handlers") or []
    assert len(entries) == 1
    assert entries[0]["handler"]["name"] == HandlerPersonaClassify.__name__
    assert entries[0]["handler"]["module"] == HandlerPersonaClassify.__module__

    assert not _required_non_injectable_params(HandlerPersonaClassify)

    handle = getattr(HandlerPersonaClassify, "handle", None)
    assert callable(handle), "HandlerPersonaClassify lost its def-B entrypoint"
    hints = inspect.signature(handle)
    params = [p for name, p in hints.parameters.items() if name != "self"]
    assert len(params) == 1, f"handle() is not def-B shaped: {hints}"

    # Behavioral parity with the pure function it fronts, on a real payload.
    request = ModelPersonaClassifyRequest(user_id="u-15268", signals=[])
    result = HandlerPersonaClassify().handle(request)
    assert isinstance(result, ModelPersonaClassifyResult)
    assert result == classify_persona(request)


def test_intent_storage_entries_are_nested_and_operation_keyed() -> None:
    """OMN-15268: the flat trio now carries a nested handler ref and an operation.

    ``handler`` is a REQUIRED field on ``ModelHandlerRoutingEntry``, so the flat
    ``handler_key``-only shape could not be parsed into the routing subcontract
    at all. The operation values are pinned against the DTO the runtime
    validates, not against the contract's own prose.
    """
    from omnimemory.nodes.node_intent_storage_effect.adapters import (
        HandlerIntentStorageAdapter,
    )
    from omnimemory.nodes.node_intent_storage_effect.models import (
        ModelIntentStorageRequest,
    )

    entries = _routing("node_intent_storage_effect").get("handlers") or []
    assert len(entries) == 3
    operations = {entry["operation"] for entry in entries}
    assert operations == {"store", "get_session", "get_distribution"}

    for entry in entries:
        handler = entry.get("handler")
        assert isinstance(handler, dict), (
            "flat handler_key shape is back — it does not satisfy the required "
            "`handler` field on ModelHandlerRoutingEntry."
        )
        assert handler["name"] == HandlerIntentStorageAdapter.__name__
        module = handler["module"]
        assert module, "empty handler.module never imports"
        assert getattr(importlib.import_module(module), handler["name"]) is (
            HandlerIntentStorageAdapter
        )

    # The seam: the routed operations are exactly the DTO's Literal members —
    # checked against the model the runtime validates, not the contract prose.
    declared = get_args(ModelIntentStorageRequest.model_fields["operation"].annotation)
    assert operations == set(declared), (
        f"routed operations {sorted(operations)} drifted from "
        f"ModelIntentStorageRequest.operation {sorted(declared)}"
    )

    assert not _required_non_injectable_params(HandlerIntentStorageAdapter)
    assert callable(getattr(HandlerIntentStorageAdapter, "handle", None))
