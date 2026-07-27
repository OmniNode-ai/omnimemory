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

**Known gap, deliberately visible.** ``_PREEXISTING_UNIMPORTABLE_TARGETS``
freezes the omnimemory entries whose declared target does not resolve to an
object at all (phantom class name, or flat schema with no nested
``handler: {name, module}``). Those entries cannot be ctor-checked because there
is nothing to inspect, so recording them here is what keeps this gate from
silently skipping them. It is asserted by **set equality**, not membership: the
set growing OR shrinking fails the test. Burn-down is OMN-15268.
"""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Callable
from pathlib import Path
from typing import Any

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
# predicate has nothing to inspect. Frozen so the ctor gate does not silently
# skip them. Burn-down: OMN-15268. Format: "<node>::<operation>".
_PREEXISTING_UNIMPORTABLE_TARGETS: frozenset[str] = frozenset(
    {
        # Phantom class names: module.Class.method appended to the module path,
        # or a class that does not exist in the named module.
        "node_agent_coordinator_orchestrator::subscribe",
        "node_agent_coordinator_orchestrator::unsubscribe",
        "node_agent_coordinator_orchestrator::list_subscriptions",
        "node_agent_coordinator_orchestrator::notify",
        "node_agent_learning_retrieval_effect::retrieve",
        "node_persona_builder_compute::classify",
        "node_navigation_history_reducer::qdrant",
        "node_navigation_history_reducer::http",
        # Flat schema: no nested handler: {name, module} to resolve at all.
        "node_intent_storage_effect::store",
        "node_intent_storage_effect::get_session",
        "node_intent_storage_effect::get_distribution",
    }
)

_NODES_DIR = Path(__file__).parent.parent / "src" / "omnimemory" / "nodes"


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
    burning one down (OMN-15268) also fails until the baseline is edited in the
    same PR. Without this the ctor gate above would silently skip these entries,
    and a silent skip is a check that does not exist.
    """
    unimportable = {
        route_key
        for _, route_key, entry in _iter_routing_entries()
        if _resolve(entry) is None
    }
    assert unimportable == set(_PREEXISTING_UNIMPORTABLE_TARGETS), (
        "handler_routing entries with unresolvable targets drifted from the "
        "OMN-15268 baseline.\n"
        f"  newly unresolvable: {sorted(unimportable - _PREEXISTING_UNIMPORTABLE_TARGETS)}\n"
        f"  fixed (remove from _PREEXISTING_UNIMPORTABLE_TARGETS): "
        f"{sorted(_PREEXISTING_UNIMPORTABLE_TARGETS - unimportable)}"
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
