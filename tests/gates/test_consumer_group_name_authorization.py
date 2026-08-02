# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""AC3 consumer-group authorization gate for OmniMemory (OMN-15639).

Every Kafka consumer group name this repo can mint must be derivable from the
canonical grammar in ``omnibase_core.utils.util_consumer_group`` and must match
one of the six MSK IAM group patterns granted to the managed data plane.

Two independent assertions, both fail-closed:

``test_no_unmigrated_group_literals_in_src``
    Default-deny AST walk of ``src/omnimemory``. Every ``group_id=`` /
    ``consumer_group=`` / ``kafka_group_id=`` keyword argument, and every
    default value on a field whose name matches ``(consumer_)?group(_id)?``,
    must be a call to (or a local binding of) one of the canonical derivation
    helpers. A newly added string literal FAILS by default -- that is the whole
    point of the gate. ``_LEGACY_UNMIGRATED`` is the escape hatch and MUST stay
    empty at end state.

``test_derived_memory_group_is_iam_authorized``
    Positive + negative control against the pinned IAM pattern set that core
    vendors from Terraform. Proves the checker actually discriminates: the
    pre-OMN-15639 literal shape must be REJECTED, and the derived name must be
    ACCEPTED. Without the negative half a permissive translator would pass.

The pattern data has exactly one home (``omnibase_core`` packaged data); this
module imports it rather than re-vendoring it.

Reference: OMN-15639, seam table section 3 (IAM pattern set) and the AC3 gate
design (call-site enumeration, default-deny).
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src" / "omnimemory"

# Keyword arguments that carry a Kafka consumer group name to a bus/transport.
_GROUP_KEYWORDS = frozenset({"group_id", "consumer_group", "kafka_group_id"})

# Field/variable names that hold a consumer group name.
_GROUP_NAME_PATTERN = frozenset(
    {"group", "group_id", "consumer_group", "consumer_group_id"}
)

# The only functions permitted to mint a consumer group name. Sourced from
# omnibase_core.utils.util_consumer_group -- see the OMN-15639 seam table.
_CANONICAL_DERIVATIONS = frozenset(
    {
        "compute_consumer_group_id",
        "derive_prefixed_group_id",
        "apply_instance_discriminator",
    }
)

# Call sites not yet migrated to the canonical derivation. MUST be empty at end
# state; an entry here is proof-debt, not an exemption. Format: "<relpath>:<line>".
_LEGACY_UNMIGRATED: frozenset[str] = frozenset()


def _iter_source_files() -> list[Path]:
    return sorted(p for p in SRC_ROOT.rglob("*.py") if "__pycache__" not in p.parts)


def _callee_name(node: ast.AST) -> str | None:
    """Return the bare function name for a Call node's callee, if any."""
    if not isinstance(node, ast.Call):
        return None
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _collect_derived_bindings(tree: ast.Module) -> set[str]:
    """Names bound (anywhere in the module) to a canonical derivation call.

    Deliberately module-wide and therefore *permissive on binding lookup*: the
    fail-closed property lives in the value check below, and a module-wide
    binding set only ever matters when the module already contains a canonical
    derivation call. It never lets a bare literal through.
    """
    derived: set[str] = set()
    for node in ast.walk(tree):
        targets: list[ast.expr] = []
        value: ast.expr | None = None
        if isinstance(node, ast.Assign):
            targets, value = list(node.targets), node.value
        elif (isinstance(node, ast.AnnAssign) and node.value is not None) or isinstance(
            node, ast.NamedExpr
        ):
            targets, value = [node.target], node.value
        if value is None or _callee_name(value) not in _CANONICAL_DERIVATIONS:
            continue
        for target in targets:
            if isinstance(target, ast.Name):
                derived.add(target.id)
    return derived


def _value_is_canonical(value: ast.expr, derived_bindings: set[str]) -> bool:
    """True when `value` provably came from a canonical derivation helper."""
    # (i) direct call to a canonical derivation
    if _callee_name(value) in _CANONICAL_DERIVATIONS:
        return True
    # (ii) a local name bound to such a call. Anything else -- notably an
    # Attribute, which could resolve to any string -- fails closed.
    return isinstance(value, ast.Name) and value.id in derived_bindings


def _scan_file(path: Path) -> list[str]:
    """Return ``<relpath>:<line> <detail>`` for every non-conformant group site."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    derived_bindings = _collect_derived_bindings(tree)
    relpath = path.relative_to(REPO_ROOT).as_posix()
    findings: list[str] = []

    for node in ast.walk(tree):
        # --- keyword arguments: subscribe(group_id=...), Consumer(group_id=...)
        if isinstance(node, ast.Call):
            for keyword in node.keywords:
                if keyword.arg not in _GROUP_KEYWORDS:
                    continue
                if _value_is_canonical(keyword.value, derived_bindings):
                    continue
                findings.append(
                    f"{relpath}:{keyword.value.lineno} "
                    f"{keyword.arg}= is not derived from "
                    f"{sorted(_CANONICAL_DERIVATIONS)} "
                    f"(got {type(keyword.value).__name__})"
                )

        # --- defaults / assignments on a group-named field
        assignments: list[tuple[str, ast.expr]] = []
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    assignments.append((target.id, node.value))
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            if isinstance(node.target, ast.Name):
                assignments.append((node.target.id, node.value))

        for name, value in assignments:
            if name.lower() not in _GROUP_NAME_PATTERN:
                continue
            if _value_is_canonical(value, derived_bindings):
                continue
            findings.append(
                f"{relpath}:{value.lineno} {name} is assigned a "
                f"non-derived value (got {type(value).__name__})"
            )

    return findings


def test_no_unmigrated_group_literals_in_src() -> None:
    """Default-deny: no consumer group name in src/ escapes the canonical grammar."""
    assert SRC_ROOT.is_dir(), f"source root not found: {SRC_ROOT}"

    findings: list[str] = []
    for path in _iter_source_files():
        findings.extend(_scan_file(path))

    unexpected = [
        finding
        for finding in findings
        if finding.split(" ", 1)[0] not in _LEGACY_UNMIGRATED
    ]

    assert not unexpected, (
        "Non-conformant consumer group name(s) in src/omnimemory. Every group "
        "name must come from omnibase_core.utils.util_consumer_group "
        "(compute_consumer_group_id / derive_prefixed_group_id / "
        "apply_instance_discriminator) so it is IAM-authorized by "
        "construction:\n  " + "\n  ".join(unexpected)
    )


def test_legacy_unmigrated_allowlist_is_empty() -> None:
    """The escape hatch is proof-debt, not an exemption: it must stay empty."""
    assert not _LEGACY_UNMIGRATED, (
        "_LEGACY_UNMIGRATED must be empty at end state (OMN-15639); "
        f"still holding: {sorted(_LEGACY_UNMIGRATED)}"
    )


def test_derived_memory_group_is_iam_authorized() -> None:
    """Positive + negative control against core's pinned MSK IAM pattern set.

    Imported inside the test so that the default-deny AST assertion above stays
    collectable and independently meaningful; this assertion is the one that
    binds omnimemory to the shared pattern data.
    """
    from omnibase_core.enums.enum_consumer_group_purpose import (
        EnumConsumerGroupPurpose,
    )
    from omnibase_core.models.event_bus.model_consumer_group_scope import (
        ModelConsumerGroupScope,
    )
    from omnibase_core.utils.util_consumer_group import (
        compute_consumer_group_id,
        is_authorized_group_name,
    )

    from omnimemory.runtime.plugin import MEMORY_CONSUMER_GROUP_TAG

    class _Identity:
        env = "onex-dev"
        service = "omnimemory"
        node_name = "memory_domain_plugin"
        version = "v1"

    derived = compute_consumer_group_id(
        _Identity(),
        EnumConsumerGroupPurpose.CONSUME,
        scope=ModelConsumerGroupScope(ephemeral_tag=MEMORY_CONSUMER_GROUP_TAG),
    )

    assert is_authorized_group_name(derived), (
        f"derived omnimemory consumer group {derived!r} is not matched by any "
        "pinned MSK IAM group pattern"
    )

    # Negative controls -- the pre-OMN-15639 shape and its neighbours must be
    # REJECTED. If these pass, the ARN-glob translator is wrong (a '.' in the
    # pattern is literal, so 'onex-dev-...' and 'onex-...' must not match).
    for rejected in (
        "onex-runtime-memory",
        "test-consumer-memory",
        "omnimemory-memory",
        "onex-dev-omnimemory-memory",
        "local.omnimemory.memory_domain_plugin.consume.v1",
    ):
        assert not is_authorized_group_name(rejected), (
            f"{rejected!r} must NOT be authorized -- the IAM pattern matcher is "
            "too permissive (patterns are ARN globs; '.' and '-' are literal)"
        )
