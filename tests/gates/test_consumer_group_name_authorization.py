# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""AC3 consumer-group authorization gate for OmniMemory (OMN-15639).

Every Kafka consumer group name this repo can mint must be derivable from the
canonical grammar in ``omnibase_core.event_bus.util_consumer_group`` and must match
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

``test_checker_*`` (checker-of-the-checker)
    Positive and negative controls over the AST walk itself, run against inline
    source fixtures. A default-deny gate whose discrimination is never exercised
    is indistinguishable from a gate that always passes; these prove it catches
    literals, f-strings, and laundering while admitting real derivations. Added
    in the OMN-15639 round-2 remediation after a verifier demonstrated the
    ``apply_instance_discriminator`` laundering hole live.

``test_derived_memory_group_is_iam_authorized``
    Positive + negative control against the pinned IAM pattern set that core
    vendors from Terraform. Proves the checker actually discriminates: the
    pre-OMN-15639 literal shape must be REJECTED, and the derived name must be
    ACCEPTED. Without the negative half a permissive translator would pass.

``test_seam_scope_field_name_is_ephemeral_tag``
    Seam pin (OMN-14208). ``src/omnimemory/runtime/plugin.py`` constructs
    ``ModelConsumerGroupScope(ephemeral_tag=...)``; the OMN-15639 seam table
    pins the module path and the function signatures but NOT that model's field
    names. This asserts the field name field-by-field so a core-lane rename
    fails here loudly instead of surfacing as a runtime ``TypeError`` while both
    lanes' own suites stay green.

``test_unauthorized_env_token_is_not_authorized``
    Names the deployed-env-token residual in executable form: the derived
    grammar is only IAM-conformant when the runtime env token is one MSK grants.
    The kernel's default env is ``local`` (``ONEX_ENVIRONMENT``), and
    ``local.omnimemory.*`` is NOT in the granted pattern set -- only
    ``local.runtime_config.*`` is. A green suite here therefore does NOT prove
    the deployed name is authorized; the deployment must set the env token to
    ``onex-dev``. See "Residual proof legs" in the PR body.

The pattern data has exactly one home (``omnibase_core`` packaged data); this
module imports it rather than re-vendoring it.

Reference: OMN-15639, seam table section 3 (IAM pattern set) and the AC3 gate
design (call-site enumeration, default-deny).
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src" / "omnimemory"

# Keyword arguments that carry a Kafka consumer group name to a bus/transport.
_GROUP_KEYWORDS = frozenset({"group_id", "consumer_group", "kafka_group_id"})

# Field/variable names that hold a consumer group name, including the known
# near-miss spellings CodeRabbit flagged. Do not match arbitrary "*_group"
# locals; unrelated variables such as invalid_group are not Kafka group names.
_GROUP_NAME_PATTERN = re.compile(
    r"^(?:group|group_id|group_name|consumer_group|consumer_group_id|"
    r"consumer_group_name|memory_group|memory_group_id|memory_group_name|"
    r"kafka_group|kafka_group_id|kafka_group_name)$"
)

# Functions that MINT a group name from structured inputs (a node identity or a
# reserved prefix). Their result is canonical by construction. Sourced from
# omnibase_core.event_bus.util_consumer_group -- see the OMN-15639 seam table.
_MINTING_DERIVATIONS = frozenset(
    {
        "compute_consumer_group_id",
        "derive_consumer_group_id",
        "derive_prefixed_group_id",
    }
)

# Functions that TRANSFORM an already-derived group name
# (``apply_instance_discriminator(group_id, instance_id)`` appends a suffix).
# These are canonical ONLY when their first positional argument is itself
# canonical -- otherwise they launder a literal through the allowlist, which a
# round-1 verifier demonstrated live against the previous version of this gate:
#   subscribe(group_id=apply_instance_discriminator("onex-runtime-memory", None))
# passed. Now it fails. See ``test_checker_rejects_laundered_literal``.
_LAUNDERING_DERIVATIONS = frozenset({"apply_instance_discriminator"})

# Union, used only for human-readable failure messages.
_CANONICAL_DERIVATIONS = _MINTING_DERIVATIONS | _LAUNDERING_DERIVATIONS

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


def _iter_binding_pairs(tree: ast.Module) -> list[tuple[str, ast.expr]]:
    """Every ``(bound_name, value_expr)`` pair in the module."""
    pairs: list[tuple[str, ast.expr]] = []
    for node in ast.walk(tree):
        targets: list[ast.expr] = []
        value: ast.expr | None = None
        if isinstance(node, ast.Assign):
            targets, value = list(node.targets), node.value
        elif (isinstance(node, ast.AnnAssign) and node.value is not None) or isinstance(
            node, ast.NamedExpr
        ):
            targets, value = [node.target], node.value
        if value is None:
            continue
        for target in targets:
            if isinstance(target, ast.Name):
                pairs.append((target.id, value))
    return pairs


def _collect_derived_bindings(tree: ast.Module) -> set[str]:
    """Names bound (anywhere in the module) to a canonical derivation.

    Computed as a least fixpoint so that a chain
    ``base = compute_consumer_group_id(...)`` ->
    ``scoped = apply_instance_discriminator(base, instance)`` resolves, while a
    laundered literal (``apply_instance_discriminator("literal", ...)``) never
    enters the set. Monotone and bounded by the number of bound names, so the
    loop terminates.

    Deliberately module-wide and therefore *permissive on binding lookup*: the
    fail-closed property lives in the value check below, and a module-wide
    binding set only ever matters when the module already contains a canonical
    derivation call. It never lets a bare literal through.
    """
    pairs = _iter_binding_pairs(tree)
    derived: set[str] = set()
    for _ in range(len(pairs) + 1):
        grown = {
            name for name, value in pairs if _value_is_canonical(value, derived)
        } | derived
        if grown == derived:
            break
        derived = grown
    rebound = {name for name, value in pairs if not _value_is_canonical(value, derived)}
    return derived - rebound


def _value_is_canonical(value: ast.expr, derived_bindings: set[str]) -> bool:
    """True when `value` provably came from a canonical derivation helper."""
    callee = _callee_name(value)
    # (i) direct call to a minting derivation -- canonical by construction.
    if callee in _MINTING_DERIVATIONS:
        return True
    # (ii) a transforming derivation is canonical only if what it transforms is
    # canonical. Without this it launders any literal passed as arg 0. A missing
    # first positional argument (e.g. all-keyword call) fails closed.
    if callee in _LAUNDERING_DERIVATIONS:
        assert isinstance(value, ast.Call)  # guaranteed by _callee_name
        if not value.args:
            return False
        return _value_is_canonical(value.args[0], derived_bindings)
    # (iii) a local name bound to such a call. Anything else -- notably an
    # Attribute, which could resolve to any string -- fails closed.
    return isinstance(value, ast.Name) and value.id in derived_bindings


def _describe(value: ast.expr) -> str:
    """Human-readable reason a value was rejected, for the failure message."""
    callee = _callee_name(value)
    if callee in _LAUNDERING_DERIVATIONS:
        return (
            f"got {callee}(...) whose first argument is not itself derived -- "
            "it launders a hand-written name through the allowlist"
        )
    return f"got {type(value).__name__}"


def _scan_source(source: str, relpath: str) -> list[str]:
    """Return ``<relpath>:<line> <detail>`` for every non-conformant group site."""
    tree = ast.parse(source, filename=relpath)
    derived_bindings = _collect_derived_bindings(tree)
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
                    f"({_describe(keyword.value)})"
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
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = node.args
            positional = args.posonlyargs + args.args
            defaulted = positional[len(positional) - len(args.defaults) :]
            for arg, default in zip(defaulted, args.defaults, strict=True):
                assignments.append((arg.arg, default))
            for arg, kw_default in zip(args.kwonlyargs, args.kw_defaults, strict=True):
                if kw_default is not None:
                    assignments.append((arg.arg, kw_default))

        for name, value in assignments:
            if not _GROUP_NAME_PATTERN.search(name.lower()):
                continue
            if _value_is_canonical(value, derived_bindings):
                continue
            findings.append(
                f"{relpath}:{value.lineno} {name} is assigned a "
                f"non-derived value ({_describe(value)})"
            )

    return findings


def _scan_file(path: Path) -> list[str]:
    """``_scan_source`` over a real file, reported at its repo-relative path."""
    return _scan_source(
        path.read_text(encoding="utf-8"), path.relative_to(REPO_ROOT).as_posix()
    )


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
        "name must come from omnibase_core.event_bus.util_consumer_group "
        f"({' / '.join(sorted(_CANONICAL_DERIVATIONS))}) so it is IAM-authorized by "
        "construction:\n  " + "\n  ".join(unexpected)
    )


def test_legacy_unmigrated_allowlist_is_empty() -> None:
    """The escape hatch is proof-debt, not an exemption: it must stay empty."""
    assert not _LEGACY_UNMIGRATED, (
        "_LEGACY_UNMIGRATED must be empty at end state (OMN-15639); "
        f"still holding: {sorted(_LEGACY_UNMIGRATED)}"
    )


def test_checker_rejects_bare_literal_and_fstring() -> None:
    """Negative control: the default-deny walk catches hand-written names."""
    findings = _scan_source(
        "bus.subscribe(topic='t', group_id='sneaky-literal-group')\n"
        "consumer_group = 'another-sneaky'\n"
        "bus.subscribe(topic='t', group_id=f'{cfg.consumer_group}-memory')\n",
        "probe.py",
    )
    assert len(findings) == 3, findings
    assert any("group_id=" in f and "Constant" in f for f in findings)
    assert any("consumer_group is assigned" in f for f in findings)
    assert any("JoinedStr" in f for f in findings)


def test_checker_rejects_laundered_literal() -> None:
    """``apply_instance_discriminator`` must not launder a literal.

    This is the exact escape a round-1 verifier demonstrated live against the
    previous version of this gate: the call was allowed on callee name alone,
    so wrapping any string in it passed the walk.
    """
    findings = _scan_source(
        "bus.subscribe(\n"
        "    topic='t',\n"
        "    group_id=apply_instance_discriminator('onex-runtime-memory', None),\n"
        ")\n",
        "probe.py",
    )
    assert findings, (
        "apply_instance_discriminator laundered a string literal through the "
        "canonical-derivation allowlist"
    )
    assert any("launders" in finding for finding in findings), (
        "the finding must say WHY it was rejected, otherwise the next reader "
        f"sees an allowlisted callee reported as non-derived: {findings}"
    )

    # Same helper, but transforming a genuinely derived name -> allowed.
    assert not _scan_source(
        "group_id = apply_instance_discriminator(\n"
        "    compute_consumer_group_id(identity, purpose), instance_id\n"
        ")\n",
        "probe.py",
    )

    # Keyword-only call: no first positional argument to prove canonical.
    assert _scan_source(
        "bus.subscribe(group_id=apply_instance_discriminator(group_id=x, instance_id=y))\n",
        "probe.py",
    )


def test_checker_rejects_rebound_derived_name() -> None:
    """A later non-canonical rebind removes a name from the derived set."""
    findings = _scan_source(
        "base = compute_consumer_group_id(identity, purpose)\n"
        "base = 'onex-runtime-memory'\n"
        "bus.subscribe(topic='t', group_id=base)\n",
        "probe.py",
    )
    assert findings, "literal rebind retained derived authorization"


def test_checker_rejects_group_named_parameter_defaults() -> None:
    """Function defaults on group-like names are group-name producers too."""
    findings = _scan_source(
        "def subscribe(group_id='onex-runtime-memory'):\n"
        "    pass\n"
        "async def consume(*, memory_group_id='onex-runtime-memory'):\n"
        "    pass\n"
        "kafka_group = 'onex-runtime-memory'\n",
        "probe.py",
    )
    assert len(findings) == 3, findings
    assert any("group_id is assigned" in f for f in findings)
    assert any("memory_group_id is assigned" in f for f in findings)
    assert any("kafka_group is assigned" in f for f in findings)


def test_checker_accepts_derived_names_including_binding_chains() -> None:
    """Positive control: real derivations, direct and through a binding chain."""
    assert not _scan_source(
        "bus.subscribe(topic='t', group_id=compute_consumer_group_id(identity))\n",
        "probe.py",
    )
    assert not _scan_source(
        "base = compute_consumer_group_id(identity, purpose)\n"
        "scoped = apply_instance_discriminator(base, instance_id)\n"
        "bus.subscribe(topic='t', group_id=scoped)\n",
        "probe.py",
    )
    assert not _scan_source(
        "group_id = derive_prefixed_group_id(prefix, scope=scope)\n",
        "probe.py",
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
    from omnibase_core.event_bus.util_consumer_group import (
        derive_consumer_group_id,
        is_authorized_group_name,
    )
    from omnibase_core.models.event_bus.model_consumer_group_scope import (
        ModelConsumerGroupScope,
    )

    from omnimemory.runtime.plugin import MEMORY_CONSUMER_GROUP_TAG

    class _Identity:
        env = "onex-dev"
        service = "omnimemory"
        node_name = "memory_domain_plugin"
        version = "v1"

    derived = derive_consumer_group_id(
        env=_Identity.env,
        service=_Identity.service,
        node_name=_Identity.node_name,
        version=_Identity.version,
        purpose=EnumConsumerGroupPurpose.CONSUME,
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


def test_seam_scope_field_name_is_ephemeral_tag() -> None:
    """Pin the ``ModelConsumerGroupScope`` field name this repo constructs.

    OMN-14208: ``src/omnimemory/runtime/plugin.py`` calls
    ``ModelConsumerGroupScope(ephemeral_tag=MEMORY_CONSUMER_GROUP_TAG)``. The
    OMN-15639 seam table pins the util's module path and function signatures but
    NOT that model's field names, so ``ephemeral_tag`` is the one seam field
    this lane inferred (from the ``.__i.`` grammar infix) rather than read off
    the table. If the core lane names it ``ephemeral`` / ``tag`` /
    ``ephemeral_scope``, both lanes can be individually green while the pair is
    a runtime ``TypeError`` -- exactly the PAIR_INCOMPATIBLE class. Assert it
    field-by-field so the mismatch fails here, loudly, at the boundary.
    """
    from omnibase_core.models.event_bus.model_consumer_group_scope import (
        ModelConsumerGroupScope,
    )

    fields = set(ModelConsumerGroupScope.model_fields)
    assert "ephemeral_tag" in fields, (
        "seam mismatch (OMN-14208): omnimemory constructs "
        "ModelConsumerGroupScope(ephemeral_tag=...) but the core lane's model "
        f"declares fields {sorted(fields)}. Reconcile the seam table and both "
        "lanes before either merges."
    )


def test_unauthorized_env_token_is_not_authorized() -> None:
    """The derived grammar is only conformant for an MSK-granted env token.

    Residual proof leg made executable. The derived name is
    ``{env}.omnimemory.{node}.consume.{version}.__i.memory`` and ``env`` comes
    from the kernel, whose documented default is ``local``
    (``ONEX_ENVIRONMENT``; ``omnibase_infra`` ``service_kernel.py`` feeds it into
    ``ModelNodeIdentity(env=...)``). The granted pattern set authorizes
    ``local.runtime_config.*`` but NOT ``local.omnimemory.*``. So the other tests
    in this repo -- which pin ``env="onex-dev"`` -- do NOT prove the DEPLOYED
    name is authorized. This test states that dependency instead of hiding it:
    the deployment must set the env token to ``onex-dev``.
    """
    from omnibase_core.enums.enum_consumer_group_purpose import (
        EnumConsumerGroupPurpose,
    )
    from omnibase_core.event_bus.util_consumer_group import (
        derive_consumer_group_id,
        is_authorized_group_name,
    )
    from omnibase_core.models.event_bus.model_consumer_group_scope import (
        ModelConsumerGroupScope,
    )

    from omnimemory.runtime.plugin import MEMORY_CONSUMER_GROUP_TAG

    class _Identity:
        env = "local"
        service = "omnimemory"
        node_name = "memory_domain_plugin"
        version = "v1"

    derived_under_default_env = derive_consumer_group_id(
        env=_Identity.env,
        service=_Identity.service,
        node_name=_Identity.node_name,
        version=_Identity.version,
        purpose=EnumConsumerGroupPurpose.CONSUME,
        scope=ModelConsumerGroupScope(ephemeral_tag=MEMORY_CONSUMER_GROUP_TAG),
    )

    assert not is_authorized_group_name(derived_under_default_env), (
        f"{derived_under_default_env!r} is matched by a granted pattern, which "
        "contradicts the pinned set (local.* is granted only for "
        "local.runtime_config.*). Either the matcher is too permissive or the "
        "grant changed -- re-read managed-data-plane.auto.tfvars before "
        "relaxing this."
    )
