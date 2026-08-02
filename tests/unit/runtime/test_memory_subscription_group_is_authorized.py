# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""The group id PluginMemory hands to the bus must be MSK IAM-authorized.

This drives the REAL subscription-wiring path (``PluginMemory.start_consumers``)
rather than re-deriving a name in the test, then checks the group id that
actually reached ``event_bus.subscribe(group_id=...)`` against the pinned MSK
IAM group pattern set that ``omnibase_core`` vendors from Terraform.

Before OMN-15639 the plugin minted ``f"{config.consumer_group}-memory"``. That
is config-derived with a literal suffix, so it inherits whatever the kernel's
``consumer_group`` happens to be and matches NONE of the six granted patterns
for any current value -- a ``GroupAuthorizationFailedError`` on MSK.

Only the introspection publisher is stubbed (it spawns heartbeat tasks); the
group-name derivation seam under test is never mocked.

Reference: OMN-15639.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID, uuid4

import pytest

from omnimemory.runtime.introspection import IntrospectionResult
from omnimemory.runtime.plugin import PluginMemory

from .conftest import StubConfig, StubEventBus

_RUNTIME_ENV = "onex-dev"


class _NodeIdentity:
    """Kernel-shaped node identity (structurally ProtocolNodeIdentity)."""

    def __init__(self, env: str) -> None:
        self.env = env
        self.service = "omnimemory"
        self.node_name = "omnimemory"
        self.version = "v1"


@pytest.fixture
def stub_introspection(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub the introspection publisher so no heartbeat tasks leak."""

    async def _publish(
        event_bus: object,
        *,
        correlation_id: UUID | None = None,
        enable_heartbeat: bool = True,
        heartbeat_interval_seconds: float = 30.0,
    ) -> IntrospectionResult:
        return IntrospectionResult(registered_nodes=["node_1"], proxies=[])

    monkeypatch.setattr(
        "omnimemory.runtime.introspection.publish_memory_introspection",
        _publish,
    )


async def _subscribed_group_ids(config: Any, bus: StubEventBus) -> list[str]:
    plugin = PluginMemory()
    await plugin.wire_dispatchers(config)
    result = await plugin.start_consumers(config)
    assert result.success, f"start_consumers failed: {result.error_message}"
    assert bus.subscriptions, "plugin subscribed to no topics"
    return [str(sub["group_id"]) for sub in bus.subscriptions]


@pytest.mark.asyncio
@pytest.mark.usefixtures("stub_introspection")
async def test_memory_subscription_group_is_iam_authorized() -> None:
    """Every group id the plugin subscribes with matches a pinned IAM pattern."""
    from omnibase_core.utils.util_consumer_group import is_authorized_group_name

    bus = StubEventBus()
    config = StubConfig(
        event_bus=bus,
        correlation_id=uuid4(),
        consumer_group="onex-runtime",
        node_identity=_NodeIdentity(env=_RUNTIME_ENV),
    )

    group_ids = await _subscribed_group_ids(config, bus)

    for group_id in group_ids:
        assert is_authorized_group_name(group_id), (
            f"consumer group {group_id!r} reached event_bus.subscribe() but is "
            "not matched by any pinned MSK IAM group pattern -- it would fail "
            "with GroupAuthorizationFailedError on onex-dev"
        )


@pytest.mark.asyncio
@pytest.mark.usefixtures("stub_introspection")
async def test_memory_subscription_group_is_env_prefixed_and_stable() -> None:
    """The group id carries the runtime env prefix and is one shared group."""
    bus = StubEventBus()
    config = StubConfig(
        event_bus=bus,
        correlation_id=uuid4(),
        consumer_group="onex-runtime",
        node_identity=_NodeIdentity(env=_RUNTIME_ENV),
    )

    group_ids = await _subscribed_group_ids(config, bus)

    assert len(set(group_ids)) == 1, (
        f"all memory topics must share one consumer group; got {sorted(set(group_ids))}"
    )
    group_id = group_ids[0]
    assert group_id.startswith(f"{_RUNTIME_ENV}."), (
        f"group id {group_id!r} must begin with the runtime env token "
        f"{_RUNTIME_ENV!r} followed by a literal '.' -- the IAM patterns are "
        "prefix globs and the dot is literal"
    )
    assert "omnimemory" in group_id
    # The group must NOT be the pre-OMN-15639 config-derived literal.
    assert group_id != f"{config.consumer_group}-memory"


@pytest.mark.asyncio
@pytest.mark.usefixtures("stub_introspection")
async def test_missing_node_identity_does_not_mint_unauthorized_group() -> None:
    """Fail closed: without a node identity the plugin must not guess a name.

    A silent fallback here is the exact defect class OMN-15639 removes -- it
    would mint an unauthorized group and fail at the broker instead of at
    wiring time.
    """
    bus = StubEventBus()
    config = StubConfig(
        event_bus=bus,
        correlation_id=uuid4(),
        consumer_group="onex-runtime",
        node_identity=None,
    )

    plugin = PluginMemory()
    await plugin.wire_dispatchers(config)
    result = await plugin.start_consumers(config)

    assert not bus.subscriptions, (
        "plugin subscribed without a node identity; the group id could not "
        f"have been derived: {bus.subscriptions}"
    )
    assert "node_identity" in (result.message or "") or "node_identity" in (
        result.error_message or ""
    ), (
        "start_consumers must state that node_identity is missing rather than "
        f"silently proceeding; got message={result.message!r} "
        f"error={result.error_message!r}"
    )
