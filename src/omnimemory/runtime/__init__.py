# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""OmniMemory Runtime Package.

Provides contract-driven topic discovery for the OmniMemory domain.

This package contains:
    - collect_subscribe_topics_from_contracts: Discover subscribe topics from
      effect node contract.yaml files.
    - collect_publish_topics_for_dispatch: Discover publish topics keyed by
      dispatch alias from effect node contract.yaml files.
    - collect_all_publish_topics: Discover all publish topics declared across
      effect node contract.yaml files.
    - canonical_topic_to_dispatch_alias: Convert ONEX canonical topic naming
      to dispatch engine format.

Usage:
    from omnimemory.runtime.contract_topics import (
        collect_subscribe_topics_from_contracts,
        collect_publish_topics_for_dispatch,
        collect_all_publish_topics,
        canonical_topic_to_dispatch_alias,
    )

    # Get all subscribe topics declared across omnimemory effect nodes
    topics = collect_subscribe_topics_from_contracts()

    # Get publish topics keyed by dispatch alias
    publish_map = collect_publish_topics_for_dispatch()

    # Get all publish topics (full list, not just first per node)
    all_publish = collect_all_publish_topics()
"""

from omnimemory.runtime.contract_topics import (
    canonical_topic_to_dispatch_alias,
    collect_all_publish_topics,
    collect_publish_topics_for_dispatch,
    collect_subscribe_topics_from_contracts,
)

__all__ = [
    "canonical_topic_to_dispatch_alias",
    "collect_all_publish_topics",
    "collect_publish_topics_for_dispatch",
    "collect_subscribe_topics_from_contracts",
]
