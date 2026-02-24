# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""OmniMemory ONEX Nodes - Core 8 Foundation.

This package contains the Core 8 ONEX-compliant nodes for omnimemory:

EFFECT (2):
- memory_storage_effect: CRUD operations to storage backends
- memory_retrieval_effect: Semantic/temporal/contextual search

COMPUTE (2):
- semantic_analyzer_compute: Semantic analysis and embeddings
- similarity_compute: Vector similarity calculations

REDUCER (2):
- memory_consolidator_reducer: Merge similar memories
- statistics_reducer: Generate memory statistics

ORCHESTRATOR (2):
- memory_lifecycle_orchestrator: Full lifecycle management
- agent_coordinator_orchestrator: Cross-agent coordination
"""

from omnimemory.nodes.base import (
    BaseComputeNode,
    BaseEffectNode,
    BaseNode,
    BaseOrchestratorNode,
    BaseReducerNode,
    ContainerType,
)
from omnimemory.nodes.filesystem_crawler_effect import (
    HandlerFilesystemCrawler,
    ModelFilesystemCrawlerConfig,
    ModelFilesystemCrawlResult,  # omnimemory-model-exempt: handler result
)
from omnimemory.nodes.intent_event_consumer_effect import (
    HandlerIntentEventConsumer,
    ModelIntentEventConsumerConfig,
    ModelIntentEventConsumerHealth,
)

__all__: list[str] = [
    # Base classes
    "BaseNode",
    "BaseEffectNode",
    "BaseComputeNode",
    "BaseReducerNode",
    "BaseOrchestratorNode",
    "ContainerType",
    # Effect nodes
    "memory_storage_effect",
    "memory_retrieval_effect",
    "HandlerIntentEventConsumer",
    "ModelIntentEventConsumerConfig",
    "ModelIntentEventConsumerHealth",
    "HandlerFilesystemCrawler",
    "ModelFilesystemCrawlResult",
    "ModelFilesystemCrawlerConfig",
    # Compute nodes
    "semantic_analyzer_compute",
    "similarity_compute",
    # Reducer nodes
    "memory_consolidator_reducer",
    "statistics_reducer",
    # Orchestrator nodes
    "memory_lifecycle_orchestrator",
    "agent_coordinator_orchestrator",
]
