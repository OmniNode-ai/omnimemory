# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""OmniMemory ONEX Nodes - Core 8 Foundation.

This package contains the Core 8 ONEX-compliant nodes for omnimemory:

EFFECT (2):
- node_memory_storage_effect: CRUD operations to storage backends
- node_memory_retrieval_effect: Semantic/temporal/contextual search

COMPUTE (2):
- node_semantic_analyzer_compute: Semantic analysis and embeddings
- node_similarity_compute: Vector similarity calculations

REDUCER (3):
- node_memory_consolidator_reducer: Merge similar memories
- node_statistics_reducer: Generate memory statistics
- node_navigation_history_reducer: Persist navigation sessions (OMN-2584)

ORCHESTRATOR (2):
- node_memory_lifecycle_orchestrator: Full lifecycle management
- node_agent_coordinator_orchestrator: Cross-agent coordination
"""

from omnimemory.nodes.base import (
    BaseComputeNode,
    BaseEffectNode,
    BaseNode,
    BaseOrchestratorNode,
    BaseReducerNode,
    ContainerType,
)
from omnimemory.nodes.node_filesystem_crawler_effect import (
    HandlerFilesystemCrawler,
    ModelFilesystemCrawlerConfig,
    ModelFilesystemCrawlResult,  # omnimemory-model-exempt: handler result
)
from omnimemory.nodes.node_intent_event_consumer_effect import (
    HandlerIntentEventConsumer,
    ModelIntentEventConsumerConfig,
    ModelIntentEventConsumerHealth,
)
from omnimemory.nodes.node_navigation_history_reducer import (
    HandlerNavigationHistoryReducer,
    ModelNavigationHistoryRequest,
    ModelNavigationHistoryResponse,
    ModelNavigationSession,
    ModelPlanStep,
    NavigationOutcome,
    NodeNavigationHistoryReducer,
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
    "node_memory_storage_effect",
    "node_memory_retrieval_effect",
    "HandlerIntentEventConsumer",
    "ModelIntentEventConsumerConfig",
    "ModelIntentEventConsumerHealth",
    "HandlerFilesystemCrawler",
    "ModelFilesystemCrawlResult",
    "ModelFilesystemCrawlerConfig",
    # Compute nodes
    "node_semantic_analyzer_compute",
    "node_similarity_compute",
    # Reducer nodes
    "node_memory_consolidator_reducer",
    "node_statistics_reducer",
    "NodeNavigationHistoryReducer",
    "HandlerNavigationHistoryReducer",
    "ModelNavigationHistoryRequest",
    "ModelNavigationHistoryResponse",
    "ModelNavigationSession",
    "ModelPlanStep",
    "NavigationOutcome",
    # Orchestrator nodes
    "node_memory_lifecycle_orchestrator",
    "node_agent_coordinator_orchestrator",
]
