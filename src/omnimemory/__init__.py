# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""
OmniMemory - Advanced memory management and retrieval system for AI applications.

This package provides comprehensive memory management capabilities including:
- Persistent memory storage with ONEX 4-node architecture
- Vector-based semantic memory with similarity search
- Temporal memory with decay patterns and lifecycle management
- Memory consolidation, aggregation, and optimization
- Cross-modal memory integration and intelligence processing
- Contract-driven development with strong typing and validation
- Monadic error handling with NodeResult composition
- Event-driven architecture with observability patterns

Architecture:
    - Effect Nodes: Memory storage, retrieval, and persistence operations
    - Compute Nodes: Intelligence processing, semantic analysis, pattern recognition
    - Reducer Nodes: Memory consolidation, aggregation, and optimization
    - Orchestrator Nodes: Workflow coordination, agent coordination

Usage:
    >>> from omnimemory.models import core, memory, intelligence
    >>> # Use domain-specific models for memory operations
"""

__version__ = "0.6.0"
__author__ = "OmniNode-ai"
__email__ = "contact@omninode.ai"

# Import ONEX-compliant model domains
from .models import core, foundation, intelligence, memory, service

# Import protocol definitions
# Protocol categories: Base, Effect, Compute, Reducer, Orchestrator nodes
# Data models, Enums, and Error handling
from .protocols import (
    AccessLevel,
    BaseMemoryRequest,
    BaseMemoryResponse,
    ContentType,
    EnumOmniMemoryErrorCode,
    MemoryPriority,
    MemoryRecord,
    MemoryStoreRequest,
    MemoryStoreResponse,
    OperationStatus,
    ProtocolAgentCoordinator,
    ProtocolIntelligenceProcessor,
    ProtocolMemoryAggregator,
    ProtocolMemoryBase,
    ProtocolMemoryConsolidator,
    ProtocolMemoryOperations,
    ProtocolMemoryOptimizer,
    ProtocolMemoryOrchestrator,
    ProtocolMemoryPersistence,
    ProtocolMemoryRetrieval,
    ProtocolMemoryStorage,
    ProtocolOmniMemoryError,
    ProtocolPatternRecognition,
    ProtocolSemanticAnalyzer,
    ProtocolSystemError,
    ProtocolValidationError,
    ProtocolWorkflowCoordinator,
)

__all__ = [
    # Version and metadata
    "__version__",
    "__author__",
    "__email__",
    # ONEX model domains
    "core",
    "memory",
    "intelligence",
    "service",
    "foundation",
    # Base protocols
    "ProtocolMemoryBase",
    "ProtocolMemoryOperations",
    # Effect node protocols
    "ProtocolMemoryStorage",
    "ProtocolMemoryRetrieval",
    "ProtocolMemoryPersistence",
    # Compute node protocols
    "ProtocolIntelligenceProcessor",
    "ProtocolSemanticAnalyzer",
    "ProtocolPatternRecognition",
    # Reducer node protocols
    "ProtocolMemoryConsolidator",
    "ProtocolMemoryAggregator",
    "ProtocolMemoryOptimizer",
    # Orchestrator node protocols
    "ProtocolWorkflowCoordinator",
    "ProtocolAgentCoordinator",
    "ProtocolMemoryOrchestrator",
    # Data models
    "BaseMemoryRequest",
    "BaseMemoryResponse",
    "MemoryRecord",
    "MemoryStoreRequest",
    "MemoryStoreResponse",
    # Enums
    "OperationStatus",
    "ContentType",
    "MemoryPriority",
    "AccessLevel",
    # Error handling
    "ProtocolOmniMemoryError",
    "EnumOmniMemoryErrorCode",
    "ProtocolValidationError",
    "ProtocolSystemError",
]
