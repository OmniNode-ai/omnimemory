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
    - Orchestrator Nodes: Workflow coordination, agent coordination,
      system orchestration

Usage:
    >>> from omnimemory.models import core, memory, intelligence
    >>> # Use domain-specific models for memory operations
"""

__version__ = "0.1.0"
__author__ = "OmniNode-ai"
__email__ = "contact@omninode.ai"

# Import enums for public API
from .enums import (
    EnumDataType,
    EnumErrorCode,
    EnumIntelligenceOperationType,
    EnumMemoryOperationType,
    EnumMemoryStorageType,
    EnumNodeType,
    EnumOperationStatus,
    EnumPriorityLevel,
    EnumSeverity,
    EnumTrustLevel,
    MigrationStatus,
)

# Import ONEX-compliant model domains
from .models import core, foundation, intelligence, memory, service

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
    # Enums
    "EnumDataType",
    "EnumErrorCode",
    "EnumIntelligenceOperationType",
    "EnumMemoryOperationType",
    "EnumMemoryStorageType",
    "EnumNodeType",
    "EnumOperationStatus",
    "EnumPriorityLevel",
    "EnumSeverity",
    "EnumTrustLevel",
    "MigrationStatus",
]
