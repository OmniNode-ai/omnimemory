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
    - Orchestrator Nodes: Workflow coordination, agent coordination, system orchestration

Usage:
    >>> from omnimemory import OmniMemoryContainer, create_omnimemory_container
    >>> container = await create_omnimemory_container()
    >>> storage_service = await container.get_service_async(ProtocolMemoryStorage)
    >>> # Use storage_service for memory operations
"""

__version__ = "0.1.0"
__author__ = "OmniNode-ai"
__email__ = "contact@omninode.ai"

# Core container and service management
from .core import (
    OmniMemoryContainer,
    create_omnimemory_container,
    get_omnimemory_container,
    get_omnimemory_container_sync,
    OmniMemoryServiceProvider,
    MemoryServiceRegistry,
    BaseMemoryService,
    BaseEffectService,
    BaseComputeService,
    BaseReducerService,
    BaseOrchestratorService,
    ContainerError,
    ServiceResolutionError,
    ServiceRegistrationError,
)

# Protocol definitions and data models
from .protocols import (
    # Base protocols
    ProtocolMemoryBase,
    ProtocolMemoryOperations,
    
    # Effect node protocols (memory storage, retrieval, persistence)
    ProtocolMemoryStorage,
    ProtocolMemoryRetrieval,
    ProtocolMemoryPersistence,
    
    # Compute node protocols (intelligence processing, semantic analysis)
    ProtocolIntelligenceProcessor,
    ProtocolSemanticAnalyzer,
    ProtocolPatternRecognition,
    
    # Reducer node protocols (consolidation, aggregation, optimization)
    ProtocolMemoryConsolidator,
    ProtocolMemoryAggregator,
    ProtocolMemoryOptimizer,
    
    # Orchestrator node protocols (workflow, agent, memory coordination)
    ProtocolWorkflowCoordinator,
    ProtocolAgentCoordinator,
    ProtocolMemoryOrchestrator,
    
    # Data models
    BaseMemoryRequest,
    BaseMemoryResponse,
    MemoryRecord,
    UserContext,
    StoragePreferences,
    SearchFilters,
    SearchResult,
    MemoryStoreRequest,
    MemoryStoreResponse,
    MemoryRetrieveRequest,
    MemoryRetrieveResponse,
    SemanticSearchRequest,
    SemanticSearchResponse,
    TemporalSearchRequest,
    TemporalSearchResponse,
    
    # Enums
    OperationStatus,
    ContentType,
    MemoryPriority,
    AccessLevel,
    IndexingStatus,
    
    # Error handling
    OmniMemoryError,
    ValidationError,
    StorageError,
    RetrievalError,
    ProcessingError,
    CoordinationError,
    SystemError,
    OmniMemoryErrorCode,
)

__all__ = [
    # Version and metadata
    "__version__",
    "__author__", 
    "__email__",
    
    # Core container and service management
    "OmniMemoryContainer",
    "create_omnimemory_container",
    "get_omnimemory_container",
    "get_omnimemory_container_sync",
    "OmniMemoryServiceProvider",
    "MemoryServiceRegistry",
    "BaseMemoryService",
    "BaseEffectService",
    "BaseComputeService", 
    "BaseReducerService",
    "BaseOrchestratorService",
    "ContainerError",
    "ServiceResolutionError",
    "ServiceRegistrationError",
    
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
    "UserContext",
    "StoragePreferences",
    "SearchFilters",
    "SearchResult",
    "MemoryStoreRequest",
    "MemoryStoreResponse",
    "MemoryRetrieveRequest",
    "MemoryRetrieveResponse",
    "SemanticSearchRequest",
    "SemanticSearchResponse",
    "TemporalSearchRequest",
    "TemporalSearchResponse",
    
    # Enums
    "OperationStatus",
    "ContentType",
    "MemoryPriority", 
    "AccessLevel",
    "IndexingStatus",
    
    # Error handling
    "OmniMemoryError",
    "ValidationError",
    "StorageError",
    "RetrievalError",
    "ProcessingError",
    "CoordinationError",
    "SystemError", 
    "OmniMemoryErrorCode",
]