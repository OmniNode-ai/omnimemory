"""
OmniMemory Protocol Definitions

This module contains all protocol definitions for the OmniMemory system,
following ONEX 4-node architecture patterns and contract-driven development.

All protocols use typing.Protocol for structural typing and avoid isinstance
checks, supporting the ModelOnexContainer pattern for dependency injection.
"""

from .base_protocols import (  # Base protocols; Effect node protocols (memory storage, retrieval, persistence); Compute node protocols (intelligence processing, semantic analysis); Reducer node protocols (consolidation, aggregation, optimization); Orchestrator node protocols (workflow, agent, memory coordination)
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
    ProtocolPatternRecognition,
    ProtocolSemanticAnalyzer,
    ProtocolWorkflowCoordinator,
)
from .data_models import (  # Core data models; Request/Response models; Enums
    AccessLevel,
    BaseMemoryRequest,
    BaseMemoryResponse,
    ContentType,
    IndexingStatus,
    MemoryPriority,
    MemoryRecord,
    MemoryRetrieveRequest,
    MemoryRetrieveResponse,
    MemoryStoreRequest,
    MemoryStoreResponse,
    OperationStatus,
    SearchFilters,
    SearchResult,
    SemanticSearchRequest,
    SemanticSearchResponse,
    StoragePreferences,
    TemporalSearchRequest,
    TemporalSearchResponse,
    UserContext,
)
from .error_models import (  # Error handling; Error codes
    CoordinationError,
    OmniMemoryError,
    OmniMemoryErrorCode,
    ProcessingError,
    RetrievalError,
    StorageError,
    SystemError,
    ValidationError,
)

__all__ = [
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
