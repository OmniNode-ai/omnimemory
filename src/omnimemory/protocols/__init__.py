"""
OmniMemory Protocol Definitions

This module contains all protocol definitions for the OmniMemory system,
following ONEX 4-node architecture patterns and contract-driven development.

All protocols use typing.Protocol for structural typing and avoid isinstance
checks, supporting the ModelOnexContainer pattern for dependency injection.
"""

# Protocol categories: Base, Effect, Compute, Reducer, and Orchestrator node protocols
from .base_protocols import (
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
    EnumOmniMemoryErrorCode,
    ProtocolCoordinationError,
    ProtocolOmniMemoryError,
    ProtocolProcessingError,
    ProtocolRetrievalError,
    ProtocolStorageError,
    ProtocolSystemError,
    ProtocolValidationError,
)
from .protocol_embedding import ProtocolEmbeddingClient, ProtocolRateLimiter
from .protocol_embedding_provider import ProtocolEmbeddingProvider, ProtocolLLMProvider
from .protocol_handler_intent import ProtocolHandlerIntent
from .protocol_intent_graph_adapter import ProtocolIntentGraphAdapter
from .protocol_secrets_provider import ProtocolSecretsProvider

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
    "ProtocolOmniMemoryError",
    "ProtocolValidationError",
    "ProtocolStorageError",
    "ProtocolRetrievalError",
    "ProtocolProcessingError",
    "ProtocolCoordinationError",
    "ProtocolSystemError",
    "EnumOmniMemoryErrorCode",
    # Secrets provider
    "ProtocolSecretsProvider",
    # Embedding and rate limiting protocols (contract boundary)
    "ProtocolEmbeddingClient",
    "ProtocolRateLimiter",
    # Provider protocols (for handler dependencies)
    "ProtocolEmbeddingProvider",
    "ProtocolLLMProvider",
    # Handler protocols (for contract-driven handler interfaces)
    "ProtocolHandlerIntent",
    # Adapter protocols (for contract-driven dependency injection)
    "ProtocolIntentGraphAdapter",
]
