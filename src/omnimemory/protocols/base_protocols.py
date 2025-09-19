"""
Base Protocol Definitions for OmniMemory ONEX Architecture

This module defines all protocol interfaces following ONEX 4-node architecture:
- Effect: Memory storage, retrieval, and persistence operations
- Compute: Intelligence processing, semantic analysis, pattern recognition
- Reducer: Memory consolidation, aggregation, and optimization
- Orchestrator: Workflow, agent, and memory coordination

All protocols use typing.Protocol for structural typing, avoiding isinstance
checks and supporting ModelOnexContainer dependency injection patterns.
"""

from __future__ import annotations

from abc import abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Protocol
from uuid import UUID

from omnibase_core.core.monadic.model_node_result import NodeResult

from ..models.foundation import (
    ModelConfiguration,
    ModelHealthResponse,
    ModelMetadata,
    ModelMetricsResponse,
    ModelOptionalStringList,
    ModelResultCollection,
    ModelStringList,
    ModelSystemConfiguration,
)
from .data_models import (  # Requests and responses
    AgentCoordinationRequest,
    AgentCoordinationResponse,
    AggregationRequest,
    AggregationResponse,
    BackupRequest,
    BackupResponse,
    BaseMemoryRequest,
    BaseMemoryResponse,
    BroadcastRequest,
    BroadcastResponse,
    CompressionRequest,
    CompressionResponse,
    ConsolidationRequest,
    ConsolidationResponse,
    ContextMergeRequest,
    ContextMergeResponse,
    ContextualSearchRequest,
    ContextualSearchResponse,
    DeduplicationRequest,
    DeduplicationResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    InsightExtractionRequest,
    InsightExtractionResponse,
    IntelligenceProcessRequest,
    IntelligenceProcessResponse,
    LayoutOptimizationRequest,
    LayoutOptimizationResponse,
    LifecycleOrchestrationRequest,
    LifecycleOrchestrationResponse,
    MemoryDeleteRequest,
    MemoryDeleteResponse,
    MemoryRecord,
    MemoryRetrieveRequest,
    MemoryRetrieveResponse,
    MemoryStoreRequest,
    MemoryStoreResponse,
    MigrationCoordinationRequest,
    MigrationCoordinationResponse,
    ParallelCoordinationRequest,
    ParallelCoordinationResponse,
    PatternAnalysisRequest,
    PatternAnalysisResponse,
    PatternLearningRequest,
    PatternLearningResponse,
    PatternPredictionRequest,
    PatternPredictionResponse,
    PatternRecognitionRequest,
    PatternRecognitionResponse,
    PersistenceRequest,
    PersistenceResponse,
    QuotaManagementRequest,
    QuotaManagementResponse,
    RestoreRequest,
    RestoreResponse,
    RetrievalOptimizationRequest,
    RetrievalOptimizationResponse,
    SemanticAnalysisRequest,
    SemanticAnalysisResponse,
    SemanticComparisonRequest,
    SemanticComparisonResponse,
    SemanticSearchRequest,
    SemanticSearchResponse,
    StateSynchronizationRequest,
    StateSynchronizationResponse,
    StatisticsRequest,
    StatisticsResponse,
    SummarizationRequest,
    SummarizationResponse,
    TemporalSearchRequest,
    TemporalSearchResponse,
    WorkflowExecutionRequest,
    WorkflowExecutionResponse,
    WorkflowStateRequest,
    WorkflowStateResponse,
)

# === BASE PROTOCOLS ===


class ProtocolMemoryBase(Protocol):
    """
    Base protocol for all memory-related operations.

    Provides foundational capabilities that all memory components must implement,
    including health checking, configuration management, and basic observability.
    """

    @abstractmethod
    async def health_check(
        self,
        correlation_id: Optional[UUID] = None,
    ) -> NodeResult[ModelHealthResponse]:
        """
        Check the health status of the memory component.

        Returns:
            NodeResult containing ModelHealthResponse with:
            - status: overall system health (healthy/degraded/unhealthy)
            - latency_ms: response time metrics
            - resource_usage: detailed system resource metrics
            - dependencies: status of external dependencies
            - uptime, version, environment details
        """
        ...

    @abstractmethod
    async def get_metrics(
        self,
        correlation_id: Optional[UUID] = None,
    ) -> NodeResult[ModelMetricsResponse]:
        """
        Get operational metrics for the memory component.

        Returns:
            NodeResult containing ModelMetricsResponse with:
            - operation_counts: detailed counts by operation type
            - performance_metrics: latency, throughput, error rates
            - resource_metrics: memory usage, cache statistics, connections
            - custom_metrics: application-specific measurements
            - alerts: active performance warnings
        """
        ...

    @abstractmethod
    async def configure(
        self,
        config: ModelSystemConfiguration,
        correlation_id: Optional[UUID] = None,
    ) -> NodeResult[bool]:
        """
        Configure the memory component with new settings.

        Args:
            config: ModelSystemConfiguration with database, cache, performance,
                   and observability settings
            correlation_id: Request correlation ID

        Returns:
            NodeResult indicating configuration success/failure
        """
        ...


class ProtocolMemoryOperations(ProtocolMemoryBase, Protocol):
    """
    Base protocol for memory operations with common patterns.

    Extends ProtocolMemoryBase with standard CRUD operations that most
    memory components will need to implement.
    """

    @abstractmethod
    async def validate_request(
        self,
        request: BaseMemoryRequest,
    ) -> NodeResult[bool]:
        """
        Validate a memory operation request.

        Args:
            request: The request to validate

        Returns:
            NodeResult indicating validation success/failure with error details
        """
        ...

    @abstractmethod
    async def log_operation(
        self,
        operation: str,
        request: BaseMemoryRequest,
        response: BaseMemoryResponse,
        correlation_id: UUID,
    ) -> NodeResult[bool]:
        """
        Log a completed memory operation for audit and monitoring.

        Args:
            operation: Operation name
            request: Original request
            response: Operation response
            correlation_id: Request correlation ID

        Returns:
            NodeResult indicating logging success/failure
        """
        ...


# === EFFECT NODE PROTOCOLS ===


class ProtocolMemoryStorage(ProtocolMemoryOperations, Protocol):
    """
    Protocol for memory storage operations (Effect node).

    Handles the storage of memory records with metadata, provenance tracking,
    and support for different storage backends (PostgreSQL, ModelCachingSubcontract, etc.).
    """

    @abstractmethod
    async def store_memory(
        self,
        request: MemoryStoreRequest,
    ) -> NodeResult[MemoryStoreResponse]:
        """
        Store a memory record with metadata and provenance.

        Args:
            request: Memory storage request containing the memory record

        Returns:
            NodeResult with MemoryStoreResponse containing storage details
        """
        ...

    @abstractmethod
    async def retrieve_memory(
        self,
        request: MemoryRetrieveRequest,
    ) -> NodeResult[MemoryRetrieveResponse]:
        """
        Retrieve a memory record by identifier.

        Args:
            request: Memory retrieval request with memory ID

        Returns:
            NodeResult with MemoryRetrieveResponse containing the memory record
        """
        ...

    @abstractmethod
    async def delete_memory(
        self,
        request: MemoryDeleteRequest,
    ) -> NodeResult[MemoryDeleteResponse]:
        """
        Soft delete a memory record with audit trail.

        Args:
            request: Memory deletion request with memory ID

        Returns:
            NodeResult with MemoryDeleteResponse indicating success/failure
        """
        ...

    @abstractmethod
    async def update_memory(
        self,
        memory_id: UUID,
        updates: ModelMetadata,
        correlation_id: Optional[UUID] = None,
    ) -> NodeResult[MemoryRecord]:
        """
        Update an existing memory record.

        Args:
            memory_id: ID of memory to update
            updates: Dictionary of fields to update
            correlation_id: Request correlation ID

        Returns:
            NodeResult with updated MemoryRecord
        """
        ...

    @abstractmethod
    async def list_memories(
        self,
        filters: Optional[ModelMetadata] = None,
        limit: int = 100,
        offset: int = 0,
        correlation_id: Optional[UUID] = None,
    ) -> NodeResult[List[MemoryRecord]]:
        """
        List memory records with optional filtering and pagination.

        Args:
            filters: Optional filters to apply
            limit: Maximum number of records to return
            offset: Number of records to skip
            correlation_id: Request correlation ID

        Returns:
            NodeResult with list of MemoryRecord objects
        """
        ...


class ProtocolMemoryRetrieval(ProtocolMemoryOperations, Protocol):
    """
    Protocol for advanced memory retrieval operations (Effect node).

    Provides semantic search, temporal search, and contextual retrieval
    capabilities using vector embeddings and time-based indexing.
    """

    @abstractmethod
    async def semantic_search(
        self,
        request: SemanticSearchRequest,
    ) -> NodeResult[SemanticSearchResponse]:
        """
        Perform vector-based semantic similarity search.

        Args:
            request: Semantic search request with query and parameters

        Returns:
            NodeResult with SemanticSearchResponse containing matched memories
        """
        ...

    @abstractmethod
    async def temporal_search(
        self,
        request: TemporalSearchRequest,
    ) -> NodeResult[TemporalSearchResponse]:
        """
        Perform time-based memory retrieval with decay consideration.

        Args:
            request: Temporal search request with time range and criteria

        Returns:
            NodeResult with TemporalSearchResponse containing time-filtered memories
        """
        ...

    @abstractmethod
    async def contextual_search(
        self,
        request: ContextualSearchRequest,
    ) -> NodeResult[ContextualSearchResponse]:
        """
        Perform context-aware memory retrieval using multiple criteria.

        Args:
            request: Contextual search request with context parameters

        Returns:
            NodeResult with ContextualSearchResponse containing context-matched memories
        """
        ...

    @abstractmethod
    async def get_related_memories(
        self,
        memory_id: UUID,
        relationship_types: Optional[ModelOptionalStringList] = None,
        max_depth: int = 2,
        correlation_id: Optional[UUID] = None,
    ) -> NodeResult[List[MemoryRecord]]:
        """
        Get memories related to a specific memory record.

        Args:
            memory_id: ID of the source memory
            relationship_types: Types of relationships to follow
            max_depth: Maximum relationship depth to traverse
            correlation_id: Request correlation ID

        Returns:
            NodeResult with list of related MemoryRecord objects
        """
        ...


class ProtocolMemoryPersistence(ProtocolMemoryOperations, Protocol):
    """
    Protocol for memory persistence and durability management (Effect node).

    Handles long-term storage, backup/restore operations, and data durability
    across different storage systems and failure scenarios.
    """

    @abstractmethod
    async def persist_to_storage(
        self,
        request: PersistenceRequest,
    ) -> NodeResult[PersistenceResponse]:
        """
        Persist memory data to durable storage.

        Args:
            request: Persistence request with storage preferences

        Returns:
            NodeResult with PersistenceResponse containing storage details
        """
        ...

    @abstractmethod
    async def backup_memory(
        self,
        request: BackupRequest,
    ) -> NodeResult[BackupResponse]:
        """
        Create a backup of memory data with versioning.

        Args:
            request: Backup request with backup parameters

        Returns:
            NodeResult with BackupResponse containing backup details
        """
        ...

    @abstractmethod
    async def restore_memory(
        self,
        request: RestoreRequest,
    ) -> NodeResult[RestoreResponse]:
        """
        Restore memory data from a backup.

        Args:
            request: Restore request with backup identifier and options

        Returns:
            NodeResult with RestoreResponse containing restore status
        """
        ...

    @abstractmethod
    async def verify_integrity(
        self,
        memory_ids: Optional[List[UUID]] = None,
        correlation_id: Optional[UUID] = None,
    ) -> NodeResult[ModelMetadata]:
        """
        Verify the integrity of stored memory data.

        Args:
            memory_ids: Optional list of specific memory IDs to verify
            correlation_id: Request correlation ID

        Returns:
            NodeResult with integrity verification results
        """
        ...


# === COMPUTE NODE PROTOCOLS ===


class ProtocolIntelligenceProcessor(ProtocolMemoryOperations, Protocol):
    """
    Protocol for intelligence processing operations (Compute node).

    Processes raw intelligence data into structured memory records,
    extracts insights, and performs pattern analysis.
    """

    @abstractmethod
    async def process_intelligence(
        self,
        request: IntelligenceProcessRequest,
    ) -> NodeResult[IntelligenceProcessResponse]:
        """
        Process raw intelligence data into structured memory.

        Args:
            request: Intelligence processing request with raw data

        Returns:
            NodeResult with IntelligenceProcessResponse containing processed data
        """
        ...

    @abstractmethod
    async def analyze_patterns(
        self,
        request: PatternAnalysisRequest,
    ) -> NodeResult[PatternAnalysisResponse]:
        """
        Analyze patterns in intelligence data.

        Args:
            request: Pattern analysis request with data to analyze

        Returns:
            NodeResult with PatternAnalysisResponse containing discovered patterns
        """
        ...

    @abstractmethod
    async def extract_insights(
        self,
        request: InsightExtractionRequest,
    ) -> NodeResult[InsightExtractionResponse]:
        """
        Extract actionable insights from processed intelligence.

        Args:
            request: Insight extraction request with processed data

        Returns:
            NodeResult with InsightExtractionResponse containing extracted insights
        """
        ...

    @abstractmethod
    async def enrich_memory(
        self,
        memory: MemoryRecord,
        enrichment_types: ModelStringList,
        correlation_id: Optional[UUID] = None,
    ) -> NodeResult[MemoryRecord]:
        """
        Enrich a memory record with additional intelligence data.

        Args:
            memory: Memory record to enrich
            enrichment_types: Types of enrichment to apply
            correlation_id: Request correlation ID

        Returns:
            NodeResult with enriched MemoryRecord
        """
        ...


class ProtocolSemanticAnalyzer(ProtocolMemoryOperations, Protocol):
    """
    Protocol for semantic analysis and understanding (Compute node).

    Provides semantic analysis, vector embedding generation, and
    semantic similarity comparison capabilities.
    """

    @abstractmethod
    async def analyze_semantics(
        self,
        request: SemanticAnalysisRequest,
    ) -> NodeResult[SemanticAnalysisResponse]:
        """
        Analyze semantic content and relationships.

        Args:
            request: Semantic analysis request with content to analyze

        Returns:
            NodeResult with SemanticAnalysisResponse containing analysis results
        """
        ...

    @abstractmethod
    async def generate_embeddings(
        self,
        request: EmbeddingRequest,
    ) -> NodeResult[EmbeddingResponse]:
        """
        Generate vector embeddings for semantic search.

        Args:
            request: Embedding request with text to embed

        Returns:
            NodeResult with EmbeddingResponse containing vector embeddings
        """
        ...

    @abstractmethod
    async def compare_semantics(
        self,
        request: SemanticComparisonRequest,
    ) -> NodeResult[SemanticComparisonResponse]:
        """
        Compare semantic similarity between content.

        Args:
            request: Semantic comparison request with content to compare

        Returns:
            NodeResult with SemanticComparisonResponse containing similarity scores
        """
        ...

    @abstractmethod
    async def cluster_content(
        self,
        content_items: ModelStringList,
        num_clusters: Optional[int] = None,
        correlation_id: Optional[UUID] = None,
    ) -> NodeResult[ModelMetadata]:
        """
        Cluster content items by semantic similarity.

        Args:
            content_items: List of content to cluster
            num_clusters: Optional number of clusters to create
            correlation_id: Request correlation ID

        Returns:
            NodeResult with clustering results
        """
        ...


class ProtocolPatternRecognition(ProtocolMemoryOperations, Protocol):
    """
    Protocol for pattern recognition and learning (Compute node).

    Recognizes patterns in memory data, learns from historical patterns,
    and makes predictions based on learned patterns.
    """

    @abstractmethod
    async def recognize_patterns(
        self,
        request: PatternRecognitionRequest,
    ) -> NodeResult[PatternRecognitionResponse]:
        """
        Recognize patterns in memory data.

        Args:
            request: Pattern recognition request with data to analyze

        Returns:
            NodeResult with PatternRecognitionResponse containing recognized patterns
        """
        ...

    @abstractmethod
    async def learn_patterns(
        self,
        request: PatternLearningRequest,
    ) -> NodeResult[PatternLearningResponse]:
        """
        Learn new patterns from memory data.

        Args:
            request: Pattern learning request with training data

        Returns:
            NodeResult with PatternLearningResponse containing learning results
        """
        ...

    @abstractmethod
    async def predict_patterns(
        self,
        request: PatternPredictionRequest,
    ) -> NodeResult[PatternPredictionResponse]:
        """
        Predict future patterns based on learned data.

        Args:
            request: Pattern prediction request with context data

        Returns:
            NodeResult with PatternPredictionResponse containing predictions
        """
        ...

    @abstractmethod
    async def validate_patterns(
        self,
        patterns: ModelResultCollection,
        validation_data: ModelResultCollection,
        correlation_id: Optional[UUID] = None,
    ) -> NodeResult[ModelMetadata]:
        """
        Validate discovered patterns against validation data.

        Args:
            patterns: Patterns to validate
            validation_data: Data to validate patterns against
            correlation_id: Request correlation ID

        Returns:
            NodeResult with pattern validation results
        """
        ...


# === REDUCER NODE PROTOCOLS ===


class ProtocolMemoryConsolidator(ProtocolMemoryOperations, Protocol):
    """
    Protocol for memory consolidation and deduplication (Reducer node).

    Consolidates similar memories, removes duplicates while preserving
    provenance, and merges related memory contexts.
    """

    @abstractmethod
    async def consolidate_memories(
        self,
        request: ConsolidationRequest,
    ) -> NodeResult[ConsolidationResponse]:
        """
        Consolidate similar memories into unified representations.

        Args:
            request: Consolidation request with consolidation criteria

        Returns:
            NodeResult with ConsolidationResponse containing consolidation results
        """
        ...

    @abstractmethod
    async def deduplicate_memories(
        self,
        request: DeduplicationRequest,
    ) -> NodeResult[DeduplicationResponse]:
        """
        Remove duplicate memories while preserving provenance.

        Args:
            request: Deduplication request with deduplication parameters

        Returns:
            NodeResult with DeduplicationResponse containing deduplication results
        """
        ...

    @abstractmethod
    async def merge_memory_contexts(
        self,
        request: ContextMergeRequest,
    ) -> NodeResult[ContextMergeResponse]:
        """
        Merge related memory contexts.

        Args:
            request: Context merge request with merge criteria

        Returns:
            NodeResult with ContextMergeResponse containing merge results
        """
        ...

    @abstractmethod
    async def detect_conflicts(
        self,
        memories: List[MemoryRecord],
        correlation_id: Optional[UUID] = None,
    ) -> NodeResult[ModelResultCollection]:
        """
        Detect conflicts between memory records.

        Args:
            memories: List of memory records to analyze
            correlation_id: Request correlation ID

        Returns:
            NodeResult with list of detected conflicts
        """
        ...


class ProtocolMemoryAggregator(ProtocolMemoryOperations, Protocol):
    """
    Protocol for memory aggregation and summarization (Reducer node).

    Aggregates memories by criteria, creates summaries of memory clusters,
    and generates statistical analysis of memory usage.
    """

    @abstractmethod
    async def aggregate_memories(
        self,
        request: AggregationRequest,
    ) -> NodeResult[AggregationResponse]:
        """
        Aggregate memories by temporal or semantic criteria.

        Args:
            request: Aggregation request with aggregation parameters

        Returns:
            NodeResult with AggregationResponse containing aggregated data
        """
        ...

    @abstractmethod
    async def summarize_memory_clusters(
        self,
        request: SummarizationRequest,
    ) -> NodeResult[SummarizationResponse]:
        """
        Create summaries of memory clusters.

        Args:
            request: Summarization request with cluster data

        Returns:
            NodeResult with SummarizationResponse containing cluster summaries
        """
        ...

    @abstractmethod
    async def generate_memory_statistics(
        self,
        request: StatisticsRequest,
    ) -> NodeResult[StatisticsResponse]:
        """
        Generate statistical analysis of memory usage.

        Args:
            request: Statistics request with analysis parameters

        Returns:
            NodeResult with StatisticsResponse containing usage statistics
        """
        ...

    @abstractmethod
    async def create_memory_views(
        self,
        view_definition: ModelConfiguration,
        correlation_id: Optional[UUID] = None,
    ) -> NodeResult[ModelMetadata]:
        """
        Create aggregated views of memory data.

        Args:
            view_definition: Definition of the view to create
            correlation_id: Request correlation ID

        Returns:
            NodeResult with created view data
        """
        ...


class ProtocolMemoryOptimizer(ProtocolMemoryOperations, Protocol):
    """
    Protocol for memory performance optimization (Reducer node).

    Optimizes memory storage layout, compresses memories while preserving
    semantic content, and optimizes retrieval performance.
    """

    @abstractmethod
    async def optimize_memory_layout(
        self,
        request: LayoutOptimizationRequest,
    ) -> NodeResult[LayoutOptimizationResponse]:
        """
        Optimize memory storage layout for performance.

        Args:
            request: Layout optimization request with optimization parameters

        Returns:
            NodeResult with LayoutOptimizationResponse containing optimization results
        """
        ...

    @abstractmethod
    async def compress_memories(
        self,
        request: CompressionRequest,
    ) -> NodeResult[CompressionResponse]:
        """
        Compress memories while preserving semantic content.

        Args:
            request: Compression request with compression parameters

        Returns:
            NodeResult with CompressionResponse containing compression results
        """
        ...

    @abstractmethod
    async def optimize_retrieval_paths(
        self,
        request: RetrievalOptimizationRequest,
    ) -> NodeResult[RetrievalOptimizationResponse]:
        """
        Optimize memory retrieval performance.

        Args:
            request: Retrieval optimization request with optimization parameters

        Returns:
            NodeResult with RetrievalOptimizationResponse containing optimization results
        """
        ...

    @abstractmethod
    async def analyze_performance(
        self,
        time_window: datetime,
        correlation_id: Optional[UUID] = None,
    ) -> NodeResult[ModelMetadata]:
        """
        Analyze memory system performance over a time window.

        Args:
            time_window: Time window for performance analysis
            correlation_id: Request correlation ID

        Returns:
            NodeResult with performance analysis results
        """
        ...


# === ORCHESTRATOR NODE PROTOCOLS ===


class ProtocolWorkflowCoordinator(ProtocolMemoryOperations, Protocol):
    """
    Protocol for workflow coordination and execution (Orchestrator node).

    Executes complex memory workflows, coordinates parallel operations,
    and manages workflow execution state and recovery.
    """

    @abstractmethod
    async def execute_memory_workflow(
        self,
        request: WorkflowExecutionRequest,
    ) -> NodeResult[WorkflowExecutionResponse]:
        """
        Execute complex memory workflows.

        Args:
            request: Workflow execution request with workflow definition

        Returns:
            NodeResult with WorkflowExecutionResponse containing execution results
        """
        ...

    @abstractmethod
    async def coordinate_parallel_operations(
        self,
        request: ParallelCoordinationRequest,
    ) -> NodeResult[ParallelCoordinationResponse]:
        """
        Coordinate parallel memory operations.

        Args:
            request: Parallel coordination request with operation definitions

        Returns:
            NodeResult with ParallelCoordinationResponse containing coordination results
        """
        ...

    @abstractmethod
    async def manage_workflow_state(
        self,
        request: WorkflowStateRequest,
    ) -> NodeResult[WorkflowStateResponse]:
        """
        Manage workflow execution state and recovery.

        Args:
            request: Workflow state request with state management operations

        Returns:
            NodeResult with WorkflowStateResponse containing state management results
        """
        ...

    @abstractmethod
    async def monitor_workflow_progress(
        self,
        workflow_id: UUID,
        correlation_id: Optional[UUID] = None,
    ) -> NodeResult[ModelMetadata]:
        """
        Monitor the progress of a running workflow.

        Args:
            workflow_id: ID of the workflow to monitor
            correlation_id: Request correlation ID

        Returns:
            NodeResult with workflow progress information
        """
        ...


class ProtocolAgentCoordinator(ProtocolMemoryOperations, Protocol):
    """
    Protocol for cross-agent coordination and communication (Orchestrator node).

    Coordinates memory operations across multiple agents, broadcasts memory
    updates, and synchronizes agent state.
    """

    @abstractmethod
    async def coordinate_agents(
        self,
        request: AgentCoordinationRequest,
    ) -> NodeResult[AgentCoordinationResponse]:
        """
        Coordinate memory operations across multiple agents.

        Args:
            request: Agent coordination request with coordination parameters

        Returns:
            NodeResult with AgentCoordinationResponse containing coordination results
        """
        ...

    @abstractmethod
    async def broadcast_memory_updates(
        self,
        request: BroadcastRequest,
    ) -> NodeResult[BroadcastResponse]:
        """
        Broadcast memory updates to subscribed agents.

        Args:
            request: Broadcast request with update information

        Returns:
            NodeResult with BroadcastResponse containing broadcast results
        """
        ...

    @abstractmethod
    async def synchronize_agent_state(
        self,
        request: StateSynchronizationRequest,
    ) -> NodeResult[StateSynchronizationResponse]:
        """
        Synchronize memory state across agents.

        Args:
            request: State synchronization request with synchronization parameters

        Returns:
            NodeResult with StateSynchronizationResponse containing sync results
        """
        ...

    @abstractmethod
    async def register_agent(
        self,
        agent_id: UUID,
        agent_capabilities: ModelMetadata,
        correlation_id: Optional[UUID] = None,
    ) -> NodeResult[bool]:
        """
        Register an agent with the coordination system.

        Args:
            agent_id: Unique identifier for the agent
            agent_capabilities: Dictionary describing agent capabilities
            correlation_id: Request correlation ID

        Returns:
            NodeResult indicating registration success/failure
        """
        ...


class ProtocolMemoryOrchestrator(ProtocolMemoryOperations, Protocol):
    """
    Protocol for high-level memory system orchestration (Orchestrator node).

    Orchestrates complete memory lifecycle management, manages quotas and limits,
    and coordinates memory migrations between storage systems.
    """

    @abstractmethod
    async def orchestrate_memory_lifecycle(
        self,
        request: LifecycleOrchestrationRequest,
    ) -> NodeResult[LifecycleOrchestrationResponse]:
        """
        Orchestrate complete memory lifecycle management.

        Args:
            request: Lifecycle orchestration request with lifecycle parameters

        Returns:
            NodeResult with LifecycleOrchestrationResponse containing lifecycle results
        """
        ...

    @abstractmethod
    async def manage_memory_quotas(
        self,
        request: QuotaManagementRequest,
    ) -> NodeResult[QuotaManagementResponse]:
        """
        Manage memory usage quotas and limits.

        Args:
            request: Quota management request with quota parameters

        Returns:
            NodeResult with QuotaManagementResponse containing quota management results
        """
        ...

    @abstractmethod
    async def coordinate_memory_migrations(
        self,
        request: MigrationCoordinationRequest,
    ) -> NodeResult[MigrationCoordinationResponse]:
        """
        Coordinate memory migrations between storage systems.

        Args:
            request: Migration coordination request with migration parameters

        Returns:
            NodeResult with MigrationCoordinationResponse containing migration results
        """
        ...

    @abstractmethod
    async def get_system_status(
        self,
        correlation_id: Optional[UUID] = None,
    ) -> NodeResult[ModelMetadata]:
        """
        Get comprehensive memory system status.

        Args:
            correlation_id: Request correlation ID

        Returns:
            NodeResult with system status information
        """
        ...
