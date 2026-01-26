"""
Base Protocol Definitions for OmniMemory ONEX Architecture

This module defines all protocol interfaces following ONEX 4-node architecture:
- Effect: Memory storage, retrieval, and persistence operations
- Compute: Intelligence processing, semantic analysis, pattern recognition
- Reducer: Memory consolidation, aggregation, and optimization
- Orchestrator: Workflow, agent, and memory coordination

All protocols use typing.Protocol for structural typing, avoiding isinstance
checks and supporting ModelONEXContainer dependency injection patterns.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from datetime import datetime
    from uuid import UUID

    from omnibase_core.models.core.model_base_result import ModelBaseResult

    from ..models.foundation import (
        ModelConfiguration,
        ModelMetadata,
        ModelOptionalStringList,
        ModelResultCollection,
        ModelStringList,
        ModelSystemConfiguration,
    )
    from .data_models import (  # Requests and responses
        AgentCoordinationRequest,
        AggregationRequest,
        BackupRequest,
        BaseMemoryRequest,
        BaseMemoryResponse,
        BroadcastRequest,
        CompressionRequest,
        ConsolidationRequest,
        ContextMergeRequest,
        ContextualSearchRequest,
        DeduplicationRequest,
        EmbeddingRequest,
        InsightExtractionRequest,
        IntelligenceProcessRequest,
        LayoutOptimizationRequest,
        LifecycleOrchestrationRequest,
        MemoryDeleteRequest,
        MemoryRecord,
        MemoryRetrieveRequest,
        MemoryStoreRequest,
        MigrationCoordinationRequest,
        ParallelCoordinationRequest,
        PatternAnalysisRequest,
        PatternLearningRequest,
        PatternPredictionRequest,
        PatternRecognitionRequest,
        PersistenceRequest,
        QuotaManagementRequest,
        RestoreRequest,
        RetrievalOptimizationRequest,
        SemanticAnalysisRequest,
        SemanticComparisonRequest,
        SemanticSearchRequest,
        StateSynchronizationRequest,
        StatisticsRequest,
        SummarizationRequest,
        TemporalSearchRequest,
        WorkflowExecutionRequest,
        WorkflowStateRequest,
    )

# === BASE PROTOCOLS ===


class ProtocolMemoryBase(Protocol):
    """
    Base protocol for all memory-related operations.

    Provides foundational capabilities that all memory components must implement,
    including health checking, configuration management, and basic observability.
    """

    async def health_check(
        self,
        correlation_id: UUID | None = None,
    ) -> ModelBaseResult:
        """
        Check the health status of the memory component.

        Returns:
            ModelBaseResult containing ModelHealthResponse with:
            - status: overall system health (healthy/degraded/unhealthy)
            - latency_ms: response time metrics
            - resource_usage: detailed system resource metrics
            - dependencies: status of external dependencies
            - uptime, version, environment details
        """
        ...

    async def get_metrics(
        self,
        correlation_id: UUID | None = None,
    ) -> ModelBaseResult:
        """
        Get operational metrics for the memory component.

        Returns:
            ModelBaseResult containing ModelMetricsResponse with:
            - operation_counts: detailed counts by operation type
            - performance_metrics: latency, throughput, error rates
            - resource_metrics: memory usage, cache statistics, connections
            - custom_metrics: application-specific measurements
            - alerts: active performance warnings
        """
        ...

    async def configure(
        self,
        config: ModelSystemConfiguration,
        correlation_id: UUID | None = None,
    ) -> ModelBaseResult:
        """
        Configure the memory component with new settings.

        Args:
            config: ModelSystemConfiguration with database, cache, performance,
                   and observability settings
            correlation_id: Request correlation ID

        Returns:
            ModelBaseResult indicating configuration success/failure
        """
        ...


class ProtocolMemoryOperations(ProtocolMemoryBase, Protocol):
    """
    Base protocol for memory operations with common patterns.

    Extends ProtocolMemoryBase with standard CRUD operations that most
    memory components will need to implement.
    """

    async def validate_request(
        self,
        request: BaseMemoryRequest,
    ) -> ModelBaseResult:
        """
        Validate a memory operation request.

        Args:
            request: The request to validate

        Returns:
            ModelBaseResult indicating validation success/failure with error details
        """
        ...

    async def log_operation(
        self,
        operation: str,
        request: BaseMemoryRequest,
        response: BaseMemoryResponse,
        correlation_id: UUID,
    ) -> ModelBaseResult:
        """
        Log a completed memory operation for audit and monitoring.

        Args:
            operation: Operation name
            request: Original request
            response: Operation response
            correlation_id: Request correlation ID

        Returns:
            ModelBaseResult indicating logging success/failure
        """
        ...


# === EFFECT NODE PROTOCOLS ===


class ProtocolMemoryStorage(ProtocolMemoryOperations, Protocol):
    """
    Protocol for memory storage operations (Effect node).

    Handles the storage of memory records with metadata, provenance tracking,
    and support for different storage backends (PostgreSQL, Redis, etc.).
    """

    async def store_memory(
        self,
        request: MemoryStoreRequest,
    ) -> ModelBaseResult:
        """
        Store a memory record with metadata and provenance.

        Args:
            request: Memory storage request containing the memory record

        Returns:
            ModelBaseResult with MemoryStoreResponse containing storage details
        """
        ...

    async def retrieve_memory(
        self,
        request: MemoryRetrieveRequest,
    ) -> ModelBaseResult:
        """
        Retrieve a memory record by identifier.

        Args:
            request: Memory retrieval request with memory ID

        Returns:
            ModelBaseResult with MemoryRetrieveResponse containing the memory record
        """
        ...

    async def delete_memory(
        self,
        request: MemoryDeleteRequest,
    ) -> ModelBaseResult:
        """
        Soft delete a memory record with audit trail.

        Args:
            request: Memory deletion request with memory ID

        Returns:
            ModelBaseResult with MemoryDeleteResponse indicating success/failure
        """
        ...

    async def update_memory(
        self,
        memory_id: UUID,
        updates: ModelMetadata,
        correlation_id: UUID | None = None,
    ) -> ModelBaseResult:
        """
        Update an existing memory record.

        Args:
            memory_id: ID of memory to update
            updates: Dictionary of fields to update
            correlation_id: Request correlation ID

        Returns:
            ModelBaseResult with updated MemoryRecord
        """
        ...

    async def list_memories(
        self,
        filters: ModelMetadata | None = None,
        limit: int = 100,
        offset: int = 0,
        correlation_id: UUID | None = None,
    ) -> ModelBaseResult:
        """
        List memory records with optional filtering and pagination.

        Args:
            filters: Optional filters to apply
            limit: Maximum number of records to return
            offset: Number of records to skip
            correlation_id: Request correlation ID

        Returns:
            ModelBaseResult with list of MemoryRecord objects
        """
        ...


class ProtocolMemoryRetrieval(ProtocolMemoryOperations, Protocol):
    """
    Protocol for advanced memory retrieval operations (Effect node).

    Provides semantic search, temporal search, and contextual retrieval
    capabilities using vector embeddings and time-based indexing.
    """

    async def semantic_search(
        self,
        request: SemanticSearchRequest,
    ) -> ModelBaseResult:
        """
        Perform vector-based semantic similarity search.

        Args:
            request: Semantic search request with query and parameters

        Returns:
            ModelBaseResult with SemanticSearchResponse containing matched memories
        """
        ...

    async def temporal_search(
        self,
        request: TemporalSearchRequest,
    ) -> ModelBaseResult:
        """
        Perform time-based memory retrieval with decay consideration.

        Args:
            request: Temporal search request with time range and criteria

        Returns:
            ModelBaseResult with TemporalSearchResponse containing time-filtered memories
        """
        ...

    async def contextual_search(
        self,
        request: ContextualSearchRequest,
    ) -> ModelBaseResult:
        """
        Perform context-aware memory retrieval using multiple criteria.

        Args:
            request: Contextual search request with context parameters

        Returns:
            ModelBaseResult with ContextualSearchResponse containing context-matched memories
        """
        ...

    async def get_related_memories(
        self,
        memory_id: UUID,
        relationship_types: ModelOptionalStringList | None = None,
        max_depth: int = 2,
        correlation_id: UUID | None = None,
    ) -> ModelBaseResult:
        """
        Get memories related to a specific memory record.

        Args:
            memory_id: ID of the source memory
            relationship_types: Types of relationships to follow
            max_depth: Maximum relationship depth to traverse
            correlation_id: Request correlation ID

        Returns:
            ModelBaseResult with list of related MemoryRecord objects
        """
        ...


class ProtocolMemoryPersistence(ProtocolMemoryOperations, Protocol):
    """
    Protocol for memory persistence and durability management (Effect node).

    Handles long-term storage, backup/restore operations, and data durability
    across different storage systems and failure scenarios.
    """

    async def persist_to_storage(
        self,
        request: PersistenceRequest,
    ) -> ModelBaseResult:
        """
        Persist memory data to durable storage.

        Args:
            request: Persistence request with storage preferences

        Returns:
            ModelBaseResult with PersistenceResponse containing storage details
        """
        ...

    async def backup_memory(
        self,
        request: BackupRequest,
    ) -> ModelBaseResult:
        """
        Create a backup of memory data with versioning.

        Args:
            request: Backup request with backup parameters

        Returns:
            ModelBaseResult with BackupResponse containing backup details
        """
        ...

    async def restore_memory(
        self,
        request: RestoreRequest,
    ) -> ModelBaseResult:
        """
        Restore memory data from a backup.

        Args:
            request: Restore request with backup identifier and options

        Returns:
            ModelBaseResult with RestoreResponse containing restore status
        """
        ...

    async def verify_integrity(
        self,
        memory_ids: list[UUID] | None = None,
        correlation_id: UUID | None = None,
    ) -> ModelBaseResult:
        """
        Verify the integrity of stored memory data.

        Args:
            memory_ids: Optional list of specific memory IDs to verify
            correlation_id: Request correlation ID

        Returns:
            ModelBaseResult with integrity verification results
        """
        ...


# === COMPUTE NODE PROTOCOLS ===


class ProtocolIntelligenceProcessor(ProtocolMemoryOperations, Protocol):
    """
    Protocol for intelligence processing operations (Compute node).

    Processes raw intelligence data into structured memory records,
    extracts insights, and performs pattern analysis.
    """

    async def process_intelligence(
        self,
        request: IntelligenceProcessRequest,
    ) -> ModelBaseResult:
        """
        Process raw intelligence data into structured memory.

        Args:
            request: Intelligence processing request with raw data

        Returns:
            ModelBaseResult with IntelligenceProcessResponse containing processed data
        """
        ...

    async def analyze_patterns(
        self,
        request: PatternAnalysisRequest,
    ) -> ModelBaseResult:
        """
        Analyze patterns in intelligence data.

        Args:
            request: Pattern analysis request with data to analyze

        Returns:
            ModelBaseResult with PatternAnalysisResponse containing discovered patterns
        """
        ...

    async def extract_insights(
        self,
        request: InsightExtractionRequest,
    ) -> ModelBaseResult:
        """
        Extract actionable insights from processed intelligence.

        Args:
            request: Insight extraction request with processed data

        Returns:
            ModelBaseResult with InsightExtractionResponse containing extracted insights
        """
        ...

    async def enrich_memory(
        self,
        memory: MemoryRecord,
        enrichment_types: ModelStringList,
        correlation_id: UUID | None = None,
    ) -> ModelBaseResult:
        """
        Enrich a memory record with additional intelligence data.

        Args:
            memory: Memory record to enrich
            enrichment_types: Types of enrichment to apply
            correlation_id: Request correlation ID

        Returns:
            ModelBaseResult with enriched MemoryRecord
        """
        ...


class ProtocolSemanticAnalyzer(ProtocolMemoryOperations, Protocol):
    """
    Protocol for semantic analysis and understanding (Compute node).

    Provides semantic analysis, vector embedding generation, and
    semantic similarity comparison capabilities.
    """

    async def analyze_semantics(
        self,
        request: SemanticAnalysisRequest,
    ) -> ModelBaseResult:
        """
        Analyze semantic content and relationships.

        Args:
            request: Semantic analysis request with content to analyze

        Returns:
            ModelBaseResult with SemanticAnalysisResponse containing analysis results
        """
        ...

    async def generate_embeddings(
        self,
        request: EmbeddingRequest,
    ) -> ModelBaseResult:
        """
        Generate vector embeddings for semantic search.

        Args:
            request: Embedding request with text to embed

        Returns:
            ModelBaseResult with EmbeddingResponse containing vector embeddings
        """
        ...

    async def compare_semantics(
        self,
        request: SemanticComparisonRequest,
    ) -> ModelBaseResult:
        """
        Compare semantic similarity between content.

        Args:
            request: Semantic comparison request with content to compare

        Returns:
            ModelBaseResult with SemanticComparisonResponse containing similarity scores
        """
        ...

    async def cluster_content(
        self,
        content_items: ModelStringList,
        num_clusters: int | None = None,
        correlation_id: UUID | None = None,
    ) -> ModelBaseResult:
        """
        Cluster content items by semantic similarity.

        Args:
            content_items: List of content to cluster
            num_clusters: Optional number of clusters to create
            correlation_id: Request correlation ID

        Returns:
            ModelBaseResult with clustering results
        """
        ...


class ProtocolPatternRecognition(ProtocolMemoryOperations, Protocol):
    """
    Protocol for pattern recognition and learning (Compute node).

    Recognizes patterns in memory data, learns from historical patterns,
    and makes predictions based on learned patterns.
    """

    async def recognize_patterns(
        self,
        request: PatternRecognitionRequest,
    ) -> ModelBaseResult:
        """
        Recognize patterns in memory data.

        Args:
            request: Pattern recognition request with data to analyze

        Returns:
            ModelBaseResult with PatternRecognitionResponse containing recognized patterns
        """
        ...

    async def learn_patterns(
        self,
        request: PatternLearningRequest,
    ) -> ModelBaseResult:
        """
        Learn new patterns from memory data.

        Args:
            request: Pattern learning request with training data

        Returns:
            ModelBaseResult with PatternLearningResponse containing learning results
        """
        ...

    async def predict_patterns(
        self,
        request: PatternPredictionRequest,
    ) -> ModelBaseResult:
        """
        Predict future patterns based on learned data.

        Args:
            request: Pattern prediction request with context data

        Returns:
            ModelBaseResult with PatternPredictionResponse containing predictions
        """
        ...

    async def validate_patterns(
        self,
        patterns: ModelResultCollection,
        validation_data: ModelResultCollection,
        correlation_id: UUID | None = None,
    ) -> ModelBaseResult:
        """
        Validate discovered patterns against validation data.

        Args:
            patterns: Patterns to validate
            validation_data: Data to validate patterns against
            correlation_id: Request correlation ID

        Returns:
            ModelBaseResult with pattern validation results
        """
        ...


# === REDUCER NODE PROTOCOLS ===


class ProtocolMemoryConsolidator(ProtocolMemoryOperations, Protocol):
    """
    Protocol for memory consolidation and deduplication (Reducer node).

    Consolidates similar memories, removes duplicates while preserving
    provenance, and merges related memory contexts.
    """

    async def consolidate_memories(
        self,
        request: ConsolidationRequest,
    ) -> ModelBaseResult:
        """
        Consolidate similar memories into unified representations.

        Args:
            request: Consolidation request with consolidation criteria

        Returns:
            ModelBaseResult with ConsolidationResponse containing consolidation results
        """
        ...

    async def deduplicate_memories(
        self,
        request: DeduplicationRequest,
    ) -> ModelBaseResult:
        """
        Remove duplicate memories while preserving provenance.

        Args:
            request: Deduplication request with deduplication parameters

        Returns:
            ModelBaseResult with DeduplicationResponse containing deduplication results
        """
        ...

    async def merge_memory_contexts(
        self,
        request: ContextMergeRequest,
    ) -> ModelBaseResult:
        """
        Merge related memory contexts.

        Args:
            request: Context merge request with merge criteria

        Returns:
            ModelBaseResult with ContextMergeResponse containing merge results
        """
        ...

    async def detect_conflicts(
        self,
        memories: list[MemoryRecord],
        correlation_id: UUID | None = None,
    ) -> ModelBaseResult:
        """
        Detect conflicts between memory records.

        Args:
            memories: List of memory records to analyze
            correlation_id: Request correlation ID

        Returns:
            ModelBaseResult with list of detected conflicts
        """
        ...


class ProtocolMemoryAggregator(ProtocolMemoryOperations, Protocol):
    """
    Protocol for memory aggregation and summarization (Reducer node).

    Aggregates memories by criteria, creates summaries of memory clusters,
    and generates statistical analysis of memory usage.
    """

    async def aggregate_memories(
        self,
        request: AggregationRequest,
    ) -> ModelBaseResult:
        """
        Aggregate memories by temporal or semantic criteria.

        Args:
            request: Aggregation request with aggregation parameters

        Returns:
            ModelBaseResult with AggregationResponse containing aggregated data
        """
        ...

    async def summarize_memory_clusters(
        self,
        request: SummarizationRequest,
    ) -> ModelBaseResult:
        """
        Create summaries of memory clusters.

        Args:
            request: Summarization request with cluster data

        Returns:
            ModelBaseResult with SummarizationResponse containing cluster summaries
        """
        ...

    async def generate_memory_statistics(
        self,
        request: StatisticsRequest,
    ) -> ModelBaseResult:
        """
        Generate statistical analysis of memory usage.

        Args:
            request: Statistics request with analysis parameters

        Returns:
            ModelBaseResult with StatisticsResponse containing usage statistics
        """
        ...

    async def create_memory_views(
        self,
        view_definition: ModelConfiguration,
        correlation_id: UUID | None = None,
    ) -> ModelBaseResult:
        """
        Create aggregated views of memory data.

        Args:
            view_definition: Definition of the view to create
            correlation_id: Request correlation ID

        Returns:
            ModelBaseResult with created view data
        """
        ...


class ProtocolMemoryOptimizer(ProtocolMemoryOperations, Protocol):
    """
    Protocol for memory performance optimization (Reducer node).

    Optimizes memory storage layout, compresses memories while preserving
    semantic content, and optimizes retrieval performance.
    """

    async def optimize_memory_layout(
        self,
        request: LayoutOptimizationRequest,
    ) -> ModelBaseResult:
        """
        Optimize memory storage layout for performance.

        Args:
            request: Layout optimization request with optimization parameters

        Returns:
            ModelBaseResult with LayoutOptimizationResponse containing optimization results
        """
        ...

    async def compress_memories(
        self,
        request: CompressionRequest,
    ) -> ModelBaseResult:
        """
        Compress memories while preserving semantic content.

        Args:
            request: Compression request with compression parameters

        Returns:
            ModelBaseResult with CompressionResponse containing compression results
        """
        ...

    async def optimize_retrieval_paths(
        self,
        request: RetrievalOptimizationRequest,
    ) -> ModelBaseResult:
        """
        Optimize memory retrieval performance.

        Args:
            request: Retrieval optimization request with optimization parameters

        Returns:
            ModelBaseResult with RetrievalOptimizationResponse containing
            optimization results
        """
        ...

    async def analyze_performance(
        self,
        time_window: datetime,
        correlation_id: UUID | None = None,
    ) -> ModelBaseResult:
        """
        Analyze memory system performance over a time window.

        Args:
            time_window: Time window for performance analysis
            correlation_id: Request correlation ID

        Returns:
            ModelBaseResult with performance analysis results
        """
        ...


# === ORCHESTRATOR NODE PROTOCOLS ===


class ProtocolWorkflowCoordinator(ProtocolMemoryOperations, Protocol):
    """
    Protocol for workflow coordination and execution (Orchestrator node).

    Executes complex memory workflows, coordinates parallel operations,
    and manages workflow execution state and recovery.
    """

    async def execute_memory_workflow(
        self,
        request: WorkflowExecutionRequest,
    ) -> ModelBaseResult:
        """
        Execute complex memory workflows.

        Args:
            request: Workflow execution request with workflow definition

        Returns:
            ModelBaseResult with WorkflowExecutionResponse containing execution results
        """
        ...

    async def coordinate_parallel_operations(
        self,
        request: ParallelCoordinationRequest,
    ) -> ModelBaseResult:
        """
        Coordinate parallel memory operations.

        Args:
            request: Parallel coordination request with operation definitions

        Returns:
            ModelBaseResult with ParallelCoordinationResponse containing coordination results
        """
        ...

    async def manage_workflow_state(
        self,
        request: WorkflowStateRequest,
    ) -> ModelBaseResult:
        """
        Manage workflow execution state and recovery.

        Args:
            request: Workflow state request with state management operations

        Returns:
            ModelBaseResult with WorkflowStateResponse containing state management results
        """
        ...

    async def monitor_workflow_progress(
        self,
        workflow_id: UUID,
        correlation_id: UUID | None = None,
    ) -> ModelBaseResult:
        """
        Monitor the progress of a running workflow.

        Args:
            workflow_id: ID of the workflow to monitor
            correlation_id: Request correlation ID

        Returns:
            ModelBaseResult with workflow progress information
        """
        ...


class ProtocolAgentCoordinator(ProtocolMemoryOperations, Protocol):
    """
    Protocol for cross-agent coordination and communication (Orchestrator node).

    Coordinates memory operations across multiple agents, broadcasts memory
    updates, and synchronizes agent state.
    """

    async def coordinate_agents(
        self,
        request: AgentCoordinationRequest,
    ) -> ModelBaseResult:
        """
        Coordinate memory operations across multiple agents.

        Args:
            request: Agent coordination request with coordination parameters

        Returns:
            ModelBaseResult with AgentCoordinationResponse containing coordination results
        """
        ...

    async def broadcast_memory_updates(
        self,
        request: BroadcastRequest,
    ) -> ModelBaseResult:
        """
        Broadcast memory updates to subscribed agents.

        Args:
            request: Broadcast request with update information

        Returns:
            ModelBaseResult with BroadcastResponse containing broadcast results
        """
        ...

    async def synchronize_agent_state(
        self,
        request: StateSynchronizationRequest,
    ) -> ModelBaseResult:
        """
        Synchronize memory state across agents.

        Args:
            request: State synchronization request with synchronization parameters

        Returns:
            ModelBaseResult with StateSynchronizationResponse containing sync results
        """
        ...

    async def register_agent(
        self,
        agent_id: UUID,
        agent_capabilities: ModelMetadata,
        correlation_id: UUID | None = None,
    ) -> ModelBaseResult:
        """
        Register an agent with the coordination system.

        Args:
            agent_id: Unique identifier for the agent
            agent_capabilities: Dictionary describing agent capabilities
            correlation_id: Request correlation ID

        Returns:
            ModelBaseResult indicating registration success/failure
        """
        ...


class ProtocolMemoryOrchestrator(ProtocolMemoryOperations, Protocol):
    """
    Protocol for high-level memory system orchestration (Orchestrator node).

    Orchestrates complete memory lifecycle management, manages quotas and limits,
    and coordinates memory migrations between storage systems.
    """

    async def orchestrate_memory_lifecycle(
        self,
        request: LifecycleOrchestrationRequest,
    ) -> ModelBaseResult:
        """
        Orchestrate complete memory lifecycle management.

        Args:
            request: Lifecycle orchestration request with lifecycle parameters

        Returns:
            ModelBaseResult with LifecycleOrchestrationResponse containing lifecycle results
        """
        ...

    async def manage_memory_quotas(
        self,
        request: QuotaManagementRequest,
    ) -> ModelBaseResult:
        """
        Manage memory usage quotas and limits.

        Args:
            request: Quota management request with quota parameters

        Returns:
            ModelBaseResult with QuotaManagementResponse containing quota management results
        """
        ...

    async def coordinate_memory_migrations(
        self,
        request: MigrationCoordinationRequest,
    ) -> ModelBaseResult:
        """
        Coordinate memory migrations between storage systems.

        Args:
            request: Migration coordination request with migration parameters

        Returns:
            ModelBaseResult with MigrationCoordinationResponse containing migration results
        """
        ...

    async def get_system_status(
        self,
        correlation_id: UUID | None = None,
    ) -> ModelBaseResult:
        """
        Get comprehensive memory system status.

        Args:
            correlation_id: Request correlation ID

        Returns:
            ModelBaseResult with system status information
        """
        ...
