"""
Data Models for OmniMemory ONEX Architecture

This module contains all Pydantic data models used throughout the OmniMemory system,
following ONEX contract-driven development patterns with strong typing and validation.

All models support monadic patterns with NodeResult composition and provide
comprehensive validation, serialization, and observability features.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from ..models.foundation import (
    ModelConfiguration,
    ModelEventCollection,
    ModelMetadata,
    ModelOptionalStringList,
    ModelResultCollection,
    ModelStringList,
    ModelStructuredData,
    convert_dict_to_metadata,
    convert_list_to_string_list,
)

# === ENUMS ===


class OperationStatus(str, Enum):
    """Status of memory operations."""

    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class ContentType(str, Enum):
    """Type of memory content."""

    TEXT = "text"
    JSON = "json"
    BINARY = "binary"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    STRUCTURED_DATA = "structured_data"


class MemoryPriority(str, Enum):
    """Memory priority levels."""

    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"
    ARCHIVE = "archive"


class AccessLevel(str, Enum):
    """Memory access control levels."""

    PUBLIC = "public"
    INTERNAL = "internal"
    RESTRICTED = "restricted"
    CONFIDENTIAL = "confidential"
    SECRET = "secret"


class IndexingStatus(str, Enum):
    """Memory indexing status."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


# === BASE MODELS ===


class BaseMemoryModel(BaseModel):
    """Base model for all OmniMemory data structures."""

    model_config = ConfigDict(
        # ONEX compliance settings
        str_strip_whitespace=True,
        validate_assignment=True,
        validate_default=True,
        extra="forbid",
        frozen=False,
        # Performance settings
        use_enum_values=True,
        arbitrary_types_allowed=False,
        # Serialization settings
        ser_json_bytes="base64",
        ser_json_timedelta="float",
    )


class UserContext(BaseMemoryModel):
    """User context and permissions for memory operations."""

    user_id: UUID = Field(description="Unique user identifier")
    agent_id: UUID = Field(description="Agent performing the operation")
    session_id: Optional[UUID] = Field(None, description="Session identifier")
    permissions: ModelStringList = Field(
        default_factory=ModelStringList,
        description="User permissions for memory operations",
    )
    access_level: AccessLevel = Field(
        AccessLevel.INTERNAL, description="User's maximum access level"
    )
    metadata: ModelMetadata = Field(
        default_factory=ModelMetadata, description="Additional user context metadata"
    )


class StoragePreferences(BaseMemoryModel):
    """Storage location and durability preferences."""

    storage_tier: str = Field(
        "standard", description="Storage tier preference (hot/warm/cold/archive)"
    )
    durability_level: str = Field(
        "standard", description="Durability level (standard/high/critical)"
    )
    replication_factor: int = Field(
        1, ge=1, le=10, description="Number of replicas to maintain"
    )
    encryption_required: bool = Field(
        True, description="Whether encryption is required"
    )
    geographic_preference: Optional[str] = Field(
        None, description="Geographic storage preference"
    )
    retention_policy: Optional[str] = Field(
        None, description="Data retention policy identifier"
    )


class SearchFilters(BaseMemoryModel):
    """Filters for memory search operations."""

    content_types: Optional[List[ContentType]] = Field(
        None, description="Filter by content types"
    )
    priority_levels: Optional[List[MemoryPriority]] = Field(
        None, description="Filter by priority levels"
    )
    access_levels: Optional[List[AccessLevel]] = Field(
        None, description="Filter by access levels"
    )
    tags: Optional[ModelOptionalStringList] = Field(
        None, description="Filter by tags (AND logic)"
    )
    source_agents: Optional[ModelOptionalStringList] = Field(
        None, description="Filter by source agents"
    )
    date_range_start: Optional[datetime] = Field(
        None, description="Filter by creation date (start)"
    )
    date_range_end: Optional[datetime] = Field(
        None, description="Filter by creation date (end)"
    )
    has_embeddings: Optional[bool] = Field(
        None, description="Filter by embedding availability"
    )


class SearchResult(BaseMemoryModel):
    """Individual search result with scoring."""

    memory_id: UUID = Field(description="Memory identifier")
    similarity_score: float = Field(
        ge=0.0, le=1.0, description="Similarity score (0.0 to 1.0)"
    )
    relevance_score: float = Field(
        ge=0.0, le=1.0, description="Relevance score (0.0 to 1.0)"
    )
    memory_record: Optional["MemoryRecord"] = Field(
        None, description="Full memory record (if requested)"
    )
    highlight_snippets: ModelStringList = Field(
        default_factory=ModelStringList,
        description="Text snippets with search term highlights",
    )
    match_metadata: ModelMetadata = Field(
        default_factory=ModelMetadata, description="Additional match information"
    )


# === BASE REQUEST/RESPONSE MODELS ===


class BaseMemoryRequest(BaseMemoryModel):
    """Base request schema for all memory operations."""

    correlation_id: UUID = Field(
        default_factory=uuid4, description="Correlation ID for request tracking"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Request timestamp"
    )
    user_context: Optional[UserContext] = Field(
        None, description="User context and permissions"
    )
    timeout_ms: int = Field(
        30000, ge=100, le=300000, description="Request timeout in milliseconds"
    )
    metadata: ModelMetadata = Field(
        default_factory=ModelMetadata, description="Additional request metadata"
    )


class BaseMemoryResponse(BaseMemoryModel):
    """Base response schema for all memory operations."""

    correlation_id: UUID = Field(description="Correlation ID matching request")
    status: OperationStatus = Field(description="Operation execution status")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Response timestamp"
    )
    execution_time_ms: int = Field(ge=0, description="Execution time in milliseconds")
    provenance: ModelStringList = Field(
        default_factory=ModelStringList, description="Operation provenance chain"
    )
    trust_score: float = Field(
        1.0, ge=0.0, le=1.0, description="Trust score (0.0 to 1.0)"
    )
    metadata: ModelMetadata = Field(
        default_factory=ModelMetadata, description="Additional response metadata"
    )
    warnings: ModelStringList = Field(
        default_factory=ModelStringList, description="Non-fatal warnings"
    )
    events: ModelResultCollection = Field(
        default_factory=ModelResultCollection,
        description="Operation events for observability",
    )


# === CORE DATA MODELS ===


class MemoryRecord(BaseMemoryModel):
    """Core memory record with ONEX compliance."""

    memory_id: UUID = Field(
        default_factory=uuid4, description="Unique memory identifier"
    )
    content: str = Field(
        description="Memory content", max_length=1048576  # 1MB max content
    )
    content_type: ContentType = Field(description="Type of memory content")
    content_hash: Optional[str] = Field(
        None, description="SHA-256 hash of content for integrity"
    )
    embedding: Optional[List[float]] = Field(
        None,
        description="Vector embedding for semantic search",
        min_length=768,
        max_length=4096,
    )
    embedding_model: Optional[str] = Field(
        None, description="Model used to generate embedding"
    )
    tags: ModelStringList = Field(
        default_factory=ModelStringList,
        description="Memory tags for categorization",
        max_length=100,
    )
    priority: MemoryPriority = Field(
        MemoryPriority.NORMAL, description="Memory priority level"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="Creation timestamp"
    )
    updated_at: datetime = Field(
        default_factory=datetime.utcnow, description="Last update timestamp"
    )
    expires_at: Optional[datetime] = Field(
        None, description="Expiration timestamp (for temporal memory)"
    )
    provenance: ModelStringList = Field(
        default_factory=ModelStringList, description="Memory provenance chain"
    )
    source_agent: str = Field(description="Agent that created this memory")
    related_memories: List[UUID] = Field(
        default_factory=list, description="Related memory identifiers"
    )
    access_level: AccessLevel = Field(
        AccessLevel.INTERNAL, description="Memory access control level"
    )
    storage_location: Optional[str] = Field(
        None, description="Physical storage location"
    )
    index_status: IndexingStatus = Field(
        IndexingStatus.PENDING, description="Indexing status for search"
    )
    quality_score: float = Field(
        1.0, ge=0.0, le=1.0, description="Quality score based on content analysis"
    )
    usage_count: int = Field(
        0, ge=0, description="Number of times this memory has been accessed"
    )
    last_accessed: Optional[datetime] = Field(None, description="Last access timestamp")


# === MEMORY OPERATION REQUESTS/RESPONSES ===


class MemoryStoreRequest(BaseMemoryRequest):
    """Request to store memory."""

    memory: MemoryRecord = Field(description="Memory record to store")
    storage_preferences: Optional[StoragePreferences] = Field(
        None, description="Storage location and durability preferences"
    )
    generate_embedding: bool = Field(
        True, description="Whether to generate vector embedding"
    )
    embedding_model: Optional[str] = Field(
        None, description="Specific embedding model to use"
    )
    index_immediately: bool = Field(
        True, description="Whether to index for search immediately"
    )


class MemoryStoreResponse(BaseMemoryResponse):
    """Response from memory store operation."""

    memory_id: UUID = Field(description="Generated/confirmed memory identifier")
    storage_location: str = Field(description="Actual storage location")
    indexing_status: IndexingStatus = Field(description="Indexing completion status")
    embedding_generated: bool = Field(description="Whether embedding was generated")
    duplicate_detected: bool = Field(
        False, description="Whether a duplicate was detected"
    )
    storage_size_bytes: int = Field(ge=0, description="Storage size in bytes")


class MemoryRetrieveRequest(BaseMemoryRequest):
    """Request to retrieve memory by ID."""

    memory_id: UUID = Field(description="Memory identifier to retrieve")
    include_embedding: bool = Field(
        False, description="Include vector embedding in response"
    )
    include_related: bool = Field(False, description="Include related memories")
    related_depth: int = Field(
        1, ge=1, le=5, description="Depth of related memory traversal"
    )


class MemoryRetrieveResponse(BaseMemoryResponse):
    """Response from memory retrieve operation."""

    memory: Optional[MemoryRecord] = Field(description="Retrieved memory record")
    related_memories: List[MemoryRecord] = Field(
        default_factory=list, description="Related memory records (if requested)"
    )
    cache_hit: bool = Field(description="Whether result came from cache")


class MemoryDeleteRequest(BaseMemoryRequest):
    """Request to delete memory."""

    memory_id: UUID = Field(description="Memory identifier to delete")
    soft_delete: bool = Field(
        True, description="Whether to perform soft delete (preserving audit trail)"
    )
    reason: Optional[str] = Field(None, description="Reason for deletion")


class MemoryDeleteResponse(BaseMemoryResponse):
    """Response from memory delete operation."""

    memory_id: UUID = Field(description="Deleted memory identifier")
    soft_deleted: bool = Field(description="Whether soft delete was performed")
    backup_location: Optional[str] = Field(
        None, description="Location of backup (if created)"
    )


# === SEARCH OPERATION REQUESTS/RESPONSES ===


class SemanticSearchRequest(BaseMemoryRequest):
    """Request for semantic similarity search."""

    query: str = Field(description="Search query text", min_length=1, max_length=10000)
    limit: int = Field(10, ge=1, le=1000, description="Maximum number of results")
    similarity_threshold: float = Field(
        0.7, ge=0.0, le=1.0, description="Minimum similarity score"
    )
    filters: Optional[SearchFilters] = Field(
        None, description="Additional search filters"
    )
    include_embeddings: bool = Field(
        False, description="Include embeddings in response"
    )
    include_content: bool = Field(True, description="Include full content in results")
    highlight_matches: bool = Field(
        True, description="Highlight search terms in content"
    )
    embedding_model: Optional[str] = Field(
        None, description="Specific embedding model for query"
    )


class SemanticSearchResponse(BaseMemoryResponse):
    """Response from semantic search."""

    results: List[SearchResult] = Field(description="Search results with scores")
    total_matches: int = Field(ge=0, description="Total number of matches found")
    search_time_ms: int = Field(ge=0, description="Search execution time")
    index_version: str = Field(description="Search index version used")
    query_embedding: Optional[List[float]] = Field(
        None, description="Query embedding used for search"
    )


class TemporalSearchRequest(BaseMemoryRequest):
    """Request for time-based memory retrieval."""

    time_range_start: Optional[datetime] = Field(
        None, description="Start of time range"
    )
    time_range_end: Optional[datetime] = Field(None, description="End of time range")
    temporal_decay_factor: float = Field(
        1.0, ge=0.0, le=1.0, description="Temporal decay factor for scoring"
    )
    limit: int = Field(10, ge=1, le=1000, description="Maximum number of results")
    filters: Optional[SearchFilters] = Field(
        None, description="Additional search filters"
    )
    sort_by: str = Field(
        "relevance", description="Sort order (relevance/created_at/updated_at)"
    )


class TemporalSearchResponse(BaseMemoryResponse):
    """Response from temporal search."""

    results: List[SearchResult] = Field(description="Time-filtered search results")
    total_matches: int = Field(
        ge=0, description="Total number of matches in time range"
    )
    time_range_coverage: Dict[str, int] = Field(
        default_factory=dict, description="Distribution of matches across time periods"
    )


class ContextualSearchRequest(BaseMemoryRequest):
    """Request for context-aware memory retrieval."""

    context: ModelConfiguration = Field(
        default_factory=ModelConfiguration, description="Context parameters for search"
    )
    context_weight: float = Field(
        0.5, ge=0.0, le=1.0, description="Weight of context vs content similarity"
    )
    limit: int = Field(10, ge=1, le=1000, description="Maximum number of results")
    filters: Optional[SearchFilters] = Field(
        None, description="Additional search filters"
    )


class ContextualSearchResponse(BaseMemoryResponse):
    """Response from contextual search."""

    results: List[SearchResult] = Field(description="Context-matched results")
    context_analysis: ModelMetadata = Field(
        default_factory=ModelMetadata, description="Analysis of context matching"
    )


# === PLACEHOLDER MODELS FOR COMPLEX OPERATIONS ===
# These would be fully implemented based on specific requirements


class PersistenceRequest(BaseMemoryRequest):
    """Request for memory persistence operations."""

    persistence_type: str = Field(description="Type of persistence operation")
    target_storage: str = Field(description="Target storage system")
    options: ModelConfiguration = Field(default_factory=ModelConfiguration)


class PersistenceResponse(BaseMemoryResponse):
    """Response from persistence operations."""

    persistence_id: UUID = Field(description="Persistence operation ID")
    storage_location: str = Field(description="Final storage location")


class BackupRequest(BaseMemoryRequest):
    """Request for memory backup operations."""

    backup_type: str = Field(description="Type of backup")
    target_location: str = Field(description="Backup target location")
    options: ModelConfiguration = Field(default_factory=ModelConfiguration)


class BackupResponse(BaseMemoryResponse):
    """Response from backup operations."""

    backup_id: UUID = Field(description="Backup identifier")
    backup_location: str = Field(description="Backup storage location")


class RestoreRequest(BaseMemoryRequest):
    """Request for memory restore operations."""

    backup_id: UUID = Field(description="Backup to restore from")
    restore_options: ModelConfiguration = Field(default_factory=ModelConfiguration)


class RestoreResponse(BaseMemoryResponse):
    """Response from restore operations."""

    restored_memories: int = Field(description="Number of memories restored")
    restore_location: str = Field(description="Restore target location")


# Intelligence Processing Models
class IntelligenceProcessRequest(BaseMemoryRequest):
    """Request for intelligence processing."""

    raw_data: ModelStructuredData = Field(
        default_factory=ModelStructuredData, description="Raw intelligence data"
    )
    processing_options: ModelConfiguration = Field(default_factory=ModelConfiguration)


class IntelligenceProcessResponse(BaseMemoryResponse):
    """Response from intelligence processing."""

    processed_data: ModelStructuredData = Field(
        default_factory=ModelStructuredData, description="Processed intelligence data"
    )
    insights: ModelResultCollection = Field(default_factory=ModelResultCollection)


class PatternAnalysisRequest(BaseMemoryRequest):
    """Request for pattern analysis."""

    data_set: ModelStructuredData = Field(
        default_factory=ModelStructuredData, description="Data set to analyze"
    )
    analysis_type: str = Field(description="Type of pattern analysis")


class PatternAnalysisResponse(BaseMemoryResponse):
    """Response from pattern analysis."""

    patterns: ModelResultCollection = Field(
        default_factory=ModelResultCollection, description="Discovered patterns"
    )
    confidence_scores: List[float] = Field(description="Pattern confidence scores")


class InsightExtractionRequest(BaseMemoryRequest):
    """Request for insight extraction."""

    processed_data: ModelStructuredData = Field(
        default_factory=ModelStructuredData,
        description="Processed data to extract insights from",
    )
    extraction_criteria: ModelConfiguration = Field(default_factory=ModelConfiguration)


class InsightExtractionResponse(BaseMemoryResponse):
    """Response from insight extraction."""

    insights: ModelResultCollection = Field(
        default_factory=ModelResultCollection, description="Extracted insights"
    )
    insight_scores: List[float] = Field(description="Insight relevance scores")


# Semantic Analysis Models
class SemanticAnalysisRequest(BaseMemoryRequest):
    """Request for semantic analysis."""

    content: str = Field(description="Content to analyze")
    analysis_depth: str = Field("standard", description="Depth of analysis")


class SemanticAnalysisResponse(BaseMemoryResponse):
    """Response from semantic analysis."""

    semantic_features: ModelConfiguration = Field(
        default_factory=ModelConfiguration, description="Semantic features"
    )
    relationships: ModelResultCollection = Field(
        default_factory=ModelResultCollection, description="Semantic relationships"
    )


class EmbeddingRequest(BaseMemoryRequest):
    """Request for vector embedding generation."""

    text: str = Field(description="Text to embed")
    model: Optional[str] = Field(None, description="Embedding model to use")


class EmbeddingResponse(BaseMemoryResponse):
    """Response from embedding generation."""

    embedding: List[float] = Field(description="Generated vector embedding")
    model_used: str = Field(description="Embedding model used")
    dimensions: int = Field(description="Embedding dimensions")


class SemanticComparisonRequest(BaseMemoryRequest):
    """Request for semantic comparison."""

    content_a: str = Field(description="First content to compare")
    content_b: str = Field(description="Second content to compare")
    comparison_type: str = Field("similarity", description="Type of comparison")


class SemanticComparisonResponse(BaseMemoryResponse):
    """Response from semantic comparison."""

    similarity_score: float = Field(description="Semantic similarity score")
    comparison_details: ModelConfiguration = Field(
        default_factory=ModelConfiguration, description="Detailed comparison results"
    )


# Pattern Recognition Models
class PatternRecognitionRequest(BaseMemoryRequest):
    """Request for pattern recognition."""

    data: ModelStructuredData = Field(
        default_factory=ModelStructuredData, description="Data to analyze for patterns"
    )
    pattern_types: ModelStringList = Field(
        default_factory=ModelStringList, description="Types of patterns to look for"
    )


class PatternRecognitionResponse(BaseMemoryResponse):
    """Response from pattern recognition."""

    recognized_patterns: ModelResultCollection = Field(
        default_factory=ModelResultCollection, description="Recognized patterns"
    )
    pattern_confidence: List[float] = Field(description="Pattern confidence scores")


class PatternLearningRequest(BaseMemoryRequest):
    """Request for pattern learning."""

    training_data: ModelStructuredData = Field(
        default_factory=ModelStructuredData, description="Training data for learning"
    )
    learning_parameters: ModelConfiguration = Field(default_factory=ModelConfiguration)


class PatternLearningResponse(BaseMemoryResponse):
    """Response from pattern learning."""

    learned_patterns: ModelResultCollection = Field(
        default_factory=ModelResultCollection, description="Newly learned patterns"
    )
    learning_metrics: ModelConfiguration = Field(
        default_factory=ModelConfiguration, description="Learning performance metrics"
    )


class PatternPredictionRequest(BaseMemoryRequest):
    """Request for pattern prediction."""

    context_data: ModelStructuredData = Field(
        default_factory=ModelStructuredData, description="Context data for prediction"
    )
    prediction_horizon: int = Field(description="Prediction time horizon")


class PatternPredictionResponse(BaseMemoryResponse):
    """Response from pattern prediction."""

    predictions: ModelResultCollection = Field(
        default_factory=ModelResultCollection, description="Pattern predictions"
    )
    confidence_intervals: List[Dict[str, float]] = Field(
        description="Prediction confidence"
    )


# Memory Consolidation Models
class ConsolidationRequest(BaseMemoryRequest):
    """Request for memory consolidation."""

    memory_ids: List[UUID] = Field(description="Memories to consolidate")
    consolidation_strategy: str = Field(description="Consolidation strategy")


class ConsolidationResponse(BaseMemoryResponse):
    """Response from memory consolidation."""

    consolidated_memory_id: UUID = Field(description="ID of consolidated memory")
    source_memory_ids: List[UUID] = Field(description="IDs of source memories")


class DeduplicationRequest(BaseMemoryRequest):
    """Request for memory deduplication."""

    memory_scope: ModelConfiguration = Field(
        default_factory=ModelConfiguration, description="Scope of deduplication"
    )
    similarity_threshold: float = Field(
        0.95, description="Similarity threshold for duplicates"
    )


class DeduplicationResponse(BaseMemoryResponse):
    """Response from memory deduplication."""

    duplicates_removed: int = Field(description="Number of duplicates removed")
    duplicate_groups: List[List[UUID]] = Field(
        description="Groups of duplicate memories"
    )


class ContextMergeRequest(BaseMemoryRequest):
    """Request for memory context merging."""

    context_ids: List[UUID] = Field(description="Context IDs to merge")
    merge_strategy: str = Field(description="Context merge strategy")


class ContextMergeResponse(BaseMemoryResponse):
    """Response from context merging."""

    merged_context_id: UUID = Field(description="ID of merged context")
    source_context_ids: List[UUID] = Field(description="IDs of source contexts")


# Memory Aggregation Models
class AggregationRequest(BaseMemoryRequest):
    """Request for memory aggregation."""

    aggregation_criteria: ModelConfiguration = Field(
        default_factory=ModelConfiguration, description="Aggregation criteria"
    )
    aggregation_type: str = Field(description="Type of aggregation")


class AggregationResponse(BaseMemoryResponse):
    """Response from memory aggregation."""

    aggregated_data: ModelConfiguration = Field(
        default_factory=ModelConfiguration, description="Aggregated memory data"
    )
    aggregation_metadata: ModelConfiguration = Field(
        default_factory=ModelConfiguration, description="Aggregation metadata"
    )


class SummarizationRequest(BaseMemoryRequest):
    """Request for memory summarization."""

    memory_cluster: List[UUID] = Field(description="Memory cluster to summarize")
    summarization_level: str = Field("standard", description="Level of summarization")


class SummarizationResponse(BaseMemoryResponse):
    """Response from memory summarization."""

    summary: str = Field(description="Generated summary")
    key_points: ModelStringList = Field(
        default_factory=ModelStringList, description="Key points from cluster"
    )


class StatisticsRequest(BaseMemoryRequest):
    """Request for memory statistics."""

    statistics_type: ModelStringList = Field(
        default_factory=ModelStringList, description="Types of statistics to generate"
    )
    time_window: Optional[Dict[str, datetime]] = Field(
        None, description="Time window for stats"
    )


class StatisticsResponse(BaseMemoryResponse):
    """Response from memory statistics."""

    statistics: ModelConfiguration = Field(
        default_factory=ModelConfiguration, description="Generated statistics"
    )
    charts_data: Optional[ModelMetadata] = Field(
        None, description="Data for visualization"
    )


# Memory Optimization Models
class LayoutOptimizationRequest(BaseMemoryRequest):
    """Request for memory layout optimization."""

    optimization_target: str = Field(
        description="Optimization target (speed/space/balance)"
    )
    memory_scope: ModelConfiguration = Field(
        default_factory=ModelConfiguration, description="Scope of optimization"
    )


class LayoutOptimizationResponse(BaseMemoryResponse):
    """Response from layout optimization."""

    optimization_results: ModelConfiguration = Field(
        default_factory=ModelConfiguration, description="Optimization results"
    )
    performance_improvement: Dict[str, float] = Field(description="Performance gains")


class CompressionRequest(BaseMemoryRequest):
    """Request for memory compression."""

    compression_algorithm: str = Field(description="Compression algorithm to use")
    quality_threshold: float = Field(0.9, description="Minimum quality threshold")


class CompressionResponse(BaseMemoryResponse):
    """Response from memory compression."""

    compression_ratio: float = Field(description="Achieved compression ratio")
    quality_retained: float = Field(description="Quality retention score")


class RetrievalOptimizationRequest(BaseMemoryRequest):
    """Request for retrieval optimization."""

    optimization_target: str = Field(description="Optimization target")
    query_patterns: ModelResultCollection = Field(
        default_factory=ModelResultCollection, description="Common query patterns"
    )


class RetrievalOptimizationResponse(BaseMemoryResponse):
    """Response from retrieval optimization."""

    optimization_applied: ModelStringList = Field(
        default_factory=ModelStringList, description="Optimizations applied"
    )
    expected_improvement: Dict[str, float] = Field(
        description="Expected performance gains"
    )


# Workflow Coordination Models
class WorkflowExecutionRequest(BaseMemoryRequest):
    """Request for workflow execution."""

    workflow_definition: ModelConfiguration = Field(
        default_factory=ModelConfiguration, description="Workflow definition"
    )
    workflow_parameters: ModelConfiguration = Field(default_factory=ModelConfiguration)


class WorkflowExecutionResponse(BaseMemoryResponse):
    """Response from workflow execution."""

    workflow_id: UUID = Field(description="Executed workflow ID")
    execution_results: ModelConfiguration = Field(
        default_factory=ModelConfiguration, description="Workflow execution results"
    )


class ParallelCoordinationRequest(BaseMemoryRequest):
    """Request for parallel operation coordination."""

    operations: ModelResultCollection = Field(
        default_factory=ModelResultCollection, description="Operations to coordinate"
    )
    coordination_strategy: str = Field(description="Coordination strategy")


class ParallelCoordinationResponse(BaseMemoryResponse):
    """Response from parallel coordination."""

    coordination_results: ModelResultCollection = Field(
        default_factory=ModelResultCollection, description="Coordination results"
    )
    execution_summary: ModelConfiguration = Field(
        default_factory=ModelConfiguration, description="Overall execution summary"
    )


class WorkflowStateRequest(BaseMemoryRequest):
    """Request for workflow state management."""

    workflow_id: UUID = Field(description="Workflow ID to manage")
    state_operation: str = Field(description="State operation (get/set/reset)")
    state_data: Optional[ModelMetadata] = Field(None, description="State data")


class WorkflowStateResponse(BaseMemoryResponse):
    """Response from workflow state management."""

    current_state: ModelConfiguration = Field(
        default_factory=ModelConfiguration, description="Current workflow state"
    )
    state_history: ModelResultCollection = Field(
        default_factory=ModelResultCollection, description="State change history"
    )


# Agent Coordination Models
class AgentCoordinationRequest(BaseMemoryRequest):
    """Request for agent coordination."""

    agent_ids: List[UUID] = Field(description="Agents to coordinate")
    coordination_task: ModelConfiguration = Field(
        default_factory=ModelConfiguration, description="Coordination task definition"
    )


class AgentCoordinationResponse(BaseMemoryResponse):
    """Response from agent coordination."""

    coordination_plan: ModelConfiguration = Field(
        default_factory=ModelConfiguration, description="Coordination execution plan"
    )
    agent_assignments: ModelMetadata = Field(
        default_factory=ModelMetadata, description="Agent task assignments"
    )


class BroadcastRequest(BaseMemoryRequest):
    """Request for memory update broadcast."""

    update_type: str = Field(description="Type of update to broadcast")
    update_data: ModelConfiguration = Field(
        default_factory=ModelConfiguration, description="Update data to broadcast"
    )
    target_agents: Optional[List[UUID]] = Field(
        None, description="Target agents (None = all)"
    )


class BroadcastResponse(BaseMemoryResponse):
    """Response from update broadcast."""

    broadcast_id: UUID = Field(description="Broadcast operation ID")
    agents_notified: List[UUID] = Field(description="Agents successfully notified")
    failed_notifications: List[UUID] = Field(
        description="Agents that failed to receive update"
    )


class StateSynchronizationRequest(BaseMemoryRequest):
    """Request for agent state synchronization."""

    agent_ids: List[UUID] = Field(description="Agents to synchronize")
    synchronization_scope: ModelConfiguration = Field(
        default_factory=ModelConfiguration, description="Scope of synchronization"
    )


class StateSynchronizationResponse(BaseMemoryResponse):
    """Response from state synchronization."""

    synchronization_results: ModelMetadata = Field(
        default_factory=ModelMetadata, description="Sync results per agent"
    )
    conflicts_resolved: ModelResultCollection = Field(
        default_factory=ModelResultCollection,
        description="Conflicts that were resolved",
    )


# Memory Orchestration Models
class LifecycleOrchestrationRequest(BaseMemoryRequest):
    """Request for memory lifecycle orchestration."""

    lifecycle_stage: str = Field(description="Lifecycle stage to orchestrate")
    memory_scope: ModelConfiguration = Field(
        default_factory=ModelConfiguration,
        description="Scope of memories to orchestrate",
    )


class LifecycleOrchestrationResponse(BaseMemoryResponse):
    """Response from lifecycle orchestration."""

    orchestration_plan: ModelConfiguration = Field(
        default_factory=ModelConfiguration, description="Lifecycle orchestration plan"
    )
    affected_memories: List[UUID] = Field(
        description="Memories affected by orchestration"
    )


class QuotaManagementRequest(BaseMemoryRequest):
    """Request for quota management."""

    quota_type: str = Field(description="Type of quota to manage")
    quota_parameters: ModelConfiguration = Field(
        default_factory=ModelConfiguration, description="Quota parameters"
    )


class QuotaManagementResponse(BaseMemoryResponse):
    """Response from quota management."""

    current_quotas: ModelConfiguration = Field(
        default_factory=ModelConfiguration, description="Current quota status"
    )
    quota_adjustments: ModelConfiguration = Field(
        default_factory=ModelConfiguration, description="Applied quota adjustments"
    )


class MigrationCoordinationRequest(BaseMemoryRequest):
    """Request for memory migration coordination."""

    migration_plan: ModelConfiguration = Field(
        default_factory=ModelConfiguration, description="Migration plan"
    )
    source_storage: str = Field(description="Source storage system")
    target_storage: str = Field(description="Target storage system")


class MigrationCoordinationResponse(BaseMemoryResponse):
    """Response from migration coordination."""

    migration_id: UUID = Field(description="Migration operation ID")
    migration_status: ModelConfiguration = Field(
        default_factory=ModelConfiguration, description="Migration status and progress"
    )
