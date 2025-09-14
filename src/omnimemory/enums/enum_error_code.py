"""
Memory-specific error codes following ONEX standards.
General error codes should be imported from omnibase_core.
"""

from enum import Enum


class EnumErrorCode(str, Enum):
    """Memory-specific error codes for the ONEX memory system."""

    # Memory operation errors (specific to omnimemory)
    MEMORY_STORAGE_FAILED = "memory_storage_failed"
    MEMORY_RETRIEVAL_FAILED = "memory_retrieval_failed"
    MEMORY_UPDATE_FAILED = "memory_update_failed"
    MEMORY_DELETE_FAILED = "memory_delete_failed"
    MEMORY_CONSOLIDATION_FAILED = "memory_consolidation_failed"
    MEMORY_OPTIMIZATION_FAILED = "memory_optimization_failed"
    MEMORY_MIGRATION_FAILED = "memory_migration_failed"

    # Intelligence operation errors (specific to memory intelligence)
    MEMORY_ANALYSIS_FAILED = "memory_analysis_failed"
    MEMORY_PATTERN_RECOGNITION_FAILED = "memory_pattern_recognition_failed"
    MEMORY_SEMANTIC_PROCESSING_FAILED = "memory_semantic_processing_failed"
    MEMORY_EMBEDDING_GENERATION_FAILED = "memory_embedding_generation_failed"

    # Memory storage specific errors
    VECTOR_INDEX_CORRUPTION = "vector_index_corruption"
    MEMORY_QUOTA_EXCEEDED = "memory_quota_exceeded"
    TEMPORAL_MEMORY_EXPIRED = "temporal_memory_expired"
    MEMORY_DEPENDENCY_CYCLE = "memory_dependency_cycle"
    MEMORY_VERSION_CONFLICT = "memory_version_conflict"