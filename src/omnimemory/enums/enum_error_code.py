"""
Memory-specific error codes following ONEX standards.

This module ONLY contains error codes specific to OmniMemory operations.
General error codes come from omnibase_core.core.errors.core_errors when available.
"""

from __future__ import annotations

import re

try:
    from omnibase_core.core.errors.core_errors import (  # pyright: ignore[reportMissingImports]
        OnexErrorCode,
    )

    _base_class = OnexErrorCode
except ImportError:
    # Fallback for development environments without omnibase_core
    from enum import Enum

    # NOTE(OMN-0001): mypy no-redef due to fallback class shadowing.
    # This class redefines OnexErrorCode which is conditionally imported from
    # omnibase_core above. When omnibase_core is not available, we provide this
    # fallback implementation.
    class OnexErrorCode(str, Enum):  # type: ignore[no-redef]
        """Base class for ONEX error codes (fallback implementation)."""

        def get_component(self) -> str:
            """Get the component identifier for this error code."""
            raise NotImplementedError("Subclasses must implement get_component()")

        def get_number(self) -> int:
            """Get the numeric identifier for this error code."""
            raise NotImplementedError("Subclasses must implement get_number()")

        def get_description(self) -> str:
            """Get a human-readable description for this error code."""
            raise NotImplementedError("Subclasses must implement get_description()")

        def get_exit_code(self) -> int:
            """Get the appropriate CLI exit code for this error."""
            return 1  # Default to error exit code

    _base_class = OnexErrorCode


class EnumOmniMemoryErrorCode(_base_class):  # type: ignore[valid-type,misc]
    """Memory-specific error codes for the ONEX memory system."""

    # Memory operation errors (specific to omnimemory only)
    MEMORY_STORAGE_FAILED = "ONEX_OMNIMEMORY_001_MEMORY_STORAGE_FAILED"
    MEMORY_RETRIEVAL_FAILED = "ONEX_OMNIMEMORY_002_MEMORY_RETRIEVAL_FAILED"
    MEMORY_UPDATE_FAILED = "ONEX_OMNIMEMORY_003_MEMORY_UPDATE_FAILED"
    MEMORY_DELETE_FAILED = "ONEX_OMNIMEMORY_004_MEMORY_DELETE_FAILED"
    MEMORY_CONSOLIDATION_FAILED = "ONEX_OMNIMEMORY_005_MEMORY_CONSOLIDATION_FAILED"
    MEMORY_OPTIMIZATION_FAILED = "ONEX_OMNIMEMORY_006_MEMORY_OPTIMIZATION_FAILED"
    MEMORY_MIGRATION_FAILED = "ONEX_OMNIMEMORY_007_MEMORY_MIGRATION_FAILED"

    # Intelligence operation errors (specific to memory intelligence)
    MEMORY_ANALYSIS_FAILED = "ONEX_OMNIMEMORY_008_MEMORY_ANALYSIS_FAILED"
    MEMORY_PATTERN_RECOGNITION_FAILED = (
        "ONEX_OMNIMEMORY_009_MEMORY_PATTERN_RECOGNITION_FAILED"
    )
    MEMORY_SEMANTIC_PROCESSING_FAILED = (
        "ONEX_OMNIMEMORY_010_MEMORY_SEMANTIC_PROCESSING_FAILED"
    )
    MEMORY_EMBEDDING_GENERATION_FAILED = (
        "ONEX_OMNIMEMORY_011_MEMORY_EMBEDDING_GENERATION_FAILED"
    )

    # Memory storage specific errors
    VECTOR_INDEX_CORRUPTION = "ONEX_OMNIMEMORY_012_VECTOR_INDEX_CORRUPTION"
    MEMORY_QUOTA_EXCEEDED = "ONEX_OMNIMEMORY_013_MEMORY_QUOTA_EXCEEDED"
    TEMPORAL_MEMORY_EXPIRED = "ONEX_OMNIMEMORY_014_TEMPORAL_MEMORY_EXPIRED"
    MEMORY_DEPENDENCY_CYCLE = "ONEX_OMNIMEMORY_015_MEMORY_DEPENDENCY_CYCLE"
    MEMORY_VERSION_CONFLICT = "ONEX_OMNIMEMORY_016_MEMORY_VERSION_CONFLICT"

    def get_component(self) -> str:
        """Get the component identifier for this error code."""
        return "OMNIMEMORY"

    def get_number(self) -> int:
        """Get the numeric identifier for this error code."""
        match = re.search(r"ONEX_OMNIMEMORY_(\d+)_", self.value)
        return int(match.group(1)) if match else 0

    def get_description(self) -> str:
        """Get a human-readable description for this error code."""
        descriptions = {
            self.MEMORY_STORAGE_FAILED: "Failed to store memory data",
            self.MEMORY_RETRIEVAL_FAILED: "Failed to retrieve memory data",
            self.MEMORY_UPDATE_FAILED: "Failed to update existing memory",
            self.MEMORY_DELETE_FAILED: "Failed to delete memory data",
            self.MEMORY_CONSOLIDATION_FAILED: "Failed to consolidate memories",
            self.MEMORY_OPTIMIZATION_FAILED: "Failed to optimize memory storage",
            self.MEMORY_MIGRATION_FAILED: "Failed to migrate legacy memory data",
            self.MEMORY_ANALYSIS_FAILED: "Failed to analyze memory content",
            self.MEMORY_PATTERN_RECOGNITION_FAILED: "Pattern recognition failed",
            self.MEMORY_SEMANTIC_PROCESSING_FAILED: "Semantic processing failed",
            self.MEMORY_EMBEDDING_GENERATION_FAILED: "Embedding generation failed",
            self.VECTOR_INDEX_CORRUPTION: "Vector index is corrupted or invalid",
            self.MEMORY_QUOTA_EXCEEDED: "Memory storage quota exceeded",
            self.TEMPORAL_MEMORY_EXPIRED: "Temporal memory has expired",
            self.MEMORY_DEPENDENCY_CYCLE: "Circular dependency in memory",
            self.MEMORY_VERSION_CONFLICT: "Version conflict in memory data",
        }
        return descriptions.get(self, "Unknown OmniMemory error")
