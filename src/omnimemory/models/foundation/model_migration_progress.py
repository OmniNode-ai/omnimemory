"""
Migration progress tracking model for OmniMemory ONEX architecture.

This module provides models for tracking migration progress across the system:
- Progress tracking with detailed metrics
- Status monitoring and error tracking
- Estimated completion time calculation
- Batch processing support

NOTICE: This module has been refactored to follow one-model-per-file standards.
Individual models are now in separate files but re-exported here for backwards compatibility.
"""

from .model_batch_processing_metrics import ModelBatchProcessingMetrics

# Re-export individual models for backwards compatibility
from .model_file_info import ModelFileInfo
from .model_file_processing_info import ModelFileProcessingInfo
from .model_migration_progress_metrics import ModelMigrationProgressMetrics
from .model_migration_progress_tracker import ModelMigrationProgressTracker

__all__ = [
    "ModelFileInfo",
    "ModelBatchProcessingMetrics",
    "ModelFileProcessingInfo",
    "ModelMigrationProgressMetrics",
    "ModelMigrationProgressTracker",
]
