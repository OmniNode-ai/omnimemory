"""
Migration strategy enum for OmniMemory following ONEX standards.
"""

from enum import Enum


class EnumMigrationStrategy(str, Enum):
    """Migration strategies for memory operations."""

    INCREMENTAL = "incremental"
    BULK = "bulk"
    INTELLIGENT = "intelligent"
    STREAMING = "streaming"
    PARALLEL = "parallel"
    SEQUENTIAL = "sequential"
    ADAPTIVE = "adaptive"
