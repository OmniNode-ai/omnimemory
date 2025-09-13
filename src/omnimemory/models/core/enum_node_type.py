"""
Enum for ONEX node types in memory architecture.
"""

from enum import Enum


class EnumNodeType(str, Enum):
    """ONEX 4-node architecture types for memory operations."""

    EFFECT = "effect"
    COMPUTE = "compute"
    REDUCER = "reducer"
    ORCHESTRATOR = "orchestrator"