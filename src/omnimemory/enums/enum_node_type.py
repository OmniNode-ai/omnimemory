"""
Node type enumeration following ONEX standards.
"""

from enum import Enum


class EnumNodeType(str, Enum):
    """ONEX node types for the 4-node architecture."""

    EFFECT = "effect"
    COMPUTE = "compute"
    REDUCER = "reducer"
    ORCHESTRATOR = "orchestrator"
