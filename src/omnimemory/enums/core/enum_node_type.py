"""
ONEX node type enumeration.
"""

from enum import Enum


class EnumNodeType(Enum):
    """ONEX node types."""

    EFFECT = "effect"
    COMPUTE = "compute"
    REDUCER = "reducer"
    ORCHESTRATOR = "orchestrator"


# Alias for compatibility
NodeType = EnumNodeType
