"""
Core domain enums for OmniMemory following ONEX standards.

This module provides core operation enums used throughout the system.
"""

from .enum_node_type import EnumNodeType
from .enum_operation_status import EnumOperationStatus

__all__ = [
    "EnumOperationStatus",
    "EnumNodeType",
]
