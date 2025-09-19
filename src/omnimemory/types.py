"""
ONEX standardized type definitions for OmniMemory.

Provides consistent type aliases and generic types following ONEX standards
to reduce Union usage and improve type safety.
"""

from pathlib import Path
from typing import Dict, List, TypeVar, Union

# Primitive value types for configuration and data storage
PrimitiveValue = Union[str, int, float, bool]
"""Type alias for primitive configuration values."""

# Configuration dictionary type
ConfigDict = Dict[str, PrimitiveValue]
"""Type alias for configuration dictionaries with primitive values."""

# File path type for flexibility
FilePath = Union[str, Path]
"""Type alias for file path parameters that accept strings or Path objects."""

# Generic type variables for container types
T = TypeVar("T")
"""Generic type variable for container elements."""

K = TypeVar("K")
"""Generic type variable for dictionary keys."""

V = TypeVar("V")
"""Generic type variable for dictionary values."""

# Memory data value types (for cache and storage)
MemoryValue = Union[
    str, int, float, bool, Dict[str, PrimitiveValue], List[PrimitiveValue]
]
"""Type alias for memory storage values with consistent primitive types."""

# Nested memory values for complex data structures
NestedMemoryValue = Union[
    str,
    int,
    float,
    bool,
    Dict[str, Union[PrimitiveValue, "NestedMemoryValue"]],
    List[Union[PrimitiveValue, "NestedMemoryValue"]],
]
"""Type alias for complex nested memory values allowing recursive structures."""
