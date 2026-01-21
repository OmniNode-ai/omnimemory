# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Base node classes for ONEX 4-Node Architecture.

This module provides the foundational base classes for all ONEX nodes
in omnimemory. Following ONEX patterns, the base class is minimal and
only provides container injection - all business logic lives in handlers.

Node Types:
- EFFECT: External I/O operations (storage, APIs, file system)
- COMPUTE: Pure transformations and algorithms
- REDUCER: Aggregation and state management
- ORCHESTRATOR: Workflow coordination and routing

Container Support:
- Uses local ModelOnexContainer from compat layer
- Will seamlessly support omnibase_core containers when available
- Auto-injection supported via container.resolve()
"""

from __future__ import annotations

from abc import ABC

# Import local compat container first (always available)
from ..compat import ModelOnexContainer

# Type alias for container - supports both local and omnibase_core containers
# At runtime, we use ModelOnexContainer from the compat layer
# When omnibase_core is available, this can be updated to support both
ContainerType = ModelOnexContainer


class BaseNode(ABC):
    """Abstract base class for all ONEX nodes.

    This base class provides the minimal foundation for ONEX nodes,
    following the declarative pattern where:
    - Node classes are purely declarative (only __init__)
    - Business logic is defined in handlers
    - Contracts specify inputs, outputs, and behavior

    Attributes:
        _container: The ONEX node container for dependency injection
                   and handler registration. Supports both local
                   ModelOnexContainer and future omnibase_core containers.

    Example:
        ```python
        class NodeMyEffect(BaseNode):
            def __init__(self, container: ContainerType) -> None:
                super().__init__(container)
        ```
    """

    def __init__(self, container: ContainerType) -> None:
        """Initialize the base node with container injection.

        Args:
            container: ONEX node container providing dependency injection
                      and handler registration capabilities. Can be either
                      ModelOnexContainer (local) or NodeContainer (omnibase_core).
        """
        self._container = container

    @property
    def container(self) -> ContainerType:
        """Access the container for dependency resolution.

        Returns:
            The ONEX container for this node.
        """
        return self._container


class BaseEffectNode(BaseNode):
    """Base class for EFFECT nodes - external I/O operations.

    EFFECT nodes handle all external interactions:
    - Database CRUD operations
    - API calls
    - File system operations
    - Message queue operations
    """


class BaseComputeNode(BaseNode):
    """Base class for COMPUTE nodes - pure transformations.

    COMPUTE nodes perform:
    - Data transformations
    - Algorithm execution
    - Semantic analysis
    - Vector calculations
    """


class BaseReducerNode(BaseNode):
    """Base class for REDUCER nodes - aggregation and state.

    REDUCER nodes handle:
    - Data aggregation
    - State consolidation
    - Statistics generation
    - Memory merging
    """


class BaseOrchestratorNode(BaseNode):
    """Base class for ORCHESTRATOR nodes - workflow coordination.

    ORCHESTRATOR nodes manage:
    - Workflow orchestration
    - Cross-node coordination
    - Routing decisions
    - Lifecycle management
    """


__all__ = [
    "BaseComputeNode",
    "BaseEffectNode",
    "BaseNode",
    "BaseOrchestratorNode",
    "BaseReducerNode",
    "ContainerType",
]
