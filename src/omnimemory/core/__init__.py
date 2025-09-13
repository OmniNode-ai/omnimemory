"""
OmniMemory Core Module

This module provides the foundational components for the OmniMemory system,
including the ONEX-compliant container, service providers, and base classes
that implement the 4-node architecture patterns.
"""

from .container import (
    OmniMemoryContainer,
    create_omnimemory_container,
    get_omnimemory_container,
    get_omnimemory_container_sync,
)

from .service_providers import (
    OmniMemoryServiceProvider,
    MemoryServiceRegistry,
)

from .base_implementations import (
    BaseMemoryService,
    BaseEffectService,
    BaseComputeService,
    BaseReducerService,
    BaseOrchestratorService,
)

from .exceptions import (
    ContainerError,
    ServiceResolutionError,
    ServiceRegistrationError,
)

__all__ = [
    # Container
    "OmniMemoryContainer",
    "create_omnimemory_container",
    "get_omnimemory_container",
    "get_omnimemory_container_sync",
    
    # Service providers
    "OmniMemoryServiceProvider",
    "MemoryServiceRegistry",
    
    # Base implementations
    "BaseMemoryService",
    "BaseEffectService", 
    "BaseComputeService",
    "BaseReducerService",
    "BaseOrchestratorService",
    
    # Exceptions
    "ContainerError",
    "ServiceResolutionError",
    "ServiceRegistrationError",
]