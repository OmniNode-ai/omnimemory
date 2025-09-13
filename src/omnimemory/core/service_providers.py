"""
Service Providers and Registry for OmniMemory

This module provides service provider classes and a registry system for
managing service instances and their lifecycle within the ONEX container.
"""

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional, Type, TypeVar
from uuid import UUID, uuid4

from omnibase_core.core.monadic.model_node_result import NodeResult
from omnibase_spi import ProtocolLogger

from ..protocols import (
    ProtocolMemoryBase,
    OmniMemoryError,
    OmniMemoryErrorCode,
    SystemError,
)
from .exceptions import ServiceRegistrationError, ServiceResolutionError

T = TypeVar("T", bound=ProtocolMemoryBase)


class ServiceDescriptor:
    """Descriptor for a registered service."""
    
    def __init__(
        self,
        service_class: Type[T],
        protocol_type: Type[ProtocolMemoryBase],
        service_name: str,
        singleton: bool = True,
        dependencies: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize service descriptor.
        
        Args:
            service_class: Implementation class
            protocol_type: Protocol interface
            service_name: Unique service name
            singleton: Whether to create singleton instance
            dependencies: List of dependency service names
            config: Service-specific configuration
        """
        self.service_class = service_class
        self.protocol_type = protocol_type
        self.service_name = service_name
        self.singleton = singleton
        self.dependencies = dependencies or []
        self.config = config or {}
        self.registration_time = datetime.now()
        self.instance: Optional[T] = None
        self.initialization_count = 0
        self.last_access_time: Optional[datetime] = None


class OmniMemoryServiceProvider:
    """
    Advanced service provider with lifecycle management.
    
    Provides service instantiation, dependency injection, lifecycle management,
    and health monitoring for all OmniMemory services.
    """
    
    def __init__(
        self,
        logger: ProtocolLogger,
        provider_id: Optional[str] = None,
    ):
        """
        Initialize service provider.
        
        Args:
            logger: Logger instance
            provider_id: Optional provider identifier
        """
        self.provider_id = provider_id or str(uuid4())
        self.logger = logger
        self.descriptors: Dict[str, ServiceDescriptor] = {}
        self.instances: Dict[str, Any] = {}
        self.resolution_order: List[str] = []
        self.initialization_lock = asyncio.Lock()
        self.start_time = datetime.now()
    
    async def register_service(
        self,
        protocol_type: Type[ProtocolMemoryBase],
        service_class: Type[T],
        service_name: Optional[str] = None,
        singleton: bool = True,
        dependencies: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> NodeResult[bool]:
        """
        Register a service implementation.
        
        Args:
            protocol_type: Protocol interface
            service_class: Implementation class
            service_name: Optional service name (defaults to protocol name)
            singleton: Whether to create singleton instance
            dependencies: List of dependency service names
            config: Service-specific configuration
            
        Returns:
            NodeResult indicating registration success/failure
        """
        try:
            service_name = service_name or protocol_type.__name__
            
            # Validate service class implements protocol
            if not issubclass(service_class, protocol_type):
                raise ServiceRegistrationError(
                    protocol_name=protocol_type.__name__,
                    service_class=service_class.__name__,
                    context={
                        "error": "Service class does not implement protocol",
                        "protocol": protocol_type.__name__,
                        "service_class": service_class.__name__,
                    },
                )
            
            # Check for duplicate registration
            if service_name in self.descriptors:
                await self.logger.emit_log_event_async(
                    level="WARNING",
                    message=f"Overwriting existing service registration: {service_name}",
                    event_type="service_registration_overwrite",
                    service_name=service_name,
                    provider_id=self.provider_id,
                )
            
            # Create service descriptor
            descriptor = ServiceDescriptor(
                service_class=service_class,
                protocol_type=protocol_type,
                service_name=service_name,
                singleton=singleton,
                dependencies=dependencies,
                config=config,
            )
            
            self.descriptors[service_name] = descriptor
            
            await self.logger.emit_log_event_async(
                level="INFO",
                message=f"Registered service: {service_name}",
                event_type="service_registered",
                service_name=service_name,
                protocol_type=protocol_type.__name__,
                service_class=service_class.__name__,
                singleton=singleton,
                provider_id=self.provider_id,
            )
            
            return NodeResult.success(
                value=True,
                provenance=[f"service_provider.register.{service_name}"],
                trust_score=1.0,
                metadata={
                    "service_name": service_name,
                    "protocol_type": protocol_type.__name__,
                    "provider_id": self.provider_id,
                },
            )
        
        except Exception as e:
            return NodeResult.failure(
                error=ServiceRegistrationError(
                    protocol_name=protocol_type.__name__ if protocol_type else "unknown",
                    service_class=service_class.__name__ if service_class else "unknown",
                    context={
                        "provider_id": self.provider_id,
                        "error_details": str(e),
                    },
                ),
                provenance=[f"service_provider.register.failed"],
            )
    
    async def resolve_service(
        self,
        protocol_type: Type[T],
        service_name: Optional[str] = None,
        correlation_id: Optional[UUID] = None,
    ) -> NodeResult[T]:
        """
        Resolve a service instance.
        
        Args:
            protocol_type: Protocol interface to resolve
            service_name: Optional specific service name
            correlation_id: Request correlation ID
            
        Returns:
            NodeResult with resolved service instance
        """
        start_time = datetime.now()
        service_key = service_name or protocol_type.__name__
        
        try:
            async with self.initialization_lock:
                # Check if descriptor exists
                if service_key not in self.descriptors:
                    available_services = list(self.descriptors.keys())
                    raise ServiceResolutionError(
                        protocol_name=protocol_type.__name__,
                        service_name=service_name,
                        available_services=available_services,
                        context={
                            "provider_id": self.provider_id,
                            "correlation_id": str(correlation_id) if correlation_id else None,
                        },
                    )
                
                descriptor = self.descriptors[service_key]
                
                # Check for singleton instance
                if descriptor.singleton and descriptor.instance is not None:
                    descriptor.last_access_time = datetime.now()
                    
                    end_time = datetime.now()
                    resolution_time = (end_time - start_time).total_seconds() * 1000
                    
                    return NodeResult.success(
                        value=descriptor.instance,
                        provenance=[f"service_provider.resolve.cached.{service_key}"],
                        trust_score=0.95,  # Cached instances have slightly lower trust
                        metadata={
                            "service_name": service_key,
                            "protocol_type": protocol_type.__name__,
                            "provider_id": self.provider_id,
                            "resolution_time_ms": resolution_time,
                            "cache_hit": True,
                        },
                        correlation_id=str(correlation_id) if correlation_id else None,
                    )
                
                # Resolve dependencies first
                dependencies = await self._resolve_dependencies(
                    descriptor.dependencies,
                    correlation_id,
                )
                
                # Create service instance
                instance = await self._create_service_instance(
                    descriptor,
                    dependencies,
                    correlation_id,
                )
                
                # Cache singleton instance
                if descriptor.singleton:
                    descriptor.instance = instance
                
                descriptor.initialization_count += 1
                descriptor.last_access_time = datetime.now()
                
                end_time = datetime.now()
                resolution_time = (end_time - start_time).total_seconds() * 1000
                
                await self.logger.emit_log_event_async(
                    level="DEBUG",
                    message=f"Resolved service: {service_key}",
                    event_type="service_resolved",
                    service_name=service_key,
                    protocol_type=protocol_type.__name__,
                    resolution_time_ms=resolution_time,
                    provider_id=self.provider_id,
                )
                
                return NodeResult.success(
                    value=instance,
                    provenance=[f"service_provider.resolve.{service_key}"],
                    trust_score=1.0,
                    metadata={
                        "service_name": service_key,
                        "protocol_type": protocol_type.__name__,
                        "provider_id": self.provider_id,
                        "resolution_time_ms": resolution_time,
                        "cache_hit": False,
                        "initialization_count": descriptor.initialization_count,
                    },
                    correlation_id=str(correlation_id) if correlation_id else None,
                )
        
        except Exception as e:
            end_time = datetime.now()
            resolution_time = (end_time - start_time).total_seconds() * 1000
            
            await self.logger.emit_log_event_async(
                level="ERROR",
                message=f"Service resolution failed: {service_key}",
                event_type="service_resolution_failed",
                service_name=service_key,
                protocol_type=protocol_type.__name__,
                error=str(e),
                resolution_time_ms=resolution_time,
                provider_id=self.provider_id,
            )
            
            if isinstance(e, OmniMemoryError):
                return NodeResult.failure(
                    error=e,
                    provenance=[f"service_provider.resolve.failed.{service_key}"],
                    correlation_id=str(correlation_id) if correlation_id else None,
                )
            else:
                return NodeResult.failure(
                    error=ServiceResolutionError(
                        protocol_name=protocol_type.__name__,
                        service_name=service_name,
                        context={
                            "provider_id": self.provider_id,
                            "error_details": str(e),
                            "resolution_time_ms": resolution_time,
                        },
                    ),
                    provenance=[f"service_provider.resolve.failed.{service_key}"],
                    correlation_id=str(correlation_id) if correlation_id else None,
                )
    
    async def _resolve_dependencies(
        self,
        dependencies: List[str],
        correlation_id: Optional[UUID],
    ) -> Dict[str, Any]:
        """Resolve service dependencies."""
        resolved_dependencies = {}
        
        for dep_name in dependencies:
            if dep_name not in self.descriptors:
                raise ServiceResolutionError(
                    protocol_name="dependency",
                    service_name=dep_name,
                    context={
                        "provider_id": self.provider_id,
                        "missing_dependency": dep_name,
                    },
                )
            
            dep_descriptor = self.descriptors[dep_name]
            dep_result = await self.resolve_service(
                dep_descriptor.protocol_type,
                dep_name,
                correlation_id,
            )
            
            if dep_result.is_failure:
                raise ServiceResolutionError(
                    protocol_name="dependency",
                    service_name=dep_name,
                    context={
                        "provider_id": self.provider_id,
                        "dependency_error": dep_result.error.message,
                    },
                )
            
            resolved_dependencies[dep_name] = dep_result.value
        
        return resolved_dependencies
    
    async def _create_service_instance(
        self,
        descriptor: ServiceDescriptor,
        dependencies: Dict[str, Any],
        correlation_id: Optional[UUID],
    ) -> Any:
        """Create a service instance with dependency injection."""
        try:
            # Prepare constructor arguments
            constructor_args = {
                "service_name": descriptor.service_name,
                "logger": self.logger,
                "config": descriptor.config,
                **dependencies,
            }
            
            # Filter arguments based on constructor signature
            import inspect
            sig = inspect.signature(descriptor.service_class.__init__)
            filtered_args = {}
            
            for param_name in sig.parameters:
                if param_name in constructor_args:
                    filtered_args[param_name] = constructor_args[param_name]
            
            # Create instance
            instance = descriptor.service_class(**filtered_args)
            
            # Initialize if it has an initialization method
            if hasattr(instance, "initialize"):
                await instance.initialize()
            
            return instance
        
        except Exception as e:
            raise ServiceResolutionError(
                protocol_name=descriptor.protocol_type.__name__,
                service_name=descriptor.service_name,
                context={
                    "provider_id": self.provider_id,
                    "service_class": descriptor.service_class.__name__,
                    "initialization_error": str(e),
                },
            )
    
    async def get_service_health(self) -> Dict[str, Any]:
        """Get health status of all registered services."""
        service_health = {}
        
        for service_name, descriptor in self.descriptors.items():
            if descriptor.instance is not None:
                try:
                    health_result = await descriptor.instance.health_check()
                    service_health[service_name] = {
                        "status": "healthy" if health_result.is_success else "unhealthy",
                        "last_access": descriptor.last_access_time.isoformat() if descriptor.last_access_time else None,
                        "initialization_count": descriptor.initialization_count,
                    }
                except Exception as e:
                    service_health[service_name] = {
                        "status": "error",
                        "error": str(e),
                    }
            else:
                service_health[service_name] = {
                    "status": "not_initialized",
                    "registration_time": descriptor.registration_time.isoformat(),
                }
        
        return {
            "provider_id": self.provider_id,
            "uptime_seconds": (datetime.now() - self.start_time).total_seconds(),
            "registered_services": len(self.descriptors),
            "initialized_services": len([d for d in self.descriptors.values() if d.instance is not None]),
            "services": service_health,
        }
    
    def list_services(self) -> List[Dict[str, Any]]:
        """List all registered services."""
        services = []
        
        for service_name, descriptor in self.descriptors.items():
            services.append({
                "service_name": service_name,
                "protocol_type": descriptor.protocol_type.__name__,
                "service_class": descriptor.service_class.__name__,
                "singleton": descriptor.singleton,
                "dependencies": descriptor.dependencies,
                "registration_time": descriptor.registration_time.isoformat(),
                "initialization_count": descriptor.initialization_count,
                "last_access_time": descriptor.last_access_time.isoformat() if descriptor.last_access_time else None,
                "is_initialized": descriptor.instance is not None,
            })
        
        return services


class MemoryServiceRegistry:
    """
    Global registry for OmniMemory services.
    
    Provides a centralized registry for service providers and facilitates
    service discovery across the entire OmniMemory system.
    """
    
    def __init__(self):
        """Initialize service registry."""
        self.providers: Dict[str, OmniMemoryServiceProvider] = {}
        self.default_provider: Optional[OmniMemoryServiceProvider] = None
        self.registry_id = str(uuid4())
        self.start_time = datetime.now()
    
    def register_provider(
        self,
        provider: OmniMemoryServiceProvider,
        provider_name: Optional[str] = None,
        make_default: bool = False,
    ) -> None:
        """
        Register a service provider.
        
        Args:
            provider: Service provider instance
            provider_name: Optional provider name
            make_default: Whether to make this the default provider
        """
        provider_name = provider_name or provider.provider_id
        self.providers[provider_name] = provider
        
        if make_default or self.default_provider is None:
            self.default_provider = provider
    
    def get_provider(
        self,
        provider_name: Optional[str] = None,
    ) -> Optional[OmniMemoryServiceProvider]:
        """
        Get a service provider.
        
        Args:
            provider_name: Optional provider name (uses default if not specified)
            
        Returns:
            Service provider instance or None
        """
        if provider_name:
            return self.providers.get(provider_name)
        else:
            return self.default_provider
    
    async def resolve_service(
        self,
        protocol_type: Type[T],
        service_name: Optional[str] = None,
        provider_name: Optional[str] = None,
        correlation_id: Optional[UUID] = None,
    ) -> NodeResult[T]:
        """
        Resolve a service from any registered provider.
        
        Args:
            protocol_type: Protocol interface to resolve
            service_name: Optional service name
            provider_name: Optional provider name
            correlation_id: Request correlation ID
            
        Returns:
            NodeResult with resolved service instance
        """
        provider = self.get_provider(provider_name)
        
        if provider is None:
            return NodeResult.failure(
                error=ServiceResolutionError(
                    protocol_name=protocol_type.__name__,
                    service_name=service_name,
                    context={
                        "registry_id": self.registry_id,
                        "provider_name": provider_name,
                        "available_providers": list(self.providers.keys()),
                    },
                ),
                provenance=["registry.resolve.no_provider"],
                correlation_id=str(correlation_id) if correlation_id else None,
            )
        
        return await provider.resolve_service(
            protocol_type,
            service_name,
            correlation_id,
        )
    
    def get_registry_status(self) -> Dict[str, Any]:
        """Get status of the service registry."""
        return {
            "registry_id": self.registry_id,
            "uptime_seconds": (datetime.now() - self.start_time).total_seconds(),
            "provider_count": len(self.providers),
            "default_provider": self.default_provider.provider_id if self.default_provider else None,
            "providers": list(self.providers.keys()),
        }


# Global registry instance
_global_registry: Optional[MemoryServiceRegistry] = None


def get_service_registry() -> MemoryServiceRegistry:
    """Get the global service registry instance."""
    global _global_registry
    if _global_registry is None:
        _global_registry = MemoryServiceRegistry()
    return _global_registry