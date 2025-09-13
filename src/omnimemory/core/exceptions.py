"""
Container-specific exception classes for OmniMemory.

This module provides exception classes specific to container and service
resolution operations, extending the base OmniMemory error handling.
"""

from typing import Any, Dict, List, Optional
from uuid import UUID

from ..protocols.error_models import OmniMemoryError, OmniMemoryErrorCode


class ContainerError(OmniMemoryError):
    """Base exception for container-related errors."""
    
    def __init__(
        self,
        message: str,
        container_name: Optional[str] = None,
        **kwargs,
    ):
        context = kwargs.get("context", {})
        if container_name:
            context["container_name"] = container_name
        
        kwargs["context"] = context
        super().__init__(
            error_code=OmniMemoryErrorCode.INTERNAL_ERROR,
            message=message,
            **kwargs,
        )


class ServiceResolutionError(ContainerError):
    """Exception for service resolution failures."""
    
    def __init__(
        self,
        protocol_name: str,
        service_name: Optional[str] = None,
        available_services: Optional[List[str]] = None,
        **kwargs,
    ):
        message = f"Failed to resolve service for protocol: {protocol_name}"
        if service_name:
            message += f" (service: {service_name})"
        
        context = kwargs.get("context", {})
        context.update({
            "protocol_name": protocol_name,
            "service_name": service_name,
            "available_services": available_services or [],
        })
        
        kwargs["context"] = context
        kwargs["recovery_hint"] = "Check service registration and container configuration"
        
        super().__init__(
            message=message,
            **kwargs,
        )


class ServiceRegistrationError(ContainerError):
    """Exception for service registration failures."""
    
    def __init__(
        self,
        protocol_name: str,
        service_class: Optional[str] = None,
        **kwargs,
    ):
        message = f"Failed to register service for protocol: {protocol_name}"
        if service_class:
            message += f" (class: {service_class})"
        
        context = kwargs.get("context", {})
        context.update({
            "protocol_name": protocol_name,
            "service_class": service_class,
        })
        
        kwargs["context"] = context
        kwargs["recovery_hint"] = "Verify service implementation and dependencies"
        
        super().__init__(
            message=message,
            **kwargs,
        )