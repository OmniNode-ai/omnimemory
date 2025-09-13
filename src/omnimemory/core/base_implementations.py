"""
Base Implementation Classes for OmniMemory Services

This module provides abstract base classes that implement common functionality
for all OmniMemory services, following ONEX 4-node architecture patterns and
providing monadic error handling with NodeResult composition.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from omnibase_core.core.monadic.model_node_result import NodeResult
from omnibase_spi import ProtocolLogger

from ..protocols import (
    BaseMemoryRequest,
    BaseMemoryResponse,
    OperationStatus,
    ProtocolMemoryBase,
    OmniMemoryError,
    OmniMemoryErrorCode,
    SystemError,
)


class BaseMemoryService(ProtocolMemoryBase, ABC):
    """
    Abstract base class for all memory services.
    
    Provides common functionality including health checking, metrics collection,
    configuration management, and request validation that all memory services need.
    """
    
    def __init__(
        self,
        service_name: str,
        logger: ProtocolLogger,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize base memory service.
        
        Args:
            service_name: Name of the service for identification
            logger: Logger instance for service logging
            config: Optional service configuration
        """
        self.service_name = service_name
        self.logger = logger
        self.config = config or {}
        self.service_id = str(uuid4())
        self.start_time = datetime.now()
        self.operation_count = 0
        self.error_count = 0
        self.total_execution_time_ms = 0
    
    async def health_check(
        self,
        correlation_id: Optional[UUID] = None,
    ) -> NodeResult[Dict[str, Any]]:
        """
        Check the health status of the memory service.
        
        Returns:
            NodeResult containing health status information
        """
        try:
            uptime_seconds = (datetime.now() - self.start_time).total_seconds()
            avg_execution_time = (
                self.total_execution_time_ms / self.operation_count
                if self.operation_count > 0 else 0
            )
            error_rate = (
                self.error_count / self.operation_count
                if self.operation_count > 0 else 0
            )
            
            # Perform service-specific health checks
            service_health = await self._check_service_health()
            
            health_data = {
                "service_name": self.service_name,
                "service_id": self.service_id,
                "status": "healthy" if service_health else "unhealthy",
                "uptime_seconds": uptime_seconds,
                "operation_count": self.operation_count,
                "error_count": self.error_count,
                "error_rate": error_rate,
                "avg_execution_time_ms": avg_execution_time,
                "timestamp": datetime.now().isoformat(),
            }
            
            return NodeResult.success(
                value=health_data,
                provenance=[f"{self.service_name}.health_check"],
                trust_score=1.0,
                metadata={"service_id": self.service_id},
                correlation_id=str(correlation_id) if correlation_id else None,
            )
        
        except Exception as e:
            error_info = {
                "service_name": self.service_name,
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }
            
            return NodeResult.failure(
                error=SystemError(
                    message=f"Health check failed for {self.service_name}: {str(e)}",
                    system_component=self.service_name,
                    context={"service_id": self.service_id},
                    correlation_id=correlation_id,
                ),
                provenance=[f"{self.service_name}.health_check.failed"],
                correlation_id=str(correlation_id) if correlation_id else None,
            )
    
    async def get_metrics(
        self,
        correlation_id: Optional[UUID] = None,
    ) -> NodeResult[Dict[str, Any]]:
        """
        Get operational metrics for the memory service.
        
        Returns:
            NodeResult containing operational metrics
        """
        try:
            uptime_seconds = (datetime.now() - self.start_time).total_seconds()
            operations_per_second = (
                self.operation_count / uptime_seconds
                if uptime_seconds > 0 else 0
            )
            
            # Get service-specific metrics
            service_metrics = await self._collect_service_metrics()
            
            metrics = {
                "service_name": self.service_name,
                "service_id": self.service_id,
                "uptime_seconds": uptime_seconds,
                "operation_count": self.operation_count,
                "error_count": self.error_count,
                "total_execution_time_ms": self.total_execution_time_ms,
                "operations_per_second": operations_per_second,
                "avg_execution_time_ms": (
                    self.total_execution_time_ms / self.operation_count
                    if self.operation_count > 0 else 0
                ),
                "error_rate": (
                    self.error_count / self.operation_count
                    if self.operation_count > 0 else 0
                ),
                "timestamp": datetime.now().isoformat(),
                **service_metrics,
            }
            
            return NodeResult.success(
                value=metrics,
                provenance=[f"{self.service_name}.get_metrics"],
                trust_score=1.0,
                metadata={"service_id": self.service_id},
                correlation_id=str(correlation_id) if correlation_id else None,
            )
        
        except Exception as e:
            return NodeResult.failure(
                error=SystemError(
                    message=f"Metrics collection failed for {self.service_name}: {str(e)}",
                    system_component=self.service_name,
                    context={"service_id": self.service_id},
                    correlation_id=correlation_id,
                ),
                provenance=[f"{self.service_name}.get_metrics.failed"],
                correlation_id=str(correlation_id) if correlation_id else None,
            )
    
    async def configure(
        self,
        config: Dict[str, Any],
        correlation_id: Optional[UUID] = None,
    ) -> NodeResult[bool]:
        """
        Configure the memory service with new settings.
        
        Args:
            config: Configuration parameters
            correlation_id: Request correlation ID
            
        Returns:
            NodeResult indicating configuration success/failure
        """
        try:
            # Validate configuration
            validation_result = await self._validate_configuration(config)
            if not validation_result:
                raise OmniMemoryError(
                    error_code=OmniMemoryErrorCode.CONFIG_ERROR,
                    message=f"Invalid configuration for {self.service_name}",
                    context={"config": config, "service_id": self.service_id},
                )
            
            # Apply configuration
            old_config = self.config.copy()
            self.config.update(config)
            
            # Apply service-specific configuration
            await self._apply_configuration(config)
            
            await self.logger.emit_log_event_async(
                level="INFO",
                message=f"Configuration updated for {self.service_name}",
                event_type="configuration_updated",
                service_id=self.service_id,
                old_config=old_config,
                new_config=self.config,
            )
            
            return NodeResult.success(
                value=True,
                provenance=[f"{self.service_name}.configure"],
                trust_score=1.0,
                metadata={
                    "service_id": self.service_id,
                    "config_changes": list(config.keys()),
                },
                correlation_id=str(correlation_id) if correlation_id else None,
            )
        
        except Exception as e:
            # Restore old configuration on failure
            return NodeResult.failure(
                error=SystemError(
                    message=f"Configuration failed for {self.service_name}: {str(e)}",
                    system_component=self.service_name,
                    context={
                        "service_id": self.service_id,
                        "attempted_config": config,
                    },
                    correlation_id=correlation_id,
                ),
                provenance=[f"{self.service_name}.configure.failed"],
                correlation_id=str(correlation_id) if correlation_id else None,
            )
    
    async def _track_operation_start(self, operation_name: str) -> str:
        """Track the start of an operation and return tracking ID."""
        tracking_id = str(uuid4())
        self.operation_count += 1
        
        await self.logger.emit_log_event_async(
            level="DEBUG",
            message=f"Starting operation {operation_name}",
            event_type="operation_start",
            service_id=self.service_id,
            operation_name=operation_name,
            tracking_id=tracking_id,
        )
        
        return tracking_id
    
    async def _track_operation_end(
        self,
        operation_name: str,
        tracking_id: str,
        execution_time_ms: int,
        success: bool = True,
    ) -> None:
        """Track the end of an operation."""
        self.total_execution_time_ms += execution_time_ms
        if not success:
            self.error_count += 1
        
        await self.logger.emit_log_event_async(
            level="DEBUG",
            message=f"Completed operation {operation_name}",
            event_type="operation_end",
            service_id=self.service_id,
            operation_name=operation_name,
            tracking_id=tracking_id,
            execution_time_ms=execution_time_ms,
            success=success,
        )
    
    async def _create_response(
        self,
        request: BaseMemoryRequest,
        status: OperationStatus = OperationStatus.SUCCESS,
        execution_time_ms: int = 0,
        additional_data: Optional[Dict[str, Any]] = None,
    ) -> BaseMemoryResponse:
        """Create a standard response object."""
        response = BaseMemoryResponse(
            correlation_id=request.correlation_id,
            status=status,
            execution_time_ms=execution_time_ms,
            provenance=[f"{self.service_name}.operation"],
            trust_score=1.0,
            metadata={
                "service_id": self.service_id,
                "service_name": self.service_name,
                **(additional_data or {}),
            },
        )
        
        return response
    
    # Abstract methods for subclasses to implement
    
    @abstractmethod
    async def _check_service_health(self) -> bool:
        """
        Perform service-specific health checks.
        
        Returns:
            bool: True if service is healthy, False otherwise
        """
        pass
    
    @abstractmethod
    async def _collect_service_metrics(self) -> Dict[str, Any]:
        """
        Collect service-specific metrics.
        
        Returns:
            Dict containing service-specific metrics
        """
        pass
    
    @abstractmethod
    async def _validate_configuration(self, config: Dict[str, Any]) -> bool:
        """
        Validate service-specific configuration.
        
        Args:
            config: Configuration to validate
            
        Returns:
            bool: True if configuration is valid, False otherwise
        """
        pass
    
    @abstractmethod
    async def _apply_configuration(self, config: Dict[str, Any]) -> None:
        """
        Apply service-specific configuration.
        
        Args:
            config: Configuration to apply
        """
        pass


class BaseEffectService(BaseMemoryService):
    """Base class for Effect node services (memory storage, retrieval, persistence)."""
    
    async def _check_service_health(self) -> bool:
        """Default health check for Effect services."""
        # Check if storage systems are accessible
        return await self._check_storage_connectivity()
    
    async def _collect_service_metrics(self) -> Dict[str, Any]:
        """Default metrics collection for Effect services."""
        return {
            "storage_operations": await self._get_storage_operation_count(),
            "cache_hit_rate": await self._get_cache_hit_rate(),
            "storage_utilization": await self._get_storage_utilization(),
        }
    
    @abstractmethod
    async def _check_storage_connectivity(self) -> bool:
        """Check connectivity to storage systems."""
        pass
    
    @abstractmethod
    async def _get_storage_operation_count(self) -> int:
        """Get count of storage operations."""
        pass
    
    @abstractmethod
    async def _get_cache_hit_rate(self) -> float:
        """Get cache hit rate."""
        pass
    
    @abstractmethod
    async def _get_storage_utilization(self) -> Dict[str, float]:
        """Get storage utilization metrics."""
        pass


class BaseComputeService(BaseMemoryService):
    """Base class for Compute node services (intelligence processing, semantic analysis)."""
    
    async def _check_service_health(self) -> bool:
        """Default health check for Compute services."""
        # Check if processing models are loaded and responsive
        return await self._check_model_availability()
    
    async def _collect_service_metrics(self) -> Dict[str, Any]:
        """Default metrics collection for Compute services."""
        return {
            "processing_operations": await self._get_processing_operation_count(),
            "model_load_time": await self._get_model_load_time(),
            "processing_queue_size": await self._get_processing_queue_size(),
            "resource_utilization": await self._get_resource_utilization(),
        }
    
    @abstractmethod
    async def _check_model_availability(self) -> bool:
        """Check if processing models are available."""
        pass
    
    @abstractmethod
    async def _get_processing_operation_count(self) -> int:
        """Get count of processing operations."""
        pass
    
    @abstractmethod
    async def _get_model_load_time(self) -> float:
        """Get model loading time in seconds."""
        pass
    
    @abstractmethod
    async def _get_processing_queue_size(self) -> int:
        """Get current processing queue size."""
        pass
    
    @abstractmethod
    async def _get_resource_utilization(self) -> Dict[str, float]:
        """Get processing resource utilization."""
        pass


class BaseReducerService(BaseMemoryService):
    """Base class for Reducer node services (consolidation, aggregation, optimization)."""
    
    async def _check_service_health(self) -> bool:
        """Default health check for Reducer services."""
        # Check if reduction/optimization processes are running
        return await self._check_reduction_processes()
    
    async def _collect_service_metrics(self) -> Dict[str, Any]:
        """Default metrics collection for Reducer services."""
        return {
            "reduction_operations": await self._get_reduction_operation_count(),
            "data_reduction_ratio": await self._get_data_reduction_ratio(),
            "optimization_score": await self._get_optimization_score(),
        }
    
    @abstractmethod
    async def _check_reduction_processes(self) -> bool:
        """Check if reduction processes are healthy."""
        pass
    
    @abstractmethod
    async def _get_reduction_operation_count(self) -> int:
        """Get count of reduction operations."""
        pass
    
    @abstractmethod
    async def _get_data_reduction_ratio(self) -> float:
        """Get data reduction ratio achieved."""
        pass
    
    @abstractmethod
    async def _get_optimization_score(self) -> float:
        """Get optimization effectiveness score."""
        pass


class BaseOrchestratorService(BaseMemoryService):
    """Base class for Orchestrator node services (workflow, agent, memory coordination)."""
    
    async def _check_service_health(self) -> bool:
        """Default health check for Orchestrator services."""
        # Check if orchestration processes are running
        return await self._check_orchestration_processes()
    
    async def _collect_service_metrics(self) -> Dict[str, Any]:
        """Default metrics collection for Orchestrator services."""
        return {
            "orchestration_operations": await self._get_orchestration_operation_count(),
            "active_workflows": await self._get_active_workflow_count(),
            "coordination_success_rate": await self._get_coordination_success_rate(),
        }
    
    @abstractmethod
    async def _check_orchestration_processes(self) -> bool:
        """Check if orchestration processes are healthy."""
        pass
    
    @abstractmethod
    async def _get_orchestration_operation_count(self) -> int:
        """Get count of orchestration operations."""
        pass
    
    @abstractmethod
    async def _get_active_workflow_count(self) -> int:
        """Get count of active workflows."""
        pass
    
    @abstractmethod
    async def _get_coordination_success_rate(self) -> float:
        """Get coordination success rate."""
        pass