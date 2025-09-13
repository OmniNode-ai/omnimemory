"""
OmniMemory ONEX Container Implementation

This module provides the ModelOnexContainer implementation for OmniMemory,
following the patterns from omnibase_core with memory-specific enhancements
and 4-node architecture support.
"""

import asyncio
import os
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, TypeVar, Union
from uuid import UUID, uuid4

from dependency_injector import containers, providers
from omnibase_core.core.common_types import ModelStateValue
from omnibase_core.core.errors.core_errors import CoreErrorCode, OnexError
from omnibase_core.core.monadic.model_node_result import (
    ErrorInfo,
    ErrorType,
    Event,
    ExecutionContext,
    LogEntry,
    NodeResult,
)
from omnibase_core.enums.enum_log_level import EnumLogLevel as LogLevel
from omnibase_core.protocol.protocol_database_connection import ProtocolDatabaseConnection
from omnibase_core.protocol.protocol_service_discovery import ProtocolServiceDiscovery
from omnibase_core.services.protocol_service_resolver import get_service_resolver
from omnibase_spi import ProtocolLogger
from pydantic_settings import BaseSettings

from ..protocols import (
    # Effect node protocols
    ProtocolMemoryStorage,
    ProtocolMemoryRetrieval,
    ProtocolMemoryPersistence,
    # Compute node protocols
    ProtocolIntelligenceProcessor,
    ProtocolSemanticAnalyzer,
    ProtocolPatternRecognition,
    # Reducer node protocols
    ProtocolMemoryConsolidator,
    ProtocolMemoryAggregator,
    ProtocolMemoryOptimizer,
    # Orchestrator node protocols
    ProtocolWorkflowCoordinator,
    ProtocolAgentCoordinator,
    ProtocolMemoryOrchestrator,
    # Error handling
    OmniMemoryError,
    OmniMemoryErrorCode,
    SystemError,
)

if TYPE_CHECKING:
    from omnibase_core.core.node_base import ModelNodeBase

T = TypeVar("T")


class OmniMemorySettings(BaseSettings):
    """Configuration settings for OmniMemory container."""
    
    # Logging configuration
    log_level: str = "INFO"
    
    # Database configuration
    database_url: Optional[str] = None
    supabase_url: Optional[str] = None
    supabase_anon_key: Optional[str] = None
    redis_url: Optional[str] = "redis://localhost:6379"
    
    # Vector database configuration
    pinecone_api_key: Optional[str] = None
    pinecone_environment: Optional[str] = None
    pinecone_index_name: Optional[str] = "omnimemory-vectors"
    
    # Service discovery configuration
    consul_host: str = "localhost"
    consul_port: int = 8500
    consul_datacenter: str = "dc1"
    consul_timeout: int = 10
    
    # Performance configuration
    max_concurrent_operations: int = 100
    operation_timeout_ms: int = 30000
    cache_ttl_seconds: int = 300
    
    # Memory configuration
    max_memory_size_mb: int = 1024  # 1GB default
    enable_compression: bool = True
    enable_encryption: bool = True
    
    # Development settings
    development_mode: bool = False
    debug_enabled: bool = False
    
    class Config:
        env_prefix = "OMNIMEMORY_"
        env_file = ".env"


class ContainerResolutionResult:
    """
    Result wrapper for container service resolution with monadic patterns.
    
    Provides observability and error handling for dependency injection
    operations within the ONEX monadic architecture.
    """
    
    def __init__(
        self,
        service_instance: T,
        resolution_time_ms: int,
        resolution_path: List[str],
        cache_hit: bool = False,
        fallback_used: bool = False,
        warnings: Optional[List[str]] = None,
    ):
        self.service_instance = service_instance
        self.resolution_time_ms = resolution_time_ms
        self.resolution_path = resolution_path
        self.cache_hit = cache_hit
        self.fallback_used = fallback_used
        self.warnings = warnings or []
        self.timestamp = datetime.now()


class OmniMemoryServiceProvider:
    """
    Service provider with monadic composition and observable resolution.
    
    Wraps service resolution in NodeResult for consistent error handling
    and observability throughout the dependency injection process.
    """
    
    def __init__(
        self,
        container: "OmniMemoryContainer",
        correlation_id: Optional[str] = None,
    ):
        self.container = container
        self.correlation_id = correlation_id or str(uuid4())
        self.resolution_cache: Dict[str, ContainerResolutionResult] = {}
    
    async def resolve_async(
        self,
        protocol_type: type[T],
        service_name: Optional[str] = None,
        use_cache: bool = True,
    ) -> NodeResult[T]:
        """
        Resolve service with monadic composition and observable resolution.
        
        Args:
            protocol_type: Protocol interface to resolve
            service_name: Optional service name for specific resolution
            use_cache: Whether to use resolution cache
            
        Returns:
            NodeResult[T]: Monadic result with resolved service and context
        """
        start_time = datetime.now()
        protocol_name = protocol_type.__name__
        cache_key = f"{protocol_name}:{service_name or 'default'}"
        
        # Create execution context
        execution_context = ExecutionContext(
            provenance=[f"container.resolve.{protocol_name}"],
            logs=[],
            trust_score=1.0,
            timestamp=start_time,
            metadata={
                "protocol_type": protocol_name,
                "service_name": service_name,
                "correlation_id": self.correlation_id,
            },
            correlation_id=self.correlation_id,
        )
        
        try:
            # Check cache first if enabled
            if use_cache and cache_key in self.resolution_cache:
                cached_result = self.resolution_cache[cache_key]
                
                execution_context.logs.append(
                    LogEntry(
                        "INFO",
                        f"Service resolved from cache: {protocol_name}",
                        datetime.now(),
                    ),
                )
                
                return NodeResult.success(
                    value=cached_result.service_instance,
                    provenance=[*execution_context.provenance, "cache.hit"],
                    trust_score=0.95,  # Cached services have slightly lower trust
                    metadata={
                        **execution_context.metadata,
                        "resolution_time_ms": 0,  # Cache hit
                        "cache_hit": True,
                        "original_resolution_time": cached_result.resolution_time_ms,
                    },
                    events=[
                        Event(
                            type="container.service.resolved.cached",
                            payload={
                                "protocol_type": protocol_name,
                                "service_name": service_name,
                                "cache_key": cache_key,
                                "original_timestamp": cached_result.timestamp.isoformat(),
                            },
                            timestamp=datetime.now(),
                            correlation_id=self.correlation_id,
                        ),
                    ],
                    correlation_id=self.correlation_id,
                )
            
            # Resolve service from container
            resolution_result = await self._resolve_from_container(
                protocol_type,
                service_name,
            )
            
            end_time = datetime.now()
            resolution_time_ms = int((end_time - start_time).total_seconds() * 1000)
            
            # Cache successful resolution
            if use_cache:
                self.resolution_cache[cache_key] = ContainerResolutionResult(
                    service_instance=resolution_result,
                    resolution_time_ms=resolution_time_ms,
                    resolution_path=execution_context.provenance,
                    cache_hit=False,
                    fallback_used=False,
                )
            
            execution_context.logs.append(
                LogEntry(
                    "INFO",
                    f"Service resolved successfully: {protocol_name}",
                    end_time,
                ),
            )
            execution_context.timestamp = end_time
            execution_context.metadata["resolution_time_ms"] = resolution_time_ms
            
            return NodeResult.success(
                value=resolution_result,
                provenance=execution_context.provenance,
                trust_score=1.0,
                metadata=execution_context.metadata,
                events=[
                    Event(
                        type="container.service.resolved",
                        payload={
                            "protocol_type": protocol_name,
                            "service_name": service_name,
                            "resolution_time_ms": resolution_time_ms,
                            "cache_miss": True,
                        },
                        timestamp=end_time,
                        correlation_id=self.correlation_id,
                    ),
                ],
                correlation_id=self.correlation_id,
            )
        
        except Exception as e:
            # Handle resolution failure
            error_info = ErrorInfo(
                error_type=ErrorType.DEPENDENCY,
                message=f"Service resolution failed for {protocol_name}: {str(e)}",
                code="SERVICE_RESOLUTION_FAILED",
                context={
                    "protocol_type": protocol_name,
                    "service_name": service_name,
                    "correlation_id": self.correlation_id,
                },
                retryable=True,
                backoff_strategy="exponential",
                max_attempts=3,
                correlation_id=self.correlation_id,
            )
            
            execution_context.logs.append(
                LogEntry(
                    "ERROR",
                    f"Service resolution failed: {str(e)}",
                    datetime.now(),
                ),
            )
            
            return NodeResult.failure(
                error=error_info,
                provenance=[*execution_context.provenance, "resolution.failed"],
                correlation_id=self.correlation_id,
            )
    
    async def _resolve_from_container(
        self,
        protocol_type: type[T],
        service_name: Optional[str],
    ) -> T:
        """
        Resolve service from underlying container implementation.
        
        Args:
            protocol_type: Protocol interface to resolve
            service_name: Optional service name
            
        Returns:
            T: Resolved service instance
        """
        # Use protocol service resolver for external dependencies
        protocol_name = protocol_type.__name__
        
        # Check if this is a protocol we handle with service resolver
        if protocol_name in ["ProtocolServiceDiscovery", "ProtocolDatabaseConnection"]:
            service_resolver = get_service_resolver()
            return await service_resolver.resolve_service(protocol_type)
        
        # Check if container has get_service method
        if hasattr(self.container, "get_service_async"):
            return await self.container.get_service_async(protocol_type, service_name)
        if hasattr(self.container, "get_service"):
            return self.container.get_service(protocol_type, service_name)
        
        # Map protocol names to container providers
        provider_map = {
            "ProtocolLogger": "enhanced_logger",
            "Logger": "enhanced_logger",
            
            # OmniMemory-specific protocol mappings
            "ProtocolMemoryStorage": "memory_storage_service",
            "ProtocolMemoryRetrieval": "memory_retrieval_service",
            "ProtocolMemoryPersistence": "memory_persistence_service",
            "ProtocolIntelligenceProcessor": "intelligence_processor_service",
            "ProtocolSemanticAnalyzer": "semantic_analyzer_service",
            "ProtocolPatternRecognition": "pattern_recognition_service",
            "ProtocolMemoryConsolidator": "memory_consolidator_service",
            "ProtocolMemoryAggregator": "memory_aggregator_service",
            "ProtocolMemoryOptimizer": "memory_optimizer_service",
            "ProtocolWorkflowCoordinator": "workflow_coordinator_service",
            "ProtocolAgentCoordinator": "agent_coordinator_service",
            "ProtocolMemoryOrchestrator": "memory_orchestrator_service",
        }
        
        if protocol_name in provider_map:
            provider_name = provider_map[protocol_name]
            provider = getattr(self.container._base_container, provider_name, None)
            if provider:
                return provider()
        
        # Fallback error
        raise OmniMemoryError(
            error_code=OmniMemoryErrorCode.SERVICE_UNAVAILABLE,
            message=f"Unable to resolve service for protocol {protocol_name}",
            context={
                "protocol_type": protocol_name,
                "service_name": service_name,
                "available_providers": list(provider_map.keys()),
            },
        )


# === CORE CONTAINER DEFINITION ===


class _BaseOmniMemoryContainer(containers.DeclarativeContainer):
    """Base dependency injection container for OmniMemory."""
    
    # === CONFIGURATION ===
    config = providers.Configuration()
    
    # === SETTINGS ===
    settings = providers.Singleton(
        OmniMemorySettings,
    )
    
    # === CORE SERVICES ===
    
    # Enhanced logger with monadic patterns
    enhanced_logger = providers.Factory(
        lambda level: _create_enhanced_logger(level),
        level=LogLevel.INFO,
    )
    
    # === DATABASE SERVICES ===
    
    # Database connection factory
    database_connection = providers.Factory(
        lambda settings: _create_database_connection(settings),
        settings=settings,
    )
    
    # Redis connection factory
    redis_connection = providers.Factory(
        lambda settings: _create_redis_connection(settings),
        settings=settings,
    )
    
    # Vector database connection factory  
    vector_database_connection = providers.Factory(
        lambda settings: _create_vector_database_connection(settings),
        settings=settings,
    )
    
    # === MEMORY SERVICES (EFFECT NODE) ===
    
    # Memory storage service
    memory_storage_service = providers.Factory(
        lambda db, redis, logger: _create_memory_storage_service(db, redis, logger),
        db=database_connection,
        redis=redis_connection,
        logger=enhanced_logger,
    )
    
    # Memory retrieval service
    memory_retrieval_service = providers.Factory(
        lambda db, vector_db, logger: _create_memory_retrieval_service(db, vector_db, logger),
        db=database_connection,
        vector_db=vector_database_connection,
        logger=enhanced_logger,
    )
    
    # Memory persistence service
    memory_persistence_service = providers.Factory(
        lambda db, settings, logger: _create_memory_persistence_service(db, settings, logger),
        db=database_connection,
        settings=settings,
        logger=enhanced_logger,
    )
    
    # === INTELLIGENCE SERVICES (COMPUTE NODE) ===
    
    # Intelligence processor service
    intelligence_processor_service = providers.Factory(
        lambda logger, settings: _create_intelligence_processor_service(logger, settings),
        logger=enhanced_logger,
        settings=settings,
    )
    
    # Semantic analyzer service
    semantic_analyzer_service = providers.Factory(
        lambda vector_db, logger: _create_semantic_analyzer_service(vector_db, logger),
        vector_db=vector_database_connection,
        logger=enhanced_logger,
    )
    
    # Pattern recognition service
    pattern_recognition_service = providers.Factory(
        lambda db, logger, settings: _create_pattern_recognition_service(db, logger, settings),
        db=database_connection,
        logger=enhanced_logger,
        settings=settings,
    )
    
    # === CONSOLIDATION SERVICES (REDUCER NODE) ===
    
    # Memory consolidator service
    memory_consolidator_service = providers.Factory(
        lambda db, logger: _create_memory_consolidator_service(db, logger),
        db=database_connection,
        logger=enhanced_logger,
    )
    
    # Memory aggregator service
    memory_aggregator_service = providers.Factory(
        lambda db, redis, logger: _create_memory_aggregator_service(db, redis, logger),
        db=database_connection,
        redis=redis_connection,
        logger=enhanced_logger,
    )
    
    # Memory optimizer service
    memory_optimizer_service = providers.Factory(
        lambda db, redis, vector_db, logger: _create_memory_optimizer_service(db, redis, vector_db, logger),
        db=database_connection,
        redis=redis_connection,
        vector_db=vector_database_connection,
        logger=enhanced_logger,
    )
    
    # === ORCHESTRATION SERVICES (ORCHESTRATOR NODE) ===
    
    # Workflow coordinator service
    workflow_coordinator_service = providers.Factory(
        lambda logger, settings: _create_workflow_coordinator_service(logger, settings),
        logger=enhanced_logger,
        settings=settings,
    )
    
    # Agent coordinator service
    agent_coordinator_service = providers.Factory(
        lambda redis, logger, settings: _create_agent_coordinator_service(redis, logger, settings),
        redis=redis_connection,
        logger=enhanced_logger,
        settings=settings,
    )
    
    # Memory orchestrator service
    memory_orchestrator_service = providers.Factory(
        lambda db, redis, logger, settings: _create_memory_orchestrator_service(db, redis, logger, settings),
        db=database_connection,
        redis=redis_connection,
        logger=enhanced_logger,
        settings=settings,
    )
    
    # === WORKFLOW ORCHESTRATION ===
    
    # Workflow factory
    workflow_factory = providers.Factory(lambda: _create_workflow_factory())
    
    # Workflow execution coordinator
    workflow_coordinator = providers.Singleton(
        lambda factory: _create_workflow_coordinator(factory),
        factory=workflow_factory,
    )


class OmniMemoryContainer:
    """
    OmniMemory ONEX dependency injection container with monadic architecture.
    
    This container wraps the base DI container and adds:
    - Monadic service resolution through NodeResult
    - Observable dependency injection with event emission
    - Contract-driven automatic service registration
    - Workflow orchestration support
    - Enhanced error handling and recovery patterns
    - Performance monitoring and caching
    """
    
    def __init__(self, settings: Optional[OmniMemorySettings] = None):
        """Initialize enhanced container with custom methods."""
        self._settings = settings or OmniMemorySettings()
        self._base_container = _BaseOmniMemoryContainer()
        
        # Configure base container with settings
        self._base_container.config.from_dict(self._settings.dict())
        
        # Initialize performance tracking
        self._performance_metrics = {
            "total_resolutions": 0,
            "cache_hit_rate": 0.0,
            "avg_resolution_time_ms": 0.0,
            "error_rate": 0.0,
            "active_services": 0,
        }
        
        # Initialize service provider
        self._service_provider = None
    
    @property
    def base_container(self):
        """Access to base container for current standards."""
        return self._base_container
    
    @property
    def config(self):
        """Access to configuration."""
        return self._base_container.config
    
    @property
    def settings(self) -> OmniMemorySettings:
        """Access to settings."""
        return self._settings
    
    @property
    def enhanced_logger(self):
        """Access to enhanced logger."""
        return self._base_container.enhanced_logger
    
    @property
    def workflow_factory(self):
        """Access to workflow factory."""
        return self._base_container.workflow_factory
    
    @property
    def workflow_coordinator(self):
        """Access to workflow coordinator."""
        return self._base_container.workflow_coordinator
    
    @property
    def service_provider(self):
        """Access to monadic service provider."""
        if self._service_provider is None:
            self._service_provider = OmniMemoryServiceProvider(self)
        return self._service_provider
    
    async def get_service_async(
        self,
        protocol_type: type[T],
        service_name: Optional[str] = None,
        correlation_id: Optional[str] = None,
    ) -> T:
        """
        Async service resolution with monadic patterns.
        
        Args:
            protocol_type: Protocol interface to resolve
            service_name: Optional service name
            correlation_id: Optional correlation ID for tracking
            
        Returns:
            T: Resolved service instance
            
        Raises:
            OmniMemoryError: If service resolution fails
        """
        provider = OmniMemoryServiceProvider(self, correlation_id)
        result = await provider.resolve_async(protocol_type, service_name)
        
        if result.is_failure:
            raise OmniMemoryError(
                error_code=OmniMemoryErrorCode.SERVICE_UNAVAILABLE,
                message=result.error.message,
                context=result.error.context,
                correlation_id=UUID(result.error.correlation_id) if result.error.correlation_id else None,
            )
        
        return result.value
    
    def get_service_sync(
        self,
        protocol_type: type[T],
        service_name: Optional[str] = None,
    ) -> T:
        """
        Synchronous service resolution for current standards.
        
        Args:
            protocol_type: Protocol interface to resolve
            service_name: Optional service name
            
        Returns:
            T: Resolved service instance
        """
        return asyncio.run(self.get_service_async(protocol_type, service_name))
    
    # Compatibility alias
    def get_service(
        self,
        protocol_type: type[T],
        service_name: Optional[str] = None,
    ) -> T:
        """Modern standards method."""
        return self.get_service_sync(protocol_type, service_name)
    
    async def create_enhanced_nodebase(
        self,
        contract_path: Path,
        node_id: Optional[str] = None,
        workflow_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> "ModelNodeBase":
        """
        Factory method for creating Enhanced ModelNodeBase instances.
        
        Args:
            contract_path: Path to contract file
            node_id: Optional node identifier
            workflow_id: Optional workflow identifier
            session_id: Optional session identifier
            
        Returns:
            ModelNodeBase: Configured node instance
        """
        from omnibase_core.core.node_base import ModelNodeBase
        
        return ModelNodeBase(
            contract_path=contract_path,
            node_id=node_id,
            container=self,
            workflow_id=workflow_id,
            session_id=session_id,
        )
    
    def get_workflow_orchestrator(self):
        """Get workflow orchestration coordinator."""
        return self.workflow_coordinator()
    
    def get_performance_metrics(self) -> Dict[str, ModelStateValue]:
        """
        Get container performance metrics.
        
        Returns:
            Dict containing resolution times, cache hits, errors, etc.
        """
        return {
            key: ModelStateValue.from_primitive(value)
            for key, value in self._performance_metrics.items()
        }
    
    async def get_service_discovery(self) -> ProtocolServiceDiscovery:
        """Get service discovery implementation with automatic fallback."""
        return await self.get_service_async(ProtocolServiceDiscovery)
    
    async def get_database(self) -> ProtocolDatabaseConnection:
        """Get database connection implementation with automatic fallback."""
        return await self.get_service_async(ProtocolDatabaseConnection)
    
    async def get_external_services_health(self) -> Dict[str, Any]:
        """Get health status for all external services."""
        service_resolver = get_service_resolver()
        return await service_resolver.get_all_service_health()
    
    async def refresh_external_services(self) -> None:
        """Force refresh all external service connections."""
        service_resolver = get_service_resolver()
        
        # Refresh service discovery if cached
        try:
            await service_resolver.refresh_service(ProtocolServiceDiscovery)
        except Exception:
            pass  # Service may not be cached yet
        
        # Refresh database if cached
        try:
            await service_resolver.refresh_service(ProtocolDatabaseConnection)
        except Exception:
            pass  # Service may not be cached yet
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Comprehensive health check of the container and all services.
        
        Returns:
            Dict containing health status of all components
        """
        health_status = {
            "container": "healthy",
            "timestamp": datetime.now().isoformat(),
            "services": {},
            "performance": self._performance_metrics,
        }
        
        # Check core services
        try:
            logger = await self.get_service_async(ProtocolLogger)
            health_status["services"]["logger"] = "healthy"
        except Exception as e:
            health_status["services"]["logger"] = f"unhealthy: {str(e)}"
        
        # Check external services
        try:
            external_health = await self.get_external_services_health()
            health_status["external_services"] = external_health
        except Exception as e:
            health_status["external_services"] = f"check_failed: {str(e)}"
        
        return health_status


# === HELPER FUNCTIONS ===


def _create_enhanced_logger(level: LogLevel) -> ProtocolLogger:
    """Create enhanced logger with monadic patterns."""
    
    class EnhancedLogger:
        def __init__(self, level: LogLevel):
            self.level = level
        
        def emit_log_event_sync(
            self,
            level: LogLevel,
            message: str,
            event_type: str = "generic",
            **kwargs,
        ) -> None:
            """Emit log event synchronously."""
            if level.value >= self.level.value:
                print(f"[{datetime.now().isoformat()}] {level.name}: {message}")
        
        async def emit_log_event_async(
            self,
            level: LogLevel,
            message: str,
            event_type: str = "generic",
            **kwargs,
        ) -> None:
            """Emit log event asynchronously."""
            self.emit_log_event_sync(level, message, event_type, **kwargs)
        
        def emit_log_event(
            self,
            level: LogLevel,
            message: str,
            event_type: str = "generic",
            **kwargs,
        ) -> None:
            """Emit log event (defaults to sync)."""
            self.emit_log_event_sync(level, message, event_type, **kwargs)
        
        def info(self, message: str) -> None:
            self.emit_log_event_sync(LogLevel.INFO, message, "info")
        
        def warning(self, message: str) -> None:
            self.emit_log_event_sync(LogLevel.WARNING, message, "warning")
        
        def error(self, message: str) -> None:
            self.emit_log_event_sync(LogLevel.ERROR, message, "error")
    
    return EnhancedLogger(level)


# Database connection factories
def _create_database_connection(settings: OmniMemorySettings):
    """Create database connection based on settings."""
    # This would create actual database connections
    # For now, return a placeholder
    class DatabaseConnection:
        def __init__(self, url: str):
            self.url = url
    
    return DatabaseConnection(settings.database_url or "postgresql://localhost/omnimemory")


def _create_redis_connection(settings: OmniMemorySettings):
    """Create Redis connection based on settings."""
    class RedisConnection:
        def __init__(self, url: str):
            self.url = url
    
    return RedisConnection(settings.redis_url)


def _create_vector_database_connection(settings: OmniMemorySettings):
    """Create vector database connection based on settings."""
    class VectorDatabaseConnection:
        def __init__(self, api_key: str, environment: str, index_name: str):
            self.api_key = api_key
            self.environment = environment
            self.index_name = index_name
    
    return VectorDatabaseConnection(
        settings.pinecone_api_key or "test",
        settings.pinecone_environment or "test",
        settings.pinecone_index_name,
    )


# Service creation factories (these would be implemented with actual service classes)
def _create_memory_storage_service(db, redis, logger):
    """Create memory storage service."""
    class MemoryStorageService:
        def __init__(self, db, redis, logger):
            self.db = db
            self.redis = redis
            self.logger = logger
    
    return MemoryStorageService(db, redis, logger)


def _create_memory_retrieval_service(db, vector_db, logger):
    """Create memory retrieval service."""
    class MemoryRetrievalService:
        def __init__(self, db, vector_db, logger):
            self.db = db
            self.vector_db = vector_db
            self.logger = logger
    
    return MemoryRetrievalService(db, vector_db, logger)


def _create_memory_persistence_service(db, settings, logger):
    """Create memory persistence service."""
    class MemoryPersistenceService:
        def __init__(self, db, settings, logger):
            self.db = db
            self.settings = settings
            self.logger = logger
    
    return MemoryPersistenceService(db, settings, logger)


def _create_intelligence_processor_service(logger, settings):
    """Create intelligence processor service."""
    class IntelligenceProcessorService:
        def __init__(self, logger, settings):
            self.logger = logger
            self.settings = settings
    
    return IntelligenceProcessorService(logger, settings)


def _create_semantic_analyzer_service(vector_db, logger):
    """Create semantic analyzer service."""
    class SemanticAnalyzerService:
        def __init__(self, vector_db, logger):
            self.vector_db = vector_db
            self.logger = logger
    
    return SemanticAnalyzerService(vector_db, logger)


def _create_pattern_recognition_service(db, logger, settings):
    """Create pattern recognition service."""
    class PatternRecognitionService:
        def __init__(self, db, logger, settings):
            self.db = db
            self.logger = logger
            self.settings = settings
    
    return PatternRecognitionService(db, logger, settings)


def _create_memory_consolidator_service(db, logger):
    """Create memory consolidator service."""
    class MemoryConsolidatorService:
        def __init__(self, db, logger):
            self.db = db
            self.logger = logger
    
    return MemoryConsolidatorService(db, logger)


def _create_memory_aggregator_service(db, redis, logger):
    """Create memory aggregator service."""
    class MemoryAggregatorService:
        def __init__(self, db, redis, logger):
            self.db = db
            self.redis = redis
            self.logger = logger
    
    return MemoryAggregatorService(db, redis, logger)


def _create_memory_optimizer_service(db, redis, vector_db, logger):
    """Create memory optimizer service."""
    class MemoryOptimizerService:
        def __init__(self, db, redis, vector_db, logger):
            self.db = db
            self.redis = redis
            self.vector_db = vector_db
            self.logger = logger
    
    return MemoryOptimizerService(db, redis, vector_db, logger)


def _create_workflow_coordinator_service(logger, settings):
    """Create workflow coordinator service."""
    class WorkflowCoordinatorService:
        def __init__(self, logger, settings):
            self.logger = logger
            self.settings = settings
    
    return WorkflowCoordinatorService(logger, settings)


def _create_agent_coordinator_service(redis, logger, settings):
    """Create agent coordinator service."""
    class AgentCoordinatorService:
        def __init__(self, redis, logger, settings):
            self.redis = redis
            self.logger = logger
            self.settings = settings
    
    return AgentCoordinatorService(redis, logger, settings)


def _create_memory_orchestrator_service(db, redis, logger, settings):
    """Create memory orchestrator service."""
    class MemoryOrchestratorService:
        def __init__(self, db, redis, logger, settings):
            self.db = db
            self.redis = redis
            self.logger = logger
            self.settings = settings
    
    return MemoryOrchestratorService(db, redis, logger, settings)


def _create_workflow_factory():
    """Create workflow factory for workflow integration."""
    
    class WorkflowFactory:
        def create_workflow(
            self,
            workflow_type: str,
            config: Optional[Dict[str, ModelStateValue]] = None,
        ):
            """Create workflow instance by type."""
            config = config or {}
            
        def list_available_workflows(self) -> List[str]:
            """List available workflow types."""
            return [
                "simple_sequential",
                "parallel_execution",
                "conditional_branching",
                "retry_with_backoff",
                "data_pipeline",
                "memory_consolidation",
                "intelligence_processing",
            ]
    
    return WorkflowFactory()


def _create_workflow_coordinator(factory):
    """Create workflow execution coordinator."""
    
    class WorkflowCoordinator:
        def __init__(self, factory):
            self.factory = factory
            self.active_workflows: Dict[str, ModelStateValue] = {}
        
        async def execute_workflow(
            self,
            workflow_id: str,
            workflow_type: str,
            input_data: ModelStateValue,
            config: Optional[Dict[str, ModelStateValue]] = None,
        ) -> NodeResult[ModelStateValue]:
            """Execute workflow with monadic result."""
            try:
                workflow = self.factory.create_workflow(workflow_type, config)
                
                # Execute workflow using the configured type and input data
                workflow_result = await self._execute_workflow_type(
                    workflow_type,
                    input_data,
                    config,
                )
                
                return NodeResult.success(
                    value=workflow_result,
                    provenance=[f"workflow.{workflow_type}"],
                    trust_score=0.9,
                    metadata={
                        "workflow_id": workflow_id,
                        "workflow_type": workflow_type,
                    },
                )
            
            except Exception as e:
                error_info = ErrorInfo(
                    error_type=ErrorType.PERMANENT,
                    message=f"Workflow execution failed: {str(e)}",
                    correlation_id=workflow_id,
                    retryable=False,
                )
                
                return NodeResult.failure(
                    error=error_info,
                    provenance=[f"workflow.{workflow_type}.failed"],
                )
        
        async def _execute_workflow_type(
            self,
            workflow_type: str,
            input_data: Any,
            config: Optional[Dict[str, Any]],
        ) -> Any:
            """Execute a specific workflow type with input data."""
            # Simplified implementation - would be expanded with actual workflow logic
            return input_data
        
        def get_active_workflows(self) -> List[str]:
            """Get list of active workflow IDs."""
            return list(self.active_workflows.keys())
    
    return WorkflowCoordinator(factory)


# === CONTAINER FACTORY ===


async def create_omnimemory_container(
    settings: Optional[OmniMemorySettings] = None,
) -> OmniMemoryContainer:
    """
    Create and configure OmniMemory container.
    
    Args:
        settings: Optional configuration settings
        
    Returns:
        OmniMemoryContainer: Configured container instance
    """
    container = OmniMemoryContainer(settings)
    return container


# === GLOBAL CONTAINER ===

_omnimemory_container: Optional[OmniMemoryContainer] = None


async def get_omnimemory_container() -> OmniMemoryContainer:
    """Get or create global OmniMemory container instance."""
    global _omnimemory_container
    if _omnimemory_container is None:
        _omnimemory_container = await create_omnimemory_container()
    return _omnimemory_container


def get_omnimemory_container_sync() -> OmniMemoryContainer:
    """Get OmniMemory container synchronously."""
    return asyncio.run(get_omnimemory_container())