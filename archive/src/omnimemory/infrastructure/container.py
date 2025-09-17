"""
Fixed infrastructure container for OmniMemory.

This is a temporary fix for import path issues in omnibase_infra.
It creates the infrastructure container with correct import paths.
"""

from dependency_injector import containers, providers
from omnibase_core.core.onex_container import ModelONEXContainer
from omnibase_core.core.services.event_bus_service.v1_0_0.event_bus_service import (
    EventBusService,
)
from omnibase_core.core.services.event_bus_service.v1_0_0.models.model_event_bus_config import (
    ModelEventBusConfig,
)
from omnibase_core.models.core.model_event_envelope import ModelEventEnvelope
from omnibase_core.models.core.model_onex_event import (  # Fixed import path
    ModelOnexEvent,
)
from omnibase_core.protocol.protocol_event_bus import ProtocolEventBus
from omnibase_core.utils.generation.utility_schema_loader import UtilitySchemaLoader


class InfrastructureContainer(containers.DeclarativeContainer):
    """
    Infrastructure dependency injection container with fixed import paths.

    This container provides the ProtocolEventBus service required for ONEX
    EFFECT nodes to communicate via RedPanda event bus.
    """

    # Event Bus Config - Uses event_bus_url instead of bootstrap_servers
    event_bus_config = providers.Factory(
        ModelEventBusConfig,
        event_bus_url="kafka://redpanda:9092",
        auto_resolve_event_bus=True,
        enable_lifecycle_events=True,
        enable_introspection_publishing=True,
    )

    # Event Bus Service - Critical for ONEX communication
    event_bus_service = providers.Singleton(
        EventBusService,
        config=event_bus_config,
    )

    # Schema Loader Service - Required for ONEX nodes
    schema_loader_service = providers.Singleton(
        UtilitySchemaLoader,
    )

    # Register the protocol with the service
    protocol_event_bus = providers.Singleton(
        lambda container: container.event_bus_service(), container=event_bus_service
    )


def create_infrastructure_container() -> ModelONEXContainer:
    """
    Create infrastructure container with ProtocolEventBus service.

    This replaces the broken omnibase_infra.infrastructure.container.create_infrastructure_container
    function with a working version that has correct import paths.

    Returns:
        ModelONEXContainer: Container with ProtocolEventBus service registered
    """
    print("[DEBUG] Creating ONEX container...")
    # Create base ONEX container
    container = ModelONEXContainer()

    print("[DEBUG] Creating infrastructure container...")
    # Create infrastructure container
    infra_container = InfrastructureContainer()

    print("[DEBUG] Wiring infrastructure services...")
    # Wire the infrastructure services
    infra_container.wire([__name__])

    print("[DEBUG] Creating event bus service...")
    event_bus_service = infra_container.event_bus_service()
    print(f"[DEBUG] Event bus service created: {type(event_bus_service)}")

    print("[DEBUG] Creating schema loader service...")
    schema_loader_service = infra_container.schema_loader_service()
    print(f"[DEBUG] Schema loader service created: {type(schema_loader_service)}")

    print("[DEBUG] Registering ProtocolEventBus...")
    # Register ProtocolEventBus in the main container with correct name
    container.register_service("ProtocolEventBus", event_bus_service)

    print("[DEBUG] Registering ProtocolSchemaLoader...")
    # Register ProtocolSchemaLoader in the main container
    container.register_service("ProtocolSchemaLoader", schema_loader_service)

    print(
        f"[DEBUG] Container services after registration: {container._services.keys() if hasattr(container, '_services') else 'No _services attr'}"
    )
    print("[DEBUG] Infrastructure container creation complete")

    return container
