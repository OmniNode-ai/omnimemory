"""
ONEX Model Package - OmniMemory Foundation Architecture

Models are organized into functional domains following omnibase_core patterns:
- core/: Foundational models, shared types, contracts
- memory/: Memory storage, retrieval, persistence models
- intelligence/: Intelligence processing, analysis models
- service/: Service configurations, orchestration models
- container/: Container configurations and DI models
- foundation/: Base implementations and protocols
- subscription/: Agent subscriptions and notification delivery models
- adapters/: Adapter configuration models
- config/: Configuration models
- utils/: Utility models

This __init__.py maintains compatibility by re-exporting
all models at the package level following ONEX standards.
"""

# Cross-domain interface - import submodules only, no star imports
from . import (
    adapters,
    config,
    container,
    core,
    foundation,
    intelligence,
    memory,
    service,
    subscription,
    utils,
)

# Re-export domains for direct access
__all__ = [
    "adapters",
    "config",
    "container",
    "core",
    "foundation",
    "intelligence",
    "memory",
    "service",
    "subscription",
    "utils",
]
