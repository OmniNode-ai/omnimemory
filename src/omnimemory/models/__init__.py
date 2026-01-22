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

This __init__.py maintains compatibility by re-exporting
all models at the package level following ONEX standards.
"""

# Cross-domain interface - import submodules only, no star imports
from . import container, core, foundation, intelligence, memory, service, subscription

# Re-export domains for direct access
__all__ = [
    "container",
    "core",
    "foundation",
    "intelligence",
    "memory",
    "service",
    "subscription",
]
