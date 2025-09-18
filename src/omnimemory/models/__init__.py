"""
ONEX Model Package - OmniMemory Foundation Architecture

Models are organized into functional domains following omnibase_core patterns:
- core/: Foundational models, shared types, contracts
- memory/: Memory storage, retrieval, persistence models
- intelligence/: Intelligence processing, analysis models
- service/: Service configurations, orchestration models
- foundation/: Base implementations and protocols

This __init__.py provides structured access by re-exporting
all model domains at the package level following ONEX standards.
"""

# Cross-domain interface - import submodules only, no star imports
from . import core, foundation, intelligence, memory, service

# Re-export domains for direct access
__all__ = [
    "core",
    "memory",
    "intelligence",
    "service",
    "foundation",
]
