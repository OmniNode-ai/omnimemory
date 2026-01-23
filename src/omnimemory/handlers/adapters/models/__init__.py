"""Models for adapter health checks, configuration, and domain objects.

This package contains Pydantic models used by various adapters in the
handlers.adapters module for health checks, configuration, and responses.

Available modules:
    model_adapter_intent_graph_config: Configuration for AdapterIntentGraph.
    model_intent_graph_health: Health check result models.
    model_intent_domain: Intent classification storage and query models.
"""

from omnimemory.handlers.adapters.models.model_adapter_intent_graph_config import (
    ModelAdapterIntentGraphConfig,
)
from omnimemory.handlers.adapters.models.model_intent_domain import (
    ModelIntentClassificationOutput,
    ModelIntentDistributionResult,
    ModelIntentQueryResult,
    ModelIntentRecord,
    ModelIntentStorageResult,
)
from omnimemory.handlers.adapters.models.model_intent_graph_health import (
    ModelIntentGraphHealth,
)

__all__ = [
    "ModelAdapterIntentGraphConfig",
    "ModelIntentClassificationOutput",
    "ModelIntentDistributionResult",
    "ModelIntentGraphHealth",
    "ModelIntentQueryResult",
    "ModelIntentRecord",
    "ModelIntentStorageResult",
]
