> **Navigation**: [Home](../INDEX.md) > Architecture

# ONEX Four-Node Architecture in OmniMemory

> **Version**: 0.1.0
> **Last Updated**: 2026-02-19
> **Status**: Active

## Overview

OmniMemory uses the ONEX 4-node architecture to organize all memory system
operations. Every piece of code in `src/omnimemory/nodes/` belongs to one of
four archetypes:

- **EFFECT** — touches the outside world (storage, event bus, APIs)
- **COMPUTE** — transforms data without side effects (analysis, similarity)
- **REDUCER** — aggregates and consolidates state (statistics, merging)
- **ORCHESTRATOR** — coordinates multi-step workflows (lifecycle, routing)

The pattern enforces unidirectional data flow and a clean separation between
I/O and logic. Handlers hold all business logic; nodes are thin wrappers that
wire container injection to the handler.

---

## Data Flow

```
  Kafka / External API
         |
         v
  +--------------+     +--------------+     +--------------+     +------------------+
  |    EFFECT    | --> |   COMPUTE    | --> |   REDUCER    | --> |  ORCHESTRATOR    |
  |              |     |              |     |              |     |                  |
  | intent_event |     | semantic_    |     | statistics_  |     | memory_lifecycle |
  | _consumer    |     | analyzer     |     | reducer      |     | _orchestrator    |
  | _effect      |     | _compute     |     |              |     |                  |
  |              |     |              |     |              |     |                  |
  | intent_      |     | similarity_  |     |              |     | agent_           |
  | storage_     |     | compute      |     |              |     | coordinator_     |
  | effect       |     |              |     |              |     | orchestrator     |
  |              |     |              |     |              |     |                  |
  | intent_query |     |              |     |              |     |                  |
  | _effect      |     |              |     |              |     |                  |
  |              |     |              |     |              |     |                  |
  | memory_      |     |              |     |              |     |                  |
  | retrieval_   |     |              |     |              |     |                  |
  | effect       |     |              |     |              |     |                  |
  |              |     |              |     |              |     |                  |
  | memory_      |     |              |     |              |     |                  |
  | storage_     |     |              |     |              |     |                  |
  | effect       |     |              |     |              |     |                  |
  +--------------+     +--------------+     +--------------+     +------------------+
         |                    |                    |                       |
         v                    v                    v                       v
    Memgraph /           Embedding /          Aggregated             Lifecycle
    PostgreSQL /         Similarity           Statistics             Commands
    Qdrant               Vectors
```

Data flows left to right. EFFECT nodes produce raw data from external sources;
COMPUTE nodes transform it into enriched representations; REDUCER nodes
consolidate results into aggregated state; ORCHESTRATOR nodes coordinate
workflows and lifecycle decisions.

---

## Node Types in OmniMemory

### EFFECT Nodes

EFFECT nodes own all external I/O: event bus subscription, database reads and
writes, API calls, and file system operations. They are the only nodes allowed
to produce side effects.

**Characteristics:**
- Subscribe to Kafka topics or query external APIs
- Write to Memgraph, PostgreSQL, Qdrant, or the filesystem
- Include circuit breakers, retry logic, and DLQ routing
- Emit observability events on success and failure
- Delegate business logic entirely to handlers

**OmniMemory EFFECT nodes:**

| Directory | Handler | Responsibility |
|-----------|---------|----------------|
| `nodes/intent_event_consumer_effect/` | `HandlerIntentEventConsumer` | Consumes `intent-classified.v1` events from Kafka, persists to Memgraph via `HandlerIntentStorageAdapter` |
| `nodes/intent_storage_effect/` | `HandlerIntentStorageAdapter` | Writes intent records to the graph store |
| `nodes/intent_query_effect/` | `HandlerIntentQuery` | Queries stored intents, returns results over the event bus |
| `nodes/memory_retrieval_effect/` | `HandlerMemoryRetrieval` | Retrieves memory records from Qdrant, PostgreSQL, and Memgraph |
| `nodes/memory_storage_effect/` | `AdapterFilesystem` | Persists memory payloads to the filesystem adapter |

### COMPUTE Nodes

COMPUTE nodes transform data. They have no persistent side effects — all
external calls (embedding APIs, LLM endpoints) are mediated through injected
provider protocols, keeping the node logic deterministic and testable.

**Characteristics:**
- Accept strongly-typed request models, return response models
- Delegate to providers via `ProtocolEmbeddingProvider` / `ProtocolLLMProvider`
- Lazy-initialize handlers on first `execute()` call
- Convert all exceptions to error response objects — never raise to callers
- No direct Kafka or database imports

**OmniMemory COMPUTE nodes:**

| Directory | Node Class | Responsibility |
|-----------|------------|----------------|
| `nodes/semantic_analyzer_compute/` | `NodeSemanticAnalyzerCompute` | Generates embeddings, extracts entities, runs full semantic analysis |
| `nodes/similarity_compute/` | `NodeSimilarityCompute` | Computes vector similarity between memory records |

### REDUCER Nodes

REDUCER nodes aggregate state. They consume outputs from EFFECT and COMPUTE
nodes and produce consolidated views: statistics, merged records, rolled-up
counters.

**OmniMemory REDUCER nodes:**

| Directory | Responsibility |
|-----------|----------------|
| `nodes/statistics_reducer/` | Aggregates memory operation metrics and statistics |
| `nodes/memory_consolidator_reducer/` | Merges and deduplicates overlapping memory records |

> Both reducer nodes are currently scaffolded. Implementation is in progress as
> part of the Core 8 memory pipeline.

### ORCHESTRATOR Nodes

ORCHESTRATOR nodes coordinate multi-step workflows. They issue commands to
other nodes, manage lifecycle transitions, and coordinate cross-agent activity.
They do not perform I/O directly — they delegate to EFFECT nodes.

**OmniMemory ORCHESTRATOR nodes:**

| Directory | Handler | Responsibility |
|-----------|---------|----------------|
| `nodes/memory_lifecycle_orchestrator/` | `HandlerMemoryTick`, `HandlerMemoryArchive`, `HandlerMemoryExpire` | Drives time-based lifecycle: ticking expiry counters, archiving old memories, expiring stale records |
| `nodes/agent_coordinator_orchestrator/` | (in progress) | Routes memory requests across agents, coordinates cross-service memory operations |

---

## Naming Convention

All nodes follow the pattern `Node<Name><Type>` mapped to files named
`node_<name>_<type>.py`. Directories follow `<name>_<type>/`.

```
nodes/
  semantic_analyzer_compute/           # directory: <name>_<type>
    node_semantic_analyzer_compute.py  # file: node_<name>_<type>.py
    handlers/
      handler_semantic_compute.py
    models/
      model_semantic_analyzer_compute_request.py
      model_semantic_analyzer_compute_response.py
```

Handler class names follow `Handler<Name>` — they are not suffixed with the
node type because a handler may be shared across node types.

---

## Declaring an EFFECT Node

All ONEX nodes in omnimemory inherit from the base classes in
`omnimemory.nodes.base`, which provide container injection. The node itself
is a thin shell — all logic lives in the handler.

```python
# src/omnimemory/nodes/my_storage_effect/node_my_storage_effect.py

from omnimemory.nodes.base import BaseEffectNode, ContainerType
from .handlers import HandlerMyStorage
from .models import ModelMyStorageRequest, ModelMyStorageResponse


class NodeMyStorageEffect(BaseEffectNode):
    """EFFECT node for my-storage operations.

    Thin wrapper: all business logic lives in HandlerMyStorage.
    """

    def __init__(
        self,
        container: ContainerType,
        handler: HandlerMyStorage | None = None,
    ) -> None:
        super().__init__(container)
        self._handler = handler or HandlerMyStorage(container=container)

    async def execute(
        self,
        request: ModelMyStorageRequest,
    ) -> ModelMyStorageResponse:
        """Delegate to the handler. Convert exceptions to error responses."""
        try:
            return await self._handler.execute(request)
        except Exception as e:
            return ModelMyStorageResponse(
                status="error",
                error_message=f"{type(e).__name__}: {e}",
            )
```

The handler receives its dependencies via constructor injection:

```python
# src/omnimemory/nodes/my_storage_effect/handlers/handler_my_storage.py

from omnibase_core.container import ModelONEXContainer

from omnimemory.nodes.my_storage_effect.models import (
    ModelMyStorageRequest,
    ModelMyStorageResponse,
)


class HandlerMyStorage:
    def __init__(self, container: ModelONEXContainer) -> None:
        self._container = container
        self._initialized = False

    async def initialize(self, config: ModelMyStorageConfig) -> None:
        # set up DB connection, circuit breaker, etc.
        self._initialized = True

    async def execute(
        self,
        request: ModelMyStorageRequest,
    ) -> ModelMyStorageResponse:
        # Business logic here; no Kafka imports allowed (see ARCH-002)
        ...
```

---

## Runtime Wiring

The `MessageDispatchEngine` in `omnimemory.runtime.dispatch_handlers` wires
Kafka topics to the correct EFFECT node handlers at startup:

```python
from omnimemory.runtime.dispatch_handlers import create_memory_dispatch_engine

engine = create_memory_dispatch_engine(
    intent_consumer=intent_consumer_handler,
    intent_query_handler=intent_query_handler,
    publish_callback=event_bus.publish,
    publish_topics={"intent_query": "onex.evt.omnimemory.intent-query-response.v1"},  # Runtime env prefix (e.g., dev., prod.) is added by the topic builder
)
# engine is now frozen and ready for dispatch
```

The engine registers 4 handler slots covering 6 topic routes. EFFECT node
handlers receive deserialized `ModelEventEnvelope` objects; they never parse
raw Kafka bytes directly.

---

## Key Invariants

- **Handlers hold all logic.** Nodes are thin shells: `__init__` + `execute`.
- **No backwards data flow.** ORCHESTRATOR nodes do not call EFFECT nodes
  directly for reads — they issue commands that EFFECT nodes consume.
- **No direct transport imports in handlers.** See
  [ARCH-002 Kafka Abstraction](ARCH_002_KAFKA_ABSTRACTION.md).
- **All models are frozen Pydantic `BaseModel`.** No `Any` types.
- **Errors become response objects.** Nodes catch exceptions and return error
  response models; they do not propagate exceptions to callers.
