# omnimemory

Memory persistence, recall, and semantic retrieval for the OmniNode platform.

[![CI](https://github.com/OmniNode-ai/omnimemory/actions/workflows/test.yml/badge.svg)](https://github.com/OmniNode-ai/omnimemory/actions/workflows/test.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Install

```bash
uv add omnimemory
```

## Minimal Example

```python
from omnimemory.nodes.memory_storage_effect.node import NodeMemoryStorageEffect

node = NodeMemoryStorageEffect(container=container)
result = await node.execute(input_data)
```

## Architecture

Follows the [ONEX Four-Node Architecture](https://github.com/OmniNode-ai/omnibase_core/blob/main/docs/architecture/ONEX_FOUR_NODE_ARCHITECTURE.md) (EFFECT, COMPUTE, REDUCER, ORCHESTRATOR) applied to memory operations.

## Key Features

- **Memory nodes**: Storage, retrieval, intent query, and event consumer effects
- **Compute nodes**: Semantic analysis, similarity scoring, embedding generation
- **Reducer nodes**: Memory consolidation, statistics aggregation
- **Orchestrator nodes**: Memory lifecycle, agent coordination
- **Intent system**: Handler-driven intent classification and subscription
- **Vector storage**: Qdrant for semantic search, PostgreSQL for metadata
- **PII detection**: Built-in PII scanning before storage

## Documentation

- [Architecture](docs/architecture/)
- [Stub protocols](docs/stub_protocols.md)
- [PII handling](docs/pii_handling.md)
- [CLAUDE.md](CLAUDE.md) -- developer context and conventions
- [AGENT.md](AGENT.md) -- LLM navigation guide

## License

[MIT](LICENSE)
