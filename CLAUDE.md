# OmniMemory - Advanced Memory Management System

**Status**: Active Development | **Architecture**: ONEX 4.0 Compliant | **Performance Target**: Sub-100ms Operations

## 🚫 CRITICAL POLICY: NO BACKWARDS COMPATIBILITY

**NEVER KEEP BACKWARDS COMPATIBILITY EVER EVER EVER**

This project follows a **ZERO BACKWARDS COMPATIBILITY** policy:
- **Breaking changes are always acceptable**
- **No deprecated code maintenance**
- **All models MUST conform to current protocols**
- **Clean, modern architecture only**
- **Remove old patterns immediately**

## Project Overview

OmniMemory is an advanced unified memory and intelligence system designed to migrate and modernize 274+ intelligence modules from legacy omnibase_3 into a comprehensive, ONEX-compliant memory architecture. This system accelerates development across all omni agents through systematic memory management, retrieval operations, and cross-modal intelligence patterns.

### Core Mission
- **Legacy Migration**: Modernize 52,880+ lines of intelligence code from ../omnibase_3/intelligence_tools/
- **ONEX Compliance**: Full adherence to ONEX 4.0 architectural standards and patterns
- **Unified Intelligence**: Create cohesive memory system serving all omni agents
- **Performance Excellence**: Achieve sub-100ms memory operations with 1M+ ops/hour capacity

## Architecture & Design Patterns

### ONEX Standards Compliance

OmniMemory strictly follows ONEX standards from omnibase_core:

**Directory Structure Standards:**
- ✅ **models/** directory (NOT core/) - all models in `src/omnimemory/models/`
- ✅ **Pydantic BaseModel** - all models inherit from `BaseModel`
- ✅ **Strong Typing** - zero `Any` types throughout codebase
- ✅ **Field Documentation** - `Field(..., description="...")` pattern
- ✅ **Domain Organization** - models organized by functional domain

**Current Model Structure:**
```
src/omnimemory/models/         # 26 Pydantic models, zero Any types
├── core/                      # Foundation models (4 models)
├── memory/                    # Memory-specific models (6 models)
├── intelligence/              # Intelligence processing (5 models)
├── service/                   # Service configuration (4 models)
├── container/                 # Container and DI models (4 models)
└── foundation/                # Base architectural models (3 models)
```

### ONEX 4-Node Architecture Integration

```
EFFECT → COMPUTE → REDUCER → ORCHESTRATOR
```

- **EFFECT Nodes**: Memory storage, retrieval, and persistence operations
- **COMPUTE Nodes**: Intelligence processing, semantic analysis, pattern recognition
- **REDUCER Nodes**: Memory consolidation, aggregation, and optimization
- **ORCHESTRATOR Nodes**: Cross-agent coordination and workflow management

### Memory System Architecture

```mermaid
graph TB
    A[Memory Manager] --> B[Vector Memory]
    A --> C[Temporal Memory]
    A --> D[Persistent Memory]
    A --> E[Cross-Modal Memory]

    B --> F[Pinecone Vector DB]
    C --> G[ModelCachingSubcontract]
    D --> H[PostgreSQL/Supabase]
    E --> I[Multi-Modal Index]

    J[Intelligence Tools] --> A
    K[Omni Agents] --> A
    L[Legacy omnibase_3] --> M[Migration Layer] --> A
```

### Core Dependencies & Technology Stack

**Storage Layer:**
- PostgreSQL + Supabase: Persistent memory and relational data
- ModelCachingSubcontract: Standardized high-speed caching and temporal memory patterns
- Pinecone: Vector-based semantic memory and similarity search

**Framework & API:**
- FastAPI: Production-ready API layer with async support
- SQLAlchemy + Alembic: Database ORM and migrations
- Pydantic: Data validation and serialization

**ONEX Integration:**
- omnibase_spi: Service Provider Interface for ONEX compliance
- omnibase_core: Core ONEX node implementations and patterns
- MCP Protocol: Agent communication and tool integration

## Development Workflow

### Memory System Design Patterns

1. **Memory Hierarchy**: Implement tiered memory (L1: ModelCachingSubcontract, L2: PostgreSQL, L3: Vector DB)
2. **Semantic Indexing**: Vector-based memory retrieval with similarity matching
3. **Temporal Patterns**: Time-aware memory decay and consolidation
4. **Cross-Modal Integration**: Multi-modal memory bridging different data types

### ONEX Compliance Requirements

- **Contract-Driven Development**: All interfaces defined via Pydantic models
- **Async-First Design**: Full async/await support for all operations
- **Error Recovery Patterns**: Circuit breakers, timeouts, graceful degradation
- **Observability**: Comprehensive logging, metrics, and health checks
- **Security-by-Design**: Input validation, PII detection, secure communication

## Migration Strategy

### Legacy Intelligence Tools Migration

**Source**: `../omnibase_3/intelligence_tools/` (274 Python files, 52,880+ lines)

**Migration Phases**:
1. **Analysis Phase**: Catalog existing tools, dependencies, and patterns
2. **Modernization Phase**: Refactor to ONEX patterns with proper typing
3. **Integration Phase**: Unified memory interface and cross-agent communication
4. **Validation Phase**: Performance testing and ONEX compliance verification

### Migration Patterns

```python
# Legacy Pattern (omnibase_3)
def process_intelligence(data):
    # Synchronous, no typing, direct file I/O
    with open('memory.json') as f:
        return json.load(f)

# Modern ONEX Pattern (omnimemory)
async def process_intelligence(data: IntelligenceRequest) -> IntelligenceResponse:
    """Process intelligence with ONEX compliance."""
    async with memory_manager.session() as session:
        result = await session.store_and_analyze(data)
        return IntelligenceResponse.model_validate(result)
```

## Performance & Quality Targets

### Performance Specifications
- **Memory Operations**: <100ms response time (95th percentile)
- **Throughput**: 1M+ operations per hour sustained
- **Storage Efficiency**: <10MB memory footprint per 100K records
- **Vector Search**: <50ms semantic similarity queries
- **Bulk Operations**: >10K records/second batch processing

### Quality Gates
- **ONEX Compliance**: 100% contract adherence with automated validation
- **Test Coverage**: >90% code coverage with integration tests
- **Type Safety**: Full mypy strict mode compliance
- **Security**: Automated secret detection and input sanitization
- **Documentation**: Comprehensive API docs with usage examples

## Development Guidelines

### Code Organization

```
src/omnimemory/
├── core/                 # Core memory interfaces and base classes
│   ├── memory_manager.py
│   ├── interfaces.py
│   └── exceptions.py
├── storage/             # Storage layer implementations
│   ├── vector_store.py
│   ├── temporal_store.py
│   └── persistent_store.py
├── intelligence/        # Migrated intelligence tools
│   ├── analysis/
│   ├── patterns/
│   └── retrieval/
├── api/                 # FastAPI endpoints
│   ├── routes/
│   └── schemas/
└── migration/           # Legacy migration utilities
    ├── extractors/
    └── transformers/
```

### Development Commands

```bash
# Setup and development
poetry install                    # Install dependencies
poetry run pre-commit install   # Setup pre-commit hooks

# Code quality
poetry run black .              # Format code
poetry run isort .              # Sort imports
poetry run mypy src/            # Type checking
poetry run flake8 src/          # Linting

# Testing
poetry run pytest              # Run all tests
poetry run pytest -m unit     # Unit tests only
poetry run pytest -m integration  # Integration tests
poetry run pytest --cov       # Coverage report

# Migration tools
poetry run python scripts/migrate_intelligence.py  # Migrate legacy tools
poetry run python scripts/validate_onex.py         # ONEX compliance check
```

### Environment Configuration

```bash
# Database connections
DATABASE_URL="postgresql://user:pass@localhost:5432/omnimemory"
SUPABASE_URL="https://your-project.supabase.co"
SUPABASE_ANON_KEY="your-anon-key"

# Vector database
PINECONE_API_KEY="your-pinecone-key"
PINECONE_ENVIRONMENT="your-environment"

# Memory cache configuration (using ModelCachingSubcontract)
MEMORY_CACHE_ENABLED="true"

# ONEX integration
OMNIBASE_SPI_VERSION="latest"
OMNIBASE_CORE_VERSION="latest"

# Development settings
DEVELOPMENT_MODE="true"
LOG_LEVEL="INFO"
MEMORY_CACHE_SIZE="1000"
```

## Integration Patterns

### Agent Integration

```python
from omnimemory import MemoryManager, VectorMemory

class IntelligentAgent:
    def __init__(self):
        self.memory = MemoryManager()
        self.vector_memory = VectorMemory()

    async def process_with_memory(self, input_data: str) -> str:
        # Retrieve relevant memories
        context = await self.vector_memory.similarity_search(
            query=input_data,
            limit=5,
            threshold=0.8
        )

        # Process with context
        result = await self.analyze_with_context(input_data, context)

        # Store new memory
        await self.memory.store(
            key=f"processed_{datetime.now().isoformat()}",
            value=result,
            metadata={"source": "agent_processing"}
        )

        return result
```

### MCP Tool Integration

```python
@mcp_tool("omnimemory_store")
async def mcp_memory_store(
    key: str,
    value: str,
    memory_type: str = "persistent"
) -> Dict[str, Any]:
    """Store information in OmniMemory system."""
    async with get_memory_manager() as memory:
        result = await memory.store(key, value, memory_type)
        return {"success": True, "memory_id": result.id}

@mcp_tool("omnimemory_retrieve")
async def mcp_memory_retrieve(
    query: str,
    limit: int = 10,
    similarity_threshold: float = 0.7
) -> Dict[str, Any]:
    """Retrieve memories using semantic search."""
    async with get_memory_manager() as memory:
        results = await memory.semantic_search(query, limit, similarity_threshold)
        return {"memories": [r.to_dict() for r in results]}
```

## Troubleshooting & Common Issues

### Performance Optimization

- **Slow Vector Search**: Increase Pinecone index replicas, optimize embedding dimensions
- **Memory Leaks**: Use async context managers, implement proper cleanup in finally blocks
- **Database Connections**: Configure connection pooling, implement circuit breakers
- **Cache Misses**: Tune ModelCachingSubcontract configuration, implement intelligent prefetching

### Migration Issues

- **Legacy Code Compatibility**: Use adapter pattern for gradual migration
- **Type Errors**: Implement progressive typing with `# type: ignore` for interim compatibility
- **Dependency Conflicts**: Pin specific versions, use Poetry dependency groups
- **Performance Regression**: Implement benchmarking suite for before/after comparisons

### ONEX Compliance

- **Contract Validation**: Use Pydantic strict mode, implement custom validators
- **Async Patterns**: Ensure all I/O operations are async, avoid blocking calls
- **Error Handling**: Implement proper exception hierarchies, use structured logging
- **Security**: Enable all pre-commit hooks, implement input sanitization

## Success Metrics & Monitoring

### Key Performance Indicators

- **Migration Progress**: Track completion percentage of 274 intelligence tools
- **Performance Metrics**: Monitor response times, throughput, and error rates
- **ONEX Compliance**: Automated compliance scoring and validation
- **Agent Integration**: Number of omni agents successfully integrated
- **Memory Efficiency**: Storage utilization and optimization ratios

### Monitoring & Observability

```python
from omnimemory.monitoring import PerformanceMonitor, ComplianceTracker

# Performance monitoring
monitor = PerformanceMonitor()
await monitor.track_operation("memory_store", duration_ms=45)

# ONEX compliance tracking
compliance = ComplianceTracker()
score = await compliance.evaluate_operation(operation_result)
```

## Future Roadmap

### Phase 1: Foundation (Current)
- Core memory architecture implementation
- Basic ONEX compliance patterns
- Initial intelligence tool migration

### Phase 2: Intelligence Integration
- Advanced semantic search capabilities
- Cross-modal memory bridging
- Performance optimization and scaling

### Phase 3: Agent Ecosystem
- Full omni agent integration
- Advanced workflow orchestration
- Real-time collaboration patterns

### Phase 4: Advanced Intelligence
- Machine learning-enhanced memory patterns
- Predictive memory prefetching
- Self-optimizing memory hierarchies

---

**Project Repository**: `/Volumes/PRO-G40/Code/omnimemory`
**ONEX Version**: 4.0+
**Python Version**: 3.12+
**Last Updated**: 2025-09-13

For questions or contributions, refer to the project documentation in the `docs/` directory or contact the OmniNode-ai development team.