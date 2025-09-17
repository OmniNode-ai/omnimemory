# Qdrant Adapter Effect Node

## Overview

The Qdrant Adapter Effect Node provides a bridge between OmniMemory's event-driven architecture and the Qdrant vector database. It follows the ONEX infrastructure pattern, converting message bus envelopes containing vector requests into direct Qdrant client operations.

## Architecture

```
OmniMemory Event Bus → Qdrant Adapter → Qdrant Client → Vector Database
                            ↓
                    Event Publishing → Completion Events
```

### Message Flow

1. **Command Reception**: Receives vector operation commands from RedPanda topics like:
   - `dev.omnibase.onex.cmd.omnimemory-vector-search.v1`
   - `dev.omnibase.onex.cmd.omnimemory-store-memory.v1`

2. **Operation Processing**: Converts events to Qdrant operations:
   - Vector search with similarity matching
   - Vector storage (upsert operations)
   - Vector retrieval by ID
   - Batch operations
   - Collection management

3. **Event Publishing**: Publishes completion events to topics like:
   - `dev.omnibase.onex.evt.omnimemory-search-completed.v1`
   - `dev.omnibase.onex.evt.omnimemory-memory-stored.v1`

## Features

### Core Operations

- **Vector Search**: Similarity search with configurable thresholds and filters
- **Vector Storage**: Upsert operations with metadata and payload support
- **Vector Retrieval**: Get specific vectors by ID
- **Batch Operations**: Efficient bulk upsert and delete operations
- **Collection Management**: Create and manage Qdrant collections
- **Health Checks**: Comprehensive connectivity and operational health monitoring

### ONEX Compliance

- **Circuit Breaker**: Prevents cascade failures with configurable thresholds
- **Structured Logging**: Correlation ID tracking and performance metrics
- **Error Sanitization**: Prevents sensitive information leakage
- **Event Publishing**: Full integration with OmniMemory event bus
- **Health Monitoring**: Multi-dimensional health checks
- **Resource Management**: Proper async cleanup and resource lifecycle

### Security Features

- **Input Validation**: Request validation and dimension limits
- **Error Sanitization**: Automatic removal of sensitive information from logs
- **Circuit Breaker**: Protection against cascade failures
- **Timeout Management**: Configurable timeouts for all operations
- **Correlation ID Validation**: UUID format validation and injection prevention

## Configuration

### Environment Variables

```bash
# Qdrant Connection
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_URL=http://localhost:6333  # Optional, overrides host/port
QDRANT_API_KEY=your_api_key       # Optional for authentication
QDRANT_TIMEOUT_SECONDS=30

# Performance Settings
QDRANT_CONNECTION_POOL_SIZE=10
QDRANT_MAX_RETRIES=3
QDRANT_RETRY_DELAY_SECONDS=1.0

# Security Settings
QDRANT_ENABLE_ERROR_SANITIZATION=true
QDRANT_ENABLE_REQUEST_VALIDATION=true

# Circuit Breaker Settings
QDRANT_CIRCUIT_BREAKER_FAILURE_THRESHOLD=5
QDRANT_CIRCUIT_BREAKER_TIMEOUT_SECONDS=60
QDRANT_CIRCUIT_BREAKER_HALF_OPEN_MAX_CALLS=3

# Vector Settings
QDRANT_DEFAULT_COLLECTION=omnimemory_vectors
QDRANT_DEFAULT_VECTOR_SIZE=1536
QDRANT_DEFAULT_DISTANCE_METRIC=Cosine
QDRANT_MAX_VECTOR_DIMENSIONS=4096
QDRANT_MAX_BATCH_SIZE=100
QDRANT_MAX_SEARCH_LIMIT=1000
```

### Environment-Specific Configurations

#### Development
- Error sanitization disabled for debugging
- Higher failure thresholds
- Extended timeouts
- Smaller connection pools

#### Production
- Error sanitization enabled
- Strict failure thresholds
- Optimized timeouts
- Larger connection pools

## Usage Examples

### Vector Search Operation

```python
from omnimemory.nodes.node_qdrant_adapter_effect.v1_0_0.models import (
    ModelQdrantAdapterInput,
    ModelQdrantVectorOperationRequest,
)

# Create vector search request
vector_request = ModelQdrantVectorOperationRequest(
    operation_type="vector_search",
    collection_name="omnimemory_vectors",
    query_vector=[0.1, 0.2, 0.3, 0.4],
    search_limit=10,
    score_threshold=0.8,
    with_payload=True,
    with_vector=False,
    search_filter={
        "category": "document",
        "created_date": {"gte": "2024-01-01"}
    }
)

input_data = ModelQdrantAdapterInput(
    operation_type="vector_search",
    correlation_id=uuid4(),
    vector_request=vector_request
)

# Process the request
result = await qdrant_adapter.process(input_data)

# Handle results
if result.success:
    search_result = result.search_result
    print(f"Found {search_result.total_count} results")
    for point in search_result.points:
        print(f"ID: {point.id}, Score: {point.score}")
```

### Vector Storage Operation

```python
# Create vector storage request
vector_request = ModelQdrantVectorOperationRequest(
    operation_type="store_vector",
    collection_name="omnimemory_vectors",
    vector_id="doc_123",
    vector_data=[0.1, 0.2, 0.3, 0.4],
    payload={
        "text": "Sample document content",
        "category": "documentation",
        "created_date": "2024-09-15",
        "author": "system"
    }
)

input_data = ModelQdrantAdapterInput(
    operation_type="store_vector",
    correlation_id=uuid4(),
    vector_request=vector_request
)

# Process the request
result = await qdrant_adapter.process(input_data)

if result.success:
    print("Vector stored successfully")
```

### Health Check Operation

```python
# Create health check request
input_data = ModelQdrantAdapterInput(
    operation_type="health_check",
    correlation_id=uuid4(),
    health_check_type="basic",
    include_collections_info=True
)

# Process the request
result = await qdrant_adapter.process(input_data)

if result.health_status:
    print(f"Health: {result.health_status.status}")
    print(f"Connection: {result.health_status.connection_status}")
    print(f"Response Time: {result.health_status.response_time_ms}ms")
```

## Integration with OmniMemory

### Event Publishing

The adapter automatically publishes events for all operations:

```python
# Vector search completed events
topic: dev.omnibase.onex.evt.omnimemory-search-completed.v1
data: {
    "operation_type": "vector_search",
    "correlation_id": "uuid",
    "success": true,
    "vector_search_data": {
        "result_count": 5,
        "search_time_ms": 150.0,
        "similarity_threshold": 0.8,
        "results": [...]
    }
}

# Memory stored events
topic: dev.omnibase.onex.evt.omnimemory-memory-stored.v1
data: {
    "operation_type": "store_memory",
    "correlation_id": "uuid",
    "success": true,
    "store_data": {
        "memory_key": "doc_123",
        "memory_type": "vector",
        "vector_dimensions": 4,
        "storage_size": 16
    }
}
```

### Service Integration

```python
from omnibase_core.core.onex_container import ModelONEXContainer
from omnimemory.nodes.node_qdrant_adapter_effect import NodeQdrantAdapterEffect

# Create container with event bus
container = ModelONEXContainer()
container.register_service("ProtocolEventBus", your_event_bus)

# Initialize adapter
adapter = NodeQdrantAdapterEffect(container)
await adapter.initialize()

# Start service mode (listens for events automatically)
await adapter.start_service_mode()
```

## Monitoring and Observability

### Health Checks

The adapter provides multiple health check endpoints:

1. **Qdrant Connectivity**: Tests connection to Qdrant server
2. **Circuit Breaker Status**: Reports circuit breaker state
3. **Vector Operations**: Validates vector operation capabilities
4. **Event Publishing**: Confirms event bus connectivity

### Structured Logging

All operations include structured logging with:

- **Correlation ID**: Request tracing across services
- **Performance Metrics**: Execution times and resource usage
- **Error Context**: Detailed error information with sanitization
- **Circuit Breaker Events**: State changes and recovery attempts

### Metrics

Key metrics tracked:

- **Operation Latency**: P50, P95, P99 response times
- **Throughput**: Operations per second
- **Error Rates**: By operation type and error category
- **Circuit Breaker**: State transitions and failure counts
- **Vector Dimensions**: Distribution of vector sizes
- **Search Performance**: Results count and relevance scores

## Error Handling

### Circuit Breaker States

- **CLOSED**: Normal operation, all requests processed
- **OPEN**: Failures exceed threshold, requests rejected
- **HALF_OPEN**: Testing recovery, limited requests allowed

### Error Categories

- **Connectivity**: Network and connection issues
- **Authorization**: Authentication and permission errors
- **Validation**: Input validation and format errors
- **Resource**: Collection not found or resource constraints
- **Unknown**: Uncategorized errors

### Recovery Strategies

- **Automatic Retry**: Exponential backoff for transient failures
- **Circuit Breaking**: Temporary service isolation
- **Graceful Degradation**: Reduced functionality during issues
- **Event Publishing**: Error events for monitoring integration

## Performance Optimization

### Best Practices

1. **Vector Dimensions**: Use consistent dimensions (e.g., 1536 for embeddings)
2. **Batch Operations**: Use batch upsert for multiple vectors
3. **Search Filters**: Use payload filters to reduce search space
4. **Collection Design**: Separate collections by use case
5. **Connection Pooling**: Configure appropriate pool sizes
6. **Circuit Breaker**: Tune thresholds for your traffic patterns

### Configuration Tuning

- **Development**: Lower thresholds, extended timeouts, debug logging
- **Staging**: Production-like settings with monitoring
- **Production**: Optimized for performance and reliability

## Testing

### Unit Tests

```bash
# Run adapter tests
pytest tests/test_qdrant_adapter_integration.py

# Run with coverage
pytest --cov=omnimemory.nodes.node_qdrant_adapter_effect tests/
```

### Integration Tests

```bash
# Start test Qdrant instance
docker run -p 6333:6333 qdrant/qdrant

# Run integration tests
pytest tests/integration/test_qdrant_adapter.py
```

### Load Testing

```bash
# Run performance tests
pytest tests/performance/test_qdrant_adapter_load.py
```

## Troubleshooting

### Common Issues

1. **Connection Refused**: Check Qdrant server status and configuration
2. **Circuit Breaker Open**: Monitor error rates and server health
3. **Dimension Mismatch**: Ensure vector dimensions match collection config
4. **High Latency**: Check network connectivity and server resources
5. **Event Publishing Failures**: Verify RedPanda connectivity

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
export QDRANT_ENABLE_ERROR_SANITIZATION=false

# Run with detailed logging
python -m omnimemory.nodes.node_qdrant_adapter_effect
```

### Health Check Endpoints

```bash
# Check adapter health
curl http://localhost:8080/health

# Check detailed health
curl http://localhost:8080/health?include_collections=true
```

## Dependencies

### Required

- `qdrant-client`: Qdrant Python client library
- `omnibase-core`: ONEX core framework
- `pydantic`: Data validation and serialization
- `asyncio`: Asynchronous programming support

### Optional

- `prometheus-client`: Metrics collection
- `structlog`: Enhanced structured logging
- `aioredis`: Redis integration for caching

## Version History

### v1.0.0

- Initial implementation with core vector operations
- ONEX compliance with circuit breaker and structured logging
- Event publishing integration with OmniMemory
- Comprehensive health checks and monitoring
- Error sanitization and security features
- Configuration management for multiple environments