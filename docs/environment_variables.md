# OmniMemory Environment Variables

This document describes all environment variables used to configure OmniMemory.

## Overview

OmniMemory uses [pydantic-settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) for configuration management. All environment variables use the `OMNIMEMORY__` prefix with `__` as the nested delimiter.

**Example**: `OMNIMEMORY__FILESYSTEM__BASE_PATH=/data/memory`

## Configuration Hierarchy

```
SettingsMemoryService (top-level)
|-- postgres_enabled (bool)
|-- qdrant_enabled (bool)
|-- service_name (str)
|-- enable_metrics (bool)
|-- enable_logging (bool)
|-- debug_mode (bool)
|
|-- filesystem (required)
|   |-- base_path (required)
|   |-- max_file_size_bytes
|   |-- allowed_extensions
|   |-- create_if_missing
|   |-- enable_compression
|   +-- buffer_size_bytes
|
|-- postgres (optional, requires postgres_enabled=true)
|   |-- dsn (required if enabled)
|   |-- password (required if enabled)
|   |-- pool_size
|   |-- pool_timeout_seconds
|   |-- pool_recycle_seconds
|   |-- statement_timeout_seconds
|   |-- lock_timeout_seconds
|   |-- ssl_mode
|   +-- schema_name
|
|-- qdrant (optional, requires qdrant_enabled=true)
|   |-- url
|   |-- api_key
|   |-- collection_name
|   |-- vector_size
|   |-- timeout_seconds
|   |-- grpc_port
|   |-- prefer_grpc
|   |-- default_limit
|   |-- score_threshold
|   |-- distance_metric
|   +-- on_disk
|
+-- embedding (optional, requires embedding_enabled=true)
    |-- server_url (REQUIRED - no default)
    |-- timeout_seconds
    |-- max_retries
    +-- dimension
```

## Required Variables (Phase 1 Minimal)

These variables MUST be set for OmniMemory to start:

| Variable | Type | Description | Example |
|----------|------|-------------|---------|
| `OMNIMEMORY__FILESYSTEM__BASE_PATH` | Path | Base directory for memory file storage (must be absolute) | `/data/omnimemory` |

## Service-Level Configuration

Top-level settings that control service behavior.

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `OMNIMEMORY__POSTGRES_ENABLED` | bool | `false` | Enable PostgreSQL backend |
| `OMNIMEMORY__QDRANT_ENABLED` | bool | `false` | Enable Qdrant vector backend |
| `OMNIMEMORY__EMBEDDING_ENABLED` | bool | `false` | Enable real embedding server (requires `EMBEDDING__SERVER_URL`) |
| `OMNIMEMORY__SERVICE_NAME` | str | `omnimemory` | Name of the memory service instance |
| `OMNIMEMORY__ENABLE_METRICS` | bool | `true` | Enable performance metrics collection |
| `OMNIMEMORY__ENABLE_LOGGING` | bool | `true` | Enable operation logging |
| `OMNIMEMORY__DEBUG_MODE` | bool | `false` | Enable debug mode for verbose output |

## Filesystem Configuration

Filesystem backend is **required** for Phase 1. All memory operations use filesystem storage as the foundation.

| Variable | Type | Default | Constraints | Description |
|----------|------|---------|-------------|-------------|
| `OMNIMEMORY__FILESYSTEM__BASE_PATH` | Path | **required** | Must be absolute | Base directory for memory storage |
| `OMNIMEMORY__FILESYSTEM__MAX_FILE_SIZE_BYTES` | int | `10485760` (10MB) | 1 - 1073741824 (1GB) | Maximum file size in bytes |
| `OMNIMEMORY__FILESYSTEM__ALLOWED_EXTENSIONS` | list | `[".json", ".txt", ".md"]` | JSON array format | Allowed file extensions (e.g., `'[".json", ".txt"]'`) |
| `OMNIMEMORY__FILESYSTEM__CREATE_IF_MISSING` | bool | `true` | - | Create base_path if it doesn't exist |
| `OMNIMEMORY__FILESYSTEM__ENABLE_COMPRESSION` | bool | `false` | - | Enable gzip compression for stored files |
| `OMNIMEMORY__FILESYSTEM__BUFFER_SIZE_BYTES` | int | `65536` (64KB) | 4096 - 1048576 (1MB) | I/O buffer size in bytes |

## PostgreSQL Configuration (Optional)

Set `OMNIMEMORY__POSTGRES_ENABLED=true` to enable PostgreSQL backend for persistent memory storage.

| Variable | Type | Default | Constraints | Description |
|----------|------|---------|-------------|-------------|
| `OMNIMEMORY__POSTGRES__DSN` | PostgresDsn | **required if enabled** | Valid PostgreSQL DSN | PostgreSQL connection DSN |
| `OMNIMEMORY__POSTGRES__PASSWORD` | SecretStr | **required if enabled** | - | Database password (stored securely) |
| `OMNIMEMORY__POSTGRES__POOL_SIZE` | int | `5` | 1 - 50 | Connection pool size |
| `OMNIMEMORY__POSTGRES__POOL_TIMEOUT_SECONDS` | int | `30` | 1 - 300 | Pool connection acquisition timeout |
| `OMNIMEMORY__POSTGRES__POOL_RECYCLE_SECONDS` | int | `3600` | 60 - 86400 | Connection recycle time in seconds |
| `OMNIMEMORY__POSTGRES__STATEMENT_TIMEOUT_SECONDS` | int | `30` | 1 - 300 | Maximum query execution time |
| `OMNIMEMORY__POSTGRES__LOCK_TIMEOUT_SECONDS` | int | `10` | 1 - 60 | Maximum time to wait for locks |
| `OMNIMEMORY__POSTGRES__SSL_MODE` | str | `prefer` | Valid SSL mode | SSL mode (disable, allow, prefer, require, verify-ca, verify-full) |
| `OMNIMEMORY__POSTGRES__SCHEMA_NAME` | str | `omnimemory` | - | PostgreSQL schema name for memory tables |

**DSN Format**: `postgresql://user@host:port/database`

Note: The password is specified separately via `OMNIMEMORY__POSTGRES__PASSWORD` and not included in the DSN for security.

## Qdrant Configuration (Optional)

Set `OMNIMEMORY__QDRANT_ENABLED=true` to enable Qdrant vector backend for semantic memory storage.

| Variable | Type | Default | Constraints | Description |
|----------|------|---------|-------------|-------------|
| `OMNIMEMORY__QDRANT__URL` | HttpUrl | `http://localhost:6333` | Valid HTTP(S) URL | Qdrant server URL |
| `OMNIMEMORY__QDRANT__API_KEY` | SecretStr | `None` | - | Qdrant API key (optional) |
| `OMNIMEMORY__QDRANT__COLLECTION_NAME` | str | `omnimemory` | - | Default collection name for memory vectors |
| `OMNIMEMORY__QDRANT__VECTOR_SIZE` | int | `1024` | 1 - 65536 | Vector embedding dimensions |
| `OMNIMEMORY__QDRANT__TIMEOUT_SECONDS` | int | `30` | 1 - 300 | Request timeout in seconds |
| `OMNIMEMORY__QDRANT__GRPC_PORT` | int | `None` | 1 - 65535 | gRPC port for high-performance operations |
| `OMNIMEMORY__QDRANT__PREFER_GRPC` | bool | `false` | - | Prefer gRPC over HTTP for operations |
| `OMNIMEMORY__QDRANT__DEFAULT_LIMIT` | int | `10` | 1 - 1000 | Default number of results to return |
| `OMNIMEMORY__QDRANT__SCORE_THRESHOLD` | float | `0.7` | 0.0 - 1.0 | Minimum similarity score threshold |
| `OMNIMEMORY__QDRANT__DISTANCE_METRIC` | str | `Cosine` | Cosine, Euclid, Dot | Distance metric for vector similarity |
| `OMNIMEMORY__QDRANT__ON_DISK` | bool | `false` | - | Store vectors on disk instead of RAM |

## Embedding Configuration (Optional)

Set `OMNIMEMORY__EMBEDDING_ENABLED=true` to enable the real embedding server for semantic search operations. This is used by the memory retrieval handlers when `use_real_embeddings=True`.

**IMPORTANT**: The `SERVER_URL` has NO default value and MUST be provided explicitly. This prevents accidental use of wrong/hardcoded URLs.

| Variable | Type | Default | Constraints | Description |
|----------|------|---------|-------------|-------------|
| `OMNIMEMORY__EMBEDDING__SERVER_URL` | str | **REQUIRED** | Valid HTTP(S) URL | URL of the embedding server (e.g., MLX server) |
| `OMNIMEMORY__EMBEDDING__TIMEOUT_SECONDS` | float | `5.0` | > 0 | Request timeout in seconds |
| `OMNIMEMORY__EMBEDDING__MAX_RETRIES` | int | `3` | 0 - 10 | Maximum retry attempts for transient failures |
| `OMNIMEMORY__EMBEDDING__DIMENSION` | int | `1024` | > 0 | Expected embedding vector dimension |

**Example**:
```bash
# Enable real embeddings with explicit server URL
export OMNIMEMORY__EMBEDDING_ENABLED=true
export OMNIMEMORY__EMBEDDING__SERVER_URL=http://192.168.86.200:8102
```

**Note**: When using `HandlerQdrantMockConfig` with `use_real_embeddings=True`, the `embedding_server_url` must be provided explicitly (typically loaded from this environment variable). The handler will fail fast with a clear error if the URL is missing or invalid.

## Example Configurations

### Minimal (Phase 1 Local Development)

```bash
# .env.local
OMNIMEMORY__FILESYSTEM__BASE_PATH=/tmp/omnimemory
```

### Development with All Backends

```bash
# .env.development
OMNIMEMORY__FILESYSTEM__BASE_PATH=/data/omnimemory
OMNIMEMORY__FILESYSTEM__CREATE_IF_MISSING=true
OMNIMEMORY__FILESYSTEM__ENABLE_COMPRESSION=false

# Service settings
OMNIMEMORY__SERVICE_NAME=omnimemory-dev
OMNIMEMORY__DEBUG_MODE=true

# PostgreSQL
OMNIMEMORY__POSTGRES_ENABLED=true
OMNIMEMORY__POSTGRES__DSN=postgresql://omnimemory@localhost:5432/omnimemory_dev
OMNIMEMORY__POSTGRES__PASSWORD=dev_password
OMNIMEMORY__POSTGRES__POOL_SIZE=3
OMNIMEMORY__POSTGRES__SCHEMA_NAME=omnimemory_dev

# Qdrant
OMNIMEMORY__QDRANT_ENABLED=true
OMNIMEMORY__QDRANT__URL=http://localhost:6333
OMNIMEMORY__QDRANT__COLLECTION_NAME=omnimemory_dev
OMNIMEMORY__QDRANT__VECTOR_SIZE=1024

# Embedding (for real semantic search)
OMNIMEMORY__EMBEDDING_ENABLED=true
OMNIMEMORY__EMBEDDING__SERVER_URL=http://192.168.86.200:8102
```

### Production

```bash
# .env.production
OMNIMEMORY__FILESYSTEM__BASE_PATH=/var/lib/omnimemory
OMNIMEMORY__FILESYSTEM__MAX_FILE_SIZE_BYTES=52428800  # 50MB
OMNIMEMORY__FILESYSTEM__CREATE_IF_MISSING=false
OMNIMEMORY__FILESYSTEM__ENABLE_COMPRESSION=true
OMNIMEMORY__FILESYSTEM__BUFFER_SIZE_BYTES=131072  # 128KB

# Service settings
OMNIMEMORY__SERVICE_NAME=omnimemory-prod
OMNIMEMORY__ENABLE_METRICS=true
OMNIMEMORY__ENABLE_LOGGING=true
OMNIMEMORY__DEBUG_MODE=false

# PostgreSQL (credentials via Vault in production - see Secrets Management)
OMNIMEMORY__POSTGRES_ENABLED=true
OMNIMEMORY__POSTGRES__DSN=postgresql://omnimemory@db.prod.internal:5432/omnimemory
OMNIMEMORY__POSTGRES__PASSWORD=<from-vault>
OMNIMEMORY__POSTGRES__POOL_SIZE=20
OMNIMEMORY__POSTGRES__POOL_TIMEOUT_SECONDS=10
OMNIMEMORY__POSTGRES__POOL_RECYCLE_SECONDS=1800
OMNIMEMORY__POSTGRES__STATEMENT_TIMEOUT_SECONDS=60
OMNIMEMORY__POSTGRES__SSL_MODE=require
OMNIMEMORY__POSTGRES__SCHEMA_NAME=omnimemory

# Qdrant
OMNIMEMORY__QDRANT_ENABLED=true
OMNIMEMORY__QDRANT__URL=https://qdrant.prod.internal:6333
OMNIMEMORY__QDRANT__API_KEY=<from-vault>
OMNIMEMORY__QDRANT__COLLECTION_NAME=omnimemory_prod
OMNIMEMORY__QDRANT__VECTOR_SIZE=1024
OMNIMEMORY__QDRANT__TIMEOUT_SECONDS=60
OMNIMEMORY__QDRANT__PREFER_GRPC=true
OMNIMEMORY__QDRANT__GRPC_PORT=6334
OMNIMEMORY__QDRANT__ON_DISK=true
```

## Type Reference

| Type | Description | Example |
|------|-------------|---------|
| `Path` | Filesystem path (must be absolute) | `/data/omnimemory` |
| `PostgresDsn` | PostgreSQL connection string | `postgresql://user@host:5432/db` |
| `HttpUrl` | HTTP(S) URL | `http://localhost:6333` |
| `SecretStr` | Sensitive string (not logged) | Password, API key |
| `bool` | Boolean (`true`/`false`, `1`/`0`, `yes`/`no`) | `true` |
| `int` | Integer | `10485760` |
| `float` | Floating point number | `0.7` |
| `str` | String | `omnimemory` |
| `list` | JSON array format (see note below) | `[".json", ".txt"]` |

**Note on List Values**: List-type environment variables must use JSON array format. Comma-separated strings (e.g., `.json,.txt,.md`) are **not supported** by pydantic-settings by default.

```bash
# Correct - JSON array format
export OMNIMEMORY__FILESYSTEM__ALLOWED_EXTENSIONS='[".json", ".txt", ".md"]'

# Incorrect - comma-separated (will fail to parse as a list)
export OMNIMEMORY__FILESYSTEM__ALLOWED_EXTENSIONS=".json,.txt,.md"
```

> **Note**: Comma-separated value support could be added using a `@field_validator` with `mode='before'` to parse comma-separated strings. The current implementation uses the pydantic-settings v2 default behavior which expects JSON array format for list-type fields.

## Secrets Management

### Phase 1: Environment Variables

Secrets are loaded from environment variables with `SecretStr` type to prevent accidental logging.

```python
# In code - password is never exposed
config.postgres.password  # SecretStr('**********')
config.postgres.password.get_secret_value()  # Only way to access raw value
```

The `SecretStr` type ensures that:
- Secrets are never printed in logs or stack traces
- `repr()` and `str()` show masked values
- Explicit `.get_secret_value()` call required to access the actual secret

### Future: Vault Integration

OmniMemory is designed to support HashiCorp Vault via `ProtocolSecretsProvider`. The interface is defined in `src/omnimemory/protocols/secrets_provider.py`.

```python
from omnimemory.protocols import ProtocolSecretsProvider

class VaultSecretsProvider:
    """Future Vault implementation."""

    async def get_secret(self, key: str) -> SecretStr:
        # Fetch from Vault
        ...

    async def get_secret_or_default(self, key: str, default: str) -> SecretStr:
        # Fetch with fallback
        ...

    async def has_secret(self, key: str) -> bool:
        # Check existence
        ...
```

## Validation Behavior

### Fail-Fast

Missing required configuration fails immediately at startup with clear error messages:

```
pydantic.ValidationError: 1 validation error for FilesystemSettings
OMNIMEMORY__FILESYSTEM__BASE_PATH
  Field required [type=missing, input_value={}, input_type=dict]
```

### Extra Variables Rejected

Unknown environment variables with the relevant prefix are rejected (`extra="forbid"`):

```python
# This will raise ValidationError
OMNIMEMORY__UNKNOWN_SETTING=value
# Error: extra fields not permitted

# Typos are also caught
OMNIMEMORY__FILESYTEM__BASE_PATH=/data  # Note: FILESYTEM vs FILESYSTEM
# Error: extra fields not permitted
```

This prevents typos from silently being ignored and ensures configuration correctness.

### Constraint Validation

All fields with constraints are validated:

```python
# Pool size out of range
OMNIMEMORY__POSTGRES__POOL_SIZE=100
# Error: Input should be less than or equal to 50

# Invalid URL scheme
OMNIMEMORY__QDRANT__URL=ftp://localhost:6333
# Error: URL scheme should be 'http' or 'https'

# Negative buffer size
OMNIMEMORY__FILESYSTEM__BUFFER_SIZE_BYTES=-1
# Error: Input should be greater than or equal to 4096
```

## Bootstrap Validation

Beyond settings validation, the `bootstrap()` function performs runtime validation:

1. **Filesystem**: Verifies `base_path` exists (or creates it if `create_if_missing=true`), is a directory, and is writable
2. **PostgreSQL**: Validates DSN format (does not test actual connection at bootstrap)
3. **Qdrant**: Validates URL format, warns if API key missing

```python
from omnimemory import load_settings, bootstrap

# Step 1: Load and validate settings from environment
settings = load_settings()  # Validates env vars, raises ValidationError on failure

# Step 2: Convert to config model
config = settings.to_config()

# Step 3: Bootstrap validates runtime requirements
result = await bootstrap(config)

if result.success:
    print(f"Initialized backends: {result.initialized_backends}")

if result.warnings:
    for warning in result.warnings:
        print(f"Warning: {warning}")
```

### Bootstrap Error Handling

```python
from omnimemory.bootstrap import BootstrapError

try:
    result = await bootstrap(config)
except BootstrapError as e:
    print(f"Bootstrap failed for {e.config_block}: {e}")
    if e.cause:
        print(f"Underlying error: {e.cause}")
```

## Troubleshooting

### "Field required" Error

```
pydantic.ValidationError: 1 validation error for FilesystemSettings
OMNIMEMORY__FILESYSTEM__BASE_PATH
  Field required [type=missing, input_value={}, input_type=dict]
```

**Solution**: Set the required environment variable:
```bash
export OMNIMEMORY__FILESYSTEM__BASE_PATH=/data/omnimemory
```

### "extra fields not permitted" Error

```
pydantic.ValidationError: 1 validation error for SettingsMemoryService
OMNIMEMORY__TYPO_SETTING
  extra fields not permitted [type=extra_forbidden]
```

**Solution**: Check for typos in environment variable names. Use the exact variable names documented in this file.

### "Invalid URL" Error

```
pydantic.ValidationError: 1 validation error for QdrantSettings
OMNIMEMORY__QDRANT__URL
  URL scheme should be 'http' or 'https' [type=url_scheme]
```

**Solution**: Use proper URL format including scheme:
```bash
export OMNIMEMORY__QDRANT__URL=http://localhost:6333
```

### "Input should be greater/less than" Error

```
pydantic.ValidationError: 1 validation error for PostgresSettings
OMNIMEMORY__POSTGRES__POOL_SIZE
  Input should be less than or equal to 50 [type=less_than_equal]
```

**Solution**: Use a value within the documented constraints.

### Bootstrap "does not exist" Error

```
omnimemory.bootstrap.BootstrapError: Bootstrap failed [filesystem]: base_path '/nonexistent' does not exist and create_if_missing=False
```

**Solution**: Either create the directory manually, or set:
```bash
export OMNIMEMORY__FILESYSTEM__CREATE_IF_MISSING=true
```

### Bootstrap "is not writable" Error

```
omnimemory.bootstrap.BootstrapError: Bootstrap failed [filesystem]: base_path '/data/omnimemory' is not writable: [Errno 13] Permission denied
```

**Solution**: Ensure the user running OmniMemory has write permissions to the base_path directory.

### PostgreSQL Settings Ignored

If you set PostgreSQL environment variables but they seem to have no effect:

**Solution**: Ensure you enable the backend:
```bash
export OMNIMEMORY__POSTGRES_ENABLED=true
```

Without this flag, PostgreSQL settings are not loaded.

### Qdrant Warning About Missing API Key

```
Bootstrap warning: Qdrant API key not configured - using unauthenticated access
```

**Solution**: This is a warning, not an error. For production, set:
```bash
export OMNIMEMORY__QDRANT__API_KEY=your-api-key
```

## Quick Reference

### Minimum Viable Configuration

```bash
export OMNIMEMORY__FILESYSTEM__BASE_PATH=/tmp/omnimemory
```

### Enable All Backends

```bash
export OMNIMEMORY__FILESYSTEM__BASE_PATH=/data/omnimemory
export OMNIMEMORY__POSTGRES_ENABLED=true
export OMNIMEMORY__POSTGRES__DSN=postgresql://user@localhost:5432/db
export OMNIMEMORY__POSTGRES__PASSWORD=secret
export OMNIMEMORY__QDRANT_ENABLED=true
```

### Verify Configuration

```python
from omnimemory import load_settings

try:
    settings = load_settings()
    config = settings.to_config()
    print("Configuration valid!")
    print(f"Filesystem base: {config.filesystem.base_path}")
    print(f"PostgreSQL enabled: {config.postgres is not None}")
    print(f"Qdrant enabled: {config.qdrant is not None}")
except Exception as e:
    print(f"Configuration error: {e}")
```
