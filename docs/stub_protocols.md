# Stub Protocols and Compatibility Layer

## Overview

OmniMemory includes a compatibility layer (`src/omnimemory/compat/`) that provides local implementations for `omnibase_core` components that are not yet available in the installed package version. This document tracks these stubs and their migration path.

**Location**: `src/omnimemory/compat/`

---

## Current Stub Implementations

### 1. NodeResult (Monadic Pattern) - MIGRATED

**File**: `src/omnimemory/compat/node_result.py` (deprecated)

**Purpose**: Provides the monadic result pattern for node operations, enabling clean error handling without exceptions.

**Status**: **MIGRATED** - Now using `omnibase_core.models.core.model_base_result.ModelBaseResult`

**Migration Notes**:
- All protocol return types changed from `NodeResult[T]` to `ModelBaseResult`
- `result.is_success` changed to `result.success`
- Success/failure creation now uses direct construction instead of class methods
- Result values stored in `metadata.custom_fields` instead of direct `.value` access

**New Usage**:
```python
from omnibase_core.models.core.model_base_result import ModelBaseResult
from omnibase_core.models.core.model_error_details import ModelErrorDetails
from omnibase_core.models.results.model_simple_metadata import ModelGenericMetadata

# Create success result
result = ModelBaseResult(
    success=True,
    exit_code=0,
    errors=[],
    metadata=ModelGenericMetadata(custom_fields={"data": {"key": "value"}})
)

# Create failure result
result = ModelBaseResult(
    success=False,
    exit_code=1,
    errors=[ModelErrorDetails(error_message="Operation failed", error_code="ERR_001", error_type="runtime", component="memory_operation")]
)

# Check result
if result.success:
    process(result.metadata.custom_fields["data"])
else:
    handle_error(result.errors)
```

---

### 2. OnexError / BaseOnexError

**File**: `src/omnimemory/compat/onex_error.py`

**Purpose**: Structured error types for ONEX-compliant error handling with correlation IDs and error codes.

**Upstream Target**: `omnibase_core.core.errors.core_errors.OnexError`

**Status**: Local stub - awaiting upstream availability

**Usage**:
```python
from omnimemory.compat import OnexError, BaseOnexError

# Raise structured error
raise OnexError(
    message="Memory storage failed",
    error_code="MEM_001",
    correlation_id="abc-123",
    details={"operation": "store"}
)
```

---

### 3. ModelOnexContainer / ModelONEXContainer

**File**: `src/omnimemory/compat/model_onex_container.py`

**Purpose**: Dependency injection container for ONEX nodes, providing service registration and resolution.

**Upstream Target**: `omnibase_core.core.model_onex_container.ModelONEXContainer`

**Status**: Local stub - awaiting upstream availability

**Usage**:
```python
from omnimemory.compat import ModelOnexContainer

container = ModelOnexContainer()
container.register("db_handler", HandlerDb())
db = container.resolve("db_handler")
```

---

## Migration Path

### Phase 1: Monitoring (Current)

Monitor `omnibase_core` releases for availability of:
- `omnibase_core.core.errors.core_errors` (OnexError, BaseOnexError)
- `omnibase_core.core.model_onex_container` (ModelOnexContainer)

**Already migrated** (no longer monitoring):
- ~~`omnibase_core.core.monadic.model_node_result`~~ - Migrated to `ModelBaseResult` (see Section 1)

### Phase 2: Migration

When upstream components become available:

1. **Update Imports**:
   ```python
   # Before (using stub)
   from omnimemory.compat import NodeResult

   # After (using upstream ModelBaseResult)
   from omnibase_core.models.core.model_base_result import ModelBaseResult
   ```

   **Note**: NodeResult has been migrated to ModelBaseResult (see section 1 above).

2. **Run Test Suite**:
   ```bash
   pytest tests/ -v
   ```

3. **Verify API Compatibility**:
   - Ensure upstream API matches local stub
   - Update any incompatible usages

4. **Remove Stubs**:
   - Delete files from `src/omnimemory/compat/`
   - Remove exports from `__init__.py`
   - Update `docs/stub_protocols.md`

### Phase 3: Cleanup

- Remove `src/omnimemory/compat/` directory
- Update this documentation
- Remove migration notes from code

---

## Other Incomplete Features

The following features are defined but not fully implemented:

### PII Detection - Partial Implementation

**File**: `src/omnimemory/utils/pii_detector.py`

The following `PIIType` values are defined but do not have detection patterns:

| Type | Status | Required Work |
|------|--------|---------------|
| `URL` | Not Implemented | Add URL validation regex patterns |
| `PERSON_NAME` | Not Implemented | Add dictionary-based + NLP name detection |
| `ADDRESS` | Not Implemented | Add geocoding or NLP integration |

See [PII Handling Guide](./pii_handling.md) for details.

### Health Manager - Placeholder

**File**: `src/omnimemory/utils/health_manager.py`

Contains a placeholder for health check aggregation logic that returns healthy status. Full implementation pending:

```python
# Current placeholder (line 646)
# For now, return healthy as a placeholder
```

---

## Dependency on omnibase_infra

The handler reuse matrix (`docs/handler_reuse_matrix.md`) references handlers from `omnibase_infra`. This package should be added as a dependency when it becomes available on PyPI.

**Current Status**: Commented in `pyproject.toml` dev dependencies

**Add when available**:
```bash
poetry add --group dev omnibase-infra
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-01-18 | Initial documentation |
