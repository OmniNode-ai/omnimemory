> **Navigation**: [Home](../README.md) > Reference

# Stub Protocols and Compatibility Layer

## Overview

OmniMemory previously included a compatibility layer at `src/omnimemory/compat/` providing local stubs for `omnibase_core` components that were not yet available upstream. The `compat/` directory has been removed. The only remaining compat artifact is `src/omnimemory/_compat_imports.py`, which re-exports `ErrorCodeType` and `SeverityType` type aliases for backward import compatibility.

All former stubs (`NodeResult`, `OnexError`/`BaseOnexError`, `ModelONEXContainer`) have been fully migrated to upstream `omnibase_core` equivalents or removed.

### Migrated Components (for historical reference)

| Former stub | Upstream replacement | Status |
|-------------|---------------------|--------|
| `omnimemory.compat.node_result.NodeResult` | `omnibase_core.models.core.model_base_result.ModelBaseResult` | MIGRATED |
| `omnimemory.compat.onex_error.OnexError` / `BaseOnexError` | `omnibase_core.models.errors.model_onex_error.ModelOnexError` | MIGRATED |
| `omnimemory.compat.model_onex_container.ModelONEXContainer` | `omnibase_core.container.ModelONEXContainer` | MIGRATED |

---

## Other Incomplete Features

The following features are defined but not fully implemented:

### PII Detection - Partial Implementation

**File**: `src/omnimemory/adapters/adapter_pii_detector.py`

The following `PIIType` values are defined but do not have detection patterns:

| Type | Status | Required Work |
|------|--------|---------------|
| `URL` | Not Implemented | Add URL validation regex patterns |
| `PERSON_NAME` | Not Implemented | Add dictionary-based + NLP name detection |
| `ADDRESS` | Not Implemented | Add geocoding or NLP integration |

See [PII Handling Guide](./pii_handling.md) for details.

### Health Manager - Placeholder

**File**: `src/omnimemory/adapters/adapter_health_manager.py`

Contains a placeholder for health check aggregation logic that returns healthy status. Full implementation pending.

---

## Dependency on omnibase_infra

The handler reuse matrix (`docs/handler_reuse_matrix.md`) references handlers from `omnibase_infra`. This package is declared as a dependency in `pyproject.toml`.

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.1.0 | 2026-02-19 | Audit stub status: ModelONEXContainer marked MIGRATED (upstream available at omnibase_core.container); OnexError/BaseOnexError confirmed still active (upstream exposes ModelOnexError, not OnexError) |
| 1.0.0 | 2025-01-18 | Initial documentation |
