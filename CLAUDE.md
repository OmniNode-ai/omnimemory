# CLAUDE.md - OmniMemory

> **Python**: 3.12+ | **Framework**: ONEX 4.0 | **Package Manager**: Poetry | **Shared Standards**: See **`~/.claude/CLAUDE.md`** for shared development standards (Python, Git, testing, architecture principles) and infrastructure configuration (PostgreSQL, Kafka/Redpanda, Docker networking, environment variables).

---

## Table of Contents

1. [Repo Invariants](#repo-invariants)
2. [Non-Goals](#non-goals)
3. [Quick Reference](#quick-reference)
4. [Package Manager](#package-manager)
5. [Forbidden Patterns](#forbidden-patterns)
6. [Required Patterns](#required-patterns)
7. [Model Exemption Pattern](#model-exemption-pattern)
8. [Documentation](#documentation)

---

## Repo Invariants

These are non-negotiable architectural truths:

- **Zero `Any` types** — no `Any` anywhere in `src/`; use precise types or explicit `object`
- **`frozen=True` on boundary-crossing models** — all models that cross handler/node boundaries must be immutable
- **`ModelSemVer` only for version fields** — no `str`, no `str | ModelSemVer`, no validator coercion
- **No backwards compatibility** — ever; delete old code, never deprecate
- **Models live in `src/omnimemory/models/`** — organized by domain subdirectory; exceptions require `omnimemory-model-exempt` comment
- **`Field(..., description="...")` on all model fields** — no bare field declarations
- **PEP 604 unions** — `X | Y` not `Optional[X]` or `Union[X, Y]`
- **Async-first** — all I/O operations must be `async`; no blocking calls in async contexts

---

## Non-Goals

OmniMemory explicitly does **NOT**:

- **Maintain backwards compatibility** — breaking changes are always acceptable; callers update or they break
- **Accept strings where typed models are required** — no convenience coercions in validators
- **Keep deprecated code** — the moment something is outdated, delete it; no `_deprecated` suffixes or shims
- **Support legacy omnibase_3 patterns** — migrated code must fully conform to ONEX 4.0; no hybrid patterns
- **Expose untyped public APIs** — every public function and method must be fully typed

---

## Quick Reference

```bash
# Setup
poetry install
pre-commit install
pre-commit install --hook-type pre-push

# Format and lint
poetry run ruff format src/ tests/
poetry run ruff check --fix src/ tests/

# Type checking
poetry run mypy src/omnimemory

# Testing
poetry run pytest                    # All tests
poetry run pytest -m unit            # Unit tests only
poetry run pytest -m integration     # Integration tests
poetry run pytest --cov              # With coverage report

# Pre-commit validation
pre-commit run --all-files
pre-commit run --all-files --hook-stage pre-push
```

**Test markers**: `@pytest.mark.unit`, `@pytest.mark.integration`, `@pytest.mark.slow`, `@pytest.mark.benchmark`, `@pytest.mark.memgraph`, `@pytest.mark.embedding`

---

## Package Manager

This repository uses **Poetry** for dependency management. All Python commands must be run via `poetry run`.

```bash
poetry install          # Install all dependencies
poetry run <command>    # Run command in venv
poetry lock             # Regenerate lockfile
```

---

## Forbidden Patterns

### Version fields: never accept strings

```python
# WRONG - union type for backwards compatibility
version: ModelSemVer | str

# WRONG - optional string fallback
version: ModelSemVer | None = None  # when accommodating string callers

# WRONG - validator coercion for "convenience"
@field_validator("version", mode="before")
def convert_string(cls, v: object) -> ModelSemVer:
    if isinstance(v, str):
        return ModelSemVer.from_str(v)
    return v  # type: ignore[return-value]

# WRONG - helper method hiding the string/ModelSemVer duality
def get_semver(self) -> ModelSemVer:
    if isinstance(self.version, str):
        return ModelSemVer.from_str(self.version)
    return self.version
```

### Models: no Any, no bare fields, no missing frozen

```python
# WRONG - Any type
class MyModel(BaseModel):
    data: Any  # never

# WRONG - undocumented field
class MyModel(BaseModel):
    name: str  # no Field(), no description

# WRONG - boundary model not frozen
class MyRequest(BaseModel):  # crosses handler boundary
    payload: str  # missing frozen=True in ConfigDict
```

### Old code: delete, never deprecate

```python
# WRONG - keeping deprecated code
def old_process(data):  # deprecated, use new_process instead
    ...

# WRONG - compatibility shim
OldName = NewName  # backwards compat alias - never do this
```

---

## Required Patterns

### Version fields: ModelSemVer only, caller converts

```python
# CORRECT - ModelSemVer directly; callers convert before passing
class MyConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    version: ModelSemVer = Field(..., description="Semantic version of this config")
    contract_version: ModelSemVer = Field(..., description="Contract version")

# Caller responsibility:
config = MyConfig(
    version=ModelSemVer.from_str("1.0.0"),
    contract_version=ModelSemVer.from_str("2.0.0"),
)
```

### Boundary-crossing models: frozen + extra="forbid"

```python
# CORRECT - immutable, strict, documented
class MyRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    query: str = Field(..., description="Search query string")
    limit: int = Field(default=10, description="Maximum number of results to return")
```

### Models in domain subdirectory

```python
# CORRECT location: src/omnimemory/models/memory/model_search_request.py
class ModelSearchRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    query: str = Field(..., description="Semantic search query")
```

---

## Model Exemption Pattern

Some Pydantic models are exempt from the `src/omnimemory/models/` location rule when they are tightly coupled to a handler or adapter implementation. Mark them with the exemption comment:

```python
class MyHandlerConfig(  # omnimemory-model-exempt: handler config
    BaseModel
):
    """Handler-specific configuration; lives alongside handler, not in models/."""
    ...
```

**Valid exemption reasons:**

| Tag | When to use |
|-----|-------------|
| `handler metadata` | Returned by `describe()` |
| `handler health` | Returned by `health_check()` |
| `handler config` | Handler-specific configuration |
| `handler internal` | Internal implementation model not part of public API |
| `handler result` | Handler operation result |
| `handler command` | Command/request model for handler operations |
| `handler event` | Event emitted by a handler |
| `handler state` | State snapshot for handler operations |
| `adapter config` | Adapter-specific configuration |
| `adapter health` | Adapter health status |
| `adapter internal` | Internal adapter implementation |
| `projection model` | Event sourcing projection |
| `archive record format` | Archive/storage record format |

All other models belong in `src/omnimemory/models/<domain>/`.

---

## Documentation

| Topic | Document |
|-------|----------|
| Documentation index | `docs/INDEX.md` |
| Environment variables | `docs/environment_variables.md` |
| Handler reuse matrix | `docs/handler_reuse_matrix.md` |
| Performance testing | `docs/PERFORMANCE_TESTING.md` |
| PII handling | `docs/pii_handling.md` |
| Stub protocols | `docs/stub_protocols.md` |
| Architecture (ONEX 4-node) | `docs/architecture/ONEX_FOUR_NODE_ARCHITECTURE.md` |
| Architecture (Kafka abstraction) | `docs/architecture/ARCH_002_KAFKA_ABSTRACTION.md` |
| CI monitoring | `docs/ci/CI_MONITORING_GUIDE.md` |
| Runtime plugins | `docs/runtime/RUNTIME_PLUGINS.md` |

For project overview, mission, and technology stack, see `README.md`.

---

**Python**: 3.12+ | **Package Manager**: Poetry | **ONEX**: 4.0+
