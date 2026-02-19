> **Navigation**: [Home](../INDEX.md) > CI > CI Monitoring Guide

# CI Monitoring Guide

> **Purpose**: Reference for CI checks, local reproduction, and failure triage
> **Last Updated**: 2026-02-19

---

## Overview

OmniMemory CI runs on GitHub Actions and is defined in two workflow files:

| File | Trigger | Purpose |
|------|---------|---------|
| `.github/workflows/test.yml` | Push to `main`/`develop`, all PRs | Full validation pipeline |
| `.github/workflows/pre-commit.yml` | Push to `main`, all PRs | Pre-commit hook validation |

The full pipeline runs in three phases:

- **Phase 1** (parallel, ~5 min): `migration-freeze`, `lint`, `pyright`, `onex-validation`, `transport-import-guard`, `contract-validation`, `io-audit`, `check-handshake`
- **Phase 2** (after Phase 1): `test` — full test suite with coverage
- **Phase 3** (aggregation): `test-summary` — gates the overall pass/fail result

Every CI validation job has a corresponding pre-commit hook so that "works locally, fails in CI" drift is prevented. The synchronization map is documented in `.pre-commit-config.yaml` and `.github/workflows/test.yml`.

---

## CI Checks

### ARCH-002: Kafka Import Lint Guard (OMN-1750)

**What it checks**: Prevents direct Kafka client imports (`aiokafka`, `kafka`, `confluent_kafka`) from appearing in `src/omnimemory/nodes/`. This enforces ARCH-002: "Runtime owns all Kafka plumbing." Nodes must consume events through the abstract `EventBus` SPI provided by the runtime layer.

**Where defined**:
- CI job: `onex-validation` step "Validate Kafka import boundary (ARCH-002)" in `.github/workflows/test.yml`
- Script: `scripts/validation/validate_kafka_imports.py`
- Pre-commit hook: `validate-kafka-imports` in `.pre-commit-config.yaml` (stage: `pre-commit`)

**Enforced scope**: `src/omnimemory/nodes/` only. The runtime layer (`src/omnimemory/runtime/`) is intentionally excluded — it is allowed to use Kafka directly.

**How to fix a violation**:

```python
# WRONG - direct Kafka import in a node
from aiokafka import AIOKafkaConsumer

# CORRECT - for subscribing: receive an injected subscribe_callback; never import aiokafka
async def initialize(
    self,
    subscribe_callback: Callable[
        [str, Callable[[dict[str, object]], None]], Callable[[], None]
    ],
) -> None:
    unsubscribe = subscribe_callback(full_topic, self._handle_message_sync)

# CORRECT - for publishing: depend on ProtocolEventBusPublish from the runtime adapters
from omnimemory.runtime.adapters import ProtocolEventBusPublish
```

**Exemption annotation** (use sparingly, requires justification):

```python
from aiokafka import AIOKafkaConsumer  # omnimemory-kafka-exempt: <reason>
```

**Run locally**:

```bash
poetry run python scripts/validation/validate_kafka_imports.py src/
```

---

### Migration Freeze Enforcement (OMN-2074)

**What it enforces**: While `.migration_freeze` exists at the repository root, no new migration files may be added to `deployment/database/migrations/`. This freeze was activated during the DB-per-repo refactor (OMN-2055) on 2026-02-10. Modifications to existing migration files (bug fixes, comment tweaks) are allowed during the freeze.

**Where defined**:
- CI job: `migration-freeze` in `.github/workflows/test.yml`
- Script: `scripts/check_migration_freeze.sh`
- Freeze sentinel: `.migration_freeze` (root of repo)
- Pre-commit hook: `migration-freeze-check` in `.pre-commit-config.yaml` (stage: `pre-commit`, triggered by changes to `deployment/database/migrations/` or `.migration_freeze`)

**How to add a new migration correctly**:

New migrations are blocked while `.migration_freeze` exists. To proceed:

1. Check `.migration_freeze` for context on when the freeze will be lifted.
2. If the freeze must be lifted: remove `.migration_freeze` and add your migration in the same commit. The check script detects the sentinel's absence at runtime and exits cleanly.
3. If the freeze must stay active: do not add new migration files. Raise the topic in OMN-2074 or OMN-2055 to determine the correct action.

**Allowed during freeze**:
- Migration moves (reorganizing between repos)
- Ownership fixes (table transfers)
- Rollback bug fixes to existing migration files

**Run locally**:

```bash
# Pre-commit mode (checks staged files)
./scripts/check_migration_freeze.sh

# CI mode (checks diff against base branch)
./scripts/check_migration_freeze.sh --ci
```

---

### Transport Import Guard (OMN-2218)

**What it checks**: An AST-based validator that ensures nodes do not import transport or I/O libraries at runtime. This is the stricter, AST-aware counterpart to the regex-based Kafka import guard above. It covers a broader set of banned modules across all of `src/omnimemory/` (excluding `src/omnimemory/runtime/`).

**Where defined**:
- CI job: `transport-import-guard` in `.github/workflows/test.yml`
- Script: `scripts/validate_no_transport_imports.py`
- Whitelist: `tests/audit/transport_import_whitelist.yaml`
- Pre-commit hook: `validate-no-transport-imports` in `.pre-commit-config.yaml` (stage: `pre-commit`)

**Banned module categories**:

| Category | Modules |
|----------|---------|
| HTTP clients | `aiohttp`, `httpx`, `requests`, `urllib3` |
| Kafka clients | `kafka`, `aiokafka`, `confluent_kafka` |
| Redis clients | `redis`, `aioredis` |
| Database clients | `asyncpg`, `psycopg2`, `psycopg`, `aiomysql` |
| Message queues | `pika`, `aio_pika`, `kombu`, `celery` |
| gRPC | `grpc` |
| WebSocket | `websockets`, `wsproto` |

`TYPE_CHECKING`-guarded imports are allowed (they create no runtime dependency).

**How to fix a violation**:

```python
# WRONG - runtime import of a transport library in a node
import httpx

# CORRECT option 1 - use a TYPE_CHECKING guard (type-only usage)
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import httpx

# CORRECT option 2 - define a protocol and inject it
from omnimemory.protocols import HttpClientProtocol
```

**Whitelist format** (for pre-existing legitimate infrastructure files):

```yaml
schema_version: "1.0.0"
files:
  - path: "src/omnimemory/utils/health_manager.py"
    reason: "Health checks require direct asyncpg/redis connectivity probes"
    allowed_modules:
      - asyncpg
      - redis
```

**Run locally**:

```bash
poetry run python scripts/validate_no_transport_imports.py \
  --src-dir src/omnimemory \
  --exclude src/omnimemory/runtime \
  --whitelist tests/audit/transport_import_whitelist.yaml
```

---

### CI Infrastructure Alignment (OMN-2218)

**What was aligned**: Phase 7 of the CI infrastructure work (OMN-2218) added several new CI jobs and synchronized them with matching pre-commit hooks to eliminate drift between local validation and CI:

| New CI Job | Pre-commit Hook | Added |
|------------|-----------------|-------|
| `transport-import-guard` | `validate-no-transport-imports` | OMN-2218 |
| `contract-validation` | `contract-linter` | OMN-2218 |
| `io-audit` | `io-audit` | OMN-2218 |

**Current CI tooling versions** (from `.github/workflows/test.yml` and `.pre-commit-config.yaml`):

| Tool | Version | Configuration |
|------|---------|---------------|
| Python | 3.12 | `env.PYTHON_VERSION` in `test.yml` |
| Poetry | 2.2.1 | `env.POETRY_VERSION` in `test.yml` |
| ruff | 0.8.6 | `.pre-commit-config.yaml` rev; `pyproject.toml [tool.ruff]` |
| mypy | ^1.14.0 | `pyproject.toml [tool.mypy]`; CI uses Poetry env |
| pyright | 1.1.391 | `.pre-commit-config.yaml` rev; `pyrightconfig.json` |

**Version sync note**: Poetry uses caret ranges (e.g., `^0.8.0`) that allow patch updates; pre-commit pins exact versions. Both are within compatible ranges. When upgrading ruff, follow the upgrade procedure documented in `.pre-commit-config.yaml`.

**Alignment validator**: The script `scripts/validate_ci_precommit_alignment.py` verifies that CI jobs and pre-commit hooks stay synchronized. It checks that every CI validation step has a corresponding hook and vice versa.

```bash
poetry run python scripts/validate_ci_precommit_alignment.py
```

---

### Required Status Checks

**Which checks must pass before merge**: The `test-summary` job aggregates all Phase 1 and Phase 2 results. It requires all of the following to succeed (or be skipped for `check-handshake` on fork PRs):

- `migration-freeze`
- `lint` (ruff format, ruff check, mypy strict)
- `pyright`
- `onex-validation` (pydantic patterns, single-class-per-file, enum casing, no-backward-compat, secrets, naming, HTTP imports, Kafka imports, model locations)
- `transport-import-guard`
- `contract-validation`
- `io-audit`
- `check-handshake` (skipped on fork PRs — forks may not have cross-repo checkout access)
- `test`

**Architecture handshake**: The `check-handshake` job verifies that `.claude/architecture-handshake.md` matches the canonical source in the `OmniNode-ai/omnibase_core` repository. This ensures cross-repo architectural contracts stay in sync. If this check fails, update `.claude/architecture-handshake.md` from `omnibase_core/architecture-handshakes/`.

**Concurrency**: The test suite uses `cancel-in-progress: true` so that pushing new commits to an open PR cancels the previous run, conserving CI resources.

---

## Running CI Checks Locally

### Pre-commit (runs at commit time)

```bash
# Install hooks (one-time setup)
poetry install
pre-commit install
pre-commit install --hook-type pre-push

# Run all pre-commit hooks against all files
pre-commit run --all-files

# Run pre-push hooks (mypy, pyright) against all files
pre-commit run --all-files --hook-stage pre-push
```

### Individual checks

```bash
# Formatting
poetry run ruff format src/ tests/        # Auto-fix formatting
poetry run ruff format --check src/ tests/ # Check only (matches CI)

# Linting
poetry run ruff check --fix src/ tests/   # Auto-fix lint issues
poetry run ruff check src/ tests/         # Check only (matches CI)

# Type checking
poetry run mypy --show-error-codes --no-error-summary src/omnimemory
poetry run pyright src/omnimemory

# ONEX validation scripts
poetry run python scripts/validation/validate_kafka_imports.py src/
poetry run python scripts/validate_no_transport_imports.py \
  --src-dir src/omnimemory \
  --exclude src/omnimemory/runtime \
  --whitelist tests/audit/transport_import_whitelist.yaml
poetry run python scripts/validation/validate_pydantic_patterns.py src/
poetry run python scripts/validation/validate_naming.py src/

# Migration freeze
./scripts/check_migration_freeze.sh

# I/O audit
poetry run python -m omnimemory.audit

# CI/pre-commit alignment
poetry run python scripts/validate_ci_precommit_alignment.py
```

### Tests

```bash
# All tests (matches CI)
poetry run pytest tests/ -n auto --timeout=60 --tb=short

# Unit tests only
poetry run pytest -m unit

# With coverage
poetry run pytest tests/ --cov=src/omnimemory --cov-report=term-missing
```

---

## Failure Triage

| Failing job | First step |
|-------------|-----------|
| `migration-freeze` | Check if `.migration_freeze` exists and if the PR adds new files to `deployment/database/migrations/` |
| `lint` (ruff) | Run `poetry run ruff check --fix src/ tests/` and `poetry run ruff format src/ tests/` |
| `lint` (mypy) | Run `poetry run mypy --show-error-codes src/omnimemory` and address type errors |
| `pyright` | Run `poetry run pyright src/omnimemory` and address type errors |
| `onex-validation` | Run the failing validation script individually (see commands above) |
| `transport-import-guard` | Run `scripts/validate_no_transport_imports.py --verbose` to identify the import and file |
| `contract-validation` | Run `poetry run python -m omnimemory.tools.contract_linter <contract.yaml>` |
| `io-audit` | Run `poetry run python -m omnimemory.audit` to see which node has forbidden I/O |
| `check-handshake` | Diff `.claude/architecture-handshake.md` against `omnibase_core/architecture-handshakes/` |
| `test` | Run `poetry run pytest tests/ --tb=long` to see full failure output |
