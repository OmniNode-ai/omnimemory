> **Navigation**: [Home](../../README.md) > CI > CI Monitoring Guide

# CI Monitoring Guide

> **Purpose**: Reference for CI checks, local reproduction, and failure triage
> **Last Updated**: 2026-08-27

---

## Overview

OmniMemory CI runs on GitHub Actions and is defined in the following workflow files:

| File | Trigger | Purpose |
|------|---------|---------|
| `.github/workflows/ci.yml` | Push to `dev`/`main`, all PRs | Full validation pipeline |
| `.github/workflows/pre-commit.yml` | Push to `dev`/`main`, all PRs | Pre-commit hook validation |
| `.github/workflows/imperative-contract-guard.yml` | All PRs | Enforces no imperative I/O in contract-managed paths |
| `.github/workflows/omni-standards-compliance.yml` | All PRs | Platform-wide naming, typing, and pattern compliance |
| `.github/workflows/stale-todo-gate.yml` | All PRs | Blocks unresolved TODO annotations |
| `.github/workflows/docs-validate.yml` | All PRs | Validates documentation structure and links |
| `.github/workflows/cr-thread-gate.yml` | All PRs | Fails if unresolved CodeRabbit review threads exist |
| `.github/workflows/pr-title-check.yml` | All PRs | Enforces `OMN-XXXX` ticket reference in PR title |
| `.github/workflows/call-occ-autobind.yml` | All PRs | Publishes the OCC-autobind command that mints this PR's evidence companion |
| `.github/workflows/call-occ-companion-effect.yml` | All PRs | Publishes the OCC-companion-effect command that stamps `Evidence-Source: OCC#<n>` onto the PR body |

The full validation pipeline (`ci.yml`) runs in three phases:

- **Phase 1** (parallel, ~5 min): `migration-freeze`, `lint`, `pyright`, `onex-validation`, `transport-import-guard`, `contract-validation`, `io-audit`, `check-handshake`
- **Phase 2** (after Phase 1): `test` — full test suite with coverage
- **Phase 3** (aggregation): `test-summary` — gates the overall pass/fail result

Every CI validation job has a corresponding pre-commit hook so that "works locally, fails in CI" drift is prevented. The synchronization map is documented in `.pre-commit-config.yaml` and `.github/workflows/ci.yml`.

---

## CI Checks

### ARCH-002: Kafka Import Lint Guard

**What it checks**: Prevents direct Kafka client imports (`aiokafka`, `kafka`, `confluent_kafka`) from appearing in `src/omnimemory/nodes/`. This enforces ARCH-002: "Runtime owns all Kafka plumbing." Nodes must consume events through the abstract `EventBus` SPI provided by the runtime layer.

**Where defined**:
- CI job: `onex-validation` step "Validate Kafka import boundary (ARCH-002)" in `.github/workflows/ci.yml`
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

### Migration Freeze Enforcement

**What it enforces**: While `.migration_freeze` exists at the repository root, no new migration files may be added to `deployment/database/migrations/`. This freeze was activated during the DB-per-repo refactor on 2026-02-10. Modifications to existing migration files (bug fixes, comment tweaks) are allowed during the freeze.

**Where defined**:
- CI job: `migration-freeze` in `.github/workflows/ci.yml`
- Script: `scripts/check_migration_freeze.sh`
- Freeze sentinel: `.migration_freeze` (root of repo)
- Pre-commit hook: `migration-freeze-check` in `.pre-commit-config.yaml` (stage: `pre-commit`, triggered by changes to `deployment/database/migrations/` or `.migration_freeze`)

**How to add a new migration correctly**:

New migrations are blocked while `.migration_freeze` exists. To proceed:

1. Check `.migration_freeze` for context on when the freeze will be lifted.
2. If the freeze must be lifted: remove `.migration_freeze` and add your migration in the same commit. The check script detects the sentinel's absence at runtime and exits cleanly.
3. If the freeze must stay active: do not add new migration files. Raise the topic with the team to determine the correct action.

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

### Transport Import Guard

**What it checks**: An AST-based validator that ensures nodes do not import transport or I/O libraries at runtime. This is the stricter, AST-aware counterpart to the regex-based Kafka import guard above. It covers a broader set of banned modules across all of `src/omnimemory/` (excluding `src/omnimemory/runtime/`).

**Where defined**:
- CI job: `transport-import-guard` in `.github/workflows/ci.yml`
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

### CI Infrastructure Alignment

**What was aligned**: A CI infrastructure alignment pass added several new CI jobs and synchronized them with matching pre-commit hooks to eliminate drift between local validation and CI:

| New CI Job | Pre-commit Hook |
|------------|-----------------|
| `transport-import-guard` | `validate-no-transport-imports` |
| `contract-validation` | `contract-linter` |
| `io-audit` | `io-audit` |

**Current CI tooling versions** (from `.github/workflows/ci.yml` and `.pre-commit-config.yaml`):

| Tool | Version | Configuration |
|------|---------|---------------|
| Python | 3.12 | `env.PYTHON_VERSION` in `ci.yml` |
| Poetry | 2.2.1 | `env.POETRY_VERSION` in `ci.yml` |
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

### Runner routing

Most jobs in `ci.yml` hardcode `ubuntu-latest`. The jobs that do not read the shared
`OMNI_RUNNER_SELECTOR_V1` seam:

```yaml
runs-on: >-
  ${{ (fork PR)
      && fromJSON(vars.OMNI_PUBLIC_PR_RUNS_ON_JSON || '["ubuntu-latest"]')
      || fromJSON(vars.OMNI_TRUSTED_CI_RUNS_ON_JSON || '["self-hosted","omnibase-ci"]') }}
```

Fork PRs go to GitHub-hosted runners; same-repo PRs follow whatever
`OMNI_TRUSTED_CI_RUNS_ON_JSON` resolves to — historically the self-hosted `omnibase-ci`
fleet, which is the only runner class that can reach the tailnet-only lane broker.

**Correction (OMN-16682, 2026-08-27).** An earlier revision of this section claimed
`contract-compliance` in `ci.yml` "cannot complete from a hosted runner" because its
`uv sync` resolves against the tailnet mirror. That was measured and is **wrong for this
repo**: under a real seam flip to `["ubuntu-latest"]`, `ci.yml` went fully green on hosted
runners — all 16 jobs including `Contract Compliance Check` and the required `CI Summary`.
Treat the tailnet-mirror dependency as unproven here rather than as established fact, and
measure per repo instead of inheriting this claim.

**The OCC publishers are deliberately NOT on that seam.** `occ-autobind` and
`occ-companion-effect` reach the reusable workflows in `OmniNode-ai/omniclaude`, whose
trusted branch reads a dedicated `OMNI_OCC_AUTOBIND_RUNS_ON_JSON` with a literal
`["self-hosted","omnibase-ci"]` fallback. That variable is deliberately left unset, so the
literal is the operating value and the pin holds by construction.

The separation exists because these two publishers write to a tailnet-only lane broker, and
they mint the evidence line the receipt gate hard-fails without. Sharing
one variable with unrelated lint/test jobs meant a single routing change could stop every
merge in every OCC-gated repo rather than merely turning some checks red. Re-pointing either
publisher back at `OMNI_TRUSTED_CI_RUNS_ON_JSON` reads as a harmless consistency tidy-up in
review and is guarded against in the omniclaude and omnimarket test suites; do not do it.

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
