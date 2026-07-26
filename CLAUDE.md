# CLAUDE.md - OmniMemory

Document ingestion + semantic retrieval: models, protocols, and storage adapters for Qdrant, Memgraph, and PostgreSQL backends. Shared standards (Python, git, worktrees, PR/CI gates, layering) live in the root and `~/.claude/CLAUDE.md` and are not repeated here.

**Python**: `requires-python` in `pyproject.toml` (`>=3.12,<3.14` as of 2026-07-26) | **Build**: uv + hatchling. All Python commands run via `uv run`.

## Repo Invariants

Enforced by pre-commit hooks (`.pre-commit-config.yaml`) backed by `scripts/validation/`:

- **Minimize `Any`** — precise types or explicit `object`; gates are `mypy --strict` + pyright at the **pre-push** stage (`pre-commit install --hook-type pre-push`). "Zero `Any` in `src/`" is aspiration, not fact: a handful of adapter/handler files still import `typing.Any` (probe: `grep -rln 'from typing import.*Any' src/`); no ANN401-style ban exists.
- **`frozen=True, extra="forbid"` on boundary-crossing models** (`validate-pydantic-patterns` hook).
- **`ModelSemVer` only for version fields** — never `str`, never `ModelSemVer | str`, no `mode="before"` coercion validators, no helper methods hiding the duality. Callers convert with `ModelSemVer.from_str(...)` before passing.
- **No backwards compatibility, ever** — delete old code; no deprecated functions, no `OldName = NewName` alias shims (`validate-no-backward-compatibility` hook). No legacy omnibase_3 hybrid patterns — migrated code fully conforms to the ONEX 4.0 declarative pattern.
- **`Field(..., description="...")` on every model field** — no bare declarations.
- **PEP 604 unions** (`X | Y`) — ruff UP007 is organizationally mandated; never add it to the ignore list (see the NOTE in `[tool.ruff.lint]`).
- **Async-first** — all I/O is `async`; no blocking calls in async contexts.
- **Models live in `src/omnimemory/models/<domain>/`** — enforced by `scripts/validation/validate_model_locations.py`. A model tightly coupled to one handler/adapter may live alongside it with `# omnimemory-model-exempt: <reason>` on the class-def line. The validator checks marker presence only; the reason text is free-form convention (current usage: `grep -rho "omnimemory-model-exempt: .*" src/ | sort | uniq -c`).

## Commands

```bash
uv sync --group dev && pre-commit install && pre-commit install --hook-type pre-push
uv run ruff format src/ tests/ && uv run ruff check --fix src/ tests/
uv run mypy src/omnimemory --strict
uv run pytest        # markers (unit/integration/slow/memgraph/...): [tool.pytest.ini_options].markers in pyproject.toml
pre-commit run --all-files
```

## SPDX Headers

MIT SPDX headers required in `src/`, `tests/`, `scripts/`, `examples/` (spec: `omnibase_core/docs/conventions/FILE_HEADERS.md`). Stamp: `onex spdx fix src tests scripts examples` (add `--check` to dry-run). TRAP: the `# spdx-skip: <reason>` bypass is honored by the `onex spdx` CLI only — the local `validate-spdx-headers` pre-commit hook just scans the first 512 bytes for `SPDX-License-Identifier: MIT` and ignores the skip token.

## Documentation

`docs/INDEX.md` is the index (env vars, runbooks, architecture, CI, migrations all indexed there). High-traffic: `docs/environment_variables.md` (Memgraph/Qdrant/Postgres/embedding env vars), `docs/runbooks/STARTING_MEMORY_SERVICES.md`, `docs/architecture/ONEX_FOUR_NODE_ARCHITECTURE.md`.
