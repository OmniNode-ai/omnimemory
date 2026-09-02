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

## Cross-Repo

- Shared platform standards: `~/.claude/CLAUDE.md`
- Core models and ONEX kernel: [`omnibase_core`](https://github.com/OmniNode-ai/omnibase_core)
- Platform protocols: [`omnibase_spi`](https://github.com/OmniNode-ai/omnibase_spi)
- Event bus and PostgreSQL: [`omnibase_infra`](https://github.com/OmniNode-ai/omnibase_infra)

## Domain Rules

- Qdrant for vector storage, Memgraph for intent/relationship graphs, PostgreSQL for metadata.
- Document ingestion + semantic retrieval pipeline; PII detection runs before storage.
- Runnable node handlers (`contract.yaml`) are migrating to `omnimarket`; protocols, models, adapters and `PluginMemory` stay here.

## Documentation

Every OmniMemory document lives in the OmniNode knowledge base, not in this repository. The only markdown that remains here is this file, the root `README.md`, `CHANGELOG.md`, `LICENSE`, `SECURITY.md`, and the `.claude/` and `.github/` trees — enforced by the required `kb-doc-gate` check running in `strict` mode (`.kb-doc-gate.yaml`). A pointer stub is not a removal: the README's Documentation section carries the signposting role for the whole repo and is the full page index.

- Public: <https://github.com/OmniNode-ai/knowledge-base>
- Internal (repo-internal, named-author, or cross-repo content): <https://github.com/OmniNode-ai/knowledge-base-internal>

High-traffic pages: [Environment Variables](https://github.com/OmniNode-ai/knowledge-base/blob/main/reference/omnimemory-environment-variables.md), [Starting OmniMemory Services](https://github.com/OmniNode-ai/knowledge-base/blob/main/runbooks/omnimemory-starting-memory-services.md), [ONEX Four-Node Architecture](https://github.com/OmniNode-ai/knowledge-base/blob/main/architecture/omnimemory-four-node-architecture.md), [PII Handling](https://github.com/OmniNode-ai/knowledge-base/blob/main/guides/omnimemory-pii-handling.md).
