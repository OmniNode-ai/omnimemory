# AGENT.md -- omnimemory

> LLM navigation guide. Points to context sources -- does not duplicate them.

## Context

- **Documentation index**: `README.md` — every page lives in the OmniNode knowledge base; each path under `docs/` is a pointer stub
- **Architecture**: <https://github.com/OmniNode-ai/knowledge-base/blob/main/architecture/omnimemory-four-node-architecture.md>
- **PII handling**: <https://github.com/OmniNode-ai/knowledge-base/blob/main/guides/omnimemory-pii-handling.md>
- **Stub status**: <https://github.com/OmniNode-ai/knowledge-base-internal/blob/main/reference/omnimemory-stub-protocols-status.md> (migrated OMN-16602: internal implementation-status tracker)
- **Conventions**: `CLAUDE.md`

## Commands

- Tests: `uv run pytest -m unit`
- Lint: `uv run ruff check src/ tests/`
- Type check: `uv run mypy src/omnimemory/ --strict`
- Pre-commit: `pre-commit run --all-files`

## Cross-Repo

- Shared platform standards: `~/.claude/CLAUDE.md`
- Core models: `omnibase_core/CLAUDE.md`

## Rules

- Qdrant for vector storage, PostgreSQL for metadata
- Document ingestion + semantic retrieval pipeline
- PII detection required before storage
