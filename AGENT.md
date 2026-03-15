# AGENT.md -- omnimemory

> LLM navigation guide. Points to context sources -- does not duplicate them.

## Context

- **Architecture**: `docs/architecture/`
- **Stub status**: `docs/stub_protocols.md`
- **PII handling**: `docs/pii_handling.md`
- **Conventions**: `CLAUDE.md`

## Commands

- Tests: `uv run pytest -m unit`
- Lint: `uv run ruff check src/ tests/`
- Type check: `uv run mypy src/omnimemory/`
- Pre-commit: `pre-commit run --all-files`

## Cross-Repo

- Shared platform standards: `~/.claude/CLAUDE.md`
- Core models: `omnibase_core/CLAUDE.md`

## Rules

- Qdrant for vector storage, PostgreSQL for metadata
- Document ingestion + semantic retrieval pipeline
- PII detection required before storage
