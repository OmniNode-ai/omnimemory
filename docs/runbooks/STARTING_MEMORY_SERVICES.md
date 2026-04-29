> **Navigation**: [Home](../INDEX.md) > Runbooks > Starting Memory Services

# Runbook: Starting Memory Services

**Owner:** `omnimemory`
**Last verified:** 2026-04-29
**Verification:** `docker compose up -d && docker compose ps` in omnimemory root

---

## Overview

OmniMemory requires two infrastructure layers:

1. **Platform infra** (owned by `omnibase_infra`): Kafka/Redpanda + PostgreSQL
2. **Memory services** (owned by `omnimemory`): Qdrant, Memgraph, Valkey, Kreuzberg

Both layers must be running for full memory operation. Unit tests (`pytest -m unit`) do not require either layer — they run fully in-process with mock adapters. Memory services are only needed when running integration tests or workflows that exercise actual storage backends.

---

## Start Platform Infra First

```bash
# From any terminal with ONEX shell functions loaded
infra-up
```

This starts Kafka (Redpanda) and PostgreSQL on the platform server. Verify:

```bash
infra-status
```

Expected output includes `omnibase-infra-postgres` and `omnibase-infra-redpanda` in Running state.

---

## Start Memory Services

```bash
# From the omnimemory repo root
cd $OMNI_HOME/omnimemory
docker compose up -d
```

Verify all four services are healthy:

```bash
docker compose ps
```

Expected: all four containers (`omnimemory-qdrant`, `omnimemory-memgraph`, `omnimemory-valkey`, `omnimemory-kreuzberg-parser`) in Running/healthy state.

### Service health checks

```bash
# Qdrant
curl -fsS http://localhost:6333/healthz && echo "Qdrant OK"

# Memgraph (Bolt protocol — use Python or mgconsole)
docker exec omnimemory-memgraph echo "Memgraph container up"

# Valkey
docker exec omnimemory-valkey valkey-cli ping

# Kreuzberg
curl -fsS http://localhost:8090/health && echo "Kreuzberg OK"
```

---

## Start with Memgraph via infra bundle

If you are using the ONEX runtime bundle (for intent nodes and agent coordination):

```bash
infra-up-memory
```

This starts the runtime bundle including Memgraph and sets `OMNIMEMORY_*` environment variables. Use in combination with `docker compose up -d` in the omnimemory repo.

---

## Stop Memory Services

```bash
cd $OMNI_HOME/omnimemory
docker compose down
```

To stop platform infra:

```bash
infra-down
```

---

## Run Unit Tests (no external services required)

```bash
cd $OMNI_HOME/omnimemory   # or in the worktree
uv sync --group dev
uv run pytest tests/ -m unit
```

## Run All Tests

```bash
uv run pytest tests/ -v
```

Integration tests require memory services and platform infra to be running.

---

## Configuration

All service ports and connection strings are configurable via `.env`. See [Environment Variables](../environment_variables.md).

Default ports:
- Qdrant REST: `localhost:6333`
- Qdrant gRPC: `localhost:6334`
- Memgraph Bolt: `localhost:7687`
- Memgraph HTTP: `localhost:7444`
- Valkey: `localhost:6379`
- Kreuzberg parser: `localhost:8090`

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| Qdrant returns 503 | Container not healthy yet | Wait 10s and retry; check `docker compose logs omnimemory-qdrant` |
| Memgraph connection refused | Memgraph not started or port conflict | Check `docker compose ps`; ensure port 7687 is free |
| Embedding failures | Embedding service not running | Verify `LLM_EMBEDDING_URL` from `~/.omnibase/.env`; see CLAUDE.md infra section |
| `OMNIMEMORY_ENABLED` not set | Plugin does not activate | Set in `.env` or export before starting the runtime; see [Environment Variables](../environment_variables.md) |
