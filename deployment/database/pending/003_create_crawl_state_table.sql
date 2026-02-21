-- Migration: 003_create_crawl_state_table
-- Description: Create omnimemory_crawl_state table for document ingestion pipeline
-- Created: 2026-02-20
-- Ticket: OMN-2426

-- ============================================================================
-- Crawl State Table
-- ============================================================================
-- Tracks per-document crawl history across all crawler types.
-- Used for change detection: mtime fast-path, SHA-256 content fingerprinting,
-- and scope migration safety (scope_ref included in PK).
--
-- PRIMARY KEY (source_ref, crawler_type, scope_ref):
--   source_ref   = absolute path, URL, or Linear ID
--   crawler_type = one of: filesystem, git_repo, linear, watchdog
--   scope_ref    = hierarchical scope string (org/repo/subpath)
--
-- Why scope_ref in PK:
--   If scope mapping config changes (file moves between scopes or mapping is
--   reconfigured), a clean re-index is required. Without scope_ref in the PK,
--   stale crawl entries would silently block re-indexing under the new scope.

CREATE TABLE IF NOT EXISTS omnimemory_crawl_state (
    source_ref          TEXT        NOT NULL,
    crawler_type        TEXT        NOT NULL CHECK (crawler_type IN ('filesystem', 'git_repo', 'linear', 'watchdog')),
    scope_ref           TEXT        NOT NULL,
    content_fingerprint TEXT        NOT NULL CHECK (length(content_fingerprint) = 64),
    source_version      TEXT,
    last_crawled_at_utc TIMESTAMPTZ NOT NULL,
    last_changed_at_utc TIMESTAMPTZ,
    last_known_mtime    DOUBLE PRECISION,
    PRIMARY KEY (source_ref, crawler_type, scope_ref)
);

-- ============================================================================
-- Indexes for Common Access Patterns
-- ============================================================================

-- Optimizes: per-crawler state lookup (most common: filesystem walk)
-- Partial index for fetching all state entries belonging to a single crawler type
CREATE INDEX IF NOT EXISTS idx_crawl_state_crawler_type
    ON omnimemory_crawl_state(crawler_type);

-- Optimizes: scope-based queries (e.g., "all docs in omninode/omnimemory")
CREATE INDEX IF NOT EXISTS idx_crawl_state_scope_ref
    ON omnimemory_crawl_state(scope_ref);

-- Optimizes: staleness detection (find entries not crawled since a cutoff)
CREATE INDEX IF NOT EXISTS idx_crawl_state_last_crawled
    ON omnimemory_crawl_state(last_crawled_at_utc);

-- Optimizes: dedup / idempotency checks on content fingerprint
CREATE INDEX IF NOT EXISTS idx_crawl_state_content_fingerprint
    ON omnimemory_crawl_state(content_fingerprint);

-- ============================================================================
-- Comments
-- ============================================================================

COMMENT ON TABLE omnimemory_crawl_state IS
    'Per-document crawl history for document ingestion pipeline. Tracks content '
    'fingerprints and source versions for change detection across all crawler types. '
    'Part of OMN-2426 document ingestion pipeline (Stream A).';

COMMENT ON COLUMN omnimemory_crawl_state.source_ref IS
    'Unique identifier for the document source: absolute filesystem path, URL, or Linear ID.';

COMMENT ON COLUMN omnimemory_crawl_state.crawler_type IS
    'Crawler that owns this entry: filesystem | git_repo | linear | watchdog.';

COMMENT ON COLUMN omnimemory_crawl_state.scope_ref IS
    'Hierarchical scope string (org/repo/subpath). Included in PK for scope migration safety.';

COMMENT ON COLUMN omnimemory_crawl_state.content_fingerprint IS
    'SHA-256 of normalized document content (whitespace-collapsed). Used for change detection.';

COMMENT ON COLUMN omnimemory_crawl_state.source_version IS
    'Version token from the source system: git file SHA, Linear updatedAt ISO string, or NULL.';

COMMENT ON COLUMN omnimemory_crawl_state.last_crawled_at_utc IS
    'Timestamp of the most recent successful crawl, regardless of whether content changed.';

COMMENT ON COLUMN omnimemory_crawl_state.last_changed_at_utc IS
    'Timestamp of the most recent content change. NULL if content has never changed since first crawl.';

COMMENT ON COLUMN omnimemory_crawl_state.last_known_mtime IS
    'stat.st_mtime value at last crawl for FilesystemCrawler mtime fast-path. NULL for non-filesystem crawlers.';
