-- Migration: 002_create_memories_table
-- Description: Create memories table with lifecycle management columns
-- Created: 2026-01-25
-- Ticket: OMN-1453

-- ============================================================================
-- Extension: pgcrypto
-- ============================================================================
-- Required for gen_random_uuid() function on PostgreSQL versions < 13.
-- PostgreSQL 13+ includes gen_random_uuid() natively, but this extension
-- ensures compatibility with PostgreSQL 12 and earlier versions.
-- Using IF NOT EXISTS to avoid errors if already enabled.

CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- ============================================================================
-- Memories Table
-- ============================================================================
-- Stores memory content with lifecycle state management.
-- Lifecycle states: active -> stale -> expired -> archived -> deleted
-- Supports optimistic locking via lifecycle_revision column.

CREATE TABLE IF NOT EXISTS memories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content TEXT NOT NULL,
    content_type VARCHAR(50) NOT NULL DEFAULT 'text/plain',

    -- Lifecycle management (OMN-1453)
    lifecycle_state VARCHAR(20) NOT NULL DEFAULT 'active',
    expires_at TIMESTAMPTZ,
    archived_at TIMESTAMPTZ,
    lifecycle_revision INTEGER NOT NULL DEFAULT 1,
    archive_path TEXT,

    -- Optional metadata
    metadata JSONB,

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Constraints
    CONSTRAINT chk_memories_lifecycle_state CHECK (
        lifecycle_state IN ('active', 'stale', 'expired', 'archived', 'deleted')
    )
);

-- ============================================================================
-- Indexes for Lifecycle Queries
-- ============================================================================

-- Optimizes: SELECT * FROM memories WHERE lifecycle_state = 'active' AND expires_at < NOW()
-- Partial index for finding active memories that need expiration processing
CREATE INDEX IF NOT EXISTS idx_memories_lifecycle_expires
    ON memories(lifecycle_state, expires_at)
    WHERE lifecycle_state = 'active' AND expires_at IS NOT NULL;

-- Optimizes: SELECT * FROM memories WHERE lifecycle_state = 'expired' AND archived_at IS NULL
-- Partial index for finding expired memories pending archival
CREATE INDEX IF NOT EXISTS idx_memories_archive_candidates
    ON memories(lifecycle_state, archived_at)
    WHERE lifecycle_state = 'expired' AND archived_at IS NULL;

-- Optimizes: UPDATE memories SET ... WHERE id = $1 AND lifecycle_revision = $2
-- Index for optimistic locking queries (concurrent update protection)
--
-- Purpose: This composite index enables efficient optimistic locking patterns where
-- updates include both the primary key (id) and the expected revision number.
-- When multiple concurrent processes attempt to update the same memory, only one
-- succeeds (the one with the matching revision). Others get rows_affected=0,
-- indicating a conflict that requires retry or conflict resolution.
--
-- Why needed despite id being PRIMARY KEY:
--   The primary key index only covers (id). For queries like:
--     UPDATE memories SET ... WHERE id = $1 AND lifecycle_revision = $2
--   PostgreSQL must check lifecycle_revision after finding the row by id.
--   This index allows the query planner to verify both conditions in a single
--   index lookup, improving performance for high-contention workloads.
--
-- Trade-off: This index adds storage overhead and write amplification.
-- Keep if: High-concurrency lifecycle state transitions are expected.
-- Remove if: Single-writer patterns or low update frequency.
CREATE INDEX IF NOT EXISTS idx_memories_id_revision
    ON memories(id, lifecycle_revision);

-- ============================================================================
-- Additional Indexes for Common Access Patterns
-- ============================================================================

-- Optimizes: SELECT * FROM memories WHERE lifecycle_state = 'active' ORDER BY created_at DESC
-- Index for listing active memories (most common query pattern)
CREATE INDEX IF NOT EXISTS idx_memories_active_created
    ON memories(created_at DESC)
    WHERE lifecycle_state = 'active';

-- Optimizes: SELECT * FROM memories WHERE content_type = $1 AND lifecycle_state = 'active'
-- Index for content type filtering on active memories
CREATE INDEX IF NOT EXISTS idx_memories_active_content_type
    ON memories(content_type)
    WHERE lifecycle_state = 'active';

-- Optimizes: SELECT * FROM memories WHERE updated_at < NOW() - INTERVAL '30 days'
-- Index for cleanup and maintenance queries based on last update time
CREATE INDEX IF NOT EXISTS idx_memories_updated_at
    ON memories(updated_at);

-- ============================================================================
-- Trigger for updated_at Auto-Update
-- ============================================================================
-- Note: omnimemory_update_updated_at_column() function created in 001_create_subscription_tables.sql

DROP TRIGGER IF EXISTS omnimemory_trigger_memories_updated_at ON memories;
CREATE TRIGGER omnimemory_trigger_memories_updated_at
    BEFORE UPDATE ON memories
    FOR EACH ROW
    EXECUTE FUNCTION omnimemory_update_updated_at_column();

-- ============================================================================
-- Comments
-- ============================================================================

COMMENT ON TABLE memories IS
    'Memory storage with lifecycle management. States: active, stale, expired, archived, deleted.';

COMMENT ON COLUMN memories.id IS
    'Unique identifier for the memory (UUID v4)';

COMMENT ON COLUMN memories.content IS
    'The actual memory content (text, JSON, etc.)';

COMMENT ON COLUMN memories.content_type IS
    'MIME type of the content (e.g., text/plain, application/json)';

COMMENT ON COLUMN memories.lifecycle_state IS
    'Current lifecycle state: active (in use), stale (outdated), expired (past TTL), archived (cold storage), deleted (soft delete)';

COMMENT ON COLUMN memories.expires_at IS
    'Optional expiration timestamp. NULL means no expiration.';

COMMENT ON COLUMN memories.archived_at IS
    'Timestamp when memory was archived. NULL if not archived.';

COMMENT ON COLUMN memories.lifecycle_revision IS
    'Optimistic locking revision. Incremented on each lifecycle state change.';

COMMENT ON COLUMN memories.archive_path IS
    'Path to the archive file after memory is archived. NULL if not archived.';

COMMENT ON COLUMN memories.metadata IS
    'Optional JSONB metadata for storing additional memory attributes.';

COMMENT ON COLUMN memories.created_at IS
    'Timestamp when the memory was created';

COMMENT ON COLUMN memories.updated_at IS
    'Timestamp of last update (auto-updated via trigger)';
