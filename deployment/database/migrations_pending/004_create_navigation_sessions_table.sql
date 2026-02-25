-- Migration: 004_create_navigation_sessions_table
-- Description: Create navigation_sessions table for navigation history storage
-- Created: 2026-02-24
-- Ticket: OMN-2584

-- ============================================================================
-- Extension: pgcrypto
-- ============================================================================
-- Required for gen_random_uuid() on PostgreSQL < 13. Safe to repeat.

CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- ============================================================================
-- Navigation Sessions Table
-- ============================================================================
-- Stores the outcome record for every completed navigation session.
--
-- Routing:
--   outcome = 'success' → row exists here AND a point exists in Qdrant
--                         (collection: navigation_paths)
--   outcome = 'failure' → row exists here ONLY (never written to Qdrant)
--
-- Idempotency: session_id is the PRIMARY KEY, preventing duplicate inserts.
-- All INSERTs use ON CONFLICT DO NOTHING to enforce idempotency at the DB level.

CREATE TABLE IF NOT EXISTS navigation_sessions (
    -- Identity
    session_id          TEXT PRIMARY KEY,           -- UUID as text (e.g. "f47ac10b-...")
    goal_hash           TEXT NOT NULL,              -- SHA-256 of goal_condition (indexed)
    goal_condition      TEXT NOT NULL,              -- Goal state description (typed artifact)

    -- Graph state references
    start_state_id      TEXT NOT NULL,              -- Starting graph state identifier
    end_state_id        TEXT NOT NULL,              -- Terminal graph state identifier

    -- Execution summary
    step_count          INTEGER NOT NULL DEFAULT 0, -- Number of plan steps executed
    outcome             TEXT NOT NULL,              -- 'success' | 'failure'
    failure_reason      TEXT,                       -- Structured reason code (NULL on success)

    -- Graph provenance
    graph_fingerprint   TEXT NOT NULL,              -- Content hash of contract graph at navigation time

    -- Full step log (serialised plan steps for replay / debugging)
    steps_json          TEXT NOT NULL DEFAULT '[]', -- JSON array of PlanStep objects

    -- Timestamps
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Constraints
    CONSTRAINT chk_navigation_sessions_outcome CHECK (
        outcome IN ('success', 'failure')
    ),
    CONSTRAINT chk_navigation_sessions_step_count CHECK (
        step_count >= 0
    ),
    -- Failure reason must be NULL on success and non-NULL on failure
    CONSTRAINT chk_navigation_sessions_failure_reason CHECK (
        (outcome = 'success' AND failure_reason IS NULL)
        OR (outcome = 'failure' AND failure_reason IS NOT NULL)
    )
);

-- ============================================================================
-- Indexes
-- ============================================================================

-- Goal deduplication: find all sessions that targeted the same goal condition
-- (regardless of success/failure) using the compact goal_hash.
CREATE INDEX IF NOT EXISTS idx_navigation_sessions_goal_hash
    ON navigation_sessions(goal_hash);

-- Outcome filtering: retrieve all successful or failed sessions for analytics.
CREATE INDEX IF NOT EXISTS idx_navigation_sessions_outcome
    ON navigation_sessions(outcome);

-- Graph fingerprint: identify all sessions on a specific graph snapshot.
-- Used for cache invalidation when the contract graph changes.
CREATE INDEX IF NOT EXISTS idx_navigation_sessions_graph_fingerprint
    ON navigation_sessions(graph_fingerprint);

-- Temporal: most-recent sessions first (common dashboard / retrieval pattern).
CREATE INDEX IF NOT EXISTS idx_navigation_sessions_created_at
    ON navigation_sessions(created_at DESC);

-- Start-state lookup: find all sessions that began from a given state.
-- Supports the retrieval-augmented planner querying prior experience from a state.
CREATE INDEX IF NOT EXISTS idx_navigation_sessions_start_state
    ON navigation_sessions(start_state_id);

-- Composite: successful sessions from a given start state, newest first.
-- Hot path for the retrieval engine: "what succeeded from state X recently?"
CREATE INDEX IF NOT EXISTS idx_navigation_sessions_success_start_created
    ON navigation_sessions(start_state_id, created_at DESC)
    WHERE outcome = 'success';

-- ============================================================================
-- Comments
-- ============================================================================

COMMENT ON TABLE navigation_sessions IS
    'Completed navigation sessions. Success rows are mirrored in Qdrant '
    '(collection: navigation_paths) for retrieval-augmented navigation. '
    'Failure rows are stored here only — never written to Qdrant.';

COMMENT ON COLUMN navigation_sessions.session_id IS
    'UUID string, primary key. Duplicate inserts are rejected (idempotent).';

COMMENT ON COLUMN navigation_sessions.goal_hash IS
    'SHA-256 hex digest of goal_condition for indexed goal deduplication.';

COMMENT ON COLUMN navigation_sessions.goal_condition IS
    'Typed graph artifact describing the goal state. No raw model output or user content.';

COMMENT ON COLUMN navigation_sessions.start_state_id IS
    'Identifier of the graph state where navigation began.';

COMMENT ON COLUMN navigation_sessions.end_state_id IS
    'Identifier of the terminal graph state (reached goal or last attempted state).';

COMMENT ON COLUMN navigation_sessions.step_count IS
    'Number of plan steps executed. 0 is valid (immediate success or immediate failure).';

COMMENT ON COLUMN navigation_sessions.outcome IS
    'Terminal outcome: ''success'' (goal reached) or ''failure'' (goal not reached).';

COMMENT ON COLUMN navigation_sessions.failure_reason IS
    'Structured failure reason code (e.g., ''no_path_found'', ''max_steps_exceeded''). '
    'NULL when outcome = ''success''.';

COMMENT ON COLUMN navigation_sessions.graph_fingerprint IS
    'Content hash of the contract graph at navigation time. '
    'Used for cache invalidation and drift detection.';

COMMENT ON COLUMN navigation_sessions.steps_json IS
    'JSON-serialised list of PlanStep records (typed graph artifacts only). '
    'Enables deterministic replay and debug inspection.';

COMMENT ON COLUMN navigation_sessions.created_at IS
    'UTC timestamp when this session record was persisted.';
