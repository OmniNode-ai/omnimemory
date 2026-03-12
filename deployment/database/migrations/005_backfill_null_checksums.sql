-- =============================================================================
-- MIGRATION: Backfill NULL checksums and enforce NOT NULL on schema_migrations
-- =============================================================================
-- Ticket: OMN-4701 (OMN-4653 root cause fix — omnimemory DB)
-- Version: 1.0.0
--
-- PURPOSE:
--   The live omnimemory DB has 4 NULL checksum rows in schema_migrations.
--   These rows were applied before checksum tracking was implemented.
--
--   This migration:
--   1. Backfills NULL checksums with a sentinel prefix so they are
--      identifiable as pre-checksum-era rows.
--   2. Enforces NOT NULL on the checksum column to prevent future nulls.
--
-- SAFETY:
--   Wrapped in a transaction. Idempotent: UPDATE only touches NULL rows.
--   ALTER TABLE SET NOT NULL is safe once all rows have a non-null checksum.
-- =============================================================================

BEGIN;

-- Step 1: Backfill all NULL checksum rows with a stable sentinel value.
UPDATE schema_migrations
SET checksum = 'backfilled:pre-checksum-era:' || version
WHERE checksum IS NULL;

-- Step 2: Enforce NOT NULL now that all rows have a value.
ALTER TABLE schema_migrations
    ALTER COLUMN checksum SET NOT NULL;

-- Step 3: Document the column contract with a comment.
COMMENT ON COLUMN schema_migrations.checksum IS
    'SHA-256 of migration file at apply time. '
    'Prefix "backfilled:pre-checksum-era:" = applied before checksum tracking '
    '(2026-03-12 backfill, OMN-4653 / OMN-4701).';

COMMIT;
