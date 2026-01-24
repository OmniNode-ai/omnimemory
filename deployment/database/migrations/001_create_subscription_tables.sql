-- Migration: 001_create_subscription_tables
-- Description: Create tables for agent subscriptions
-- Created: 2026-01-22
-- Ticket: OMN-1393
-- Updated: 2026-01-24 - Address PR #22 review: NOT NULL constraints, column alignment (v0.2.1)
--   Notifications now use Kafka event bus, not webhooks.
--   If WebhookEmitterEffect is needed later, add webhook columns to that node.

-- ============================================================================
-- Subscriptions Table
-- ============================================================================
-- Stores agent subscriptions to memory change notifications.
-- Topic format: memory.<entity>.<event> (e.g., memory.item.created)
-- Delivery: Agents consume from Kafka directly via consumer groups.

CREATE TABLE IF NOT EXISTS subscriptions (
    id UUID PRIMARY KEY,
    agent_id VARCHAR(255) NOT NULL,
    topic VARCHAR(256) NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'active',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    suspended_reason TEXT,
    metadata JSONB,
    CONSTRAINT uq_subscriptions_agent_topic UNIQUE(agent_id, topic),
    CONSTRAINT chk_subscriptions_status CHECK (status IN ('active', 'suspended', 'deleted'))
);

-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_subscriptions_agent_id
    ON subscriptions(agent_id);

CREATE INDEX IF NOT EXISTS idx_subscriptions_topic
    ON subscriptions(topic);

CREATE INDEX IF NOT EXISTS idx_subscriptions_status
    ON subscriptions(status);

-- Partial index for active subscriptions (most common query)
CREATE INDEX IF NOT EXISTS idx_subscriptions_active_topic
    ON subscriptions(topic)
    WHERE status = 'active';

-- Optimizes: SELECT id FROM subscriptions WHERE topic = $1 AND status = 'active'
-- Composite index for topic-based subscriber lookups (hot path in notify())
CREATE INDEX IF NOT EXISTS idx_subscriptions_topic_id_active
    ON subscriptions(topic, id)
    WHERE status = 'active';

-- Optimizes: SELECT ... WHERE agent_id = $1 AND status = 'active' (no ORDER BY)
-- Covering index for non-paginated list_subscriptions() queries
CREATE INDEX IF NOT EXISTS idx_subscriptions_active_agent
    ON subscriptions(agent_id)
    WHERE status = 'active';

-- Optimizes: SELECT ... WHERE agent_id = $1 AND status = 'active' ORDER BY created_at DESC LIMIT N
-- Composite index for paginated queries (ORDER BY created_at DESC)
CREATE INDEX IF NOT EXISTS idx_subscriptions_agent_created_desc
    ON subscriptions(agent_id, created_at DESC)
    WHERE status = 'active';

-- ============================================================================
-- Helper Functions
-- ============================================================================

-- Function to update updated_at timestamp automatically
CREATE OR REPLACE FUNCTION omnimemory_update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Trigger for subscriptions table
DROP TRIGGER IF EXISTS omnimemory_trigger_subscriptions_updated_at ON subscriptions;
CREATE TRIGGER omnimemory_trigger_subscriptions_updated_at
    BEFORE UPDATE ON subscriptions
    FOR EACH ROW
    EXECUTE FUNCTION omnimemory_update_updated_at_column();

-- ============================================================================
-- Comments
-- ============================================================================

COMMENT ON TABLE subscriptions IS
    'Agent subscriptions to memory change notifications. Topic format: memory.<entity>.<event>. Delivery via Kafka.';

COMMENT ON COLUMN subscriptions.agent_id IS
    'Unique identifier of the subscribing agent';

COMMENT ON COLUMN subscriptions.topic IS
    'Memory event topic pattern (e.g., memory.item.created, memory.collection.updated)';

COMMENT ON COLUMN subscriptions.status IS
    'Subscription status: active, suspended, or deleted';

COMMENT ON COLUMN subscriptions.metadata IS
    'Optional JSON metadata for the subscription';
