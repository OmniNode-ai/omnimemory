-- Migration: 001_create_subscription_tables
-- Description: Create tables for agent subscriptions and notification delivery tracking
-- Created: 2025-01-22
-- Ticket: OMN-1393

-- ============================================================================
-- Subscriptions Table
-- ============================================================================
-- Stores agent subscriptions to memory change notifications.
-- Topic format: memory.<entity>.<event> (e.g., memory.item.created)

CREATE TABLE IF NOT EXISTS subscriptions (
    id UUID PRIMARY KEY,
    agent_id VARCHAR(255) NOT NULL,
    topic VARCHAR(255) NOT NULL,
    webhook_url TEXT NOT NULL,
    webhook_secret TEXT,
    webhook_headers JSONB,
    webhook_timeout_ms INTEGER DEFAULT 5000,
    status VARCHAR(50) DEFAULT 'active',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    suspended_reason TEXT,
    metadata JSONB,
    CONSTRAINT uq_subscriptions_agent_topic UNIQUE(agent_id, topic)
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

CREATE INDEX IF NOT EXISTS idx_subscriptions_active_agent
    ON subscriptions(agent_id)
    WHERE status = 'active';

-- ============================================================================
-- Delivery Attempts Table
-- ============================================================================
-- Tracks notification delivery attempts for retry logic and DLQ management.
-- Each delivery attempt is recorded separately to support exponential backoff.

CREATE TABLE IF NOT EXISTS delivery_attempts (
    id UUID PRIMARY KEY,
    subscription_id UUID NOT NULL REFERENCES subscriptions(id) ON DELETE CASCADE,
    event_id UUID NOT NULL,
    attempt_number INTEGER DEFAULT 1,
    status VARCHAR(50) NOT NULL,
    status_code INTEGER,
    error_message TEXT,
    response_body TEXT,
    latency_ms INTEGER,
    next_retry_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    CONSTRAINT uq_delivery_attempt UNIQUE(subscription_id, event_id, attempt_number)
);

-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_delivery_attempts_subscription_id
    ON delivery_attempts(subscription_id);

CREATE INDEX IF NOT EXISTS idx_delivery_attempts_event_id
    ON delivery_attempts(event_id);

CREATE INDEX IF NOT EXISTS idx_delivery_attempts_status
    ON delivery_attempts(status);

-- Index for time-based queries (delivery history, cleanup operations)
CREATE INDEX IF NOT EXISTS idx_delivery_attempts_created_at
    ON delivery_attempts(created_at);

-- Partial index for pending retries (used by retry scheduler)
CREATE INDEX IF NOT EXISTS idx_delivery_attempts_pending_retry
    ON delivery_attempts(next_retry_at)
    WHERE status = 'failed' AND next_retry_at IS NOT NULL;

-- Partial index for DLQ items
CREATE INDEX IF NOT EXISTS idx_delivery_attempts_dlq
    ON delivery_attempts(subscription_id, created_at)
    WHERE status = 'dlq';

-- Composite index for querying delivery status by subscription
-- Used by: retry worker, subscription health checks, delivery history
CREATE INDEX IF NOT EXISTS idx_delivery_attempts_sub_status
    ON delivery_attempts(subscription_id, status);

-- ============================================================================
-- Circuit Breaker States Table
-- ============================================================================
-- Tracks circuit breaker state per webhook endpoint.
-- Used for persistence across handler restarts (Valkey is primary cache).

CREATE TABLE IF NOT EXISTS circuit_breaker_states (
    endpoint VARCHAR(512) PRIMARY KEY,
    state VARCHAR(50) DEFAULT 'closed',
    failure_count INTEGER DEFAULT 0,
    success_count INTEGER DEFAULT 0,
    last_failure_at TIMESTAMPTZ,
    last_success_at TIMESTAMPTZ,
    last_error_message TEXT,
    open_until TIMESTAMPTZ,
    total_requests INTEGER DEFAULT 0,
    total_failures INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index for open/half_open circuits (for monitoring dashboards)
CREATE INDEX IF NOT EXISTS idx_circuit_breaker_open
    ON circuit_breaker_states(state)
    WHERE state IN ('open', 'half_open');

-- ============================================================================
-- Helper Functions
-- ============================================================================

-- Function to update updated_at timestamp automatically
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Trigger for subscriptions table
DROP TRIGGER IF EXISTS trigger_subscriptions_updated_at ON subscriptions;
CREATE TRIGGER trigger_subscriptions_updated_at
    BEFORE UPDATE ON subscriptions
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Trigger for circuit_breaker_states table
DROP TRIGGER IF EXISTS trigger_circuit_breaker_updated_at ON circuit_breaker_states;
CREATE TRIGGER trigger_circuit_breaker_updated_at
    BEFORE UPDATE ON circuit_breaker_states
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- Comments
-- ============================================================================

COMMENT ON TABLE subscriptions IS
    'Agent subscriptions to memory change notifications. Topic format: memory.<entity>.<event>';

COMMENT ON COLUMN subscriptions.agent_id IS
    'Unique identifier of the subscribing agent';

COMMENT ON COLUMN subscriptions.topic IS
    'Memory event topic pattern (e.g., memory.item.created, memory.collection.updated)';

COMMENT ON COLUMN subscriptions.webhook_url IS
    'URL to POST notifications to (HTTPS recommended in production)';

COMMENT ON COLUMN subscriptions.webhook_secret IS
    'Shared secret for HMAC-SHA256 signature verification (X-Signature-256 header)';

COMMENT ON COLUMN subscriptions.webhook_headers IS
    'Custom headers to include in webhook requests (JSON object)';

COMMENT ON COLUMN subscriptions.webhook_timeout_ms IS
    'Request timeout in milliseconds (default: 5000ms)';

COMMENT ON COLUMN subscriptions.status IS
    'Subscription status: active, suspended, or deleted';

COMMENT ON TABLE delivery_attempts IS
    'Notification delivery attempt tracking for retry logic and dead letter queue';

COMMENT ON COLUMN delivery_attempts.subscription_id IS
    'Reference to the target subscription';

COMMENT ON COLUMN delivery_attempts.event_id IS
    'UUID of the notification event being delivered';

COMMENT ON COLUMN delivery_attempts.attempt_number IS
    'Delivery attempt number (1-based, increases with each retry)';

COMMENT ON COLUMN delivery_attempts.status IS
    'Delivery status: pending, success, failed, or dlq';

COMMENT ON COLUMN delivery_attempts.next_retry_at IS
    'Scheduled time for next retry attempt (NULL if success or dlq)';

COMMENT ON TABLE circuit_breaker_states IS
    'Circuit breaker state per webhook endpoint for cascade failure protection';

COMMENT ON COLUMN circuit_breaker_states.endpoint IS
    'Webhook endpoint URL (primary key)';

COMMENT ON COLUMN circuit_breaker_states.state IS
    'Circuit state: closed (normal), open (blocking), or half_open (testing)';

COMMENT ON COLUMN circuit_breaker_states.open_until IS
    'Timestamp when circuit will transition from open to half_open';
