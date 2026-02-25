# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Integration tests package for OmniMemory.

This package contains integration tests that verify end-to-end functionality
with real or mocked external services (PostgreSQL, Valkey, webhooks, etc.).

Test Categories:
    - Subscription: HandlerSubscription integration tests
    - Orchestrator: NodeAgentCoordinatorOrchestrator integration tests
    - Nodes: ONEX node integration tests

Prerequisites:
    - PostgreSQL running (for subscription tests)
    - Valkey running (for subscription tests)
    - omnibase_infra installed (dev dependency)

Usage:
    # Run all integration tests
    pytest tests/integration/ -v

    # Run with specific markers
    pytest -m "integration" -v

    # Skip if external services unavailable (automatic)
    pytest -m integration -v

Environment Variables:
    TEST_DB_DSN: PostgreSQL connection string
    TEST_VALKEY_HOST: Valkey hostname (default: localhost)
    TEST_VALKEY_PORT: Valkey port (default: 6379)

.. versionadded:: 0.1.0
    Initial implementation for OMN-1393.
"""
