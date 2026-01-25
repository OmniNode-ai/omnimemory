# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Integration tests for ONEX nodes.

This package contains integration tests that verify node behavior with
real PostgreSQL, Valkey, and other external services.

Test Categories:
    - memory_lifecycle_orchestrator: Concurrency and atomicity tests

Prerequisites:
    - PostgreSQL running (for projection queries)
    - Valkey running (for distributed locks)
    - omnibase_infra installed (dev dependency)
    - OMN-1524 infra primitives implemented

Usage:
    # Run all node integration tests
    pytest tests/integration/nodes/ -v

    # Run with specific markers
    pytest -m "integration and concurrency" -v

Environment Variables:
    TEST_DB_DSN: PostgreSQL connection string
    TEST_VALKEY_HOST: Valkey hostname (default: localhost)
    TEST_VALKEY_PORT: Valkey port (default: 6379)

.. versionadded:: 0.1.0
    Initial implementation for OMN-1453.
"""
