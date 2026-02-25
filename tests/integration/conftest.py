# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Shared fixtures for integration tests.

This module provides reusable fixtures for integration tests that require
unique identifiers for test isolation.

These fixtures are shared across:
- test_handler_subscription.py
- test_node_agent_coordinator.py
"""

from __future__ import annotations

from uuid import uuid4

import pytest


@pytest.fixture
def unique_agent_id() -> str:
    """Generate a unique agent ID for test isolation.

    Returns:
        Unique agent identifier.
    """
    return f"test_agent_{uuid4().hex[:8]}"


@pytest.fixture
def unique_topic() -> str:
    """Generate a unique topic for test isolation.

    Returns:
        Unique topic in memory.<entity>.<event> format.
    """
    return f"memory.test_{uuid4().hex[:8]}.created"
