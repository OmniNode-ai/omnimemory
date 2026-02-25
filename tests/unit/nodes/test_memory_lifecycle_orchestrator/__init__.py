# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Unit tests for memory_lifecycle_orchestrator handlers.

This package contains unit tests for the lifecycle orchestrator handlers:
- HandlerMemoryTick: RuntimeTick evaluation for TTL and archive candidates
- HandlerMemoryExpire: ACTIVE -> EXPIRED state transition with optimistic locking
- HandlerMemoryArchive: EXPIRED -> ARCHIVED with cold storage archival

Related Tickets:
    - OMN-1453: OmniMemory P4b - Lifecycle Orchestrator Database Integration
"""
