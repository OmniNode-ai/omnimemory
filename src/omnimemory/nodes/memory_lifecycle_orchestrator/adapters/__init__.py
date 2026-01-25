# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Memory Lifecycle Orchestrator Adapters.

Storage adapters for memory lifecycle operations including archive storage
and projection readers.

Adapters (to be implemented):
    - AdapterArchiveStorage: Cold storage adapter for archived memories
    - ProjectionReaderMemoryLifecycle: Read-only projection state access

Adapter Pattern:
    Adapters wrap external dependencies (storage backends, databases) and
    provide a consistent interface for handlers. All adapters implement
    protocol-based interfaces for dependency injection and testing.

.. versionadded:: 0.1.0
    Initial implementation for OMN-1453.

Ticket: OMN-1453
"""

# TODO(OMN-1453): Add adapter imports as implemented:
#   AdapterArchiveStorage, ProjectionReaderMemoryLifecycle

__all__: list[str] = []
