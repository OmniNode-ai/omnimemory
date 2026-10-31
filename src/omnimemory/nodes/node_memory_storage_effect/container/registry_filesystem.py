# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Binding site for the filesystem storage handler (OMN-21623)."""

from __future__ import annotations

from omnibase_infra.handlers.handler_filesystem import HandlerFileSystem

__all__ = ["HandlerFileSystem", "bind_filesystem_handler"]


def bind_filesystem_handler() -> HandlerFileSystem:
    """Build the infra filesystem handler (not yet initialized)."""
    return HandlerFileSystem()
