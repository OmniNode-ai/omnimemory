# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Binding site for the infra HTTP REST handler (OMN-21623)."""

from __future__ import annotations

from omnibase_core.container import ModelONEXContainer
from omnibase_infra.handlers.handler_http import HandlerHttpRest

__all__ = ["bind_http_rest_handler"]


def bind_http_rest_handler(container: ModelONEXContainer) -> HandlerHttpRest:
    """Build the infra HTTP REST handler (not yet initialized)."""
    return HandlerHttpRest(container)
