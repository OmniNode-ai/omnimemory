# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Timestamps for activation decay against an injected reference time."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict

__all__ = ["ModelActivationDecayRequest"]


class ModelActivationDecayRequest(BaseModel):
    """Timestamps for activation decay against an injected reference time."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    created_at: datetime
    last_accessed_at: datetime | None = None
    now: datetime
