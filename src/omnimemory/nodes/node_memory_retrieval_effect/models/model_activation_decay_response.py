# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Freshest timestamp, activation multiplier, and stale classification."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict

__all__ = ["ModelActivationDecayResponse"]


class ModelActivationDecayResponse(BaseModel):
    """Freshest timestamp, activation multiplier, and stale classification."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    freshest_timestamp: datetime
    activation: float
    is_stale: bool
