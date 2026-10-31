# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Ranked legs to gate and fuse, with an optional smoothing constant."""

from pydantic import BaseModel, ConfigDict

from .model_ranked_leg import ModelRankedLeg

__all__ = ["ModelRankFusionRequest"]


class ModelRankFusionRequest(BaseModel):
    """Ranked legs to gate and fuse, with an optional smoothing constant."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    legs: tuple[ModelRankedLeg, ...]
    k: int | None = None
