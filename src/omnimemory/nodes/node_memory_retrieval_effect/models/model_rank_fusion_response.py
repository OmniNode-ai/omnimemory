# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Relevant ids per leg, accumulated RRF scores, and final ranking."""

from pydantic import BaseModel, ConfigDict

__all__ = ["ModelRankFusionResponse"]


class ModelRankFusionResponse(BaseModel):
    """Relevant ids per leg, accumulated RRF scores, and final ranking."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    relevant_legs: tuple[tuple[str, ...], ...]
    scores: dict[str, float]
    ranking: tuple[str, ...]
