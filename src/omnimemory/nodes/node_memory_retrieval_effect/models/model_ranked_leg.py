# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Results in rank order and the relevance floor for their own leg."""

from pydantic import BaseModel, ConfigDict

from .model_ranked_entry import ModelRankedEntry

__all__ = ["ModelRankedLeg"]


class ModelRankedLeg(BaseModel):
    """Results in rank order and the relevance floor for their own leg."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    entries: tuple[ModelRankedEntry, ...]
    min_score: float
