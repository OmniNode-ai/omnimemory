# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""One document and its score from a single retrieval leg."""

from pydantic import BaseModel, ConfigDict

__all__ = ["ModelRankedEntry"]


class ModelRankedEntry(BaseModel):
    """One document and its score from a single retrieval leg."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    document_id: str
    score: float
