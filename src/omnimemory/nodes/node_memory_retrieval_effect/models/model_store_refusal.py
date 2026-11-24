# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Typed refusal naming the store that could not answer a memory search (OMN-26659)."""

from pydantic import BaseModel, ConfigDict, Field

from omnimemory.enums.enum_memory_store import EnumMemoryStore
from omnimemory.enums.enum_store_refusal_reason import EnumStoreRefusalReason

__all__ = ["ModelStoreRefusal"]


class ModelStoreRefusal(BaseModel):
    """A search the runtime declined to answer, and the store that is the reason.

    Carried on a ``refused`` retrieval response in place of rows. The runtime
    never substitutes data for a store it cannot reach: the caller is told
    which store, and why, and gets nothing else.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    store: EnumMemoryStore = Field(
        ...,
        description="The store that is not configured or not reachable",
    )
    reason: EnumStoreRefusalReason = Field(
        ...,
        description="Whether the store is not configured or not reachable",
    )
    detail: str = Field(
        ...,
        min_length=1,
        description="What was missing or what the store raised, for the operator",
    )
