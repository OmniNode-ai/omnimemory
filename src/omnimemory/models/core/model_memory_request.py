# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""
Memory request model following ONEX standards.
"""

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from ...enums.enum_memory_operation_type import EnumMemoryOperationType
from ..foundation.model_memory_data import ModelMemoryRequestData
from ..foundation.model_semver import ModelSemVer
from .model_memory_context import ModelMemoryContext
from .model_memory_parameters import ModelMemoryOptions, ModelMemoryParameters


class ModelMemoryRequest(BaseModel):
    """Base memory request model following ONEX standards."""

    model_config = ConfigDict(frozen=False, extra="forbid")

    # Contract version for schema tracking
    contract_version: ModelSemVer = Field(
        default_factory=lambda: ModelSemVer.parse("1.0.0"),
        description="Schema version for this request contract as ModelSemVer",
    )

    # Request identification
    request_id: UUID = Field(
        description="Unique identifier for this request",
    )
    operation_type: EnumMemoryOperationType = Field(
        description="Type of memory operation requested",
    )

    # Context information
    context: ModelMemoryContext = Field(
        description="Context information for the request",
    )

    # Request payload
    data: ModelMemoryRequestData | None = Field(
        default=None,
        description="Structured request data payload following ONEX standards",
    )

    # Request parameters - using structured model instead of dict
    parameters: ModelMemoryParameters = Field(
        default_factory=ModelMemoryParameters,
        description="Structured operation parameters following ONEX standards",
    )

    # Request options - using structured model instead of dict
    options: ModelMemoryOptions = Field(
        default_factory=ModelMemoryOptions,
        description="Boolean options for the request following ONEX standards",
    )
