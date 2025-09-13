"""
Memory request model following ONEX standards.
"""

from uuid import UUID

from pydantic import BaseModel, Field

from .enum_memory_operation_type import EnumMemoryOperationType
from .model_memory_context import ModelMemoryContext


class ModelMemoryRequest(BaseModel):
    """Base memory request model following ONEX standards."""

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
    data: dict[str, str] = Field(
        default_factory=dict,
        description="Request data payload with string values for type safety",
    )

    # Request parameters
    parameters: dict[str, str] = Field(
        default_factory=dict,
        description="Operation parameters with string values for type safety",
    )

    # Request options
    options: dict[str, bool] = Field(
        default_factory=dict,
        description="Boolean options for the request",
    )

    # Validation requirements
    validate_input: bool = Field(
        default=True,
        description="Whether to validate input data",
    )
    require_confirmation: bool = Field(
        default=False,
        description="Whether the operation requires explicit confirmation",
    )