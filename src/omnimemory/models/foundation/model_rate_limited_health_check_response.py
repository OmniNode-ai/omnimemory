"""
Rate-limited health check response model following ONEX standards.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

from .model_health_response_main import ModelHealthResponse


class ModelRateLimitedHealthCheckResponse(BaseModel):
    """Rate-limited health check response."""

    health_check: Optional[ModelHealthResponse] = Field(
        default=None, description="Health check result if within rate limit"
    )
    rate_limited: bool = Field(description="Whether the request was rate limited")
    rate_limit_reset_time: Optional[datetime] = Field(
        default=None, description="When the rate limit will reset"
    )
    remaining_requests: Optional[int] = Field(
        default=None, description="Number of requests remaining in the current window"
    )
    error_message: Optional[str] = Field(
        default=None, description="Error message if rate limited"
    )
