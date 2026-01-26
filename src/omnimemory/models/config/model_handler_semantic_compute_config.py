# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Handler configuration model for semantic compute operations.

This model configures the semantic compute handler's behavior and wraps
the policy config for runtime decisions.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnimemory.models.config.model_semantic_compute_policy_config import (
    ModelSemanticComputePolicyConfig,
)
from omnimemory.models.foundation.model_semver import ModelSemVer


class ModelHandlerSemanticComputeConfig(BaseModel):
    """Configuration for the semantic compute handler.

    This model configures the handler's behavior and wraps the policy config.
    The handler uses this config to initialize and the policy uses the
    nested policy_config for runtime decisions.

    Example::

        config = ModelHandlerSemanticComputeConfig(
            handler_name="my-semantic-handler",
            policy_config=ModelSemanticComputePolicyConfig(
                cache_embeddings=True,
                entity_extraction_mode=EnumEntityExtractionMode.DETERMINISTIC,
            ),
        )
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    handler_name: str = Field(
        default="semantic-compute",
        min_length=1,
        max_length=100,
        description="Name identifier for this handler instance",
    )

    handler_version: ModelSemVer = Field(
        default_factory=lambda: ModelSemVer.parse("1.0.0"),
        description="Semantic version of the handler",
    )

    policy_config: ModelSemanticComputePolicyConfig = Field(
        default_factory=ModelSemanticComputePolicyConfig,
        description="Policy configuration for runtime decisions",
    )

    enable_caching: bool = Field(
        default=True,
        description="Enable in-memory caching of results",
    )

    max_cache_size: int = Field(
        default=1000,
        ge=0,
        le=100000,
        description="Maximum number of cached items (0 to disable)",
    )
