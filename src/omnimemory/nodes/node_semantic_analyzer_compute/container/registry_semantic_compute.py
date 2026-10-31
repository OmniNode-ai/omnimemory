# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Binding site for the semantic compute handler and its policy (OMN-21623)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omnibase_core.container import ModelONEXContainer

    from omnimemory.models.config import ModelSemanticComputePolicyConfig
    from omnimemory.nodes.node_semantic_analyzer_compute.handlers.handler_semantic_compute import (
        HandlerSemanticCompute,
        HandlerSemanticComputePolicy,
    )

__all__ = ["bind_semantic_compute", "bind_semantic_compute_policy"]


def bind_semantic_compute(container: ModelONEXContainer) -> HandlerSemanticCompute:
    """Build the semantic compute handler (not yet initialized)."""
    from omnimemory.nodes.node_semantic_analyzer_compute.handlers.handler_semantic_compute import (
        HandlerSemanticCompute,
    )

    return HandlerSemanticCompute(container=container)


def bind_semantic_compute_policy(
    config: ModelSemanticComputePolicyConfig,
) -> HandlerSemanticComputePolicy:
    """Build the semantic compute policy from its configuration."""
    from omnimemory.nodes.node_semantic_analyzer_compute.handlers.handler_semantic_compute import (
        HandlerSemanticComputePolicy,
    )

    return HandlerSemanticComputePolicy(config)
