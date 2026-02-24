# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Handlers for the semantic analyzer compute node.

This module exports the handler implementations for semantic analysis
operations. Following ONEX patterns, handlers live inside their node's
directory structure.

.. versionadded:: 0.1.0
    Initial implementation for OMN-1390.
"""

# Re-export config from models for convenience
from omnimemory.models.config import ModelHandlerSemanticComputeConfig
from omnimemory.nodes.semantic_analyzer_compute.handlers.handler_semantic_compute import (
    HandlerSemanticCompute,
    HandlerSemanticComputePolicy,
)

__all__ = [
    "HandlerSemanticCompute",
    "HandlerSemanticComputePolicy",
    "ModelHandlerSemanticComputeConfig",
]
