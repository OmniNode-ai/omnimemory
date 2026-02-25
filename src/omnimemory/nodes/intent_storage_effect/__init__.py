# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Intent Storage Effect - ONEX Node (Demo Critical Path).

Stores intent classifications in Memgraph, linking them to sessions
for analytics and pattern learning.

This is a fully declarative ONEX node:
- Node behavior defined in contract.yaml
- Business logic implemented in adapters/
- No node.py class needed

Node Type: EFFECT
"""

from .adapters import HandlerIntentStorageAdapter
from .models import ModelIntentStorageRequest, ModelIntentStorageResponse

__all__ = [
    # Models
    "ModelIntentStorageRequest",
    "ModelIntentStorageResponse",
    # Handlers
    "HandlerIntentStorageAdapter",
]
