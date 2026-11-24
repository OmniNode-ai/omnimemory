# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Test-only stand-ins for the memory retrieval stores (OMN-26659).

Nothing under ``src/`` imports this package: the runtime resolves its stores
from settings and answers a typed refusal when one is not configured.
"""
