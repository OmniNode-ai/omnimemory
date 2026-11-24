# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The stores a memory search is answered from (OMN-26659)."""

from enum import Enum


class EnumMemoryStore(str, Enum):
    """A store the retrieval handler can be asked to search.

    A refusal names the store by this value, so a caller can tell which
    backend is missing or down without parsing a message.
    """

    VECTOR = "vector"
    DATABASE = "database"
    GRAPH = "graph"
