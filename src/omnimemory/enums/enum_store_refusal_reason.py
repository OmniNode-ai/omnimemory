# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Why a memory search was refused instead of answered (OMN-26659)."""

from enum import Enum


class EnumStoreRefusalReason(str, Enum):
    """The ways a store can fail to answer a search."""

    NOT_CONFIGURED = "not_configured"
    UNREACHABLE = "unreachable"
    FAILED = "failed"
