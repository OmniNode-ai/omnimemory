# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""KreuzbergParseEffect node package.

Subscribes to document-discovered and document-changed events,
calls the kreuzberg HTTP service to extract plain text, and emits
document-indexed (kreuzberg variant) or document-parse-failed events.

Related: OMN-2733
"""
