# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Shared type aliases for the crawl domain."""

from typing import Literal

__all__ = ["TriggerSource"]

# Valid trigger sources for crawl events.
# Used by event models and the FilesystemCrawler handler.
TriggerSource = Literal["scheduled", "manual", "git_hook", "filesystem_watch"]
