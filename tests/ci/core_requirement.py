# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Resolve the omnibase_core requirement a CI job installs into a runnable one (OMN-22360).

``ci.yml`` builds the omnibase_core git URL from the GitHub context
(``github.server_url`` and ``github.repository_owner``) so the workflow carries no
first-party URL literal. A test that runs the requirement outside GitHub Actions
needs the same URL; it is read from the omnibase_core entry of
``.pre-commit-config.yaml``, the one place this repository spells it.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CORE_REPO_SUFFIX = "/omnibase_core"
GITHUB_CONTEXT_URL = re.compile(
    r"\$\{\{ github\.server_url \}\}/\$\{\{ github\.repository_owner \}\}/omnibase_core"
)


def core_repo_url() -> str:
    """The omnibase_core repository URL of the pre-commit config's core hook entries."""
    config = yaml.safe_load(
        (ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
    )
    urls = {
        str(entry["repo"]).rstrip("/").removesuffix(".git")
        for entry in config["repos"]
        if str(entry.get("repo", "")).rstrip("/").endswith(CORE_REPO_SUFFIX)
    }
    assert len(urls) == 1, urls
    return urls.pop()


def runnable_core_requirement(requirement: str) -> str:
    """Replace the GitHub-context URL of ``requirement`` with the real core URL."""
    return GITHUB_CONTEXT_URL.sub(core_repo_url(), requirement)
