# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The onex-validate-links hook runs the link check module (OMN-25070).

omnibase_core will drop the onex-validate-links console script from its package,
so the hook must run the module.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

CONFIG = Path(__file__).resolve().parents[2] / ".pre-commit-config.yaml"
MODULE_COMMAND = (
    "python -m omnibase_core.nodes.node_markdown_links_check_compute"
    ".runtime_markdown_links_check --verbose"
)


def _hooks() -> list[dict[str, object]]:
    with CONFIG.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    hooks: list[dict[str, object]] = []
    for repo in config.get("repos", []) or []:
        if isinstance(repo, dict):
            hooks.extend(h for h in repo.get("hooks", []) or [] if isinstance(h, dict))
    return hooks


@pytest.mark.unit
def test_exactly_one_onex_validate_links_hook() -> None:
    validate_hooks = [h for h in _hooks() if h.get("id") == "onex-validate-links"]

    assert len(validate_hooks) == 1
    entry = validate_hooks[0].get("entry")
    assert isinstance(entry, str)
    assert entry.endswith(MODULE_COMMAND), entry


@pytest.mark.unit
def test_no_hook_invokes_onex_validate_links_console_script() -> None:
    for hook in _hooks():
        entry = hook.get("entry")
        if not isinstance(entry, str):
            continue
        tokens = entry.split()
        if tokens[:2] == ["uv", "run"]:
            tokens = tokens[2:]
        assert tokens[:1] != ["onex-validate-links"], (hook.get("id"), entry)
