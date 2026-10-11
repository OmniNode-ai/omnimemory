# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The navigation history reducer's embedding endpoint: env, then overlay, else none (OMN-20936)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from omnimemory.runtime.plugin import _navigation_embedding_url

pytestmark = pytest.mark.unit

_OVERLAY_URL = "http://192.0.2.31:8100"
_ENV_URL = "http://192.0.2.32:8100"


def _write_overlay(root: Path, body: str) -> None:
    directory = root / "node_navigation_history_reducer"
    directory.mkdir(parents=True)
    (directory / "overlay.yaml").write_text(body, encoding="utf-8")


@pytest.fixture(autouse=True)
def _clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLM_EMBEDDING_URL", raising=False)
    monkeypatch.delenv("ONEX_SKILL_OVERLAY_ROOTS", raising=False)


def test_with_the_overlay_root_the_overlay_url_resolves(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _write_overlay(tmp_path, f'embedding_url: "{_OVERLAY_URL}"\n')
    monkeypatch.setenv("ONEX_SKILL_OVERLAY_ROOTS", str(tmp_path))
    assert _navigation_embedding_url() == _OVERLAY_URL


def test_without_the_overlay_root_no_endpoint_is_configured() -> None:
    assert _navigation_embedding_url() == ""


def test_a_root_without_the_overlay_file_is_no_overlay(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("ONEX_SKILL_OVERLAY_ROOTS", str(tmp_path))
    assert _navigation_embedding_url() == ""


def test_the_environment_variable_wins_over_the_overlay(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _write_overlay(tmp_path, f'embedding_url: "{_OVERLAY_URL}"\n')
    monkeypatch.setenv("ONEX_SKILL_OVERLAY_ROOTS", str(tmp_path))
    monkeypatch.setenv("LLM_EMBEDDING_URL", _ENV_URL)
    assert _navigation_embedding_url() == _ENV_URL


def test_the_first_root_that_holds_the_overlay_wins(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    empty, first, second = (tmp_path / name for name in ("empty", "first", "second"))
    empty.mkdir()
    _write_overlay(first, f'embedding_url: "{_OVERLAY_URL}"\n')
    _write_overlay(second, f'embedding_url: "{_ENV_URL}"\n')
    monkeypatch.setenv(
        "ONEX_SKILL_OVERLAY_ROOTS",
        os.pathsep.join(str(p) for p in (empty, first, second)),
    )
    assert _navigation_embedding_url() == _OVERLAY_URL


@pytest.mark.parametrize(
    "body",
    [
        "embedding_url: 8100\n",
        "embedding_url: [a, b]\n",
        'other: "x"\nembedding_url: "http://192.0.2.31:8100"\n',
        "- just\n- a list\n",
        "embedding_url: {unterminated\n",
    ],
)
def test_a_malformed_overlay_is_refused_not_skipped(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, body: str
) -> None:
    _write_overlay(tmp_path, body)
    monkeypatch.setenv("ONEX_SKILL_OVERLAY_ROOTS", str(tmp_path))
    with pytest.raises(ValueError, match="overlay"):
        _navigation_embedding_url()
