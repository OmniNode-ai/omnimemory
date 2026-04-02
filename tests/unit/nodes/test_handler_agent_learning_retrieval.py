# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Tests for agent learning retrieval handler logic."""

from datetime import datetime, timedelta, timezone

import pytest

from omnimemory.nodes.node_agent_learning_retrieval_effect.handlers.handler_agent_learning_retrieval import (
    build_context_query_text,
    build_error_query_text,
    compute_freshness_score,
    rank_and_merge,
)


@pytest.mark.unit
class TestComputeFreshnessScore:
    def test_today(self) -> None:
        now = datetime.now(tz=timezone.utc)
        assert compute_freshness_score(now, now) == pytest.approx(1.0)

    def test_one_week(self) -> None:
        now = datetime.now(tz=timezone.utc)
        score = compute_freshness_score(now - timedelta(weeks=1), now)
        assert 0.85 < score < 0.95

    def test_four_weeks(self) -> None:
        now = datetime.now(tz=timezone.utc)
        score = compute_freshness_score(now - timedelta(weeks=4), now)
        assert 0.60 < score < 0.72

    def test_future_date_clamps_to_one(self) -> None:
        now = datetime.now(tz=timezone.utc)
        score = compute_freshness_score(now + timedelta(days=5), now)
        assert score == pytest.approx(1.0)


@pytest.mark.unit
class TestBuildQueryTexts:
    def test_error_query(self) -> None:
        text = build_error_query_text("ImportError: no module named foo")
        assert "ImportError" in text
        assert text.startswith("Error: ")

    def test_error_query_truncates(self) -> None:
        long_error = "x" * 5000
        text = build_error_query_text(long_error)
        assert len(text) <= 2007  # "Error: " prefix + 2000 chars

    def test_context_query(self) -> None:
        text = build_context_query_text(
            repo="omnibase_infra",
            file_paths=("src/foo.py", "tests/test_foo.py"),
            task_type="ci_fix",
        )
        assert "omnibase_infra" in text
        assert "ci_fix" in text
        assert "src/foo.py" in text

    def test_context_query_no_files(self) -> None:
        text = build_context_query_text(
            repo="omnimemory",
            file_paths=(),
            task_type="bug_fix",
        )
        assert "omnimemory" in text
        assert "Files:" not in text

    def test_context_query_no_task_type(self) -> None:
        text = build_context_query_text(
            repo="omnimemory",
            file_paths=("a.py",),
            task_type=None,
        )
        assert "Task type:" not in text


@pytest.mark.unit
class TestRankAndMerge:
    def test_sorts_by_combined_score(self) -> None:
        matches: list[dict[str, object]] = [
            {"combined_score": 0.5, "match_type": "task_context"},
            {"combined_score": 0.9, "match_type": "error_signature"},
            {"combined_score": 0.7, "match_type": "task_context"},
        ]
        ranked = rank_and_merge(matches, max_results=3)
        assert ranked[0]["combined_score"] == 0.9
        assert ranked[1]["combined_score"] == 0.7
        assert ranked[2]["combined_score"] == 0.5

    def test_respects_max_results(self) -> None:
        matches: list[dict[str, object]] = [
            {"combined_score": float(i) / 10, "match_type": "task_context"}
            for i in range(10)
        ]
        ranked = rank_and_merge(matches, max_results=3)
        assert len(ranked) == 3

    def test_empty_input(self) -> None:
        ranked = rank_and_merge([], max_results=5)
        assert ranked == []
