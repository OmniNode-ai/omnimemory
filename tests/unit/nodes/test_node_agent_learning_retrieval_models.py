# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Tests for agent learning retrieval node models."""

from datetime import datetime, timezone
from uuid import uuid4

import pytest

from omnimemory.nodes.node_agent_learning_retrieval_effect.models.model_request import (
    EnumRetrievalMatchType,
    ModelAgentLearningRetrievalRequest,
)
from omnimemory.nodes.node_agent_learning_retrieval_effect.models.model_response import (
    EnumRetrievalTaskType,
    ModelAgentLearningRetrievalResponse,
    ModelRetrievedLearning,
)


@pytest.mark.unit
class TestRetrievalRequest:
    def test_error_query(self) -> None:
        req = ModelAgentLearningRetrievalRequest(
            match_type=EnumRetrievalMatchType.ERROR_SIGNATURE,
            error_text="ImportError: foo",
            repo="omnibase_core",
        )
        assert req.min_similarity_error == 0.85

    def test_auto_query(self) -> None:
        req = ModelAgentLearningRetrievalRequest(
            repo="omnidash",
            error_text="TypeError: undefined",
            file_paths=("src/app/page.tsx",),
        )
        assert req.match_type == EnumRetrievalMatchType.AUTO

    def test_frozen(self) -> None:
        req = ModelAgentLearningRetrievalRequest(repo="test")
        with pytest.raises(Exception):
            req.repo = "other"  # type: ignore[misc]


@pytest.mark.unit
class TestRetrievalResponse:
    def test_empty_response(self) -> None:
        resp = ModelAgentLearningRetrievalResponse(query_ms=12)
        assert resp.matches == ()

    def test_with_matches(self) -> None:
        match = ModelRetrievedLearning(
            learning_id=uuid4(),
            match_type=EnumRetrievalMatchType.ERROR_SIGNATURE,
            similarity=0.92,
            freshness_score=0.95,
            combined_score=0.87,
            repo="omnibase_infra",
            resolution_summary="Fixed by adding missing import.",
            task_type=EnumRetrievalTaskType.CI_FIX,
            age_days=2,
            created_at=datetime.now(tz=timezone.utc),
        )
        resp = ModelAgentLearningRetrievalResponse(
            matches=(match,),
            query_ms=45,
            error_matches_count=1,
        )
        assert len(resp.matches) == 1
        assert resp.matches[0].similarity == 0.92
