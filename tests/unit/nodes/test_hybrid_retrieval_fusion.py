# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""OMN-16765 — retrieval-quality gate for hybrid rank fusion.

This is the Step 2 deliverable: the failing test, written before the fusion
implementation exists.

Design decisions this file encodes, all argued in
``docs/design/OMN-16765-hybrid-retrieval.md``:

* **Marker is ``unit``, not ``integration``.** ``pyproject.toml`` defines
  ``unit`` as "no external dependencies". This test opens no connection and
  touches no network — both retrieval legs are pre-computed and checked into
  ``tests/fixtures/omn16765/hybrid_retrieval_corpus.json``. Sections 4 and 5 of
  the ticket say "integration test"; that wording predates the ruling requiring
  the measurement be deterministic and CI-runnable, and ``ci.yml`` starts no
  service containers, so an integration-shaped test could not pass the merge
  gate at all.
* **Metric is NDCG@10.** The corpus carries graded relevance (2 = the answer,
  1 = related, 0 = irrelevant), so precision@k — which is blind to both grade
  and rank order within the cut — is the wrong instrument for evaluating a
  *ranking* change.
* **Three query families, asserted differently.** ``exact_token`` must
  *improve*; ``paraphrase`` and ``agreement`` must not *regress*. A fusion
  measured only on the cases it was designed to fix will always look good.

The NDCG helpers below carry their own self-tests. Those pass immediately and
are guards on the metric, so a failure in the fusion assertions cannot be
confused with a broken metric.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Sequence

# Kept beside its only consumer rather than under a top-level tests/fixtures/.
# scripts/ci/test_selection_adjacency.yaml treats top-level tests/<dir>/ as a
# declared test family, and a pure data directory is not one — living under
# tests/unit/ means a change to the corpus falls back to selecting the unit
# family, which is the behaviour we want.
CORPUS_PATH = (
    Path(__file__).parent / "fixtures" / "omn16765" / "hybrid_retrieval_corpus.json"
)

NDCG_K = 10

# Absolute NDCG@10 margin by which fusion must beat the vector-only arm on
# exact-token queries. Conservative: on this corpus the correct document sits
# at vector rank 4-6, and the lexical leg ranks it first.
EXACT_TOKEN_MIN_GAIN = 0.15

# Tolerance for the non-regression families. Not zero, because rank fusion can
# legitimately perturb an already-correct ordering by a hair. Deliberately
# tight: a leg returning irrelevant results still contributes full rank-1
# weight under naive RRF, and if that sinks a correct semantic ranking, this
# test is supposed to catch it rather than wave it through.
NON_REGRESSION_TOLERANCE = 0.02


# =============================================================================
# Metric
# =============================================================================


def dcg_at_k(gains: Sequence[int], k: int) -> float:
    """Discounted cumulative gain over graded relevance.

    Uses the exponential-gain form, ``(2**rel - 1) / log2(i + 1)``, which is
    the standard choice when grades are ordinal rather than binary: it makes
    the gap between "the answer" and "merely related" larger than the gap
    between "related" and "irrelevant".
    """
    return sum(
        (2**rel - 1) / math.log2(i + 2) for i, rel in enumerate(gains[:k]) if rel > 0
    )


def ndcg_at_k(ranked_ids: Sequence[str], labels: dict[str, int], k: int) -> float:
    """Normalized DCG@k. Returns 0.0 when no relevant document exists."""
    gains = [labels.get(doc_id, 0) for doc_id in ranked_ids]
    ideal = sorted(labels.values(), reverse=True)
    idcg = dcg_at_k(ideal, k)
    if idcg == 0.0:
        return 0.0
    return dcg_at_k(gains, k) / idcg


@pytest.mark.unit
def test_ndcg_perfect_ranking_is_one() -> None:
    labels = {"a": 2, "b": 1, "c": 1}
    assert ndcg_at_k(["a", "b", "c"], labels, NDCG_K) == pytest.approx(1.0)


@pytest.mark.unit
def test_ndcg_reversed_ranking_is_worse_than_perfect() -> None:
    labels = {"a": 2, "b": 1, "c": 0}
    perfect = ndcg_at_k(["a", "b", "c"], labels, NDCG_K)
    reversed_ = ndcg_at_k(["c", "b", "a"], labels, NDCG_K)
    assert reversed_ < perfect


@pytest.mark.unit
def test_ndcg_is_zero_when_nothing_relevant_exists() -> None:
    assert ndcg_at_k(["a", "b"], {}, NDCG_K) == 0.0


@pytest.mark.unit
def test_ndcg_rewards_promoting_the_graded_answer() -> None:
    """The property the whole gate rests on: rank order inside the cut matters.

    This is precisely what precision@k cannot see — both rankings below have
    the same set of documents in the top 3.
    """
    labels = {"right": 2, "related": 1}
    top = ndcg_at_k(["right", "related", "junk"], labels, NDCG_K)
    buried = ndcg_at_k(["junk", "related", "right"], labels, NDCG_K)
    assert top > buried


@pytest.mark.unit
def test_ndcg_ignores_documents_beyond_k() -> None:
    labels = {"right": 2}
    within = ndcg_at_k(["x"] * 9 + ["right"], labels, NDCG_K)
    beyond = ndcg_at_k(["x"] * 10 + ["right"], labels, NDCG_K)
    assert within > 0.0
    assert beyond == 0.0


# =============================================================================
# Corpus
# =============================================================================


def load_corpus() -> list[dict[str, object]]:
    """Load the checked-in labelled corpus."""
    with CORPUS_PATH.open(encoding="utf-8") as fh:
        return list(json.load(fh)["queries"])


def ranked_ids(leg: Sequence[dict[str, object]]) -> list[str]:
    """Extract document ids from a pre-computed leg, preserving rank order."""
    return [str(entry["doc_id"]) for entry in leg]


@pytest.mark.unit
def test_corpus_is_present_and_well_formed() -> None:
    queries = load_corpus()
    assert len(queries) == 24
    for q in queries:
        assert q["family"] in {"exact_token", "paraphrase", "agreement"}
        assert q["query"]
        assert q["labels"], f"{q['query_id']} has no relevance labels"
        assert 2 in q["labels"].values(), (  # type: ignore[union-attr]
            f"{q['query_id']} has no grade-2 answer, so NDCG cannot discriminate"
        )


@pytest.mark.unit
def test_corpus_families_are_balanced() -> None:
    queries = load_corpus()
    counts: dict[str, int] = {}
    for q in queries:
        counts[str(q["family"])] = counts.get(str(q["family"]), 0) + 1
    assert counts == {"exact_token": 8, "paraphrase": 8, "agreement": 8}


@pytest.mark.unit
def test_corpus_is_discriminating_by_construction() -> None:
    """Section 4 of the ticket: a corpus where both arms score the same proves nothing.

    On exact-token queries the lexical leg must rank the correct answer strictly
    above where the vector leg puts it. If this ever stops holding, the corpus
    has lost its power to detect a fusion regression and the gate below is
    measuring nothing.
    """
    for q in load_corpus():
        if q["family"] != "exact_token":
            continue
        labels: dict[str, int] = q["labels"]  # type: ignore[assignment]
        answer = next(doc for doc, grade in labels.items() if grade == 2)
        vector_rank = ranked_ids(q["vector"]).index(answer)  # type: ignore[arg-type]
        lexical_rank = ranked_ids(q["lexical"]).index(answer)  # type: ignore[arg-type]
        assert lexical_rank < vector_rank, (
            f"{q['query_id']}: lexical must out-rank vector for this family"
        )


# =============================================================================
# The gate — RED until handler_fusion lands
# =============================================================================


def scored(leg: Sequence[dict[str, object]]) -> list[tuple[str, float]]:
    """Extract ``(doc_id, score)`` pairs from a pre-computed leg, in rank order."""
    return [(str(e["doc_id"]), float(e["score"])) for e in leg]  # type: ignore[arg-type]


def _scores_by_family(*, gated: bool) -> dict[str, tuple[float, float]]:
    """Mean vector-only and fused NDCG@10 per family.

    Args:
        gated: When True, each leg is filtered against its own relevance floor
            before fusing. When False, raw RRF is applied to every returned
            document — the architecture plan's algorithm exactly as written.
            Both are measured so the effect of gating is a number rather than
            an assertion.
    """
    from omnimemory.nodes.node_memory_retrieval_effect.handlers.handler_fusion import (
        MIN_LEXICAL_TS_RANK,
        MIN_VECTOR_COSINE,
        fuse_rrf,
        select_relevant,
    )

    totals: dict[str, list[tuple[float, float]]] = {}

    for q in load_corpus():
        labels: dict[str, int] = q["labels"]  # type: ignore[assignment]
        vector_scored = scored(q["vector"])  # type: ignore[arg-type]
        lexical_scored = scored(q["lexical"])  # type: ignore[arg-type]

        if gated:
            vector_leg = select_relevant(vector_scored, MIN_VECTOR_COSINE)
            lexical_leg = select_relevant(lexical_scored, MIN_LEXICAL_TS_RANK)
        else:
            vector_leg = [doc for doc, _ in vector_scored]
            lexical_leg = [doc for doc, _ in lexical_scored]

        baseline = ndcg_at_k([doc for doc, _ in vector_scored], labels, NDCG_K)
        fused = ndcg_at_k(fuse_rrf(vector_leg, lexical_leg), labels, NDCG_K)

        totals.setdefault(str(q["family"]), []).append((baseline, fused))

    return {
        family: (
            sum(b for b, _ in pairs) / len(pairs),
            sum(f for _, f in pairs) / len(pairs),
        )
        for family, pairs in totals.items()
    }


@pytest.mark.unit
def test_fusion_beats_vector_only_on_exact_token_queries() -> None:
    """The claim this ticket exists to prove."""
    baseline, fused = _scores_by_family(gated=True)["exact_token"]
    assert fused > baseline, (
        f"fusion did not improve exact-token retrieval: {fused:.4f} vs {baseline:.4f}"
    )
    assert fused - baseline >= EXACT_TOKEN_MIN_GAIN, (
        f"gain {fused - baseline:.4f} below required {EXACT_TOKEN_MIN_GAIN}"
    )


@pytest.mark.unit
def test_fusion_does_not_regress_paraphrase_queries() -> None:
    """The converse case — where the lexical leg has nothing useful to add."""
    baseline, fused = _scores_by_family(gated=True)["paraphrase"]
    assert fused >= baseline - NON_REGRESSION_TOLERANCE, (
        f"fusion regressed paraphrase retrieval: {fused:.4f} vs {baseline:.4f}"
    )


@pytest.mark.unit
def test_fusion_does_not_disturb_queries_both_legs_already_agree_on() -> None:
    baseline, fused = _scores_by_family(gated=True)["agreement"]
    assert fused >= baseline - NON_REGRESSION_TOLERANCE, (
        f"fusion disturbed an agreed ranking: {fused:.4f} vs {baseline:.4f}"
    )


@pytest.mark.unit
def test_ungated_rrf_regresses_paraphrase_queries() -> None:
    """Records *why* relevance gating exists, as a measurement rather than prose.

    The architecture plan specifies RRF over both legs' results with no
    relevance floor. Applied literally that regresses paraphrase retrieval,
    because RRF awards a leg's rank-1 document the full ``1/(k+1)`` weight even
    when that leg matched a single stray lexeme and its "best" result is
    irrelevant. The semantic leg is correct on these queries and gets displaced.

    This test asserts the regression is real. If a future change to the fusion
    makes ungated RRF safe, this test fails and the gating can be reconsidered —
    which is the intended signal, not a nuisance.
    """
    baseline, fused = _scores_by_family(gated=False)["paraphrase"]
    assert fused < baseline - NON_REGRESSION_TOLERANCE, (
        "ungated RRF no longer regresses paraphrase retrieval "
        f"({fused:.4f} vs {baseline:.4f}); relevance gating may no longer be needed"
    )


@pytest.mark.unit
def test_fusion_is_deterministic() -> None:
    """Same inputs, same output ordering, every run — including ties.

    RRF produces exact ties whenever two documents hold the same rank in their
    respective legs, so tie-breaking has to be defined rather than left to dict
    ordering.
    """
    from omnimemory.nodes.node_memory_retrieval_effect.handlers.handler_fusion import (
        fuse_rrf,
    )

    q = load_corpus()[0]
    vector = ranked_ids(q["vector"])  # type: ignore[arg-type]
    lexical = ranked_ids(q["lexical"])  # type: ignore[arg-type]
    first = fuse_rrf(vector, lexical)
    for _ in range(5):
        assert fuse_rrf(vector, lexical) == first


@pytest.mark.unit
def test_fusion_preserves_every_input_document() -> None:
    """Fusion reorders and deduplicates; it must not drop candidates.

    Dropping is :func:`select_relevant`'s job and happens before fusion, so
    ``fuse_rrf`` itself is total over its inputs.
    """
    from omnimemory.nodes.node_memory_retrieval_effect.handlers.handler_fusion import (
        fuse_rrf,
    )

    for q in load_corpus():
        vector = ranked_ids(q["vector"])  # type: ignore[arg-type]
        lexical = ranked_ids(q["lexical"])  # type: ignore[arg-type]
        fused = fuse_rrf(vector, lexical)
        assert set(fused) == set(vector) | set(lexical), q["query_id"]
        assert len(fused) == len(set(fused)), f"{q['query_id']}: duplicate in output"


@pytest.mark.unit
def test_select_relevant_preserves_rank_order_and_drops_below_floor() -> None:
    from omnimemory.nodes.node_memory_retrieval_effect.handlers.handler_fusion import (
        select_relevant,
    )

    entries = [("a", 0.9), ("b", 0.1), ("c", 0.6), ("d", 0.05)]
    assert select_relevant(entries, 0.5) == ["a", "c"]
    assert select_relevant(entries, 0.0) == ["a", "b", "c", "d"]
    assert select_relevant(entries, 1.0) == []


@pytest.mark.unit
def test_fuse_rrf_rejects_negative_k() -> None:
    from omnimemory.nodes.node_memory_retrieval_effect.handlers.handler_fusion import (
        fuse_rrf,
    )

    with pytest.raises(ValueError, match="non-negative"):
        fuse_rrf(["a"], k=-1)


@pytest.mark.unit
def test_fuse_rrf_scores_and_fuse_rrf_cannot_diverge() -> None:
    """fuse_rrf derives its order from fuse_rrf_scores; the handler uses both.

    If the two ever disagreed, the handler would attach a score from one and an
    ordering from the other -- exactly the defect the normalisation exists to
    prevent.
    """
    from omnimemory.nodes.node_memory_retrieval_effect.handlers.handler_fusion import (
        fuse_rrf,
        fuse_rrf_scores,
    )

    for q in load_corpus():
        vector = ranked_ids(q["vector"])  # type: ignore[arg-type]
        lexical = ranked_ids(q["lexical"])  # type: ignore[arg-type]
        order = fuse_rrf(vector, lexical)
        scores = fuse_rrf_scores(vector, lexical)

        assert set(order) == set(scores), q["query_id"]
        values = [scores[doc_id] for doc_id in order]
        assert values == sorted(values, reverse=True), q["query_id"]


@pytest.mark.unit
def test_fuse_rrf_scores_rewards_agreement_between_legs() -> None:
    """A document both legs return must outscore one only a single leg returned.

    This is the whole mechanism of rank fusion; if it stops holding, RRF has
    been broken rather than merely retuned.
    """
    from omnimemory.nodes.node_memory_retrieval_effect.handlers.handler_fusion import (
        fuse_rrf_scores,
    )

    scores = fuse_rrf_scores(["both", "vector_only"], ["both", "lexical_only"])
    assert scores["both"] > scores["vector_only"]
    assert scores["both"] > scores["lexical_only"]
