# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""OMN-16765 step 4 — does activation decay improve ranking, and what does the
frozen-core limitation cost?

Two questions, asserted rather than argued, the same way the fusion variants
were:

1. **Does decay help at all?** Measured on the ``recency`` family, which is
   built so retrieval alone ranks the *wrong* document first — the stale
   document deliberately outscores the fresh one on both legs. If decay did
   nothing, these queries would score badly. That construction is what stops
   this being a corpus where decay wins by definition.

2. **What does `created_at`-only cost?** ``ModelMemorySnapshot`` is frozen,
   lives in ``omnibase_core``, and carries no ``last_accessed_at``. RC-05..08
   give both documents an *identical* ``created_at``, so a ``created_at``-only
   decay cannot separate them at all — only the access term can. The delta
   between the two variants on that arm is the measured cost of not plumbing
   the field, and is the evidence that decides whether that cross-repo change
   is worth filing.

Timestamps in the corpus are authored, not observed access data. That is
recorded here and in the corpus ``_meta`` because the number this file produces
is meant to justify (or not justify) a change to a published core model.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

import pytest

from omnimemory.nodes.node_memory_retrieval_effect.handlers.handler_decay import (
    activation_decay,
)
from omnimemory.nodes.node_memory_retrieval_effect.handlers.handler_fusion import (
    MIN_LEXICAL_TS_RANK,
    MIN_VECTOR_COSINE,
    fuse_rrf,
    fuse_rrf_scores,
    select_relevant,
)

from .test_hybrid_retrieval_fusion import (
    NDCG_K,
    load_corpus,
    ndcg_at_k,
    scored,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

# Decay must not merely help on average — it must clear this margin on the
# family built to need it. Conservative: on this corpus the correct document
# sits at retrieval rank 2 behind a deliberately higher-scoring stale one.
RECENCY_MIN_GAIN = 0.15

# The three original families discriminate on relevance, not recency, and all
# their documents share one timestamp. Decay must leave them alone.
UNRELATED_FAMILY_TOLERANCE = 0.01


def corpus_documents() -> Mapping[str, dict[str, str | None]]:
    import json

    from .test_hybrid_retrieval_fusion import CORPUS_PATH

    with CORPUS_PATH.open(encoding="utf-8") as fh:
        return dict(json.load(fh)["documents"])


def reference_now() -> datetime:
    import json

    from .test_hybrid_retrieval_fusion import CORPUS_PATH

    with CORPUS_PATH.open(encoding="utf-8") as fh:
        return datetime.fromisoformat(json.load(fh)["_meta"]["reference_now"])


def _rank(
    query: dict[str, object],
    documents: Mapping[str, dict[str, str | None]],
    now: datetime,
    *,
    decay: bool,
    use_access_time: bool,
) -> list[str]:
    """Fuse, then optionally modulate by decay — post-fusion, multiplicative.

    Args:
        decay: When False, the ranking is fusion alone (the current shipped
            behaviour, and the baseline every assertion here compares against).
        use_access_time: When False, decay reads ``created_at`` only — the
            snapshot path. When True it also reads ``last_accessed_at``, which
            is the shape available only where the model carries the field.
    """
    vector = scored(query["vector"])  # type: ignore[arg-type]
    lexical = scored(query["lexical"])  # type: ignore[arg-type]
    legs = (
        select_relevant(vector, MIN_VECTOR_COSINE),
        select_relevant(lexical, MIN_LEXICAL_TS_RANK),
    )
    if not decay:
        return fuse_rrf(*legs)

    fused = fuse_rrf_scores(*legs)
    modulated: dict[str, float] = {}
    for doc_id, score in fused.items():
        meta = documents[doc_id]
        created = datetime.fromisoformat(str(meta["created_at"]))
        accessed_raw = meta["last_accessed_at"]
        accessed = (
            datetime.fromisoformat(str(accessed_raw))
            if use_access_time and accessed_raw is not None
            else None
        )
        modulated[doc_id] = score * activation_decay(created, accessed, now=now)
    return sorted(modulated, key=lambda d: (-modulated[d], d))


def _mean_ndcg(family: str, *, decay: bool, use_access_time: bool) -> float:
    documents, now = corpus_documents(), reference_now()
    scores = [
        ndcg_at_k(
            _rank(q, documents, now, decay=decay, use_access_time=use_access_time),
            q["labels"],  # type: ignore[arg-type]
            NDCG_K,
        )
        for q in load_corpus()
        if q["family"] == family
    ]
    return sum(scores) / len(scores)


# =============================================================================
# 1. Does decay improve ranking where recency is the discriminator?
# =============================================================================


@pytest.mark.unit
def test_retrieval_alone_gets_the_recency_family_wrong() -> None:
    """The premise the rest of this file rests on.

    If fusion alone already scored well here, decay would have nothing to fix
    and every gain below would be an artefact of a corpus that never posed the
    problem.
    """
    baseline = _mean_ndcg("recency", decay=False, use_access_time=False)
    assert baseline < 0.75, (
        f"recency family is too easy without decay ({baseline:.4f}); "
        "the stale document is supposed to out-rank the fresh one on retrieval"
    )


@pytest.mark.unit
def test_decay_improves_the_recency_family() -> None:
    baseline = _mean_ndcg("recency", decay=False, use_access_time=False)
    decayed = _mean_ndcg("recency", decay=True, use_access_time=True)
    assert decayed - baseline >= RECENCY_MIN_GAIN, (
        f"decay gain {decayed - baseline:.4f} below required {RECENCY_MIN_GAIN} "
        f"({baseline:.4f} -> {decayed:.4f})"
    )


@pytest.mark.unit
@pytest.mark.parametrize("family", ["exact_token", "paraphrase", "agreement"])
def test_decay_does_not_disturb_the_relevance_families(family: str) -> None:
    """Decay is a recency signal; on families that turn on relevance it should
    be close to inert. Their documents share one timestamp, so any movement
    here would mean decay is reordering on something other than age."""
    baseline = _mean_ndcg(family, decay=False, use_access_time=False)
    decayed = _mean_ndcg(family, decay=True, use_access_time=True)
    assert abs(decayed - baseline) <= UNRELATED_FAMILY_TOLERANCE, (
        f"{family}: decay moved a relevance-discriminated family "
        f"{baseline:.4f} -> {decayed:.4f}"
    )


# =============================================================================
# 2. What does the created_at-only path cost?
# =============================================================================


@pytest.mark.unit
def test_created_at_only_cannot_separate_equally_aged_documents() -> None:
    """RC-05..08 give both documents the same created_at deliberately.

    This is the frozen-core path. It is not that ``created_at``-only decay is
    merely weaker here — it has no signal at all, because the only thing
    distinguishing the two documents is when they were last read.
    """
    documents, now = corpus_documents(), reference_now()
    access_discriminated = [
        q
        for q in load_corpus()
        if q["family"] == "recency" and "accessed" in str(q.get("note", ""))
    ]
    assert len(access_discriminated) == 4

    for q in access_discriminated:
        without = _rank(q, documents, now, decay=True, use_access_time=False)
        no_decay = _rank(q, documents, now, decay=False, use_access_time=False)
        assert without == no_decay, (
            f"{q['query_id']}: created_at-only decay changed the order, but "
            "both documents share a created_at — it should be inert here"
        )


@pytest.mark.unit
def test_access_time_recovers_what_created_at_alone_cannot() -> None:
    """The measurement that decides the core plumb.

    Asserts the delta exists and is material. The magnitude is reported by
    ``pytest -s`` via the failure message on regression, and recorded in the
    design note.
    """
    documents, now = corpus_documents(), reference_now()
    queries = [
        q
        for q in load_corpus()
        if q["family"] == "recency" and "accessed" in str(q.get("note", ""))
    ]

    def mean(*, use_access_time: bool) -> float:
        return sum(
            ndcg_at_k(
                _rank(q, documents, now, decay=True, use_access_time=use_access_time),
                q["labels"],  # type: ignore[arg-type]
                NDCG_K,
            )
            for q in queries
        ) / len(queries)

    created_only = mean(use_access_time=False)
    with_access = mean(use_access_time=True)
    assert with_access > created_only, (
        f"last_accessed_at bought nothing: {created_only:.4f} -> {with_access:.4f}"
    )


# =============================================================================
# 3. The limitation this design carries — asserted, not omitted
# =============================================================================


@pytest.mark.unit
def test_decay_can_promote_a_fresh_irrelevant_document() -> None:
    """Multiplicative post-fusion decay trades relevance for recency, and past
    a certain age gap it trades badly.

    Found while building the recency family: a filler document 14 days old
    displaced a 400-day-old *correct* answer, because ``exp(-0.015 * 400)`` is
    ~0.002 against ~0.81. Fusion had ranked the right document second; decay
    pushed it to third.

    This is inherent to the plan's §4.2 shape — a multiplicative modulation
    cannot distinguish "old and wrong" from "old and right", it only sees age.
    Documented here rather than engineered away, because the corpus change that
    made the arm above pass (removing the filler) would otherwise have quietly
    deleted the evidence for it.

    The practical bound: with lambda = 0.015/day, a document roughly 100 days
    older than another loses about 4x of its fused score. Where a corpus has a
    wide age spread and relevance does not correlate with age, decay should be
    expected to cost accuracy rather than buy it.
    """
    now = reference_now()
    from datetime import timedelta

    old_and_correct = activation_decay(now - timedelta(days=400), None, now=now)
    fresh_and_wrong = activation_decay(now - timedelta(days=14), None, now=now)

    # The fused scores RRF would produce for adjacent ranks are within ~2% of
    # each other, so a decay ratio this large overwhelms any ranking signal.
    assert fresh_and_wrong / old_and_correct > 100, (
        "the age gap no longer dominates the fused score; the caveat recorded "
        "in this test may no longer hold and should be re-derived"
    )
