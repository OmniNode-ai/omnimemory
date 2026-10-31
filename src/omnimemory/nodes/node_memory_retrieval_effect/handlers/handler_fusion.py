# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Rank fusion for hybrid retrieval (OMN-16765).

Combines the ranked output of the semantic (Qdrant) and full-text (Postgres)
legs into a single ordering, behind the existing retrieval protocol.

Two concerns are kept separate on purpose:

* Rank accumulation in :meth:`HandlerRankFusion.handle` is **pure rank
  fusion**. It consumes ranks and nothing else.
  The two legs produce numerically incomparable quantities — cosine similarity
  in ``[0, 1]`` versus an unbounded Postgres ``ts_rank`` whose magnitude depends
  on document length and term-frequency normalisation — so any score-combining
  scheme would first need a mapping onto a shared scale, and that mapping is a
  fitted parameter with no principled value. Reciprocal Rank Fusion sidesteps it
  entirely by discarding magnitudes.

* Relevance gating in :meth:`HandlerRankFusion.handle` decides **whether a leg
  has anything to contribute**,
  using each leg's own scores against its own floor. This never compares one
  leg's scores to another's, so it does not reintroduce the scale problem RRF
  exists to avoid.

The split matters because RRF alone cannot tell a leg that answered well from a
leg that answered badly: it gives whatever sits at rank 1 the full ``1/(k+1)``
weight either way. On a paraphrase query the lexical leg may match a stray
lexeme and return a document that is merely present rather than relevant, and
unweighted RRF will promote it to the top of the fused list on the strength of
being that leg's best of a bad set. Gating before fusing is what stops a leg
with no signal from displacing a leg that has some.

.. versionadded:: 0.18.0
    Initial implementation for OMN-16765.
"""

from __future__ import annotations

from ..models import ModelRankFusionRequest, ModelRankFusionResponse

__all__ = [
    "MIN_LEXICAL_TS_RANK",
    "MIN_VECTOR_COSINE",
    "RRF_K_DEFAULT",
    "HandlerRankFusion",
]

RRF_K_DEFAULT = 60
"""Smoothing constant from the original RRF formulation.

Larger values flatten the contribution of top ranks; smaller values sharpen it.
60 is the published default and the value named in the architecture plan. It is
recorded here as a default rather than a tuned result: no evidence in this
repository justifies moving it, and changing it without evidence would be worse
than taking the documented value.
"""

MIN_VECTOR_COSINE = 0.5
"""Cosine similarity below which the semantic leg is treated as not matching.

Cosine is bounded in ``[0, 1]`` and 0.5 is the midpoint. The existing layered
search in ``adapter_learning_scoring`` already treats 0.70 as a broad
context match and 0.85 as a high-precision one, so 0.5 sits deliberately below
both: this floor is meant to exclude noise, not to make a relevance judgement
that belongs to the ranking.
"""

MIN_LEXICAL_TS_RANK = 0.15
"""Postgres ``ts_rank`` below which the lexical leg is treated as not matching.

``ts_rank`` is unbounded, but its low end is interpretable: a value this small
means almost no query lexeme was found in the document. It is the difference
between "this document is about the query" and "this document happens to
contain one of these words".
"""


class HandlerRankFusion:
    """Gate each leg on its own scores, then fuse ranks without retaining state."""

    def handle(self, request: ModelRankFusionRequest) -> ModelRankFusionResponse:
        """Return relevant ids, reciprocal rank scores, and deterministic order.

        Each surviving document accumulates ``1 / (k + rank)`` per occurrence,
        counting ranks from 1 after filtering. Agreement across legs therefore
        adds contributions without comparing their incomparable raw scores.
        Scores are meaningful only relative to others within this query.
        Ties are broken by document id ascending, never by insertion order.

        Raises:
            ValueError: If ``k`` is negative.
        """
        k = RRF_K_DEFAULT if request.k is None else request.k
        if k < 0:
            raise ValueError(f"RRF k must be non-negative, got {k}")

        relevant_legs = tuple(
            tuple(
                entry.document_id
                for entry in leg.entries
                if entry.score >= leg.min_score
            )
            for leg in request.legs
        )
        scores: dict[str, float] = {}
        for leg in relevant_legs:
            for rank, doc_id in enumerate(leg, start=1):
                scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
        ranking = tuple(sorted(scores, key=lambda doc_id: (-scores[doc_id], doc_id)))
        return ModelRankFusionResponse(
            relevant_legs=relevant_legs,
            scores=scores,
            ranking=ranking,
        )
