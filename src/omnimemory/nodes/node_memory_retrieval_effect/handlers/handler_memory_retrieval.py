# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Memory Retrieval Handler - answers searches from injected stores or refuses by name.

The handler routes a request to the store that serves its operation:
- search: the vector store (semantic similarity)
- search_text: the database store (full-text)
- search_graph: the graph store (relationship traversal)
- search_hybrid: the vector and database stores, fused by rank

It builds none of those stores. They are injected behind their protocols
(``omnimemory.protocols.protocol_retrieval_store``) and resolved from settings
by ``container.registry_memory_retrieval``. A store that is absent, or that
cannot be reached, is answered with a ``refused`` response that names it and
carries no rows (OMN-26659).

Example::

    handler = HandlerMemoryRetrieval(vector_store=vector_store)
    response = await handler.execute(
        ModelMemoryRetrievalRequest(operation="search", query_text="auth")
    )
    # response.status == "success", or "refused" naming the vector store.

.. versionadded:: 0.1.0
    Initial implementation for OMN-1387.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, assert_never

if TYPE_CHECKING:
    from omnimemory.protocols.protocol_retrieval_store import (
        ProtocolDatabaseStore,
        ProtocolGraphStore,
        ProtocolRetrievalStore,
        ProtocolVectorStore,
    )

    from .handler_decay import HandlerActivationDecay

from omnimemory.enums.enum_memory_store import EnumMemoryStore
from omnimemory.enums.enum_store_refusal_reason import EnumStoreRefusalReason

from ..container.registry_memory_retrieval import (
    bind_activation_decay,
    bind_rank_fusion,
)
from ..models import (
    ModelActivationDecayRequest,
    ModelMemoryRetrievalRequest,
    ModelMemoryRetrievalResponse,
    ModelRankedEntry,
    ModelRankedLeg,
    ModelRankFusionRequest,
    ModelStoreRefusal,
)
from .handler_fusion import (
    MIN_LEXICAL_TS_RANK,
    MIN_VECTOR_COSINE,
    HandlerRankFusion,
)

logger = logging.getLogger(__name__)

__all__ = [
    "HandlerMemoryRetrieval",
]


def _refusal(
    store: EnumMemoryStore, reason: EnumStoreRefusalReason, detail: str
) -> ModelMemoryRetrievalResponse:
    """Build a refused response: the store named, no rows."""
    return ModelMemoryRetrievalResponse(
        status="refused",
        refusal=ModelStoreRefusal(store=store, reason=reason, detail=detail),
    )


class HandlerMemoryRetrieval:
    """Routes retrieval requests to the injected stores, or refuses by name.

    Supported operations:
        - search: Semantic similarity search (vector store)
        - search_text: Full-text search (database store)
        - search_graph: Relationship traversal (graph store)
        - search_hybrid: Fused semantic and full-text retrieval

    Every store is optional at construction. A request whose store is absent
    is refused with ``reason=not_configured``; a store that fails to connect or
    raises while answering is refused with ``reason=unreachable``. Neither
    returns rows, and neither is answered from anything but the real store.
    """

    def __init__(
        self,
        vector_store: ProtocolVectorStore | None = None,
        database_store: ProtocolDatabaseStore | None = None,
        graph_store: ProtocolGraphStore | None = None,
        fusion: HandlerRankFusion | None = None,
        decay: HandlerActivationDecay | None = None,
    ) -> None:
        """Hold the injected stores and the ranking collaborators.

        Args:
            vector_store: Store behind ``search``; None when not configured.
            database_store: Store behind ``search_text``; None when not configured.
            graph_store: Store behind ``search_graph``; None when not configured.
            fusion: Stateless rank fusion collaborator, constructed when None.
            decay: Stateless activation decay collaborator, constructed when None.
        """
        self._fusion = bind_rank_fusion() if fusion is None else fusion
        self._decay = bind_activation_decay() if decay is None else decay
        self._stores: dict[EnumMemoryStore, ProtocolRetrievalStore | None] = {
            EnumMemoryStore.VECTOR: vector_store,
            EnumMemoryStore.DATABASE: database_store,
            EnumMemoryStore.GRAPH: graph_store,
        }
        self._connected: set[EnumMemoryStore] = set()
        self._connect_lock = asyncio.Lock()

    @property
    def configured_stores(self) -> frozenset[EnumMemoryStore]:
        """The stores this handler was given."""
        return frozenset(k for k, v in self._stores.items() if v is not None)

    async def _connect(
        self, store: EnumMemoryStore
    ) -> ModelMemoryRetrievalResponse | None:
        """Connect one store on first use.

        Returns:
            None when the store is connected; a refusal naming it when it is
            not configured or cannot be reached. A failed connect is not
            cached, so the next request tries again.
        """
        backend = self._stores[store]
        if backend is None:
            return _refusal(
                store,
                EnumStoreRefusalReason.NOT_CONFIGURED,
                f"no {store.value} store is configured for this runtime",
            )
        if store in self._connected:
            return None
        async with self._connect_lock:
            if store in self._connected:
                return None
            try:
                await backend.initialize()
            except Exception as exc:
                logger.exception("Memory %s store could not be reached", store.value)
                return _refusal(
                    store,
                    EnumStoreRefusalReason.UNREACHABLE,
                    f"{type(exc).__name__}: {exc}",
                )
            self._connected.add(store)
        return None

    async def _ask(
        self, store: EnumMemoryStore, request: ModelMemoryRetrievalRequest
    ) -> ModelMemoryRetrievalResponse:
        """Send one request to one store; refuse by name when it cannot answer."""
        refusal = await self._connect(store)
        if refusal is not None:
            return refusal
        backend = self._stores[store]
        if backend is None:  # pragma: no cover - _connect refused above
            raise RuntimeError(f"{store.value} store vanished after connect")
        try:
            return await backend.execute(request)
        except Exception as exc:
            logger.exception(
                "Memory %s store failed answering %s", store.value, request.operation
            )
            # A dropped connection or a timeout means the service is gone; any
            # other error means the store was reached and could not answer.
            reachable = not isinstance(exc, OSError)
            if not reachable:
                self._connected.discard(store)
            return _refusal(
                store,
                (
                    EnumStoreRefusalReason.FAILED
                    if reachable
                    else EnumStoreRefusalReason.UNREACHABLE
                ),
                f"{type(exc).__name__}: {exc}",
            )

    async def handle(
        self, request: ModelMemoryRetrievalRequest
    ) -> ModelMemoryRetrievalResponse:
        """Canonical definition-B dispatch entrypoint (CLAUDE.md rule 7a).

        This is the entrypoint the runtime auto-wiring binds — it resolves
        ``handle_async`` then ``handle`` — and the name the omnimarket
        dispatch-entrypoint gate (OMN-14617) looks for on every handler
        declared in ``handler_routing.handlers[]``.

        Added, never renamed: ``execute`` remains the implementation and keeps
        its existing call sites (the mock sub-handlers, this module, and
        ``node_memory_retrieval_effect/__init__.py``).

        Args:
            request: The retrieval request.

        Returns:
            Response with search results or error information.
        """
        return await self.execute(request)

    async def execute(
        self, request: ModelMemoryRetrievalRequest
    ) -> ModelMemoryRetrievalResponse:
        """Execute a memory retrieval operation.

        Routes the request to the store that serves its operation and returns
        that store's answer, or a refusal naming the store that could not give one.

        Args:
            request: The retrieval request.

        Returns:
            Response with search results, a store's own error, or a refusal.
        """
        match request.operation:
            case "search":
                return await self._ask(EnumMemoryStore.VECTOR, request)
            case "search_text":
                return await self._ask(EnumMemoryStore.DATABASE, request)
            case "search_graph":
                return await self._ask(EnumMemoryStore.GRAPH, request)
            case "search_hybrid":
                return await self._execute_hybrid(request)
            case _:
                assert_never(request.operation)

    async def _execute_hybrid(
        self, request: ModelMemoryRetrievalRequest
    ) -> ModelMemoryRetrievalResponse:
        """Fan out to the semantic and full-text legs, then fuse their rankings.

        The two legs are dispatched concurrently — they share no state and
        neither depends on the other's result. Each is sent as the operation it
        already implements, so this adds no new behaviour to either sub-handler.

        Ranking is a two-step process, and the steps are deliberately distinct:
        each leg is first filtered against its own relevance floor, then the
        survivors are fused by rank. Fusing first would let a leg that matched
        nothing useful contribute its best-of-a-bad-set at full rank-1 weight
        and displace a leg that answered correctly. See ``HandlerRankFusion``.

        Args:
            request: The originating ``search_hybrid`` request.

        Returns:
            One response carrying the fused, deduplicated ranking, truncated to
            the request's limit. If either leg errors, that error is returned
            rather than a half-fused ranking presented as a whole answer.
        """
        for needed in (EnumMemoryStore.VECTOR, EnumMemoryStore.DATABASE):
            refusal = await self._connect(needed)
            if refusal is not None:
                return refusal

        vector_response, lexical_response = await asyncio.gather(
            self._ask(
                EnumMemoryStore.VECTOR,
                request.model_copy(update={"operation": "search"}),
            ),
            self._ask(
                EnumMemoryStore.DATABASE,
                request.model_copy(update={"operation": "search_text"}),
            ),
        )

        for leg_response in (vector_response, lexical_response):
            if leg_response.status in ("error", "refused"):
                return leg_response

        by_id = {
            str(result.snapshot.snapshot_id): result
            for result in (*lexical_response.results, *vector_response.results)
        }

        fusion_response = self._fusion.handle(
            ModelRankFusionRequest(
                legs=tuple(
                    ModelRankedLeg(
                        entries=tuple(
                            ModelRankedEntry(
                                document_id=str(result.snapshot.snapshot_id),
                                score=result.score,
                            )
                            for result in response.results
                        ),
                        min_score=min_score,
                    )
                    for response, min_score in (
                        (vector_response, MIN_VECTOR_COSINE),
                        (lexical_response, MIN_LEXICAL_TS_RANK),
                    )
                )
            )
        )
        fused_scores = fusion_response.scores

        # Activation decay, applied AFTER fusion as a multiplicative modulation
        # on the fused score -- not as a third ranked list inside the fusion.
        # That ordering is the architecture plan's §4.2 composed read path.
        #
        # `now` is captured once and shared by every result, so two documents
        # of identical age cannot be scored against different reference points
        # within one response.
        #
        # It is also truncated to the hour, which makes the operation
        # reproducible: without it, two calls microseconds apart return scores
        # differing at ~1e-10, so `handle()` and `execute()` stop being
        # observably identical and any response comparison becomes flaky.
        # Truncating is not a workaround for that test -- sub-second precision
        # is meaningless against a decay constant expressed per *day*. At
        # lambda = 0.015/day a whole hour of drift moves the multiplier by
        # ~0.06%, well inside the noise the constant already carries.
        #
        # ModelMemorySnapshot carries created_at and no last_accessed_at, so on
        # this path the decay degrades to created_at alone -- the ruled shape
        # (OMN-16765). HandlerActivationDecay takes the freshest timestamp available
        # rather than assuming which fields exist.
        now = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
        ranked_scores = {
            doc_id: score
            * self._decay.handle(
                ModelActivationDecayRequest(
                    created_at=by_id[doc_id].snapshot.created_at, now=now
                )
            ).activation
            for doc_id, score in fused_scores.items()
        }
        ranked_ids = sorted(ranked_scores, key=lambda d: (-ranked_scores[d], d))

        # Re-score against the final ranking, rather than carrying each leg's
        # own number through. Returning a raw backend score alongside a fused
        # ordering lets the two contradict each other: a document promoted
        # because both legs agreed on it can sit above one with a higher raw
        # cosine, and a caller sorting by `score` would then reorder the very
        # ranking this operation exists to produce.
        #
        # Normalised against the top hit because the field is declared
        # `ge=0.0, le=1.0` and documented as a relevance score. Raw RRF values
        # are ~0.016-0.033, which satisfy the bound while reading as "3%
        # relevant". Dividing by the maximum is monotonic, so the ordering is
        # untouched; only the scale changes.
        top_score = max(ranked_scores.values(), default=0.0)
        results = [
            by_id[doc_id].model_copy(
                update={
                    "score": ranked_scores[doc_id] / top_score if top_score else 0.0
                }
            )
            for doc_id in ranked_ids
        ][: request.limit]

        logger.debug(
            "Hybrid search fused %d vector and %d lexical results into %d "
            "(decay applied on created_at)",
            len(vector_response.results),
            len(lexical_response.results),
            len(results),
        )

        return ModelMemoryRetrievalResponse(
            status="success" if results else "no_results",
            results=results,
            total_count=len(results),
            query_embedding_used=vector_response.query_embedding_used,
        )

    async def shutdown(self) -> None:
        """Release every store this handler connected."""
        connected = [self._stores[k] for k in self._connected]
        self._connected.clear()
        await asyncio.gather(*(b.shutdown() for b in connected if b is not None))
        logger.info("Memory retrieval handler shutdown complete")
