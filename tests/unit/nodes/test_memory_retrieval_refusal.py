# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""A memory search answers from the injected stores or refuses by name (OMN-26659).

AC1: the retrieval handler is built only from injected stores behind their
     protocols, resolved from settings.
AC2: a store that is not configured, or not reachable, answers a typed refusal
     naming it, with no rows.
"""

from __future__ import annotations

import json
from uuid import UUID, uuid4

import pytest
from omnibase_core.enums.enum_subject_type import EnumSubjectType
from omnibase_core.models.omnimemory import (
    ModelCostLedger,
    ModelMemorySnapshot,
    ModelSubjectRef,
)
from omnibase_core.models.primitives.model_semver import ModelSemVer
from pydantic import ValidationError

from omnimemory.enums.enum_memory_store import EnumMemoryStore
from omnimemory.enums.enum_store_refusal_reason import EnumStoreRefusalReason
from omnimemory.nodes.node_memory_retrieval_effect.container.registry_memory_retrieval import (
    bind_memory_retrieval,
)
from omnimemory.nodes.node_memory_retrieval_effect.handlers import (
    HandlerMemoryRetrieval,
    HandlerQdrant,
)
from omnimemory.nodes.node_memory_retrieval_effect.models import (
    ModelMemoryRetrievalRequest,
    ModelMemoryRetrievalResponse,
    ModelStoreRefusal,
)
from omnimemory.nodes.node_memory_retrieval_effect.models.model_retrieval_store_settings import (
    ModelRetrievalStoreSettings,
)
from omnimemory.protocols.protocol_retrieval_store import (
    ProtocolDatabaseStore,
    ProtocolGraphStore,
    ProtocolVectorStore,
)
from tests.stand_ins.stand_in_retrieval_handler import StandInHandlerMemoryRetrieval

_OPERATION_STORE = {
    "search": EnumMemoryStore.VECTOR,
    "search_text": EnumMemoryStore.DATABASE,
    "search_graph": EnumMemoryStore.GRAPH,
}


def _request(operation: str) -> ModelMemoryRetrievalRequest:
    return ModelMemoryRetrievalRequest.model_validate(
        {
            "operation": operation,
            "query_text": "authentication",
            "snapshot_id": str(uuid4()) if operation == "search_graph" else None,
        }
    )


def _snapshot(subject_text: str) -> ModelMemorySnapshot:
    return ModelMemorySnapshot(
        snapshot_id=uuid4(),
        subject=ModelSubjectRef(
            subject_type=EnumSubjectType.AGENT,
            subject_id=uuid4(),
            subject_key=subject_text,
        ),
        cost_ledger=ModelCostLedger(budget_total=100.0),
        schema_version=ModelSemVer(major=1, minor=0, patch=0),
    )


class _DownStore:
    """A store whose backing service cannot be reached."""

    def __init__(self, *, fail_on: str) -> None:
        self.fail_on = fail_on
        self.initialize_calls = 0

    async def initialize(self) -> None:
        self.initialize_calls += 1
        if self.fail_on == "initialize":
            raise ConnectionError("connection refused")

    async def execute(
        self, request: ModelMemoryRetrievalRequest
    ) -> ModelMemoryRetrievalResponse:
        if self.fail_on == "execute":
            raise TimeoutError("timed out")
        if self.fail_on == "reached":
            raise ValueError("payload is not a memory snapshot")
        return ModelMemoryRetrievalResponse(status="no_results")

    async def shutdown(self) -> None:
        return None


@pytest.mark.unit
class TestUnconfiguredRuntimeRefuses:
    """AC2: an unconfigured runtime returns a refusal, never a result."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("operation", list(_OPERATION_STORE))
    async def test_no_store_refuses_naming_the_store(self, operation: str) -> None:
        response = await HandlerMemoryRetrieval().execute(_request(operation))

        assert response.status == "refused"
        assert response.results == []
        assert response.total_count == 0
        assert response.refusal is not None
        assert response.refusal.store is _OPERATION_STORE[operation]
        assert response.refusal.reason is EnumStoreRefusalReason.NOT_CONFIGURED

    @pytest.mark.asyncio
    async def test_hybrid_without_either_store_names_the_vector_store(self) -> None:
        response = await HandlerMemoryRetrieval().execute(_request("search_hybrid"))

        assert response.status == "refused"
        assert response.refusal is not None
        assert response.refusal.store is EnumMemoryStore.VECTOR

    @pytest.mark.asyncio
    async def test_hybrid_names_the_store_that_is_missing(self) -> None:
        vector = StandInHandlerMemoryRetrieval().vector
        response = await HandlerMemoryRetrieval(vector_store=vector).execute(
            _request("search_hybrid")
        )

        assert response.status == "refused"
        assert response.refusal is not None
        assert response.refusal.store is EnumMemoryStore.DATABASE
        assert response.results == []

    @pytest.mark.asyncio
    async def test_missing_store_is_refused_even_when_another_is_configured(
        self,
    ) -> None:
        handler = HandlerMemoryRetrieval(
            vector_store=StandInHandlerMemoryRetrieval().vector
        )

        response = await handler.execute(_request("search_graph"))

        assert response.status == "refused"
        assert response.refusal is not None
        assert response.refusal.store is EnumMemoryStore.GRAPH


@pytest.mark.unit
class TestUnreachableStoreRefuses:
    """AC2: a store that cannot be reached is refused by name with no rows."""

    @pytest.mark.asyncio
    async def test_store_that_fails_to_connect_is_refused(self) -> None:
        down = _DownStore(fail_on="initialize")
        handler = HandlerMemoryRetrieval(vector_store=down)

        response = await handler.execute(_request("search"))

        assert response.status == "refused"
        assert response.results == []
        assert response.refusal is not None
        assert response.refusal.store is EnumMemoryStore.VECTOR
        assert response.refusal.reason is EnumStoreRefusalReason.UNREACHABLE
        assert "connection refused" in response.refusal.detail

    @pytest.mark.asyncio
    async def test_failed_connect_is_retried_on_the_next_search(self) -> None:
        down = _DownStore(fail_on="initialize")
        handler = HandlerMemoryRetrieval(vector_store=down)

        await handler.execute(_request("search"))
        await handler.execute(_request("search"))

        assert down.initialize_calls == 2

    @pytest.mark.asyncio
    async def test_store_that_raises_while_answering_is_refused(self) -> None:
        handler = HandlerMemoryRetrieval(database_store=_DownStore(fail_on="execute"))

        response = await handler.execute(_request("search_text"))

        assert response.status == "refused"
        assert response.results == []
        assert response.refusal is not None
        assert response.refusal.store is EnumMemoryStore.DATABASE
        assert response.refusal.reason is EnumStoreRefusalReason.UNREACHABLE
        assert "timed out" in response.refusal.detail

    @pytest.mark.asyncio
    async def test_store_that_is_reached_but_errors_is_refused_as_failed(self) -> None:
        handler = HandlerMemoryRetrieval(vector_store=_DownStore(fail_on="reached"))

        response = await handler.execute(_request("search"))

        assert response.status == "refused"
        assert response.results == []
        assert response.refusal is not None
        assert response.refusal.store is EnumMemoryStore.VECTOR
        assert response.refusal.reason is EnumStoreRefusalReason.FAILED
        assert "not a memory snapshot" in response.refusal.detail

    @pytest.mark.asyncio
    async def test_hybrid_refuses_when_one_leg_is_unreachable(self) -> None:
        stand_in = StandInHandlerMemoryRetrieval()
        handler = HandlerMemoryRetrieval(
            vector_store=stand_in.vector,
            database_store=_DownStore(fail_on="initialize"),
        )

        response = await handler.execute(_request("search_hybrid"))

        assert response.status == "refused"
        assert response.refusal is not None
        assert response.refusal.store is EnumMemoryStore.DATABASE
        assert response.results == []


@pytest.mark.unit
class TestInjectedStoresAnswer:
    """AC1: a configured store answers through its protocol, and only it does."""

    @pytest.mark.asyncio
    async def test_injected_vector_store_serves_search(self) -> None:
        handler = StandInHandlerMemoryRetrieval()
        await handler.initialize()
        handler.seed_snapshots([_snapshot("authentication login flow")])

        response = await handler.execute(
            ModelMemoryRetrievalRequest(
                operation="search",
                query_text="authentication login",
                similarity_threshold=0.0,
            )
        )

        assert response.status == "success"
        assert response.refusal is None
        assert len(response.results) == 1

    def test_stores_satisfy_their_protocols(self) -> None:
        stand_in = StandInHandlerMemoryRetrieval()

        assert isinstance(stand_in.vector, ProtocolVectorStore)
        assert isinstance(stand_in.database, ProtocolDatabaseStore)
        assert isinstance(stand_in.graph, ProtocolGraphStore)

    def test_production_vector_store_satisfies_its_protocol(self) -> None:
        handler = bind_memory_retrieval(
            ModelRetrievalStoreSettings(
                qdrant_host="qdrant.test",
                embedding_server_url="http://embed.test:8100",
            )
        )

        assert handler.configured_stores == {EnumMemoryStore.VECTOR}
        assert isinstance(handler._stores[EnumMemoryStore.VECTOR], HandlerQdrant)
        assert isinstance(handler._stores[EnumMemoryStore.VECTOR], ProtocolVectorStore)


@pytest.mark.unit
class TestStoresResolveFromSettings:
    """AC1: stores come from settings; an empty settings object configures none."""

    def test_empty_settings_configure_no_store(self) -> None:
        handler = bind_memory_retrieval(ModelRetrievalStoreSettings())

        assert handler.configured_stores == frozenset()

    def test_host_without_embedding_server_is_not_a_vector_store(self) -> None:
        handler = bind_memory_retrieval(
            ModelRetrievalStoreSettings(qdrant_host="qdrant.test")
        )

        assert handler.configured_stores == frozenset()

    def test_settings_read_the_deployed_environment_names(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("QDRANT_HOST", "qdrant.lane")
        monkeypatch.setenv("QDRANT_PORT", "6334")
        monkeypatch.setenv("LLM_EMBEDDING_URL", "http://embed.lane:8100")

        settings = ModelRetrievalStoreSettings()

        assert settings.qdrant_host == "qdrant.lane"
        assert settings.qdrant_port == 6334
        assert settings.embedding_server_url == "http://embed.lane:8100"

    def test_unset_environment_configures_nothing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        for name in ("QDRANT_HOST", "QDRANT_PORT", "LLM_EMBEDDING_URL"):
            monkeypatch.delenv(name, raising=False)

        assert bind_memory_retrieval(
            ModelRetrievalStoreSettings()
        ).configured_stores == (frozenset())

    @pytest.mark.asyncio
    async def test_unreachable_qdrant_from_settings_is_refused_not_answered(
        self,
    ) -> None:
        handler = bind_memory_retrieval(
            ModelRetrievalStoreSettings(
                qdrant_host="qdrant.invalid",
                qdrant_port=6333,
                embedding_server_url="http://embed.invalid:8100",
            )
        )

        response = await handler.execute(_request("search"))

        assert response.status == "refused"
        assert response.results == []
        assert response.refusal is not None
        assert response.refusal.store is EnumMemoryStore.VECTOR
        assert response.refusal.reason is EnumStoreRefusalReason.UNREACHABLE


@pytest.mark.unit
class TestRefusalShape:
    """A refusal names its store and carries no rows, by construction."""

    def test_refused_response_must_name_a_store(self) -> None:
        with pytest.raises(ValidationError):
            ModelMemoryRetrievalResponse(status="refused")

    def test_only_a_refused_response_carries_a_refusal(self) -> None:
        refusal = ModelStoreRefusal(
            store=EnumMemoryStore.GRAPH,
            reason=EnumStoreRefusalReason.NOT_CONFIGURED,
            detail="no graph store is configured for this runtime",
        )
        with pytest.raises(ValidationError):
            ModelMemoryRetrievalResponse(status="success", refusal=refusal)

    def test_refused_response_carries_no_rows(self) -> None:
        refusal = ModelStoreRefusal(
            store=EnumMemoryStore.VECTOR,
            reason=EnumStoreRefusalReason.UNREACHABLE,
            detail="connection refused",
        )
        with pytest.raises(ValidationError):
            ModelMemoryRetrievalResponse(
                status="refused", refusal=refusal, query_embedding_used=[0.1]
            )


@pytest.mark.unit
class TestDispatchRefusesWhenUnconfigured:
    """The runtime dispatch path answers a refusal, never a stand-in's rows."""

    @pytest.mark.asyncio
    async def test_engine_default_handler_refuses_search(self) -> None:
        from omnibase_core.models.effect.model_effect_context import ModelEffectContext
        from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope

        from omnimemory.runtime.dispatch_handlers import (
            create_memory_retrieval_dispatch_handler,
        )

        correlation_id = UUID("12345678-1234-1234-1234-123456789abc")
        dispatch = create_memory_retrieval_dispatch_handler(
            retrieval_handler=bind_memory_retrieval(ModelRetrievalStoreSettings()),
            correlation_id=correlation_id,
        )
        envelope: ModelEventEnvelope[object] = ModelEventEnvelope(
            payload={"operation": "search", "query_text": "authentication"},
            correlation_id=correlation_id,
        )
        context = ModelEffectContext(correlation_id=correlation_id, envelope_id=uuid4())

        result = await dispatch(envelope, context)

        assert isinstance(result, str)
        parsed = json.loads(result)
        assert parsed["status"] == "refused"
        assert parsed["refusal"]["store"] == "vector"
        assert parsed["refusal"]["reason"] == "not_configured"
        assert parsed["results"] == []
