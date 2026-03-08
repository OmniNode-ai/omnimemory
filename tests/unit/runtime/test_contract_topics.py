# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Unit tests for contract-driven topic discovery.

Validates:
    - collect_subscribe_topics_from_contracts returns correct topics
    - collect_publish_topics_for_dispatch returns correct dispatch map
    - collect_all_publish_topics returns all declared publish topics
    - canonical_topic_to_dispatch_alias converts correctly
    - No hardcoded topic constants remain in registry files

Related:
    - OMN-2213: Phase 2 -- Contract-driven topic discovery for omnimemory
"""

from __future__ import annotations

import pytest

from omnimemory.runtime.contract_topics import (
    _OMNIMEMORY_EVENT_BUS_NODE_PACKAGES,
    _derive_dispatch_key,
    canonical_topic_to_dispatch_alias,
    collect_all_publish_topics,
    collect_publish_topics_for_dispatch,
    collect_subscribe_topics_from_contracts,
)

# =============================================================================
# Expected subscribe topics (must match contract.yaml declarations)
# =============================================================================

# intent_event_consumer_effect
EXPECTED_INTENT_CLASSIFIED = "onex.evt.omniintelligence.intent-classified.v1"

# intent_query_effect
EXPECTED_INTENT_QUERY_REQUESTED = "onex.cmd.omnimemory.intent-query-requested.v1"

# memory_retrieval_effect
EXPECTED_MEMORY_RETRIEVAL_REQUESTED = (
    "onex.cmd.omnimemory.memory-retrieval-requested.v1"
)

# memory_lifecycle_orchestrator
EXPECTED_RUNTIME_TICK = "onex.cmd.omnimemory.runtime-tick.v1"
EXPECTED_ARCHIVE_MEMORY = "onex.cmd.omnimemory.archive-memory.v1"
EXPECTED_EXPIRE_MEMORY = "onex.cmd.omnimemory.expire-memory.v1"

# filesystem_crawler_effect
EXPECTED_CRAWL_TICK = "onex.cmd.omnimemory.crawl-tick.v1"

EXPECTED_SUBSCRIBE_TOPICS = {
    EXPECTED_INTENT_CLASSIFIED,
    EXPECTED_INTENT_QUERY_REQUESTED,
    EXPECTED_MEMORY_RETRIEVAL_REQUESTED,
    EXPECTED_RUNTIME_TICK,
    EXPECTED_ARCHIVE_MEMORY,
    EXPECTED_EXPIRE_MEMORY,
    EXPECTED_CRAWL_TICK,
}

# =============================================================================
# Expected publish topics (first per node, for dispatch map)
# =============================================================================

EXPECTED_DISPATCH_MAP = {
    "intent_event_consumer": "onex.evt.omnimemory.intent-stored.v1",
    "intent_query": "onex.evt.omnimemory.intent-query-response.v1",
    "intent_storage": "onex.evt.omnimemory.intent-stored.v1",
    "memory_retrieval": "onex.evt.omnimemory.memory-retrieval-response.v1",
    "memory_storage": "onex.evt.omnimemory.memory-stored.v1",
    "memory_lifecycle": "onex.evt.omnimemory.memory-expired.v1",
    "filesystem_crawler": "onex.evt.omnimemory.document-discovered.v1",
}

# =============================================================================
# Expected ALL publish topics (full list across all contracts)
# =============================================================================

EXPECTED_ALL_PUBLISH_TOPICS = {
    # intent_event_consumer_effect
    "onex.evt.omnimemory.intent-stored.v1",
    "onex.evt.omniintelligence.intent-classified.v1.dlq",
    # intent_query_effect
    "onex.evt.omnimemory.intent-query-response.v1",
    # intent_storage_effect (intent-stored.v1 also from consumer, set deduplicates)
    "onex.evt.omnimemory.intent-store-failed.v1",
    # memory_retrieval_effect
    "onex.evt.omnimemory.memory-retrieval-response.v1",
    # memory_storage_effect
    "onex.evt.omnimemory.memory-stored.v1",
    "onex.evt.omnimemory.memory-retrieved.v1",
    "onex.evt.omnimemory.memory-updated.v1",
    "onex.evt.omnimemory.memory-deleted.v1",
    # memory_lifecycle_orchestrator
    "onex.evt.omnimemory.memory-expired.v1",
    "onex.evt.omnimemory.memory-archived.v1",
    "onex.evt.omnimemory.memory-archive-initiated.v1",
    "onex.evt.omnimemory.lifecycle-transition-failed.v1",
    # filesystem_crawler_effect
    "onex.evt.omnimemory.document-discovered.v1",
    "onex.evt.omnimemory.document-changed.v1",
    "onex.evt.omnimemory.document-removed.v1",
    "onex.evt.omnimemory.document-indexed.v1",
}


# =============================================================================
# Tests: collect_subscribe_topics_from_contracts
# =============================================================================


@pytest.mark.unit
class TestCollectSubscribeTopics:
    """Validate contract-driven subscribe topic collection."""

    def test_returns_exactly_seven_topics(self) -> None:
        """All omnimemory nodes declare 7 subscribe topics total."""
        topics = collect_subscribe_topics_from_contracts()
        assert len(topics) == 7

    def test_contains_intent_classified_topic(self) -> None:
        """Intent classified topic from intent_event_consumer_effect."""
        topics = collect_subscribe_topics_from_contracts()
        assert EXPECTED_INTENT_CLASSIFIED in topics

    def test_contains_intent_query_requested_topic(self) -> None:
        """Intent query requested topic from intent_query_effect."""
        topics = collect_subscribe_topics_from_contracts()
        assert EXPECTED_INTENT_QUERY_REQUESTED in topics

    def test_contains_memory_retrieval_requested_topic(self) -> None:
        """Memory retrieval requested topic from memory_retrieval_effect."""
        topics = collect_subscribe_topics_from_contracts()
        assert EXPECTED_MEMORY_RETRIEVAL_REQUESTED in topics

    def test_contains_runtime_tick_topic(self) -> None:
        """Runtime tick topic from memory_lifecycle_orchestrator."""
        topics = collect_subscribe_topics_from_contracts()
        assert EXPECTED_RUNTIME_TICK in topics

    def test_contains_archive_memory_topic(self) -> None:
        """Archive memory command topic from memory_lifecycle_orchestrator."""
        topics = collect_subscribe_topics_from_contracts()
        assert EXPECTED_ARCHIVE_MEMORY in topics

    def test_contains_expire_memory_topic(self) -> None:
        """Expire memory command topic from memory_lifecycle_orchestrator."""
        topics = collect_subscribe_topics_from_contracts()
        assert EXPECTED_EXPIRE_MEMORY in topics

    def test_contains_crawl_tick_topic(self) -> None:
        """Crawl tick command topic from filesystem_crawler_effect."""
        topics = collect_subscribe_topics_from_contracts()
        assert EXPECTED_CRAWL_TICK in topics

    def test_all_expected_topics_present(self) -> None:
        """All 6 expected subscribe topics must be in the discovered set."""
        topics = set(collect_subscribe_topics_from_contracts())
        assert topics == EXPECTED_SUBSCRIBE_TOPICS

    def test_returns_list_type(self) -> None:
        """Return type must be a list for ordered iteration."""
        topics = collect_subscribe_topics_from_contracts()
        assert isinstance(topics, list)

    def test_no_duplicates(self) -> None:
        """No duplicate topics should be returned."""
        topics = collect_subscribe_topics_from_contracts()
        assert len(topics) == len(set(topics))

    def test_custom_node_packages_override(self) -> None:
        """Providing node_packages overrides the default list."""
        topics = collect_subscribe_topics_from_contracts(
            node_packages=["omnimemory.nodes.node_intent_query_effect"],
        )
        assert topics == [EXPECTED_INTENT_QUERY_REQUESTED]

    def test_node_with_empty_subscribe_returns_nothing(self) -> None:
        """Nodes with empty subscribe_topics contribute nothing."""
        topics = collect_subscribe_topics_from_contracts(
            node_packages=["omnimemory.nodes.node_intent_storage_effect"],
        )
        assert topics == []


# =============================================================================
# Tests: collect_publish_topics_for_dispatch
# =============================================================================


@pytest.mark.unit
class TestCollectPublishTopicsForDispatch:
    """Validate contract-driven publish topic collection for dispatch engine."""

    def test_returns_dict(self) -> None:
        """Return type must be a dict."""
        result = collect_publish_topics_for_dispatch()
        assert isinstance(result, dict)

    def test_contains_all_expected_keys(self) -> None:
        """All expected dispatch keys must be present."""
        result = collect_publish_topics_for_dispatch()
        assert set(result.keys()) == set(EXPECTED_DISPATCH_MAP.keys())

    def test_intent_event_consumer_topic(self) -> None:
        """intent_event_consumer dispatch key maps to correct topic."""
        result = collect_publish_topics_for_dispatch()
        assert (
            result["intent_event_consumer"]
            == EXPECTED_DISPATCH_MAP["intent_event_consumer"]
        )

    def test_intent_query_topic(self) -> None:
        """intent_query dispatch key maps to correct topic."""
        result = collect_publish_topics_for_dispatch()
        assert result["intent_query"] == EXPECTED_DISPATCH_MAP["intent_query"]

    def test_intent_storage_topic(self) -> None:
        """intent_storage dispatch key maps to correct topic."""
        result = collect_publish_topics_for_dispatch()
        assert result["intent_storage"] == EXPECTED_DISPATCH_MAP["intent_storage"]

    def test_memory_retrieval_topic(self) -> None:
        """memory_retrieval dispatch key maps to correct topic."""
        result = collect_publish_topics_for_dispatch()
        assert result["memory_retrieval"] == EXPECTED_DISPATCH_MAP["memory_retrieval"]

    def test_memory_storage_topic(self) -> None:
        """memory_storage dispatch key maps to correct topic."""
        result = collect_publish_topics_for_dispatch()
        assert result["memory_storage"] == EXPECTED_DISPATCH_MAP["memory_storage"]

    def test_memory_lifecycle_topic(self) -> None:
        """memory_lifecycle dispatch key maps to correct topic."""
        result = collect_publish_topics_for_dispatch()
        assert result["memory_lifecycle"] == EXPECTED_DISPATCH_MAP["memory_lifecycle"]

    def test_all_values_are_strings(self) -> None:
        """All publish topic values must be strings."""
        result = collect_publish_topics_for_dispatch()
        for key, value in result.items():
            assert isinstance(value, str), f"Value for '{key}' is not a string: {value}"

    def test_all_values_contain_evt(self) -> None:
        """All publish topics must be .evt. topics."""
        result = collect_publish_topics_for_dispatch()
        for key, value in result.items():
            assert ".evt." in value, (
                f"Publish topic '{key}' is not an event topic: {value}"
            )

    def test_custom_node_packages_override(self) -> None:
        """Providing node_packages overrides the default list."""
        result = collect_publish_topics_for_dispatch(
            node_packages=["omnimemory.nodes.node_intent_query_effect"],
        )
        assert result == {
            "intent_query": "onex.evt.omnimemory.intent-query-response.v1"
        }


# =============================================================================
# Tests: collect_all_publish_topics
# =============================================================================


@pytest.mark.unit
class TestCollectAllPublishTopics:
    """Validate full publish topic collection across all contracts."""

    def test_returns_list(self) -> None:
        """Return type must be a list."""
        result = collect_all_publish_topics()
        assert isinstance(result, list)

    def test_all_expected_topics_present(self) -> None:
        """All expected publish topics must be in the discovered set."""
        topics = set(collect_all_publish_topics())
        assert topics == EXPECTED_ALL_PUBLISH_TOPICS

    def test_includes_dlq_topics(self) -> None:
        """DLQ topics declared in publish_topics must be included."""
        topics = collect_all_publish_topics()
        assert "onex.evt.omniintelligence.intent-classified.v1.dlq" in topics

    def test_includes_all_memory_storage_topics(self) -> None:
        """All 4 memory storage CRUD event topics must be present."""
        topics = set(collect_all_publish_topics())
        expected_crud = {
            "onex.evt.omnimemory.memory-stored.v1",
            "onex.evt.omnimemory.memory-retrieved.v1",
            "onex.evt.omnimemory.memory-updated.v1",
            "onex.evt.omnimemory.memory-deleted.v1",
        }
        assert expected_crud.issubset(topics)


# =============================================================================
# Tests: canonical_topic_to_dispatch_alias
# =============================================================================


@pytest.mark.unit
class TestCanonicalTopicToDispatchAlias:
    """Validate canonical-to-dispatch topic conversion."""

    def test_converts_cmd_to_commands(self) -> None:
        """``.cmd.`` should be converted to ``.commands.``."""
        result = canonical_topic_to_dispatch_alias(
            "onex.cmd.omnimemory.intent-query-requested.v1"
        )
        assert result == "onex.commands.omnimemory.intent-query-requested.v1"

    def test_converts_evt_to_events(self) -> None:
        """``.evt.`` should be converted to ``.events.``."""
        result = canonical_topic_to_dispatch_alias(
            "onex.evt.omnimemory.intent-stored.v1"
        )
        assert result == "onex.events.omnimemory.intent-stored.v1"

    def test_no_cmd_or_evt_unchanged(self) -> None:
        """Topics without .cmd. or .evt. should pass through unchanged."""
        topic = "some.other.topic.v1"
        assert canonical_topic_to_dispatch_alias(topic) == topic

    def test_internal_topic_unchanged(self) -> None:
        """Internal topics without .cmd./.evt. pass through."""
        topic = "onex.internal.some-internal-event.v1"
        assert canonical_topic_to_dispatch_alias(topic) == topic

    @pytest.mark.parametrize(
        ("canonical", "expected_alias"),
        [
            (
                EXPECTED_INTENT_QUERY_REQUESTED,
                "onex.commands.omnimemory.intent-query-requested.v1",
            ),
            (
                EXPECTED_MEMORY_RETRIEVAL_REQUESTED,
                "onex.commands.omnimemory.memory-retrieval-requested.v1",
            ),
            (
                EXPECTED_ARCHIVE_MEMORY,
                "onex.commands.omnimemory.archive-memory.v1",
            ),
            (
                EXPECTED_EXPIRE_MEMORY,
                "onex.commands.omnimemory.expire-memory.v1",
            ),
            (
                EXPECTED_RUNTIME_TICK,
                "onex.commands.omnimemory.runtime-tick.v1",
            ),
            (
                EXPECTED_INTENT_CLASSIFIED,
                "onex.events.omniintelligence.intent-classified.v1",
            ),
        ],
    )
    def test_all_omnimemory_subscribe_topics_convert(
        self,
        canonical: str,
        expected_alias: str,
    ) -> None:
        """All subscribe topics must produce correct dispatch aliases."""
        assert canonical_topic_to_dispatch_alias(canonical) == expected_alias


# =============================================================================
# Tests: _derive_dispatch_key
# =============================================================================


@pytest.mark.unit
class TestDeriveDispatchKey:
    """Validate dispatch key derivation from package paths."""

    def test_strips_effect_suffix(self) -> None:
        """_effect suffix should be stripped."""
        assert (
            _derive_dispatch_key("omnimemory.nodes.node_intent_query_effect")
            == "intent_query"
        )

    def test_strips_orchestrator_suffix(self) -> None:
        """_orchestrator suffix should be stripped."""
        assert (
            _derive_dispatch_key("omnimemory.nodes.node_memory_lifecycle_orchestrator")
            == "memory_lifecycle"
        )

    def test_strips_compute_suffix(self) -> None:
        """_compute suffix should be stripped."""
        assert (
            _derive_dispatch_key("omnimemory.nodes.node_similarity_compute")
            == "similarity"
        )

    def test_strips_reducer_suffix(self) -> None:
        """_reducer suffix should be stripped."""
        assert (
            _derive_dispatch_key("omnimemory.nodes.node_memory_consolidator_reducer")
            == "memory_consolidator"
        )

    def test_no_matching_suffix(self) -> None:
        """Package without known suffix should use full tail."""
        assert _derive_dispatch_key("omnimemory.nodes.some_node") == "some_node"


# =============================================================================
# Tests: Node package registry completeness
# =============================================================================


@pytest.mark.unit
class TestNodePackageRegistry:
    """Validate that _OMNIMEMORY_EVENT_BUS_NODE_PACKAGES is complete."""

    def test_contains_intent_event_consumer(self) -> None:
        """intent_event_consumer_effect must be in the package list."""
        assert (
            "omnimemory.nodes.node_intent_event_consumer_effect"
            in _OMNIMEMORY_EVENT_BUS_NODE_PACKAGES
        )

    def test_contains_intent_query(self) -> None:
        """intent_query_effect must be in the package list."""
        assert (
            "omnimemory.nodes.node_intent_query_effect"
            in _OMNIMEMORY_EVENT_BUS_NODE_PACKAGES
        )

    def test_contains_intent_storage(self) -> None:
        """intent_storage_effect must be in the package list."""
        assert (
            "omnimemory.nodes.node_intent_storage_effect"
            in _OMNIMEMORY_EVENT_BUS_NODE_PACKAGES
        )

    def test_contains_memory_retrieval(self) -> None:
        """memory_retrieval_effect must be in the package list."""
        assert (
            "omnimemory.nodes.node_memory_retrieval_effect"
            in _OMNIMEMORY_EVENT_BUS_NODE_PACKAGES
        )

    def test_contains_memory_storage(self) -> None:
        """memory_storage_effect must be in the package list."""
        assert (
            "omnimemory.nodes.node_memory_storage_effect"
            in _OMNIMEMORY_EVENT_BUS_NODE_PACKAGES
        )

    def test_contains_memory_lifecycle(self) -> None:
        """memory_lifecycle_orchestrator must be in the package list."""
        assert (
            "omnimemory.nodes.node_memory_lifecycle_orchestrator"
            in _OMNIMEMORY_EVENT_BUS_NODE_PACKAGES
        )

    def test_contains_filesystem_crawler(self) -> None:
        """filesystem_crawler_effect must be in the package list."""
        assert (
            "omnimemory.nodes.node_filesystem_crawler_effect"
            in _OMNIMEMORY_EVENT_BUS_NODE_PACKAGES
        )

    def test_exactly_seven_packages(self) -> None:
        """Exactly 7 omnimemory nodes have event_bus enabled."""
        assert len(_OMNIMEMORY_EVENT_BUS_NODE_PACKAGES) == 7


# =============================================================================
# Tests: No hardcoded topic constants in registry
# =============================================================================


@pytest.mark.unit
class TestNoHardcodedTopics:
    """Validate that hardcoded topic constants have been removed."""

    def test_registry_has_no_get_topic_suffixes(self) -> None:
        """RegistryIntentQueryEffect must not have get_topic_suffixes method."""
        from omnimemory.nodes.node_intent_query_effect.registry import (
            RegistryIntentQueryEffect,
        )

        assert not hasattr(RegistryIntentQueryEffect, "get_topic_suffixes"), (
            "get_topic_suffixes should have been removed from "
            "RegistryIntentQueryEffect -- topics are now contract-driven"
        )

    def test_registry_has_no_topic_string_constants(self) -> None:
        """RegistryIntentQueryEffect must not define any topic string constants.

        Topic strings must live in contract.yaml, not in Python registry code.
        This validates OMN-1538 acceptance criterion: no hardcoded topic strings
        in Python code.
        """
        import inspect

        from omnimemory.nodes.node_intent_query_effect.registry import (
            RegistryIntentQueryEffect,
        )

        source = inspect.getsource(RegistryIntentQueryEffect)
        # Topic strings follow the pattern onex.<kind>.<service>.<event>.<version>
        assert "onex.cmd." not in source, (
            "Found hardcoded onex.cmd.* topic in RegistryIntentQueryEffect -- "
            "topics must be declared in contract.yaml"
        )
        assert "onex.evt." not in source, (
            "Found hardcoded onex.evt.* topic in RegistryIntentQueryEffect -- "
            "topics must be declared in contract.yaml"
        )


# =============================================================================
# Tests: OMN-1538 -- Topic validation at contract load time
# =============================================================================


@pytest.mark.unit
class TestContractTopicValidation:
    """Validate that intent_query_effect contract topics conform to ONEX naming.

    Acceptance criterion (OMN-1538, AC3): Topics validated against ONEX
    convention at contract load time via ModelEventBusSubcontract.
    """

    def test_intent_query_effect_subscribe_topics_are_valid_onex_names(self) -> None:
        """Subscribe topics in intent_query_effect contract must pass ONEX validation."""
        from omnibase_core.validation import validate_topic_suffix

        topics = collect_subscribe_topics_from_contracts(
            node_packages=["omnimemory.nodes.node_intent_query_effect"],
        )
        assert len(topics) >= 1, (
            "intent_query_effect must declare at least 1 subscribe topic"
        )
        for topic in topics:
            result = validate_topic_suffix(topic)
            assert result.is_valid, (
                f"Subscribe topic '{topic}' in intent_query_effect contract.yaml "
                f"does not conform to ONEX naming convention: {result.error_message}"
            )

    def test_intent_query_effect_publish_topics_are_valid_onex_names(self) -> None:
        """Publish topics in intent_query_effect contract must pass ONEX validation."""
        from omnibase_core.validation import validate_topic_suffix

        result = collect_publish_topics_for_dispatch(
            node_packages=["omnimemory.nodes.node_intent_query_effect"],
        )
        assert result, "intent_query_effect must declare at least 1 publish topic"
        for _key, topic in result.items():
            validation = validate_topic_suffix(topic)
            assert validation.is_valid, (
                f"Publish topic '{topic}' in intent_query_effect contract.yaml "
                f"does not conform to ONEX naming convention: {validation.error_message}"
            )

    def test_intent_query_effect_subscribe_topic_is_cmd_kind(self) -> None:
        """Subscribe topics for intent_query_effect must be .cmd. (command) topics."""
        topics = collect_subscribe_topics_from_contracts(
            node_packages=["omnimemory.nodes.node_intent_query_effect"],
        )
        for topic in topics:
            assert ".cmd." in topic, (
                f"Subscribe topic '{topic}' must use .cmd. kind "
                f"(intent_query_effect subscribes to commands)"
            )

    def test_intent_query_effect_publish_topic_is_evt_kind(self) -> None:
        """Publish topics for intent_query_effect must be .evt. (event) topics."""
        result = collect_publish_topics_for_dispatch(
            node_packages=["omnimemory.nodes.node_intent_query_effect"],
        )
        for _key, topic in result.items():
            assert ".evt." in topic, (
                f"Publish topic '{topic}' must use .evt. kind "
                f"(intent_query_effect publishes response events)"
            )

    def test_intent_query_effect_subscribe_topic_has_omnimemory_domain(self) -> None:
        """Subscribe topic must be in omnimemory domain."""
        topics = collect_subscribe_topics_from_contracts(
            node_packages=["omnimemory.nodes.node_intent_query_effect"],
        )
        for topic in topics:
            assert "omnimemory" in topic, (
                f"Subscribe topic '{topic}' must be in the omnimemory domain"
            )

    def test_intent_query_effect_publish_topic_has_omnimemory_domain(self) -> None:
        """Publish topic must be in omnimemory domain."""
        result = collect_publish_topics_for_dispatch(
            node_packages=["omnimemory.nodes.node_intent_query_effect"],
        )
        for _key, topic in result.items():
            assert "omnimemory" in topic, (
                f"Publish topic '{topic}' must be in the omnimemory domain"
            )
