# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Contract-driven topic discovery for the OmniMemory domain.

Reads ``event_bus.subscribe_topics`` and ``event_bus.publish_topics`` from
omnimemory node ``contract.yaml`` files and returns the collected lists.
This replaces formerly-hardcoded topic constants (e.g.
``RegistryIntentQueryEffect.get_topic_suffixes()``).

Design decisions:
    - Topics are declared in each node's contract.yaml (source of truth).
    - This module reads those contracts via ``importlib.resources`` (ONEX I/O
      audit compliant -- package resource reads, not arbitrary filesystem I/O).
    - The module also provides ``canonical_topic_to_dispatch_alias`` to convert
      ONEX canonical topic naming (``.cmd.`` / ``.evt.``) to the dispatch engine
      format (``.commands.`` / ``.events.``).

Related:
    - OMN-2213: Phase 2 -- Contract-driven topic discovery for omnimemory
    - OMN-2033: Reference implementation in omniintelligence
"""

from __future__ import annotations

import importlib.resources
import logging
from collections.abc import Callable

import yaml

logger = logging.getLogger(__name__)

# ============================================================================
# Node packages that declare event_bus topics
# ============================================================================
# All omnimemory nodes with ``event_bus.event_bus_enabled: true`` in their
# contract.yaml. Both effect nodes and orchestrator nodes are included since
# the memory_lifecycle_orchestrator also subscribes to Kafka topics.

_OMNIMEMORY_EVENT_BUS_NODE_PACKAGES: list[str] = [
    "omnimemory.nodes.intent_event_consumer_effect",
    "omnimemory.nodes.intent_query_effect",
    "omnimemory.nodes.intent_storage_effect",
    "omnimemory.nodes.memory_retrieval_effect",
    "omnimemory.nodes.memory_storage_effect",
    "omnimemory.nodes.memory_lifecycle_orchestrator",
]


# ============================================================================
# Public API
# ============================================================================


def collect_subscribe_topics_from_contracts(
    *,
    node_packages: list[str] | None = None,
) -> list[str]:
    """Collect subscribe topics from omnimemory node contracts.

    Scans ``contract.yaml`` files from omnimemory nodes and extracts
    ``event_bus.subscribe_topics`` from each enabled node.  Returns the
    aggregate list of all topics in package-declaration order.

    This is the single replacement for formerly-hardcoded topic constants
    such as ``RegistryIntentQueryEffect.get_topic_suffixes()["subscribe"]``.

    Args:
        node_packages: Override list of node packages to scan.  Defaults to
            the built-in omnimemory event bus nodes.

    Returns:
        Ordered list of subscribe topic strings.

    Note:
        If a ``contract.yaml`` is missing, contains invalid YAML, or a
        node package is not installed, a warning is logged and the
        package is skipped.  This prevents a single missing or corrupt
        contract from blocking discovery of topics from all other valid
        contracts.
    """
    packages = node_packages or _OMNIMEMORY_EVENT_BUS_NODE_PACKAGES
    all_topics: list[str] = []

    for package in packages:
        topics = _safe_read_topics(_read_subscribe_topics, package)
        if topics is not None:
            all_topics.extend(topics)

    logger.debug(
        "Collected %d omnimemory subscribe topics from %d contracts",
        len(all_topics),
        len(packages),
    )

    return all_topics


def collect_publish_topics_for_dispatch(
    *,
    node_packages: list[str] | None = None,
) -> dict[str, str]:
    """Collect publish topics from contracts and map to dispatch engine keys.

    Reads ``event_bus.publish_topics`` from omnimemory node contracts
    and returns a dict mapping dispatch keys to topic strings.

    The dispatch key is derived from the package name by stripping the
    ``omnimemory.nodes.`` prefix and any trailing ``_effect`` /
    ``_orchestrator`` suffix.  For example:
        - ``omnimemory.nodes.intent_query_effect`` -> ``"intent_query"``
        - ``omnimemory.nodes.memory_lifecycle_orchestrator`` -> ``"memory_lifecycle"``

    Only the first publish topic per contract is used in the returned dict.
    Use ``collect_all_publish_topics()`` if you need the full list.

    Args:
        node_packages: Override list of node packages to scan.  Defaults to
            the built-in omnimemory event bus nodes.

    Returns:
        Dict mapping dispatch key to first publish topic string.
        Empty dict if no publish topics are declared.

    Note:
        If a ``contract.yaml`` is missing, contains invalid YAML, or a
        node package is not installed, a warning is logged and the
        package is skipped.  This prevents a single missing or corrupt
        contract from blocking discovery of topics from all other valid
        contracts.
    """
    packages = node_packages or _OMNIMEMORY_EVENT_BUS_NODE_PACKAGES
    result: dict[str, str] = {}

    for package in packages:
        topics = _safe_read_topics(_read_publish_topics, package)
        if topics:
            key = _derive_dispatch_key(package)
            result[key] = topics[0]

    logger.debug(
        "Collected %d publish topics for dispatch engine: %s",
        len(result),
        result,
    )

    return result


def collect_all_publish_topics(
    *,
    node_packages: list[str] | None = None,
) -> list[str]:
    """Collect all publish topics from omnimemory node contracts.

    Unlike ``collect_publish_topics_for_dispatch`` which returns only the
    first topic per node, this function returns every publish topic declared
    across all contracts.

    Args:
        node_packages: Override list of node packages to scan.  Defaults to
            the built-in omnimemory event bus nodes.

    Returns:
        Ordered list of all publish topic strings.  Results may contain
        duplicate topics if multiple nodes publish to the same topic.

    Note:
        If a ``contract.yaml`` is missing, contains invalid YAML, or a
        node package is not installed, a warning is logged and the
        package is skipped.  This prevents a single missing or corrupt
        contract from blocking discovery of topics from all other valid
        contracts.
    """
    packages = node_packages or _OMNIMEMORY_EVENT_BUS_NODE_PACKAGES
    all_topics: list[str] = []

    for package in packages:
        topics = _safe_read_topics(_read_publish_topics, package)
        if topics is not None:
            all_topics.extend(topics)

    return all_topics


def canonical_topic_to_dispatch_alias(topic: str) -> str:
    """Convert ONEX canonical topic naming to dispatch engine format.

    ONEX canonical naming uses ``.cmd.`` for commands and ``.evt.`` for
    events.  ``MessageDispatchEngine`` expects ``.commands.`` and
    ``.events.`` segments.  This function bridges the naming gap.

    Args:
        topic: Canonical topic string (e.g.
            ``onex.cmd.omnimemory.intent-query-requested.v1``).

    Returns:
        Dispatch-compatible topic string (e.g.
            ``onex.commands.omnimemory.intent-query-requested.v1``).
    """
    return topic.replace(".cmd.", ".commands.").replace(".evt.", ".events.")


# ============================================================================
# Internal helpers
# ============================================================================


def _safe_read_topics(
    reader: Callable[[str], list[str]],
    package: str,
) -> list[str] | None:
    """Call *reader* for *package*, handling common contract read errors.

    Wraps ``FileNotFoundError``, ``ModuleNotFoundError``, and
    ``yaml.YAMLError`` with appropriate warning-level log messages and
    returns ``None`` so the caller can skip the package.

    Args:
        reader: A callable that accepts a package name and returns a list
            of topic strings (e.g. ``_read_subscribe_topics``).
        package: Fully-qualified Python package path containing a
            ``contract.yaml`` file.

    Returns:
        The topic list on success, or ``None`` if the contract could not
        be read.
    """
    try:
        return reader(package)
    except FileNotFoundError:
        logger.warning(
            "contract.yaml not found in package %s, skipping",
            package,
        )
        return None
    except ModuleNotFoundError:
        logger.warning(
            "Package %s is not installed/importable, skipping",
            package,
        )
        return None
    except yaml.YAMLError:
        logger.warning(
            "contract.yaml in package %s contains invalid YAML, skipping",
            package,
        )
        return None


def _derive_dispatch_key(package: str) -> str:
    """Derive a dispatch key from a fully-qualified node package path.

    Strips ``omnimemory.nodes.`` prefix and common suffixes
    (``_effect``, ``_orchestrator``, ``_compute``, ``_reducer``).

    Args:
        package: Fully-qualified Python package path.

    Returns:
        Short dispatch key string.

    Examples:
        >>> _derive_dispatch_key("omnimemory.nodes.intent_query_effect")
        'intent_query'
        >>> _derive_dispatch_key("omnimemory.nodes.memory_lifecycle_orchestrator")
        'memory_lifecycle'
    """
    tail = package.rsplit(".", 1)[-1]
    for suffix in ("_effect", "_orchestrator", "_compute", "_reducer"):
        if tail.endswith(suffix):
            tail = tail[: -len(suffix)]
            break
    return tail


def _read_event_bus_topics(package: str, field: str) -> list[str]:
    """Read a topic list from a node package's ``event_bus`` contract section.

    Shared implementation for both subscribe and publish topic discovery.
    Uses ``importlib.resources`` for ONEX I/O audit compliance.

    Args:
        package: Fully-qualified Python package path containing
            a ``contract.yaml`` file.
        field: Topic field name (``"subscribe_topics"`` or ``"publish_topics"``).

    Returns:
        List of topic strings (empty if event bus is disabled or field absent).
    """
    package_files = importlib.resources.files(package)
    contract_file = package_files.joinpath("contract.yaml")
    content = contract_file.read_text(encoding="utf-8")
    contract: object = yaml.safe_load(content)

    if not isinstance(contract, dict):
        logger.warning(
            "contract.yaml in %s is not a valid mapping (got %s), skipping",
            package,
            type(contract).__name__,
        )
        return []

    event_bus: object = contract.get("event_bus", {})
    if not isinstance(event_bus, dict):
        logger.warning(
            "event_bus in %s contract.yaml is not a mapping (got %s), skipping",
            package,
            type(event_bus).__name__,
        )
        return []

    if not event_bus.get("event_bus_enabled", False):
        return []

    topics_raw: object = event_bus.get(field, [])
    if not isinstance(topics_raw, list):
        logger.warning(
            "%s in %s contract.yaml is not a list, skipping",
            field,
            package,
        )
        return []

    topics: list[str] = [t for t in topics_raw if isinstance(t, str)]
    if len(topics) != len(topics_raw):
        logger.warning(
            "%s in %s contract.yaml contains non-string entries, skipping invalid items",
            field,
            package,
        )

    if topics:
        logger.debug(
            "Discovered %s from %s: %s",
            field,
            package,
            topics,
        )
    return topics


def _read_subscribe_topics(package: str) -> list[str]:
    """Read ``event_bus.subscribe_topics`` from a node package's contract."""
    return _read_event_bus_topics(package, "subscribe_topics")


def _read_publish_topics(package: str) -> list[str]:
    """Read ``event_bus.publish_topics`` from a node package's contract."""
    return _read_event_bus_topics(package, "publish_topics")


__all__ = [
    "canonical_topic_to_dispatch_alias",
    "collect_all_publish_topics",
    "collect_publish_topics_for_dispatch",
    "collect_subscribe_topics_from_contracts",
]
