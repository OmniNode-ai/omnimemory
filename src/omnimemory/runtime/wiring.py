# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Memory domain handler wiring via contract-driven discovery.

Discovers handler classes from contract.yaml files in omnimemory node
packages and standalone handler contracts, then verifies importability.

Replaces the former hardcoded ``_HANDLER_SPECS`` list with contract-driven
discovery that reads ``handler_config`` and ``handler_routing`` sections
from YAML contracts.

Note:
    OmniMemory handlers require runtime dependencies (storage adapters,
    containers, event buses) and cannot be fully instantiated at wiring
    time.  This module verifies importability only.  Full instantiation
    happens when the kernel creates handler instances with injected deps.

Related:
    - OMN-7150, OMN-7151, OMN-7152: Handler wiring migration
    - OMN-2216: Phase 5 -- Runtime plugin PluginMemory
    - omnimemory/runtime/contract_topics.py (reference pattern)
"""

from __future__ import annotations

import asyncio
import importlib
import importlib.resources
import logging
import re
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from omnibase_infra.runtime.models import ModelDomainPluginConfig

logger = logging.getLogger(__name__)

# Node packages containing handler declarations in contract.yaml.
# Each package must have a contract.yaml with either:
#   - handler_config: {handler_class, handler_module}
#   - handler_routing: {handlers: [{handler_key, ...}]}
_OMNIMEMORY_HANDLER_NODE_PACKAGES: list[str] = [
    "omnimemory.nodes.node_intent_event_consumer_effect",
    "omnimemory.nodes.node_intent_query_effect",
    "omnimemory.nodes.node_similarity_compute",
    "omnimemory.nodes.node_semantic_analyzer_compute",
    "omnimemory.nodes.node_navigation_history_reducer",
]

# Standalone handler contracts (not in node packages).
# Each entry is (package, filename) for importlib.resources access.
_OMNIMEMORY_STANDALONE_HANDLER_CONTRACTS: list[tuple[str, str]] = [
    ("omnimemory.handlers", "handler_subscription_contract.yaml"),
    ("omnimemory.handlers.adapters", "adapter_intent_graph_contract.yaml"),
    ("omnimemory.runtime", "handler_lifecycle_contract.yaml"),
]


def _class_to_module(class_name: str) -> str:
    """Convert CamelCase class name to snake_case module name.

    Example: HandlerIntentQuery -> handler_intent_query
    """
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", class_name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def _read_handler_spec_from_contract(
    contract: dict[str, object],
    package: str,
) -> tuple[str, str] | None:
    """Extract (module_path, class_name) from a contract dict.

    Supports both handler_config (flat) and handler_routing (structured)
    patterns. Returns None if no handler declaration found.
    """
    # Try handler_config first (flat pattern)
    handler_config = contract.get("handler_config")
    if isinstance(handler_config, dict):
        handler_class = handler_config.get("handler_class")
        handler_module = handler_config.get("handler_module")
        if handler_class and handler_module:
            return (str(handler_module), str(handler_class))

    # Try handler_routing (structured pattern) — extract first handler
    handler_routing = contract.get("handler_routing")
    if isinstance(handler_routing, dict):
        handlers = handler_routing.get("handlers", [])
        if handlers and isinstance(handlers, list):
            first = handlers[0]
            if isinstance(first, dict):
                # handler_routing uses handler_key (class name only),
                # sometimes with method suffix (e.g., "HandlerSimilarityCompute.cosine_distance")
                handler_key = first.get("handler_key")
                if handler_key and isinstance(handler_key, str):
                    # Strip method suffix if present
                    class_name = handler_key.split(".")[0]
                    # Infer module from package: package.handlers.handler_<snake_name>
                    return (
                        f"{package}.handlers.{_class_to_module(class_name)}",
                        class_name,
                    )

    return None


def _discover_from_node_package(package: str) -> tuple[str, str] | None:
    """Read handler spec from a node package's contract.yaml."""
    try:
        pkg_files = importlib.resources.files(package)
        contract_file = pkg_files.joinpath("contract.yaml")
        content = contract_file.read_text(encoding="utf-8")
    except (FileNotFoundError, ModuleNotFoundError, TypeError):
        logger.warning("No contract.yaml found in %s, skipping", package)
        return None

    contract: object = yaml.safe_load(content)
    if not isinstance(contract, dict):
        logger.warning("contract.yaml in %s is not a valid mapping, skipping", package)
        return None

    spec = _read_handler_spec_from_contract(contract, package)
    if spec is None:
        logger.warning(
            "No handler_config or handler_routing in %s contract.yaml, skipping",
            package,
        )
    return spec


def _discover_from_standalone_contract(
    package: str,
    filename: str,
) -> tuple[str, str] | None:
    """Read handler spec from a standalone handler contract YAML."""
    try:
        pkg_files = importlib.resources.files(package)
        contract_file = pkg_files.joinpath(filename)
        content = contract_file.read_text(encoding="utf-8")
    except (FileNotFoundError, ModuleNotFoundError, TypeError):
        logger.warning(
            "Standalone contract %s/%s not found, skipping", package, filename
        )
        return None

    contract: object = yaml.safe_load(content)
    if not isinstance(contract, dict):
        logger.warning("%s/%s is not a valid mapping, skipping", package, filename)
        return None

    spec = _read_handler_spec_from_contract(contract, package)
    if spec is None:
        logger.warning("No handler_config in %s/%s, skipping", package, filename)
    return spec


async def wire_memory_handlers(
    config: ModelDomainPluginConfig,
) -> list[str]:
    """Wire memory domain handlers by verifying importability.

    Discovers handler classes from contract.yaml files in omnimemory
    node packages and standalone handler contracts, then verifies each
    handler is importable and callable.

    Args:
        config: Plugin configuration with container and correlation_id.

    Returns:
        List of handler names successfully verified.

    Raises:
        ImportError: If any required handler module cannot be imported.
    """
    correlation_id = config.correlation_id
    services_registered: list[str] = []

    # Collect all handler specs from contracts
    specs: list[tuple[str, str]] = []

    for package in _OMNIMEMORY_HANDLER_NODE_PACKAGES:
        spec = _discover_from_node_package(package)
        if spec:
            specs.append(spec)

    for package, filename in _OMNIMEMORY_STANDALONE_HANDLER_CONTRACTS:
        spec = _discover_from_standalone_contract(package, filename)
        if spec:
            specs.append(spec)

    # Verify importability for each discovered handler
    for module_path, attr_name in specs:
        try:
            mod = await asyncio.to_thread(importlib.import_module, module_path)
        except ModuleNotFoundError as e:
            raise ImportError(
                f"Failed to import handler module '{module_path}' "
                f"(correlation_id={correlation_id})"
            ) from e
        try:
            handler_attr = getattr(mod, attr_name)
        except AttributeError as e:
            raise ImportError(f"{attr_name} not found in {module_path}") from e

        if not callable(handler_attr):
            raise ImportError(f"{attr_name} in {module_path} is not callable")

        logger.debug(
            "Verified %s importable (correlation_id=%s)",
            attr_name,
            correlation_id,
        )

        services_registered.append(attr_name)

    logger.info(
        "Memory handlers wired: %d services (correlation_id=%s)",
        len(services_registered),
        correlation_id,
        extra={"services": services_registered},
    )

    return services_registered


__all__: list[str] = [
    "wire_memory_handlers",
]
