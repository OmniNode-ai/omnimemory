# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Read the contract-declared Qdrant endpoint for the memory-retrieval effect.

The memory-retrieval node's ``descriptor.qdrant_host`` / ``descriptor.qdrant_port``
are the single source of truth for the Qdrant endpoint the retrieval handlers
(and the navigation-history reducer that writes to the same store) connect to
(OMN-13562 / OMN-13556 Wave-1 endpoint→overlay migration). They are declared with
the ``${env.VAR}`` overlay convention so an operator overlay / the per-lane
service env supplies the real endpoint per lane — never a hardcoded host in
source.

Resolution goes through ``expand_contract_env_refs`` — the one sanctioned
env-reading boundary in the overlay package — so callers never read
``os.environ`` directly. The host is resolved fail-closed: an unset/empty host
raises rather than silently defaulting to localhost (which would mask a
missing-config deploy and connect to the wrong endpoint). The port carries a
``6333`` inline default (the Qdrant HTTP default) to preserve prior behaviour.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from omnibase_infra.runtime.overlay.contract_env_ref import expand_contract_env_refs

_CONTRACT = Path(__file__).resolve().parent / "contract.yaml"


def _load_descriptor(contract_path: Path) -> dict[str, object]:
    # node-purity-ok: reads the node's OWN contract.yaml (contract-owned endpoint
    # policy), not runtime data I/O — the sanctioned descriptor-load boundary for
    # the ${env.VAR} overlay seam (OMN-13562 Wave-1), mirroring the infra
    # node_vector_store_effect contract_descriptor pattern.
    with contract_path.open(encoding="utf-8") as contract_file:  # node-purity-ok
        raw = yaml.safe_load(contract_file)
    if not isinstance(raw, dict):
        raise ValueError(f"contract {contract_path} must contain a mapping")
    descriptor = raw.get("descriptor")
    if not isinstance(descriptor, dict):
        raise ValueError(
            f"contract {contract_path} must declare a descriptor mapping with "
            "qdrant_host / qdrant_port"
        )
    return descriptor


def contract_qdrant_host(contract_path: Path = _CONTRACT) -> str:
    """Return the resolved ``descriptor.qdrant_host`` for the retrieval node.

    Contract-declared (overridable by an operator overlay contract) via the
    ``${env.QDRANT_HOST}`` convention — never hardcoded in source. Fails closed:
    raises ``ValueError`` when the field is absent or resolves to an empty string,
    so callers never silently fall back to ``localhost`` when ``QDRANT_HOST`` is
    unset.
    """
    descriptor = _load_descriptor(contract_path)
    declared = descriptor.get("qdrant_host")
    if not isinstance(declared, str):
        raise ValueError(
            f"contract {contract_path} must declare a string "
            "descriptor.qdrant_host (the ${env.QDRANT_HOST} overlay value the "
            "retrieval handlers use as the Qdrant host)"
        )
    resolved: str = expand_contract_env_refs(declared).strip()
    if not resolved:
        raise ValueError(
            "descriptor.qdrant_host resolved empty — set QDRANT_HOST (the Qdrant "
            "host the memory-retrieval effect connects to). The effect fails "
            "closed rather than silently default to localhost."
        )
    return resolved


def contract_qdrant_port(contract_path: Path = _CONTRACT) -> int:
    """Return the resolved ``descriptor.qdrant_port`` for the retrieval node.

    Contract-declared via the ``${env.QDRANT_PORT:6333}`` convention — the inline
    ``6333`` default (the Qdrant HTTP default port) preserves prior behaviour.
    Raises ``ValueError`` when the resolved value is not a valid integer.
    """
    descriptor = _load_descriptor(contract_path)
    declared = descriptor.get("qdrant_port")
    if not isinstance(declared, str):
        raise ValueError(
            f"contract {contract_path} must declare a string "
            "descriptor.qdrant_port (the ${env.QDRANT_PORT:6333} overlay value "
            "the retrieval handlers use as the Qdrant port)"
        )
    resolved: str = expand_contract_env_refs(declared).strip()
    try:
        return int(resolved)
    except ValueError as exc:
        raise ValueError(
            f"descriptor.qdrant_port resolved to {resolved!r}, which is not a "
            "valid integer — set QDRANT_PORT to the Qdrant HTTP port."
        ) from exc


__all__: list[str] = ["contract_qdrant_host", "contract_qdrant_port"]
