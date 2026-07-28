# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Load and validate the static module adjacency map."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ModelAdjacencyEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    reverse_deps: list[str] = Field(default_factory=list)


class ModelTestFamily(BaseModel):
    """Declaration for a test location that lives outside ``tests/unit/``.

    omnimemory keeps structural test modules at ``tests/`` root and per-area
    packages at ``tests/nodes/``, ``tests/handlers/`` etc. (deliberate layout,
    documented in ``tests/conftest.py``). The ``tests/unit/<module>/`` mapping
    cannot reach them, so each such location declares the source prefixes that
    must pull it in. Undeclared locations escalate to the full suite.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    triggers: list[str] = Field(
        default_factory=list,
        description=(
            "Repo-relative path prefixes whose change selects this test family. "
            "A directory prefix ends with '/'; a file prefix is an exact path."
        ),
    )
    reason: str | None = Field(
        default=None,
        description=(
            "Required when triggers is empty: why no source change selects this "
            "family. Keeps 'runs nothing' an explicit, reviewed decision."
        ),
    )

    @model_validator(mode="after")
    def validate_empty_triggers_are_justified(self) -> ModelTestFamily:
        if not self.triggers and not self.reason:
            raise ValueError(
                "a test family with no triggers must state a reason; silent "
                "never-selected families are the OMN-15271 defect"
            )
        return self


class ModelThresholds(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    modules_changed_for_full_suite: int = Field(..., ge=1)


class ModelAdjacencyMap(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = Field(..., ge=1)
    shared_modules: list[str]
    thresholds: ModelThresholds
    test_infrastructure_paths: list[str]
    adjacency: dict[str, ModelAdjacencyEntry]
    test_families: dict[str, ModelTestFamily] = Field(
        ...,
        description=(
            "Test locations outside tests/unit/, keyed by repo-relative path "
            "('tests/<file>.py' or 'tests/<dir>/'), mapped to the source "
            "prefixes that select them (OMN-15271)."
        ),
    )

    @model_validator(mode="after")
    def validate_test_family_keys(self) -> ModelAdjacencyMap:
        for family in self.test_families:
            if not family.startswith("tests/"):
                raise ValueError(
                    f"test_families key must start with 'tests/': {family}"
                )
            module = family.removeprefix("tests/unit/").rstrip("/")
            if (
                family.startswith("tests/unit/")
                and family.endswith("/")
                and module in self.adjacency
            ):
                raise ValueError(
                    f"tests/unit/{module}/ is already resolved from the adjacency "
                    f"map; declaring it in test_families is misleading: {family}"
                )
            if not family.endswith(("/", ".py")):
                raise ValueError(
                    f"test_families key must be a directory ('.../') or a .py file: {family}"
                )
        return self

    @model_validator(mode="after")
    def validate_shared_modules_in_adjacency(self) -> ModelAdjacencyMap:
        for shared in self.shared_modules:
            if shared not in self.adjacency:
                raise ValueError(f"shared_module '{shared}' has no adjacency entry")
        for module, entry in self.adjacency.items():
            for dep in entry.reverse_deps:
                if dep not in self.adjacency:
                    raise ValueError(
                        f"adjacency['{module}'].reverse_deps references unknown module '{dep}'"
                    )
        return self


def load_adjacency_map(path: Path) -> ModelAdjacencyMap:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return ModelAdjacencyMap.model_validate(raw)
