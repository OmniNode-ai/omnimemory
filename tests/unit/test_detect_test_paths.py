# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for change-aware test path detection (OMN-10762)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.ci.detect_test_paths import (
    _resolve,
    _split_count_for,
    compute_selection,
)
from scripts.ci.test_selection_loader import ModelAdjacencyMap, load_adjacency_map
from scripts.ci.test_selection_models import EnumFullSuiteReason, ModelTestSelection

ADJACENCY_PATH = (
    Path(__file__).parent.parent.parent
    / "scripts"
    / "ci"
    / "test_selection_adjacency.yaml"
)


@pytest.fixture
def adjacency_map() -> ModelAdjacencyMap:
    return load_adjacency_map(ADJACENCY_PATH)


@pytest.mark.unit
class TestAdjacencyMapLoads:
    def test_loads_without_error(self, adjacency_map: ModelAdjacencyMap) -> None:
        assert adjacency_map.schema_version == 1

    def test_all_shared_modules_in_adjacency(
        self, adjacency_map: ModelAdjacencyMap
    ) -> None:
        for module in adjacency_map.shared_modules:
            assert module in adjacency_map.adjacency, (
                f"shared_module '{module}' missing from adjacency"
            )

    def test_all_reverse_deps_valid(self, adjacency_map: ModelAdjacencyMap) -> None:
        for module, entry in adjacency_map.adjacency.items():
            for dep in entry.reverse_deps:
                assert dep in adjacency_map.adjacency, (
                    f"adjacency['{module}'].reverse_deps references unknown module '{dep}'"
                )


@pytest.mark.unit
class TestResolveTestPaths:
    def test_src_change_maps_to_unit_dir(
        self, adjacency_map: ModelAdjacencyMap
    ) -> None:
        result = _resolve(["src/omnimemory/tools/some_tool.py"], adjacency_map)
        assert "tests/unit/tools/" in result

    def test_src_change_expands_reverse_deps(
        self, adjacency_map: ModelAdjacencyMap
    ) -> None:
        # tools -> reverse_deps: [handlers, nodes]
        result = _resolve(["src/omnimemory/tools/some_tool.py"], adjacency_map)
        assert "tests/unit/handlers/" in result
        assert "tests/unit/nodes/" in result

    def test_unit_test_change_included_directly(
        self, adjacency_map: ModelAdjacencyMap
    ) -> None:
        result = _resolve(["tests/unit/handlers/test_foo.py"], adjacency_map)
        assert "tests/unit/handlers/" in result

    def test_integration_change_ignored(self, adjacency_map: ModelAdjacencyMap) -> None:
        result = _resolve(["tests/integration/test_ingestion.py"], adjacency_map)
        assert result == []

    def test_doc_only_change_ignored(self, adjacency_map: ModelAdjacencyMap) -> None:
        result = _resolve(["docs/README.md"], adjacency_map)
        assert result == []

    def test_unknown_src_module_ignored(self, adjacency_map: ModelAdjacencyMap) -> None:
        result = _resolve(["src/omnimemory/nonexistent_module/foo.py"], adjacency_map)
        assert result == []

    def test_result_is_sorted(self, adjacency_map: ModelAdjacencyMap) -> None:
        result = _resolve(
            ["src/omnimemory/utils/util_foo.py", "src/omnimemory/audit/audit_bar.py"],
            adjacency_map,
        )
        assert result == sorted(result)


@pytest.mark.unit
class TestComputeSelection:
    def test_feature_flag_off_returns_full_suite(self) -> None:
        sel = compute_selection(
            changed_files=["src/omnimemory/tools/foo.py"],
            adjacency_path=ADJACENCY_PATH,
            ref_name="jonah/omn-12345-feature",
            feature_flag_enabled=False,
        )
        assert sel.is_full_suite
        assert sel.full_suite_reason == EnumFullSuiteReason.FEATURE_FLAG_OFF

    def test_main_branch_returns_full_suite(self) -> None:
        sel = compute_selection(
            changed_files=["src/omnimemory/tools/foo.py"],
            adjacency_path=ADJACENCY_PATH,
            ref_name="main",
        )
        assert sel.is_full_suite
        assert sel.full_suite_reason == EnumFullSuiteReason.MAIN_BRANCH

    def test_merge_group_returns_full_suite(self) -> None:
        sel = compute_selection(
            changed_files=["src/omnimemory/tools/foo.py"],
            adjacency_path=ADJACENCY_PATH,
            ref_name="jonah/omn-10762-feature",
            event_name="merge_group",
        )
        assert sel.is_full_suite
        assert sel.full_suite_reason == EnumFullSuiteReason.MERGE_GROUP

    def test_scheduled_returns_full_suite(self) -> None:
        sel = compute_selection(
            changed_files=[],
            adjacency_path=ADJACENCY_PATH,
            ref_name="jonah/omn-10762-feature",
            event_name="schedule",
        )
        assert sel.is_full_suite
        assert sel.full_suite_reason == EnumFullSuiteReason.SCHEDULED

    def test_test_infra_change_returns_full_suite(self) -> None:
        sel = compute_selection(
            changed_files=["pyproject.toml"],
            adjacency_path=ADJACENCY_PATH,
            ref_name="jonah/omn-10762-feature",
        )
        assert sel.is_full_suite
        assert sel.full_suite_reason == EnumFullSuiteReason.TEST_INFRASTRUCTURE

    def test_shared_module_change_returns_full_suite(self) -> None:
        sel = compute_selection(
            changed_files=["src/omnimemory/models/memory/model_foo.py"],
            adjacency_path=ADJACENCY_PATH,
            ref_name="jonah/omn-10762-feature",
        )
        assert sel.is_full_suite
        assert sel.full_suite_reason == EnumFullSuiteReason.SHARED_MODULE

    def test_leaf_module_change_returns_smart_selection(self) -> None:
        sel = compute_selection(
            changed_files=["src/omnimemory/handlers/handler_foo.py"],
            adjacency_path=ADJACENCY_PATH,
            ref_name="jonah/omn-10762-feature",
        )
        assert not sel.is_full_suite
        assert sel.full_suite_reason is None
        assert "tests/unit/handlers/" in sel.selected_paths

    def test_deleted_util_drops_missing_dirs_keeps_existing(self) -> None:
        # Deleting src/omnimemory/utils/audit_logger.py resolves to
        # {utils, handlers, nodes, tools}. Only handlers/ and nodes/ have a
        # tests/unit/<module>/ dir; tools/ and utils/ must be dropped so pytest
        # does not abort collection on a missing path (exit 5). OMN-11576.
        sel = compute_selection(
            changed_files=["src/omnimemory/utils/audit_logger.py"],
            adjacency_path=ADJACENCY_PATH,
            ref_name="jonah/omn-11576-delete-dead-audit-logger",
        )
        assert not sel.is_full_suite
        assert "tests/unit/handlers/" in sel.selected_paths
        assert "tests/unit/nodes/" in sel.selected_paths
        assert "tests/unit/tools/" not in sel.selected_paths
        assert "tests/unit/utils/" not in sel.selected_paths

    def test_selected_paths_all_exist_on_disk(self) -> None:
        # Mixed resolution: handlers/nodes exist, tools/utils do not. Only the
        # existing directories survive filtering.
        sel = compute_selection(
            changed_files=["src/omnimemory/handlers/handler_foo.py"],
            adjacency_path=ADJACENCY_PATH,
            ref_name="jonah/omn-11576-feature",
        )
        repo_root = ADJACENCY_PATH.parent.parent.parent
        for path in sel.selected_paths:
            assert (repo_root / path).is_dir(), f"selected non-existent path: {path}"

    def test_doc_only_returns_fallback_unit_dir(self) -> None:
        sel = compute_selection(
            changed_files=["docs/README.md"],
            adjacency_path=ADJACENCY_PATH,
            ref_name="jonah/omn-10762-feature",
        )
        assert not sel.is_full_suite
        assert sel.selected_paths == ["tests/unit/"]

    def test_matrix_length_equals_split_count(self) -> None:
        sel = compute_selection(
            changed_files=["src/omnimemory/audit/audit_io.py"],
            adjacency_path=ADJACENCY_PATH,
            ref_name="jonah/omn-10762-feature",
        )
        assert len(sel.matrix) == sel.split_count

    def test_matrix_is_one_indexed(self) -> None:
        sel = compute_selection(
            changed_files=["src/omnimemory/audit/audit_io.py"],
            adjacency_path=ADJACENCY_PATH,
            ref_name="jonah/omn-10762-feature",
        )
        assert sel.matrix[0] == 1


@pytest.mark.unit
class TestSplitCountFor:
    def test_one_path_gives_one_split(self) -> None:
        assert _split_count_for(["tests/unit/audit/"]) == 1

    def test_two_paths_give_one_split(self) -> None:
        assert _split_count_for(["tests/unit/audit/", "tests/unit/utils/"]) == 1

    def test_three_paths_give_two_splits(self) -> None:
        assert (
            _split_count_for(["tests/unit/a/", "tests/unit/b/", "tests/unit/c/"]) == 2
        )

    def test_many_paths_capped_at_four(self) -> None:
        paths = [f"tests/unit/mod{i}/" for i in range(20)]
        assert _split_count_for(paths) == 4


@pytest.mark.unit
class TestModelTestSelection:
    def test_full_suite_requires_reason(self) -> None:
        with pytest.raises(Exception):
            ModelTestSelection(
                selected_paths=["tests/"],
                split_count=10,
                is_full_suite=True,
                full_suite_reason=None,
                matrix=list(range(1, 11)),
            )

    def test_non_full_suite_forbids_reason(self) -> None:
        with pytest.raises(Exception):
            ModelTestSelection(
                selected_paths=["tests/unit/audit/"],
                split_count=1,
                is_full_suite=False,
                full_suite_reason=EnumFullSuiteReason.MAIN_BRANCH,
                matrix=[1],
            )

    def test_matrix_must_match_split_count(self) -> None:
        with pytest.raises(Exception):
            ModelTestSelection(
                selected_paths=["tests/unit/audit/"],
                split_count=2,
                is_full_suite=False,
                full_suite_reason=None,
                matrix=[1],  # wrong length
            )

    def test_valid_smart_selection_serializes(self) -> None:
        sel = ModelTestSelection(
            selected_paths=["tests/unit/audit/"],
            split_count=1,
            is_full_suite=False,
            full_suite_reason=None,
            matrix=[1],
        )
        data = json.loads(sel.model_dump_json())
        assert data["split_count"] == 1
        assert data["is_full_suite"] is False
