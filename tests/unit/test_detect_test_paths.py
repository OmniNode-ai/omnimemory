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

    def test_integration_change_selects_itself(
        self, adjacency_map: ModelAdjacencyMap
    ) -> None:
        # OMN-15271: previously asserted `== []`. A changed test file that
        # contributes nothing is the fail-open this ticket closes; integration
        # tests skip cleanly without live backends, so selecting one is safe.
        result = _resolve(["tests/integration/test_ingestion.py"], adjacency_map)
        assert result == ["tests/integration/test_ingestion.py"]

    def test_doc_only_change_ignored(self, adjacency_map: ModelAdjacencyMap) -> None:
        result = _resolve(["docs/README.md"], adjacency_map)
        assert result == []

    def test_unknown_src_module_selects_only_whole_src_gates(
        self, adjacency_map: ModelAdjacencyMap
    ) -> None:
        # OMN-15639: previously asserted `== []`. An unknown src module still
        # pulls in NO adjacency-resolved tests/unit/<module>/ dir -- that is the
        # original intent and is unchanged. It now additionally selects
        # tests/gates/, whose trigger is the whole src tree because the
        # consumer-group authorization gate AST-walks all of src default-deny.
        # A new module is exactly where an unauthorized group literal would
        # hide, so firing the gate there is the fail-closed behaviour.
        result = _resolve(["src/omnimemory/nonexistent_module/foo.py"], adjacency_map)
        assert result == ["tests/gates/"]

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
        # existing paths survive filtering. Since OMN-15271 a selection may also
        # name a single test file, so directories and files are checked apart.
        sel = compute_selection(
            changed_files=["src/omnimemory/handlers/handler_foo.py"],
            adjacency_path=ADJACENCY_PATH,
            ref_name="jonah/omn-11576-feature",
        )
        repo_root = ADJACENCY_PATH.parent.parent.parent
        for path in sel.selected_paths:
            target = repo_root / path
            exists = target.is_dir() if path.endswith("/") else target.is_file()
            assert exists, f"selected non-existent path: {path}"

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


# ---------------------------------------------------------------------------
# OMN-15271: test locations outside tests/unit/ must be selectable.
#
# The pre-fix selector inspected exactly two prefixes (src/omnimemory/ and
# tests/unit/) and contributed NOTHING for anything else, so the ~25 files at
# tests/ root -- documented as a deliberate layout in tests/conftest.py -- were
# unselectable on the PR-into-dev path. A PR that added or edited one of them
# went green without ever collecting it, and a src/omnimemory/nodes/** change
# never pulled in the root-level structural gates that read those nodes.
# ---------------------------------------------------------------------------

REPO_ROOT = ADJACENCY_PATH.parent.parent.parent

# Recorded change set of omnimemory#417 (OMN-15235), CI run 30310898812.
OMN_15235_CHANGED_FILES = [
    "src/omnimemory/nodes/node_memory_retrieval_effect/contract.yaml",
    "src/omnimemory/nodes/node_memory_retrieval_effect/handlers/handler_qdrant.py",
    "tests/unit/nodes/test_handler_routing_boot_resolvable.py",
]

# Root-level structural gates that read src/omnimemory/nodes/** directly.
NODE_STRUCTURAL_GATES = (
    "tests/test_contract_validation.py",
    "tests/test_contract_version.py",
    "tests/test_node_enforcement.py",
    "tests/test_node_imports.py",
)


def _covers(selected_paths: list[str], test_file: str) -> bool:
    """True when pytest would collect `test_file` from this selection."""
    return any(
        test_file == path or (path.endswith("/") and test_file.startswith(path))
        for path in selected_paths
    )


@pytest.mark.unit
class TestChangedTestFileAlwaysSelected:
    """Acceptance 1: a changed tests/**/*.py always contributes itself. Never zero."""

    def test_root_level_test_change_selects_that_file(self) -> None:
        sel = compute_selection(
            changed_files=["tests/test_contract_validation.py"],
            adjacency_path=ADJACENCY_PATH,
            ref_name="jonah/omn-15271-selector",
        )
        assert not sel.is_full_suite
        assert "tests/test_contract_validation.py" in sel.selected_paths

    def test_root_level_test_change_survives_alongside_src_change(self) -> None:
        # The dangerous shape: a src change produces a non-empty selection, so
        # the tests/unit/ fallback never fires and the changed root-level test
        # is silently dropped.
        sel = compute_selection(
            changed_files=[
                "src/omnimemory/handlers/handler_foo.py",
                "tests/test_concurrency.py",
            ],
            adjacency_path=ADJACENCY_PATH,
            ref_name="jonah/omn-15271-selector",
        )
        assert not sel.is_full_suite
        assert "tests/unit/handlers/" in sel.selected_paths
        assert "tests/test_concurrency.py" in sel.selected_paths

    def test_non_unit_test_package_change_selects_that_file(self) -> None:
        # tests/nodes/ and tests/handlers/ are real test packages outside
        # tests/unit/ and were equally unselectable. Changed alone, nothing
        # triggers the package, so the file must carry itself.
        sel = compute_selection(
            changed_files=["tests/handlers/test_handler_intent.py"],
            adjacency_path=ADJACENCY_PATH,
            ref_name="jonah/omn-15271-selector",
        )
        assert not sel.is_full_suite
        assert "tests/handlers/test_handler_intent.py" in sel.selected_paths

    def test_non_unit_test_package_change_covered_alongside_src_change(self) -> None:
        # With the package's own trigger firing, the covering directory is the
        # selection; the file must not be dropped either way.
        sel = compute_selection(
            changed_files=[
                "src/omnimemory/handlers/handler_foo.py",
                "tests/handlers/test_handler_intent.py",
            ],
            adjacency_path=ADJACENCY_PATH,
            ref_name="jonah/omn-15271-selector",
        )
        assert _covers(sel.selected_paths, "tests/handlers/test_handler_intent.py")

    def test_integration_test_change_selects_that_file(self) -> None:
        # Integration tests skip cleanly without live backends, so selecting a
        # changed one is safe; leaving it unselected is the fail-open.
        sel = compute_selection(
            changed_files=[
                "src/omnimemory/handlers/handler_foo.py",
                "tests/integration/test_handler_subscription.py",
            ],
            adjacency_path=ADJACENCY_PATH,
            ref_name="jonah/omn-15271-selector",
        )
        assert "tests/integration/test_handler_subscription.py" in sel.selected_paths

    def test_file_directly_under_tests_unit_selects_itself(self) -> None:
        # tests/unit/test_x.py has no <module> component: the old
        # parts[2] mapping produced the non-existent directory
        # "tests/unit/test_x.py/", which the on-disk filter then dropped.
        sel = compute_selection(
            changed_files=[
                "src/omnimemory/handlers/handler_foo.py",
                "tests/unit/test_entry_points.py",
            ],
            adjacency_path=ADJACENCY_PATH,
            ref_name="jonah/omn-15271-selector",
        )
        assert "tests/unit/test_entry_points.py" in sel.selected_paths
        assert "tests/unit/test_entry_points.py/" not in sel.selected_paths

    def test_deleted_test_file_is_not_passed_to_pytest(self) -> None:
        # A deleted path handed to pytest aborts the run (exit 4), so a changed
        # test file that no longer exists must not be selected. Same rationale
        # as the OMN-11576 missing-directory filter.
        sel = compute_selection(
            changed_files=["tests/test_deleted_by_this_pr.py"],
            adjacency_path=ADJACENCY_PATH,
            ref_name="jonah/omn-15271-selector",
        )
        assert "tests/test_deleted_by_this_pr.py" not in sel.selected_paths
        for path in sel.selected_paths:
            assert (REPO_ROOT / path).exists(), f"selected non-existent path: {path}"


@pytest.mark.unit
class TestOMN15235Replay:
    """Recorded live instance: omnimemory#417 / CI run 30310898812."""

    def test_recorded_change_set_selects_root_level_node_gates(self) -> None:
        sel = compute_selection(
            changed_files=OMN_15235_CHANGED_FILES,
            adjacency_path=ADJACENCY_PATH,
            ref_name="jonah/omn-15235-boot-resolvable",
        )
        assert not sel.is_full_suite
        assert "tests/unit/nodes/" in sel.selected_paths
        for gate in NODE_STRUCTURAL_GATES:
            assert gate in sel.selected_paths, (
                f"{gate} reads src/omnimemory/nodes/** but was not selected for "
                "a contract.yaml change"
            )

    def test_contract_yaml_alone_selects_root_level_node_gates(self) -> None:
        # The ticket's second recorded row: a future bad handler_routing entry
        # touches only the contract, and resolved to ['tests/unit/nodes/'].
        sel = compute_selection(
            changed_files=[
                "src/omnimemory/nodes/node_memory_retrieval_effect/contract.yaml"
            ],
            adjacency_path=ADJACENCY_PATH,
            ref_name="jonah/omn-15235-boot-resolvable",
        )
        assert not sel.is_full_suite
        for gate in NODE_STRUCTURAL_GATES:
            assert gate in sel.selected_paths


@pytest.mark.unit
class TestUndeclaredFamilyEscalates:
    """Fail-closed: an undeclared test family forces the full suite."""

    def test_undeclared_root_level_test_escalates(self, tmp_path: Path) -> None:
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "test_brand_new_gate.py").write_text("")
        sel = compute_selection(
            changed_files=["src/omnimemory/handlers/handler_foo.py"],
            adjacency_path=ADJACENCY_PATH,
            ref_name="jonah/omn-15271-selector",
            repo_root=tmp_path,
        )
        assert sel.is_full_suite
        assert sel.full_suite_reason == EnumFullSuiteReason.UNDECLARED_TEST_FAMILY

    def test_undeclared_test_package_escalates(self, tmp_path: Path) -> None:
        (tmp_path / "tests" / "brand_new_pkg").mkdir(parents=True)
        (tmp_path / "tests" / "brand_new_pkg" / "test_x.py").write_text("")
        sel = compute_selection(
            changed_files=["src/omnimemory/handlers/handler_foo.py"],
            adjacency_path=ADJACENCY_PATH,
            ref_name="jonah/omn-15271-selector",
            repo_root=tmp_path,
        )
        assert sel.is_full_suite
        assert sel.full_suite_reason == EnumFullSuiteReason.UNDECLARED_TEST_FAMILY


@pytest.mark.unit
class TestTestFamilyDeclarationsMatchDisk:
    def test_every_family_on_disk_is_declared(
        self, adjacency_map: ModelAdjacencyMap
    ) -> None:
        from scripts.ci.detect_test_paths import discover_test_families

        undeclared = discover_test_families(REPO_ROOT, adjacency_map) - set(
            adjacency_map.test_families
        )
        assert not undeclared, (
            f"undeclared test families (add them to test_selection_adjacency.yaml "
            f"with their triggers): {sorted(undeclared)}"
        )

    def test_every_declared_family_exists_on_disk(
        self, adjacency_map: ModelAdjacencyMap
    ) -> None:
        stale = {
            family
            for family in adjacency_map.test_families
            if not (REPO_ROOT / family).exists()
        }
        assert not stale, f"declared test families missing from disk: {sorted(stale)}"

    def test_every_trigger_prefix_exists_on_disk(
        self, adjacency_map: ModelAdjacencyMap
    ) -> None:
        # A trigger that no longer resolves silently stops firing.
        stale = {
            (family, trigger)
            for family, entry in adjacency_map.test_families.items()
            for trigger in entry.triggers
            if not (REPO_ROOT / trigger).exists()
        }
        assert not stale, f"test_families triggers missing from disk: {sorted(stale)}"


@pytest.mark.unit
class TestNarrowingNoRegression:
    """Narrowing on pure-src diffs must survive the OMN-15271 change."""

    def test_leaf_src_change_still_narrows(self) -> None:
        sel = compute_selection(
            changed_files=["src/omnimemory/handlers/handler_foo.py"],
            adjacency_path=ADJACENCY_PATH,
            ref_name="jonah/omn-15271-selector",
        )
        assert not sel.is_full_suite
        assert "tests/unit/handlers/" in sel.selected_paths
        assert "tests/unit/nodes/" in sel.selected_paths
        assert "tests/" not in sel.selected_paths
        assert "tests/unit/" not in sel.selected_paths

    def test_leaf_src_change_pulls_no_unrelated_root_files(self) -> None:
        sel = compute_selection(
            changed_files=["src/omnimemory/audit/audit_io.py"],
            adjacency_path=ADJACENCY_PATH,
            ref_name="jonah/omn-15271-selector",
        )
        assert not sel.is_full_suite
        assert not [p for p in sel.selected_paths if p.startswith("tests/test_")]

    def test_selector_change_still_runs_the_selector_tests(self) -> None:
        # Adding family triggers means fewer changes fall through to the
        # tests/unit/ fallback, so anything the fallback used to cover must now
        # be reachable by trigger. A scripts/ci/ edit must still run this file.
        sel = compute_selection(
            changed_files=["scripts/ci/detect_test_paths.py"],
            adjacency_path=ADJACENCY_PATH,
            ref_name="jonah/omn-15271-selector",
        )
        assert not sel.is_full_suite
        assert _covers(sel.selected_paths, "tests/unit/test_detect_test_paths.py")
        assert _covers(sel.selected_paths, "tests/scripts/test_check_dep_provenance.py")

    def test_contract_only_change_still_runs_node_uuid_gate(self) -> None:
        # tests/unit/test_no_duplicate_node_uuids.py sits directly under
        # tests/unit/, so tests/unit/nodes/ never covered it.
        sel = compute_selection(
            changed_files=[
                "src/omnimemory/nodes/node_memory_retrieval_effect/contract.yaml"
            ],
            adjacency_path=ADJACENCY_PATH,
            ref_name="jonah/omn-15271-selector",
        )
        assert _covers(sel.selected_paths, "tests/unit/test_no_duplicate_node_uuids.py")

    def test_main_branch_still_full_suite_with_root_changes(self) -> None:
        sel = compute_selection(
            changed_files=["tests/test_contract_validation.py"],
            adjacency_path=ADJACENCY_PATH,
            ref_name="main",
        )
        assert sel.is_full_suite
        assert sel.full_suite_reason == EnumFullSuiteReason.MAIN_BRANCH
