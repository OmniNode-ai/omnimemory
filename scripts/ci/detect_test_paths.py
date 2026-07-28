# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Change-aware test path resolution for omnimemory CI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.ci.test_selection_loader import (
    ModelAdjacencyMap,
    load_adjacency_map,
)
from scripts.ci.test_selection_models import (
    EnumFullSuiteReason,
    ModelTestSelection,
)

SRC_PREFIX = "src/omnimemory/"
TEST_PREFIX = "tests/"
TEST_UNIT_PREFIX = "tests/unit/"

FULL_SUITE_BRANCHES = {"main"}

# tests/unit/ is not itself a family: its <module>/ subdirectories are resolved
# from the adjacency map, and its remaining contents are enumerated separately.
NON_FAMILY_TEST_DIRS = {"unit"}


def resolve_test_paths(
    changed_files: list[str],
    adjacency_path: Path,
) -> list[str]:
    """Map changed file paths to deterministic test paths.

    Behavior:
      - Source changes under src/omnimemory/<module>: include
        tests/unit/<module>/ plus that module's reverse dependents.
      - Source changes matching a declared `test_families` trigger: include that
        family (a tests/ root module or a tests/<area>/ package). OMN-15271.
      - Test changes under tests/unit/<module>/: include that directory.
      - A changed test file that nothing above covers contributes ITSELF, so a
        changed test can never be selected away (OMN-15271). Non-collectible
        test support files (conftest.py, helpers) contribute their directory.
      - Files outside src/ and tests/: no contribution; caller decides whether
        to escalate to full suite.

    Not covered: non-.py fixtures under tests/. Those still fall through to the
    caller's tests/unit/ fallback.
    """
    config = load_adjacency_map(adjacency_path)
    return _resolve(changed_files, config)


def _matches_prefix(path: str, prefix: str) -> bool:
    """True when `path` is `prefix` itself or lives under it."""
    return path == prefix or path.startswith(prefix.rstrip("/") + "/")


def discover_test_families(repo_root: Path, config: ModelAdjacencyMap) -> set[str]:
    """Every test location on disk that `test_families` must declare.

    Only ``tests/unit/<module>/`` directories whose ``<module>`` is an adjacency
    key are auto-covered — the adjacency map resolves those from source changes.
    Everything else (tests/ root modules, tests/<area>/ packages, files directly
    under tests/unit/, and tests/unit/<dir>/ with no matching source module) has
    no path into the selection until it is declared.
    """
    tests_dir = repo_root / TEST_PREFIX
    if not tests_dir.is_dir():
        return set()

    families = {f"{TEST_PREFIX}{path.name}" for path in tests_dir.glob("test_*.py")}
    families |= {
        f"{TEST_PREFIX}{path.name}/"
        for path in _child_dirs(tests_dir)
        if path.name not in NON_FAMILY_TEST_DIRS
    }

    unit_dir = tests_dir / "unit"
    if unit_dir.is_dir():
        families |= {
            f"{TEST_UNIT_PREFIX}{path.name}" for path in unit_dir.glob("test_*.py")
        }
        families |= {
            f"{TEST_UNIT_PREFIX}{path.name}/"
            for path in _child_dirs(unit_dir)
            if path.name not in config.adjacency
        }
    return families


def _child_dirs(parent: Path) -> list[Path]:
    """Real (non-hidden, non-cache) child directories of `parent`."""
    return [
        path
        for path in parent.iterdir()
        if path.is_dir()
        and not path.name.startswith(".")
        and path.name != "__pycache__"
    ]


def _resolve(changed_files: list[str], config: ModelAdjacencyMap) -> list[str]:
    direct_modules: set[str] = set()
    selected: set[str] = set()

    for path in changed_files:
        if path.startswith(SRC_PREFIX):
            module = path[len(SRC_PREFIX) :].split("/", 1)[0]
            if module in config.adjacency:
                direct_modules.add(module)
        elif path.startswith(TEST_UNIT_PREFIX):
            parts = path.split("/")
            # tests/unit/<module>/<file>: only a real subdirectory maps to a
            # directory. tests/unit/<file>.py has no <module> component and is
            # picked up by the self-selection pass below.
            if len(parts) >= 4:
                selected.add(f"{TEST_UNIT_PREFIX}{parts[2]}/")

    expanded: set[str] = set(direct_modules)
    for module in direct_modules:
        expanded.update(config.adjacency[module].reverse_deps)

    for module in expanded:
        selected.add(f"{TEST_UNIT_PREFIX}{module}/")

    # Declared test families outside tests/unit/ fire on their source triggers.
    for family, entry in config.test_families.items():
        if any(
            _matches_prefix(changed, trigger)
            for changed in changed_files
            for trigger in entry.triggers
        ):
            selected.add(family)

    # A changed test file always contributes itself unless an already-selected
    # directory covers it. This is the fail-closed floor: never zero for a
    # changed test (OMN-15271).
    for path in changed_files:
        if not path.startswith(TEST_PREFIX) or not path.endswith(".py"):
            continue
        name = path.rsplit("/", 1)[-1]
        # pytest collects `test_*.py` only; anything else (conftest.py, shared
        # helpers) is selected via the directory whose tests it feeds.
        candidate = path if name.startswith("test_") else f"{path.rsplit('/', 1)[0]}/"
        if any(
            candidate.startswith(chosen) for chosen in selected if chosen.endswith("/")
        ):
            continue
        selected.add(candidate)

    return sorted(selected)


def compute_selection(
    changed_files: list[str],
    adjacency_path: Path,
    ref_name: str,
    event_name: str = "pull_request",
    feature_flag_enabled: bool = True,
    repo_root: Path = REPO_ROOT,
) -> ModelTestSelection:
    config = load_adjacency_map(adjacency_path)

    # 0. Feature flag short-circuit: off → legacy full suite.
    if not feature_flag_enabled:
        return _full_suite(EnumFullSuiteReason.FEATURE_FLAG_OFF)

    # 1. Branch / event escalation.
    if ref_name in FULL_SUITE_BRANCHES:
        return _full_suite(EnumFullSuiteReason.MAIN_BRANCH)
    if event_name == "merge_group":
        return _full_suite(EnumFullSuiteReason.MERGE_GROUP)
    if event_name == "schedule":
        return _full_suite(EnumFullSuiteReason.SCHEDULED)

    # 2. Test infrastructure escalation.
    for changed in changed_files:
        if any(
            changed == infra or changed.startswith(infra.rstrip("/") + "/")
            for infra in config.test_infrastructure_paths
        ):
            return _full_suite(EnumFullSuiteReason.TEST_INFRASTRUCTURE)

    # 3. Shared module escalation.
    changed_modules = {
        path[len(SRC_PREFIX) :].split("/", 1)[0]
        for path in changed_files
        if path.startswith(SRC_PREFIX)
    } & set(config.adjacency.keys())
    if changed_modules & set(config.shared_modules):
        return _full_suite(EnumFullSuiteReason.SHARED_MODULE)

    # 4. Threshold escalation: too many distinct modules.
    if len(changed_modules) >= config.thresholds.modules_changed_for_full_suite:
        return _full_suite(EnumFullSuiteReason.THRESHOLD_MODULES)

    # 4b. Undeclared test family escalation (OMN-15271). A test location that
    # exists on disk but is not declared in test_families has no path into the
    # selection, so narrowing cannot be proven safe: escalate.
    if discover_test_families(repo_root, config) - set(config.test_families):
        return _full_suite(EnumFullSuiteReason.UNDECLARED_TEST_FAMILY)

    # 5. Smart selection.
    selected = _resolve(changed_files, config)
    # Drop paths absent from disk. A module may have reverse_deps or be changed
    # without owning a tests/unit/<module>/ dir, and a changed test file may be
    # one this PR deletes; passing a missing path to pytest aborts the run
    # before any real test executes (exit code 5 / 4).
    selected = [p for p in selected if _exists(repo_root, p)]
    if not selected:
        # Conservative one-shard fallback over the full tests/unit/ tree.
        # Fires for doc-only or workflow-only changes, or when every resolved
        # module lacks a dedicated test directory. Changed test files never
        # reach here — they self-select above.
        selected = ["tests/unit/"]
    split_count = _split_count_for(selected)

    return ModelTestSelection(
        selected_paths=selected,
        split_count=split_count,
        is_full_suite=False,
        full_suite_reason=None,
        matrix=list(range(1, split_count + 1)),
    )


def _exists(repo_root: Path, selected_path: str) -> bool:
    """Directory selections must be directories; file selections must be files."""
    target = repo_root / selected_path
    return target.is_dir() if selected_path.endswith("/") else target.is_file()


def _full_suite(reason: EnumFullSuiteReason) -> ModelTestSelection:
    return ModelTestSelection(
        selected_paths=["tests/"],
        split_count=10,
        is_full_suite=True,
        full_suite_reason=reason,
        matrix=list(range(1, 11)),
    )


def _split_count_for(selected_paths: list[str]) -> int:
    """Map selected path count to split count (conservative starting heuristic)."""
    n = len(selected_paths)
    if n <= 2:
        return 1
    if n <= 5:
        return 2
    if n <= 10:
        return 3
    return 4


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Resolve change-aware test paths")
    parser.add_argument(
        "--changed-files-from",
        type=Path,
        required=True,
        help="Path to a file with one changed-file path per line.",
    )
    parser.add_argument("--ref-name", required=True)
    parser.add_argument("--event-name", default="pull_request")
    parser.add_argument(
        "--adjacency",
        type=Path,
        default=Path(__file__).parent / "test_selection_adjacency.yaml",
    )
    parser.add_argument(
        "--feature-flag",
        choices=("on", "off"),
        default="on",
        help="When 'off', emit a FEATURE_FLAG_OFF full-suite selection regardless of changed files.",
    )
    args = parser.parse_args(argv)

    changed = [
        line.strip()
        for line in args.changed_files_from.read_text().splitlines()
        if line.strip()
    ]
    selection = compute_selection(
        changed_files=changed,
        adjacency_path=args.adjacency,
        ref_name=args.ref_name,
        event_name=args.event_name,
        feature_flag_enabled=(args.feature_flag == "on"),
    )
    sys.stdout.write(selection.model_dump_json())
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
