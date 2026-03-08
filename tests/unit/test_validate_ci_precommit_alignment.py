# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Unit tests for CI/pre-commit alignment validation.

Tests validate that ``scripts/validate_ci_precommit_alignment.py`` correctly
detects drift between CI workflow jobs and pre-commit hooks, and that its
``EXPECTED_ALIGNMENTS`` list stays in sync with the actual configuration
files in the repository.
"""

from __future__ import annotations

import importlib.util
import sys
import textwrap
from pathlib import Path

import pytest
import yaml

# ---------------------------------------------------------------------------
# Import functions from the script under test.
# ``scripts/`` is not a Python package, so we load the module by file path.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "validate_ci_precommit_alignment.py"
_spec = importlib.util.spec_from_file_location(
    "validate_ci_precommit_alignment", _SCRIPT_PATH
)
assert _spec is not None
assert _spec.loader is not None
_module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _module
_spec.loader.exec_module(_module)

load_file = _module.load_file
extract_hook_ids = _module.extract_hook_ids
extract_ci_job_ids = _module.extract_ci_job_ids
check_ci_contains = _module.check_ci_contains
EXPECTED_ALIGNMENTS: list[tuple[str, str, str]] = _module.EXPECTED_ALIGNMENTS


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def precommit_content() -> str:
    """Load the actual .pre-commit-config.yaml content from the repo."""
    path = _REPO_ROOT / ".pre-commit-config.yaml"
    content = path.read_text(encoding="utf-8")
    return content


@pytest.fixture
def ci_content() -> str:
    """Load the actual .github/workflows/test.yml content from the repo."""
    path = _REPO_ROOT / ".github" / "workflows" / "test.yml"
    content = path.read_text(encoding="utf-8")
    return content


@pytest.fixture
def precommit_data() -> dict:
    """Parse the actual .pre-commit-config.yaml as a YAML dict."""
    path = _REPO_ROOT / ".pre-commit-config.yaml"
    content = path.read_text(encoding="utf-8")
    return yaml.safe_load(content)


@pytest.fixture
def ci_data() -> dict:
    """Parse the actual .github/workflows/test.yml as a YAML dict."""
    path = _REPO_ROOT / ".github" / "workflows" / "test.yml"
    content = path.read_text(encoding="utf-8")
    return yaml.safe_load(content)


# =============================================================================
# Tests: extract_hook_ids
# =============================================================================


@pytest.mark.unit
class TestExtractHookIds:
    """Tests for pre-commit hook ID extraction via regex."""

    def test_extracts_simple_hook_ids(self) -> None:
        content = textwrap.dedent("""\
            repos:
              - repo: https://example.com
                hooks:
                  - id: ruff-format
                  - id: ruff
              - repo: local
                hooks:
                  - id: mypy
        """)
        result = extract_hook_ids(content)
        assert result == {"ruff-format", "ruff", "mypy"}

    def test_ignores_non_id_lines(self) -> None:
        content = textwrap.dedent("""\
            repos:
              - repo: https://example.com
                hooks:
                  - id: ruff
                    name: ruff linter
                    args: [--check]
        """)
        result = extract_hook_ids(content)
        assert result == {"ruff"}

    def test_empty_content_returns_empty_set(self) -> None:
        result = extract_hook_ids("")
        assert result == set()

    def test_no_hooks_returns_empty_set(self) -> None:
        content = textwrap.dedent("""\
            repos:
              - repo: https://example.com
                rev: v1.0.0
        """)
        result = extract_hook_ids(content)
        assert result == set()


# =============================================================================
# Tests: extract_ci_job_ids
# =============================================================================


@pytest.mark.unit
class TestExtractCiJobIds:
    """Tests for CI workflow job ID extraction via YAML parsing."""

    def test_extracts_job_ids(self) -> None:
        content = textwrap.dedent("""\
            name: CI
            on: push
            jobs:
              lint:
                runs-on: ubuntu-latest
                steps: []
              test:
                runs-on: ubuntu-latest
                steps: []
        """)
        result = extract_ci_job_ids(content)
        assert result == {"lint", "test"}

    def test_empty_jobs_returns_empty_set(self) -> None:
        content = textwrap.dedent("""\
            name: CI
            on: push
            jobs: {}
        """)
        result = extract_ci_job_ids(content)
        assert result == set()

    def test_no_jobs_key_returns_empty_set(self) -> None:
        content = textwrap.dedent("""\
            name: CI
            on: push
        """)
        result = extract_ci_job_ids(content)
        assert result == set()

    def test_non_dict_content_returns_empty_set(self) -> None:
        result = extract_ci_job_ids("- just a list\n- of items\n")
        assert result == set()

    def test_jobs_not_dict_returns_empty_set(self) -> None:
        content = textwrap.dedent("""\
            name: CI
            on: push
            jobs:
              - not
              - a
              - dict
        """)
        result = extract_ci_job_ids(content)
        assert result == set()


# =============================================================================
# Tests: check_ci_contains
# =============================================================================


@pytest.mark.unit
class TestCheckCiContains:
    """Tests for the simple substring search."""

    def test_finds_exact_match(self) -> None:
        assert check_ci_contains("run: ruff format --check", "ruff format --check")

    def test_finds_substring(self) -> None:
        assert check_ci_contains("poetry run mypy src/omnimemory", "mypy")

    def test_returns_false_when_missing(self) -> None:
        assert not check_ci_contains("run: ruff check", "mypy")

    def test_empty_content(self) -> None:
        assert not check_ci_contains("", "something")

    def test_empty_search_term(self) -> None:
        # Empty string is always a substring
        assert check_ci_contains("anything", "")


# =============================================================================
# Tests: load_file
# =============================================================================


@pytest.mark.unit
class TestLoadFile:
    """Tests for file loading with error handling."""

    def test_loads_existing_file(self, tmp_path: Path) -> None:
        target = tmp_path / "test.txt"
        target.write_text("hello world", encoding="utf-8")
        result = load_file(target)
        assert result == "hello world"

    def test_returns_none_for_missing_file(self, tmp_path: Path) -> None:
        target = tmp_path / "nonexistent.txt"
        result = load_file(target)
        assert result is None

    def test_returns_none_for_directory(self, tmp_path: Path) -> None:
        result = load_file(tmp_path)
        assert result is None


# =============================================================================
# Tests: EXPECTED_ALIGNMENTS structural integrity
# =============================================================================


@pytest.mark.unit
class TestExpectedAlignmentsStructure:
    """Verify the EXPECTED_ALIGNMENTS list has valid structure."""

    def test_not_empty(self) -> None:
        assert len(EXPECTED_ALIGNMENTS) > 0

    def test_all_entries_are_three_tuples(self) -> None:
        for i, entry in enumerate(EXPECTED_ALIGNMENTS):
            assert isinstance(entry, tuple), f"Entry {i} is not a tuple: {entry!r}"
            assert len(entry) == 3, (
                f"Entry {i} has {len(entry)} elements, expected 3: {entry!r}"
            )

    def test_all_entries_have_nonempty_strings(self) -> None:
        for i, (hook_id, ci_job, ci_desc) in enumerate(EXPECTED_ALIGNMENTS):
            assert isinstance(hook_id, str), f"Entry {i}: hook_id is not a string"
            assert hook_id.strip(), f"Entry {i}: hook_id is empty"
            assert isinstance(ci_job, str), f"Entry {i}: ci_job is not a string"
            assert ci_job.strip(), f"Entry {i}: ci_job is empty"
            assert isinstance(ci_desc, str), (
                f"Entry {i}: ci_description is not a string"
            )
            assert ci_desc.strip(), f"Entry {i}: ci_description is empty"

    def test_no_duplicate_hook_ids(self) -> None:
        hook_ids = [entry[0] for entry in EXPECTED_ALIGNMENTS]
        duplicates = [h for h in hook_ids if hook_ids.count(h) > 1]
        assert not duplicates, (
            f"Duplicate hook IDs in EXPECTED_ALIGNMENTS: {set(duplicates)}"
        )


# =============================================================================
# Tests: EXPECTED_ALIGNMENTS vs actual pre-commit config
# =============================================================================


@pytest.mark.unit
class TestAlignmentsMatchPrecommit:
    """Every hook_id in EXPECTED_ALIGNMENTS must exist in .pre-commit-config.yaml."""

    def test_all_expected_hooks_exist_in_precommit(
        self, precommit_content: str
    ) -> None:
        actual_hook_ids = extract_hook_ids(precommit_content)
        missing: list[str] = []
        for hook_id, _ci_job, _ci_desc in EXPECTED_ALIGNMENTS:
            if hook_id not in actual_hook_ids:
                missing.append(hook_id)
        assert not missing, (
            f"EXPECTED_ALIGNMENTS references hooks not found in "
            f".pre-commit-config.yaml: {missing}"
        )


@pytest.mark.unit
class TestPrecommitHooksCoveredByAlignments:
    """Validation-related pre-commit hooks should be covered by EXPECTED_ALIGNMENTS.

    Not every hook needs a CI counterpart (e.g., clean-tmp-directory, yamlfmt,
    trailing-whitespace are formatting/cleanup hooks). But every hook that
    performs validation/linting should have a corresponding entry.
    """

    # Hooks that are intentionally NOT tracked in EXPECTED_ALIGNMENTS because
    # they are formatting/cleanup hooks without CI counterparts, or are
    # standard pre-commit-hooks that are not project-specific validation.
    EXCLUDED_HOOKS: frozenset[str] = frozenset(
        {
            "yamlfmt",
            "trailing-whitespace",
            "end-of-file-fixer",
            "check-merge-conflict",
            "check-added-large-files",
            "check-yaml",
            "check-toml",
            "debug-statements",
            "onex-validate-clean-root",
            "no-internal-ips",
            "no-planning-docs",
            "no-env-file",
            "validate-string-versions",
        }
    )

    def test_all_validation_hooks_tracked(self, precommit_content: str) -> None:
        actual_hook_ids = extract_hook_ids(precommit_content)
        tracked_hook_ids = {entry[0] for entry in EXPECTED_ALIGNMENTS}

        untracked = actual_hook_ids - tracked_hook_ids - self.EXCLUDED_HOOKS
        assert not untracked, (
            f"Pre-commit hooks exist that are NOT in EXPECTED_ALIGNMENTS and "
            f"NOT in EXCLUDED_HOOKS. Either add them to EXPECTED_ALIGNMENTS "
            f"(if they have CI counterparts) or to EXCLUDED_HOOKS in this test "
            f"(if they are formatting/cleanup only): {sorted(untracked)}"
        )


# =============================================================================
# Tests: EXPECTED_ALIGNMENTS vs actual CI workflow
# =============================================================================


@pytest.mark.unit
class TestAlignmentsMatchCiWorkflow:
    """Every CI job name in EXPECTED_ALIGNMENTS must exist in test.yml."""

    def test_all_expected_ci_jobs_exist(self, ci_content: str) -> None:
        actual_ci_jobs = extract_ci_job_ids(ci_content)
        expected_ci_jobs = {entry[1] for entry in EXPECTED_ALIGNMENTS}
        missing = expected_ci_jobs - actual_ci_jobs
        assert not missing, (
            f"EXPECTED_ALIGNMENTS references CI jobs not found in "
            f".github/workflows/test.yml: {missing}"
        )

    def test_all_ci_descriptions_found_in_workflow(self, ci_content: str) -> None:
        missing: list[tuple[str, str]] = []
        for hook_id, _ci_job, ci_desc in EXPECTED_ALIGNMENTS:
            if not check_ci_contains(ci_content, ci_desc):
                missing.append((hook_id, ci_desc))
        assert not missing, (
            f"EXPECTED_ALIGNMENTS references CI descriptions not found in "
            f".github/workflows/test.yml: {missing}"
        )


# =============================================================================
# Tests: Drift detection logic (synthetic scenarios)
# =============================================================================


@pytest.mark.unit
class TestDriftDetectionMissingHook:
    """When a hook_id from EXPECTED_ALIGNMENTS is missing from pre-commit
    config, the script should detect drift."""

    def test_detects_missing_hook(self) -> None:
        # Pre-commit content that is missing the "ruff" hook
        precommit = textwrap.dedent("""\
            repos:
              - repo: https://example.com
                hooks:
                  - id: ruff-format
        """)
        hook_ids = extract_hook_ids(precommit)
        # "ruff" is in EXPECTED_ALIGNMENTS but not in this precommit config
        assert "ruff" not in hook_ids
        assert "ruff-format" in hook_ids


@pytest.mark.unit
class TestDriftDetectionMissingCiReference:
    """When a CI description from EXPECTED_ALIGNMENTS is missing from the CI
    workflow, the script should detect drift."""

    def test_detects_missing_ci_description(self) -> None:
        ci = textwrap.dedent("""\
            name: CI
            on: push
            jobs:
              lint:
                runs-on: ubuntu-latest
                steps:
                  - name: ruff format
                    run: ruff format --check
        """)
        # "mypy" is a CI description in EXPECTED_ALIGNMENTS but not here
        assert not check_ci_contains(ci, "mypy")
        assert check_ci_contains(ci, "ruff format --check")


@pytest.mark.unit
class TestDriftDetectionMissingCiJob:
    """When a CI job from EXPECTED_ALIGNMENTS is missing from the workflow
    YAML, the script should detect drift."""

    def test_detects_missing_ci_job(self) -> None:
        ci = textwrap.dedent("""\
            name: CI
            on: push
            jobs:
              lint:
                runs-on: ubuntu-latest
                steps: []
        """)
        job_ids = extract_ci_job_ids(ci)
        assert "lint" in job_ids
        assert "pyright" not in job_ids


# =============================================================================
# Tests: Parametrized validation of each alignment entry against real files
# =============================================================================


def _alignment_id(val: tuple[str, str, str]) -> str:
    """Generate a readable test ID from an alignment tuple."""
    return val[0]


@pytest.mark.unit
class TestEachAlignmentEntry:
    """Parametrized test that checks each EXPECTED_ALIGNMENTS entry against
    the actual repository configuration files."""

    @pytest.mark.parametrize(
        "alignment",
        EXPECTED_ALIGNMENTS,
        ids=[a[0] for a in EXPECTED_ALIGNMENTS],
    )
    def test_hook_exists_in_precommit(
        self, alignment: tuple[str, str, str], precommit_content: str
    ) -> None:
        hook_id, _ci_job, _ci_desc = alignment
        actual_hook_ids = extract_hook_ids(precommit_content)
        assert hook_id in actual_hook_ids, (
            f"Hook '{hook_id}' listed in EXPECTED_ALIGNMENTS but not found in "
            f".pre-commit-config.yaml. Available hooks: {sorted(actual_hook_ids)}"
        )

    @pytest.mark.parametrize(
        "alignment",
        EXPECTED_ALIGNMENTS,
        ids=[a[0] for a in EXPECTED_ALIGNMENTS],
    )
    def test_ci_job_exists_in_workflow(
        self, alignment: tuple[str, str, str], ci_content: str
    ) -> None:
        _hook_id, ci_job, _ci_desc = alignment
        actual_ci_jobs = extract_ci_job_ids(ci_content)
        assert ci_job in actual_ci_jobs, (
            f"CI job '{ci_job}' listed in EXPECTED_ALIGNMENTS but not found in "
            f".github/workflows/test.yml. Available jobs: {sorted(actual_ci_jobs)}"
        )

    @pytest.mark.parametrize(
        "alignment",
        EXPECTED_ALIGNMENTS,
        ids=[a[0] for a in EXPECTED_ALIGNMENTS],
    )
    def test_ci_description_found_in_workflow(
        self, alignment: tuple[str, str, str], ci_content: str
    ) -> None:
        hook_id, _ci_job, ci_desc = alignment
        assert check_ci_contains(ci_content, ci_desc), (
            f"CI description '{ci_desc}' (for hook '{hook_id}') not found anywhere "
            f"in .github/workflows/test.yml"
        )


# =============================================================================
# Tests: Required checks manifest alignment
# =============================================================================


@pytest.mark.unit
class TestRequiredChecksManifest:
    """The .github/required-checks.yaml must exist and contain entries
    that align with the CI jobs referenced by EXPECTED_ALIGNMENTS."""

    REQUIRED_CHECKS_PATH = _REPO_ROOT / ".github" / "required-checks.yaml"

    def test_required_checks_file_exists(self) -> None:
        assert self.REQUIRED_CHECKS_PATH.exists(), (
            f"Required checks manifest not found: {self.REQUIRED_CHECKS_PATH}"
        )

    def test_required_checks_contains_ci_job_ids(self) -> None:
        content = self.REQUIRED_CHECKS_PATH.read_text(encoding="utf-8")
        data = yaml.safe_load(content)
        assert isinstance(data, dict), "required-checks.yaml should be a YAML dict"

        required = data.get("required_checks", [])
        manifest_job_ids = {
            entry["job_id"]
            for entry in required
            if isinstance(entry, dict) and "job_id" in entry
        }

        expected_ci_jobs = {entry[1] for entry in EXPECTED_ALIGNMENTS}

        # Every CI job referenced by an alignment should appear in the
        # required-checks manifest (or be documented as excluded).
        excluded = data.get("excluded_checks", [])
        excluded_names = {
            entry.get("name", "") for entry in excluded if isinstance(entry, dict)
        }

        missing = expected_ci_jobs - manifest_job_ids
        if missing:
            # Filter out any that might be covered by exclusions
            # (unlikely, but be thorough)
            assert not missing, (
                f"CI jobs referenced by EXPECTED_ALIGNMENTS are not in "
                f"required-checks.yaml: {missing}. "
                f"Manifest job_ids: {sorted(manifest_job_ids)}, "
                f"Excluded: {sorted(excluded_names)}"
            )


# =============================================================================
# Tests: Alignment count sanity
# =============================================================================


@pytest.mark.unit
class TestAlignmentCount:
    """Sanity check that the alignment list has a reasonable number of entries.

    This catches accidental truncation or mass deletion of entries.
    """

    # At time of writing there are 17 entries. Set a lower bound at 15 to
    # allow minor removals but catch catastrophic loss.
    MINIMUM_EXPECTED_ENTRIES = 15

    def test_alignment_count_is_reasonable(self) -> None:
        assert len(EXPECTED_ALIGNMENTS) >= self.MINIMUM_EXPECTED_ENTRIES, (
            f"EXPECTED_ALIGNMENTS has only {len(EXPECTED_ALIGNMENTS)} entries, "
            f"expected at least {self.MINIMUM_EXPECTED_ENTRIES}. "
            f"Was the list accidentally truncated?"
        )

    def test_alignment_count_matches_known_value(self) -> None:
        """Guard against silent addition or removal of entries.

        Update the expected count here when legitimately adding or removing
        alignment entries.
        """
        expected_count = 18
        assert len(EXPECTED_ALIGNMENTS) == expected_count, (
            f"EXPECTED_ALIGNMENTS has {len(EXPECTED_ALIGNMENTS)} entries, "
            f"expected {expected_count}. If you added or removed entries, "
            f"update expected_count in this test."
        )
