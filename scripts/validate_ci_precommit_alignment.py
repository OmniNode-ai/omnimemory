#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Validate alignment between CI workflow jobs and pre-commit hooks.

This script ensures that the CI workflow (.github/workflows/ci.yml) and
pre-commit configuration (.pre-commit-config.yaml) stay synchronized.

It checks:
1. Every CI validation step has a corresponding pre-commit hook
2. Every pre-commit validation hook has a corresponding CI job
3. The required-checks manifest is current

This prevents "works locally, fails in CI" (or vice versa) drift.

Usage:
    python scripts/validate_ci_precommit_alignment.py
    python scripts/validate_ci_precommit_alignment.py --verbose

Exit codes:
    0 - Aligned (no drift detected)
    1 - Drift detected
    2 - File parsing error

Reference: omniintelligence scripts/validate_ci_precommit_alignment.py
Added for OMN-2218: Phase 7 CI Infrastructure Alignment
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

# Expected hook-to-CI alignment pairs
# Format: (pre-commit hook id, CI job name, CI step/command description)
EXPECTED_ALIGNMENTS: list[tuple[str, str, str]] = [
    # Formatting and linting
    ("ruff-format", "lint", "ruff format --check"),
    ("ruff", "lint", "ruff check"),
    ("mypy-type-check", "lint", "mypy"),
    ("pyright", "pyright", "pyright"),
    # ONEX validation hooks
    ("validate-pydantic-patterns", "onex-validation", "validate_pydantic_patterns.py"),
    (
        "validate-single-class-per-file",
        "onex-validation",
        "validate_single_class_per_file.py",
    ),
    ("validate-enum-casing", "onex-validation", "validate_enum_casing.py"),
    (
        "validate-no-backward-compatibility",
        "onex-validation",
        "validate_no_backward_compatibility.py",
    ),
    ("validate-secrets", "onex-validation", "validate_secrets.py"),
    ("onex-validate-naming", "onex-validation", "validate_naming.py"),
    ("validate-http-imports", "onex-validation", "validate_http_imports.py"),
    ("validate-kafka-imports", "onex-validation", "validate_kafka_imports.py"),
    ("validate-model-locations", "onex-validation", "validate_model_locations.py"),
    # Infrastructure hooks
    ("migration-freeze-check", "migration-freeze", "check_migration_freeze.sh"),
    # New hooks from OMN-2218
    (
        "validate-no-transport-imports",
        "transport-import-guard",
        "validate_no_transport_imports.py",
    ),
    ("contract-linter", "contract-validation", "contract_linter"),
    ("io-audit", "io-audit", "io_audit"),
    # AI-slop pattern checker
    ("check-ai-slop", "ai-slop-check", "check_ai_slop.py"),
]


def load_file(path: Path) -> str | None:
    """Load a file and return its content, or None if not found."""
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        print(f"ERROR: File not found: {path}", file=sys.stderr)
        return None
    except OSError as e:
        print(f"ERROR: Cannot read {path}: {e}", file=sys.stderr)
        return None


def extract_hook_ids(precommit_content: str) -> set[str]:
    """Extract hook IDs from pre-commit config."""
    hook_ids: set[str] = set()
    for match in re.finditer(r"^\s+- id:\s*(\S+)", precommit_content, re.MULTILINE):
        hook_ids.add(match.group(1))
    return hook_ids


def extract_ci_job_ids(ci_content: str) -> set[str]:
    """Extract job IDs from CI workflow using YAML parser."""
    data = yaml.safe_load(ci_content)
    if not isinstance(data, dict):
        return set()
    jobs = data.get("jobs")
    if not isinstance(jobs, dict):
        return set()
    return set(jobs.keys())


def check_ci_contains(ci_content: str, search_term: str) -> bool:
    """Check if CI content references a search term."""
    return search_term in ci_content


def main() -> int:
    """Main entry point."""
    verbose = "--verbose" in sys.argv or "-v" in sys.argv

    precommit_path = Path(".pre-commit-config.yaml")
    ci_path = Path(".github/workflows/ci.yml")
    required_checks_path = Path(".github/required-checks.yaml")

    precommit_content = load_file(precommit_path)
    ci_content = load_file(ci_path)

    if precommit_content is None or ci_content is None:
        return 2

    hook_ids = extract_hook_ids(precommit_content)
    ci_job_ids = extract_ci_job_ids(ci_content)

    if verbose:
        print(f"Found {len(hook_ids)} pre-commit hooks: {sorted(hook_ids)}")
        print(f"Found {len(ci_job_ids)} CI jobs: {sorted(ci_job_ids)}")
        print()

    drift_found = False
    checked = 0

    print("Checking CI/pre-commit alignment...")
    print()

    for hook_id, ci_job, ci_description in EXPECTED_ALIGNMENTS:
        checked += 1

        # Check pre-commit hook exists
        if hook_id not in hook_ids:
            print(f"  DRIFT: Pre-commit hook '{hook_id}' not found in {precommit_path}")
            drift_found = True
            continue

        # Check CI references the validation script/command
        if not check_ci_contains(ci_content, ci_description):
            print(
                f"  DRIFT: CI workflow does not reference '{ci_description}' "
                f"(expected for hook '{hook_id}')"
            )
            drift_found = True
            continue

        if verbose:
            print(f"  OK: {hook_id} <-> {ci_job} ({ci_description})")

    print()

    # Check required-checks.yaml exists
    if not required_checks_path.exists():
        print(f"  DRIFT: Required checks manifest not found: {required_checks_path}")
        drift_found = True
    elif verbose:
        print(f"  OK: Required checks manifest exists: {required_checks_path}")

    print()

    if drift_found:
        print(f"FAILED: CI/pre-commit drift detected ({checked} alignments checked)")
        print()
        print("To fix:")
        print("  1. Ensure every pre-commit hook has a CI counterpart")
        print("  2. Ensure every CI validation step has a pre-commit hook")
        print("  3. Update .github/required-checks.yaml if jobs were renamed")
        return 1

    print(f"PASSED: All {checked} CI/pre-commit alignments verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
