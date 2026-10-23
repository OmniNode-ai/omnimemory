# SPDX-FileCopyrightText: 2026 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-21331: the Canonical File Shape job reads omnibase_core through a dev-merge tag."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
CI = ROOT / ".github/workflows/ci.yml"
CORE_GIT = "omnibase-core @ git+https://github.com/OmniNode-ai/omnibase_core@"
DEV_MERGE_TAG = re.compile(r"dev-\d{14}-[0-9a-f]{12}")


def _job() -> dict[str, object]:
    data = yaml.safe_load(CI.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    jobs = data["jobs"]
    assert isinstance(jobs, dict)
    job = jobs["canonical-file-shape"]
    assert isinstance(job, dict)
    return job


def test_canonical_file_shape_core_is_read_through_a_dev_merge_tag() -> None:
    env = _job()["env"]
    assert isinstance(env, dict)
    core = env["CANONICAL_FILE_SHAPE_CORE"]
    assert isinstance(core, str)
    assert core.startswith(CORE_GIT)
    ref = core.removeprefix(CORE_GIT)
    assert DEV_MERGE_TAG.fullmatch(ref), ref
    assert not re.search(r"[0-9a-f]{40}", core)


def test_canonical_file_shape_job_plants_a_bad_sample() -> None:
    assert "zz_canary_planted" in CI.read_text(encoding="utf-8")
