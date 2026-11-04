# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Run the runtime_profiles contract validator with the omnimemory allowlist.

Mirrors the required CI gate `.github/workflows/validator-runtime-profiles.yml`
exactly: it constructs `ValidatorRuntimeProfiles` with the repo-local allowlist
and validates the source tree, both passed as the arguments `<allowlist> <src-root>`.

This script exists so the local pre-commit hook and CI run the SAME invocation.
The validator's module `__main__` entry point resolves its allowlist from the
omnibase_core *package* directory, which does not contain the omnimemory
repo's frozen-violator allowlist — so invoking the bare module locally flags
every pre-existing allowlisted contract while CI passes. Pointing the hook at
this script removes that local/CI drift (OMN-13297, mirrors OMN-12955 for
omnimarket).
"""

from __future__ import annotations

import sys
from pathlib import Path

from omnibase_core.validation.validator_runtime_profiles import (
    ValidatorRuntimeProfiles,
)


def main() -> int:
    allowlist, src_root = (Path(arg) for arg in sys.argv[1:3])
    result = ValidatorRuntimeProfiles(allowlist_path=allowlist).validate(src_root)
    for issue in result.issues:
        print(
            f"[{issue.severity.value}] {issue.file_path}:{issue.line_number}: {issue.message}"
        )
    return 0 if result.is_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
