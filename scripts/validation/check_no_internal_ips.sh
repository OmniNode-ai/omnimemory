#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
set -euo pipefail

if [[ $# -gt 0 ]]; then
    targets=("$@")
else
    targets=(src/ tests/ docs/ .github/)
fi

matches="$(grep -rHInE '(192\.168\.|10\.(0|1|2)\.|172\.(1[6-9]|2[0-9]|3[0-1])\.)' "${targets[@]}" 2>/dev/null | grep -v 'onex-allow-internal-ip' || true)"
if [[ -n "$matches" ]]; then
    printf '%s\n' "$matches" >&2
    echo 'ERROR: hardcoded internal IP found (add # onex-allow-internal-ip to suppress)' >&2
    exit 1
fi
