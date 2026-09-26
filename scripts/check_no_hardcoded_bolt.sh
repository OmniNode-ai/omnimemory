#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Reject bolt://localhost literal strings in production Python source.
# Docstrings (triple-quoted blocks) and test files are exempt.
# This is the memory-pipeline equivalent of kafka-no-hardcoded-fallback.
#
# Usage: called by pre-commit with staged filenames; no args scans src/.
# Exempt: files matching *test_*, *conftest*, */tests/*
set -euo pipefail

VIOLATIONS=0

scan_file() {
    local file="$1"
    # Skip test files
    case "$(basename "$file")" in
        test_* | *_test.py | conftest.py) return ;;
    esac
    case "/$file/" in
        */tests/*) return ;;
    esac

    if grep -n 'bolt://localhost' "$file" 2>/dev/null; then
        echo "ERROR: Hardcoded bolt://localhost in $file"
        echo "       Read OMNIMEMORY_MEMGRAPH_HOST/PORT from env instead of hardcoding."
        echo "       (Check plugin config, Settings model, or env helper for the project-standard pattern.)"
        VIOLATIONS=$((VIOLATIONS + 1))
    fi
}

if [[ $# -gt 0 ]]; then
    for file in "$@"; do
        scan_file "$file"
    done
else
    while IFS= read -r -d '' file; do
        scan_file "$file"
    done < <(find src/ -name "*.py" -type f -print0)
fi

if [[ $VIOLATIONS -gt 0 ]]; then
    echo ""
    echo "Found $VIOLATIONS file(s) with hardcoded bolt://localhost URIs."
    echo "These will fail inside Docker where 'localhost' resolves to the container itself."
    exit 1
fi
