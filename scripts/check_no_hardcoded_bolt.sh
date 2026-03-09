#!/usr/bin/env bash
# Reject bolt://localhost literal strings in production Python source.
# Docstrings (triple-quoted blocks) and test files are exempt.
# This is the memory-pipeline equivalent of kafka-no-hardcoded-fallback.
#
# Usage: called by pre-commit (pass_filenames: false, scans src/ directly)
# Exempt: files matching *test_*, *conftest*, */tests/*
set -euo pipefail

VIOLATIONS=0

while IFS= read -r -d '' file; do
    # Skip test files
    case "$file" in
        *test_* | *conftest* | */tests/*) continue ;;
    esac

    if grep -n 'bolt://localhost' "$file" 2>/dev/null; then
        echo "ERROR: Hardcoded bolt://localhost in $file"
        echo "       Read OMNIMEMORY_MEMGRAPH_HOST/PORT from env instead of hardcoding."
        echo "       (Check plugin config, Settings model, or env helper for the project-standard pattern.)"
        VIOLATIONS=$((VIOLATIONS + 1))
    fi
done < <(find src/ -name "*.py" -type f -print0)

if [[ $VIOLATIONS -gt 0 ]]; then
    echo ""
    echo "Found $VIOLATIONS file(s) with hardcoded bolt://localhost URIs."
    echo "These will fail inside Docker where 'localhost' resolves to the container itself."
    exit 1
fi
