#!/usr/bin/env bash
# check_migration_freeze.sh — Block new migrations while .migration_freeze exists.
#
# Usage:
#   Pre-commit: ./scripts/check_migration_freeze.sh           (checks staged files)
#   CI:         ./scripts/check_migration_freeze.sh --ci       (checks diff vs base branch)
#
# Exit codes:
#   0 — No freeze active, or no new migrations detected
#   1 — Freeze violation: new migration files added

set -euo pipefail

FREEZE_FILE=".migration_freeze"
# Path to migrations directory. git diff returns empty (not error) if this path
# does not exist yet, which is correct — no migrations means no violations.
MIGRATIONS_DIR="deployment/database/migrations"

# If no freeze file, nothing to enforce.
if [ ! -f "$FREEZE_FILE" ]; then
    echo "No migration freeze active — skipping check."
    exit 0
fi

echo "Migration freeze is ACTIVE ($FREEZE_FILE exists)"

MODE="${1:-precommit}"

if [ "$MODE" = "--ci" ]; then
    # CI mode: compare against base branch
    # GITHUB_BASE_REF is set automatically for pull_request events.
    # DEFAULT_BRANCH can be passed from the workflow for push events.
    BASE_BRANCH="${GITHUB_BASE_REF:-${DEFAULT_BRANCH:-main}}"
    # Defensive fetch: ensure origin/<base> ref is up-to-date even if
    # the CI runner's checkout didn't fully resolve it.
    if ! git fetch origin "${BASE_BRANCH}" --quiet 2>/dev/null; then
        echo "Warning: git fetch origin ${BASE_BRANCH} failed; using existing refs." >&2
    fi
    # Detect added (A) or renamed (R) files in the migrations directory.
    # Modified (M) files are intentionally allowed — fixing existing
    # migrations (rollback bug fixes, comment tweaks) is safe during freeze.
    # Three-dot diff finds the merge-base automatically. If no common ancestor
    # exists (orphan branch), git diff falls back to two-dot behavior — safe.
    # Uses awk (not grep|awk) to avoid grep exit-code 1 on no-match with pipefail.
    NEW_MIGRATIONS=$(git diff --name-status "origin/${BASE_BRANCH}...HEAD" -- "$MIGRATIONS_DIR" \
        | awk '/^[AR]/ {print $NF}')
else
    # Pre-commit mode: check staged files
    # Uses awk (not grep|awk) to avoid grep exit-code 1 on no-match with pipefail.
    NEW_MIGRATIONS=$(git diff --cached --name-status -- "$MIGRATIONS_DIR" \
        | awk '/^[AR]/ {print $NF}')
fi

if [ -n "$NEW_MIGRATIONS" ]; then
    echo ""
    echo "ERROR: Migration freeze violation!"
    echo "Blocked: new migration files (A=added) or renames (R) while $FREEZE_FILE exists."
    echo "Allowed: modifications (M) to existing migrations (bug fixes, comments)."
    echo ""
    echo "Violating files:"
    echo "$NEW_MIGRATIONS" | sed 's/^/  /'
    echo ""
    echo "See $FREEZE_FILE for details on the active freeze."
    exit 1
fi

echo "No new migrations detected — freeze check passed."
exit 0
