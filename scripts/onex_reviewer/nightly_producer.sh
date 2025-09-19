#!/usr/bin/env bash
# ONEX Nightly Producer Script
# Creates incremental diff for nightly review

set -euo pipefail

# Configuration
PREV_FILE=.onex_nightly_prev
REPO_NAME="$(basename "$(git rev-parse --show-toplevel)")"

echo "ONEX Nightly Producer for $REPO_NAME"

# Fetch latest from origin
echo "Fetching latest from origin..."
git fetch origin

# Check for previous marker
if [[ ! -f "$PREV_FILE" ]]; then
  echo "No previous marker found. Creating initial marker..."
  echo "$(git rev-parse origin/main)" > "$PREV_FILE"
  echo "Initial marker set. Run again for incremental changes."
  exit 0
fi

# Get previous and current commits
PREV="$(cat "$PREV_FILE")"
HEADSHA="$(git rev-parse origin/main)"

# Check if there are changes
if [[ "$PREV" == "$HEADSHA" ]]; then
  echo "No changes since last run ($PREV)"
  exit 0
fi

# Create output directory
OUT_DIR=".onex_nightly/$REPO_NAME/$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$OUT_DIR"

echo "Generating diff from $PREV to $HEADSHA..."

# Generate statistics
git diff --stat "$PREV...$HEADSHA" > "$OUT_DIR/nightly.stats"

# Generate name status
git diff --name-status "$PREV...$HEADSHA" > "$OUT_DIR/nightly.names"

# Generate unified diff
git diff -U1 "$PREV...$HEADSHA" > "$OUT_DIR/nightly.diff" || true

# Count changes
CHANGED_FILES=$(git diff --name-only "$PREV...$HEADSHA" | wc -l)

# Create manifest
echo "Creating manifest..."
cat > "$OUT_DIR/manifest.json" <<EOF
{
  "repo": "$REPO_NAME",
  "commit_range": "$PREV...$HEADSHA",
  "date": "$(date -u +%Y-%m-%d)",
  "changed_files": $CHANGED_FILES,
  "output_dir": "$OUT_DIR"
}
EOF

# Update marker (only on success)
echo "$HEADSHA" > "$PREV_FILE"

echo ""
echo "Nightly production complete!"
echo "- Changed files: $CHANGED_FILES"
echo "- Commit range: $PREV...$HEADSHA"
echo "- Output: $OUT_DIR"
echo ""
echo "To run nightly review:"
echo "python -m onex_reviewer.run_nightly \"$OUT_DIR\""