#!/usr/bin/env bash
# ONEX Baseline Producer Script
# Creates sharded baseline inputs for reviewing entire codebase

set -euo pipefail

# Configuration
EXCLUDES_REGEX='(^|/)(archive|archived|deprecated|dist|build|node_modules|venv|\.venv|\.mypy_cache|\.pytest_cache)(/|$)'
INCLUDE_EXT='py|pyi|yaml|yml|toml|json|ini|cfg|sh|bash|ps1|psm1|go|rs|ts|tsx|js|jsx|proto|sql|md'
BYTES_PER_SHARD=$((200*1024))  # 200KB per shard
EMPTY_TREE=4b825dc642cb6eb9a060e54bf8d69288fbee4904

# Get repository name
REPO_NAME="$(basename "$(git rev-parse --show-toplevel)")"

# Create output directory
OUT_DIR=".onex_baseline/$REPO_NAME/$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$OUT_DIR/shards"

echo "ONEX Baseline Producer for $REPO_NAME"
echo "Output directory: $OUT_DIR"

# Get list of files to review (excluding archived/deprecated)
echo "Collecting files to review..."
git ls-files -z \
  | tr -d '\r' \
  | xargs -0 -I{} bash -c '[[ "{}" =~ '"$EXCLUDES_REGEX"' ]] || echo "{}"' \
  | grep -E "\.(${INCLUDE_EXT})$" \
  > "$OUT_DIR/files.list" || true

FILE_COUNT=$(wc -l < "$OUT_DIR/files.list")
echo "Found $FILE_COUNT files to review"

# Generate git names (all as Added since baseline is against empty tree)
awk '{print "A\t"$0}' "$OUT_DIR/files.list" > "$OUT_DIR/nightly.names"

# Generate unified diff against empty tree
echo "Generating diff against empty tree..."
git diff -U0 --no-color "$EMPTY_TREE..HEAD" -- $(cat "$OUT_DIR/files.list") > "$OUT_DIR/nightly.diff" || true

# Generate statistics
git diff --stat "$EMPTY_TREE..HEAD" -- $(cat "$OUT_DIR/files.list") > "$OUT_DIR/nightly.stats" || true

# Split diff into shards
echo "Sharding diff (max $BYTES_PER_SHARD bytes per shard)..."

# Use csplit to split on diff boundaries
csplit -s -f "$OUT_DIR/tmpdiff_" -b "%03d" "$OUT_DIR/nightly.diff" '/^diff --git /' '{*}' 2>/dev/null || true

# Combine into sized shards
shard_idx=0
shard_bytes=0

for part in "$OUT_DIR"/tmpdiff_*; do
  if [[ ! -f "$part" ]]; then
    continue
  fi

  bytes=$(wc -c < "$part")

  # Start new shard if needed
  if (( shard_bytes + bytes > BYTES_PER_SHARD || shard_bytes == 0 )); then
    shard_idx=$((shard_idx + 1))
    shard_file="$OUT_DIR/shards/diff_shard_${shard_idx}.diff"
    > "$shard_file"
    shard_bytes=0
  fi

  # Add to current shard
  cat "$part" >> "$shard_file"
  shard_bytes=$((shard_bytes + bytes))
done

# Clean up temp files
rm -f "$OUT_DIR"/tmpdiff_*

# Create manifest
echo "Creating manifest..."
> "$OUT_DIR/manifest.json"
echo "{" >> "$OUT_DIR/manifest.json"
echo "  \"repo\": \"$REPO_NAME\"," >> "$OUT_DIR/manifest.json"
echo "  \"commit_range\": \"$EMPTY_TREE...$(git rev-parse HEAD)\"," >> "$OUT_DIR/manifest.json"
echo "  \"date\": \"$(date -u +%Y-%m-%d)\"," >> "$OUT_DIR/manifest.json"
echo "  \"total_files\": $FILE_COUNT," >> "$OUT_DIR/manifest.json"
echo "  \"total_shards\": $shard_idx," >> "$OUT_DIR/manifest.json"
echo "  \"output_dir\": \"$OUT_DIR\"" >> "$OUT_DIR/manifest.json"
echo "}" >> "$OUT_DIR/manifest.json"

echo ""
echo "Baseline production complete!"
echo "- Total files: $FILE_COUNT"
echo "- Total shards: $shard_idx"
echo "- Output: $OUT_DIR"
echo ""
echo "To run baseline review:"
echo "python -m onex_reviewer.run_baseline \"$OUT_DIR\""