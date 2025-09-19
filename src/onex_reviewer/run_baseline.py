#!/usr/bin/env python3
"""Run ONEX baseline review on sharded diffs."""

import json
import sys
from pathlib import Path
from datetime import datetime
from typing import List

from .agents.baseline import BaselineAgent
from .models.inputs import ReviewInput
from .models.outputs import ReviewOutput
from .models.finding import Finding


def load_policy() -> str:
    """Load policy configuration."""
    policy_path = Path(__file__).parent / "config" / "policy.yaml"
    return policy_path.read_text()


def process_shard(agent: BaselineAgent, base_dir: Path, shard_path: Path, manifest: dict) -> ReviewOutput:
    """Process a single diff shard."""
    # Load shard diff
    diff_content = shard_path.read_text()

    # Load other inputs
    stats = (base_dir / "nightly.stats").read_text()
    names = (base_dir / "nightly.names").read_text()

    # Extract shard index from filename
    shard_index = int(shard_path.stem.split("_")[-1])

    # Create review input
    review_input = ReviewInput(
        repo=manifest["repo"],
        commit_range=manifest["commit_range"],
        today=manifest["date"],
        policy_yaml=load_policy(),
        git_stats=stats,
        git_names=names,
        git_diff=diff_content,
        is_baseline=True,
        shard_index=shard_index
    )

    # Process with agent
    return agent.process(review_input)


def main(output_dir: str):
    """Main entry point for baseline review."""
    base_dir = Path(output_dir)

    # Load manifest
    manifest_path = base_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"Error: Manifest not found at {manifest_path}")
        sys.exit(1)

    manifest = json.loads(manifest_path.read_text())
    print(f"ONEX Baseline Review for {manifest['repo']}")
    print(f"Processing {manifest['total_shards']} shards...")

    # Initialize agent
    agent = BaselineAgent()

    # Process all shards
    all_findings: List[Finding] = []
    shard_dir = base_dir / "shards"

    for shard_path in sorted(shard_dir.glob("diff_shard_*.diff")):
        print(f"Processing {shard_path.name}...")
        output = process_shard(agent, base_dir, shard_path, manifest)
        all_findings.extend(output.findings)

    # Generate final summary
    final_output = ReviewOutput(
        findings=all_findings,
        summary="",  # Will regenerate
        risk_score=0
    )

    # Recalculate risk score and summary
    final_output.risk_score = agent._calculate_risk_score(all_findings)

    # Create a dummy input for summary generation
    dummy_input = ReviewInput(
        repo=manifest["repo"],
        commit_range=manifest["commit_range"],
        today=manifest["date"],
        policy_yaml="",
        git_stats="",
        git_names="",
        git_diff=""
    )
    final_output.summary = agent._generate_summary(all_findings, final_output.risk_score, dummy_input)

    # Output results
    output_file = base_dir / "baseline_review.out"
    with open(output_file, "w") as f:
        f.write(final_output.to_output_string())

    # Print summary
    print("\n" + "="*60)
    print("BASELINE REVIEW COMPLETE")
    print("="*60)
    print(f"Total findings: {len(all_findings)}")
    print(f"Risk score: {final_output.risk_score}")
    print(f"Output: {output_file}")
    print("\nSummary:")
    print(final_output.summary)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m onex_reviewer.run_baseline <output_dir>")
        sys.exit(1)

    main(sys.argv[1])