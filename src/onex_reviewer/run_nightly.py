#!/usr/bin/env python3
"""Run ONEX nightly review on incremental changes."""

import json
import sys
from pathlib import Path
from datetime import datetime

from .agents.nightly import NightlyAgent
from .models.inputs import ReviewInput
from .models.outputs import ReviewOutput


def load_policy() -> str:
    """Load policy configuration."""
    policy_path = Path(__file__).parent / "config" / "policy.yaml"
    return policy_path.read_text()


def main(output_dir: str):
    """Main entry point for nightly review."""
    base_dir = Path(output_dir)

    # Load manifest
    manifest_path = base_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"Error: Manifest not found at {manifest_path}")
        sys.exit(1)

    manifest = json.loads(manifest_path.read_text())
    print(f"ONEX Nightly Review for {manifest['repo']}")
    print(f"Reviewing changes: {manifest['commit_range']}")
    print(f"Changed files: {manifest['changed_files']}")

    # Load inputs
    stats = (base_dir / "nightly.stats").read_text()
    names = (base_dir / "nightly.names").read_text()
    diff = (base_dir / "nightly.diff").read_text()

    # Create review input
    review_input = ReviewInput(
        repo=manifest["repo"],
        commit_range=manifest["commit_range"],
        today=manifest["date"],
        policy_yaml=load_policy(),
        git_stats=stats,
        git_names=names,
        git_diff=diff,
        is_baseline=False
    )

    # Initialize agent and process
    agent = NightlyAgent()
    output = agent.process(review_input)

    # Save output
    output_file = base_dir / "nightly_review.out"
    with open(output_file, "w") as f:
        f.write(output.to_output_string())

    # Save NDJSON separately for easier processing
    ndjson_file = base_dir / "findings.ndjson"
    with open(ndjson_file, "w") as f:
        for finding in output.findings:
            f.write(json.dumps(finding.to_ndjson_dict(), ensure_ascii=True) + "\n")

    # Print summary
    print("\n" + "="*60)
    print("NIGHTLY REVIEW COMPLETE")
    print("="*60)
    print(f"Total findings: {len(output.findings)}")
    print(f"Risk score: {output.risk_score}")
    print(f"Output: {output_file}")
    print(f"NDJSON: {ndjson_file}")
    print("\nSummary:")
    print(output.summary)

    # Exit with non-zero if critical findings
    critical_count = sum(1 for f in output.findings if f.severity == "error")
    if critical_count > 0:
        print(f"\n⚠️  {critical_count} critical errors found!")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m onex_reviewer.run_nightly <output_dir>")
        sys.exit(1)

    main(sys.argv[1])