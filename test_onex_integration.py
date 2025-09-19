#!/usr/bin/env python3
"""Integration test for ONEX reviewer system."""

import subprocess
import json
import tempfile
from pathlib import Path


def test_baseline_producer():
    """Test baseline producer script."""
    print("Testing baseline producer...")

    # Run baseline producer
    result = subprocess.run(
        ["./scripts/onex_reviewer/baseline_producer.sh"],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print(f"Error running baseline producer:")
        print(f"  stdout: {result.stdout}")
        print(f"  stderr: {result.stderr}")
        return False

    # Find output directory
    import re
    match = re.search(r"Output directory: (.+)", result.stdout)
    if not match:
        print("Could not find output directory")
        return False

    output_dir = Path(match.group(1))

    # Check expected files exist
    expected_files = [
        "manifest.json",
        "nightly.diff",
        "nightly.stats",
        "nightly.names"
    ]

    for file in expected_files:
        if not (output_dir / file).exists():
            print(f"Missing expected file: {file}")
            return False

    # Check manifest
    manifest = json.loads((output_dir / "manifest.json").read_text())
    print(f"  Repository: {manifest['repo']}")
    print(f"  Total files: {manifest['total_files']}")
    print(f"  Total shards: {manifest['total_shards']}")

    return True


def test_nightly_producer():
    """Test nightly producer script."""
    print("\nTesting nightly producer...")

    # First run to initialize
    result = subprocess.run(
        ["./scripts/onex_reviewer/nightly_producer.sh"],
        capture_output=True,
        text=True
    )

    if "Initial marker set" in result.stdout:
        print("  Initial marker created")
    elif "No changes since last run" in result.stdout:
        print("  No changes to review")
    else:
        # Check for actual output
        import re
        match = re.search(r"Output: (.+)", result.stdout)
        if match:
            output_dir = Path(match.group(1))
            manifest = json.loads((output_dir / "manifest.json").read_text())
            print(f"  Changed files: {manifest['changed_files']}")

    return True


def main():
    """Run integration tests."""
    print("ONEX Reviewer Integration Tests")
    print("=" * 60)

    try:
        # Test baseline producer
        if not test_baseline_producer():
            print("❌ Baseline producer test failed")
            return 1
        print("✅ Baseline producer test passed")

        # Test nightly producer
        if not test_nightly_producer():
            print("❌ Nightly producer test failed")
            return 1
        print("✅ Nightly producer test passed")

        print("\n" + "=" * 60)
        print("✅ ALL INTEGRATION TESTS PASSED!")

    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())