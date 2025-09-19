#!/usr/bin/env python3
"""Test script for ONEX reviewer implementation."""

import json
from datetime import datetime
from pathlib import Path

from src.onex_reviewer.agents.baseline import BaselineAgent
from src.onex_reviewer.agents.nightly import NightlyAgent
from src.onex_reviewer.models.inputs import ReviewInput
from src.onex_reviewer.rules.engine import RuleEngine


def create_sample_diff():
    """Create a sample diff with various violations."""
    return """diff --git a/src/omnibase_spi/protocols/protocol_config.py b/src/omnibase_spi/protocols/protocol_config.py
index abc123..def456 100644
--- a/src/omnibase_spi/protocols/protocol_config.py
+++ b/src/omnibase_spi/protocols/protocol_config.py
@@ -1,5 +1,10 @@
+from typing import Protocol, Any
+import os  # SPI forbidden library
+from omniagent import something  # Boundary violation in omnibase_core
+
+class ConfigProvider(Protocol):  # Missing Protocol prefix and @runtime_checkable
+    def get_config(self, key) -> Any:  # Missing parameter annotation, uses Any
+        pass
+
+def process_data(data):  # Missing annotations
+    return data

diff --git a/src/omnibase_core/model_user.py b/src/omnibase_core/model_user.py
index 111222..333444 100644
--- a/src/omnibase_core/model_user.py
+++ b/src/omnibase_core/model_user.py
@@ -1,3 +1,8 @@
+from typing import Optional
+from omnimcp import tools  # Boundary violation
+
+class UserManager:  # Missing Model prefix
+    def __init__(self):
+        pass
+
+# onex:ignore ONEX.NAMING.MODEL_001  # Malformed waiver - missing reason and expires"""


def create_sample_policy():
    """Create sample policy YAML."""
    return """ruleset_version: "0.1"
repos:
  omnibase_core:
    forbids:
      - "^from\\\\s+omniagent\\\\b"
      - "^from\\\\s+omnimcp\\\\b"
  omnibase_spi:
    forbids:
      - "^import\\\\s+(os|pathlib|sqlite3)\\\\b"
      - "^from\\\\s+(requests|httpx)\\\\b" """


def test_rule_engine():
    """Test the rule engine directly."""
    print("Testing Rule Engine...")
    print("=" * 60)

    engine = RuleEngine()

    # Create test input
    review_input = ReviewInput(
        repo="omnibase_core",
        commit_range="abc123...def456",
        today="2025-09-19",
        policy_yaml=create_sample_policy(),
        git_stats="2 files changed, 15 insertions(+)",
        git_names="M\tsrc/omnibase_spi/protocols/protocol_config.py\nM\tsrc/omnibase_core/model_user.py",
        git_diff=create_sample_diff(),
        is_baseline=False
    )

    # Apply rules
    findings = engine.apply_rules(review_input)

    print(f"Found {len(findings)} violations:\n")
    for finding in findings:
        print(f"[{finding.severity.upper()}] {finding.rule_id}")
        print(f"  File: {finding.path}:{finding.line}")
        print(f"  Message: {finding.message}")
        print(f"  Evidence: {finding.evidence}")
        print(f"  Fix: {finding.suggested_fix}")
        print()

    return findings


def test_agents():
    """Test both baseline and nightly agents."""
    print("\nTesting Agents...")
    print("=" * 60)

    # Test input
    review_input = ReviewInput(
        repo="omnibase_core",
        commit_range="abc123...def456",
        today="2025-09-19",
        policy_yaml=create_sample_policy(),
        git_stats="2 files changed, 15 insertions(+)",
        git_names="M\tsrc/omnibase_spi/protocols/protocol_config.py\nM\tsrc/omnibase_core/model_user.py",
        git_diff=create_sample_diff(),
        is_baseline=False
    )

    # Test nightly agent
    print("\nNightly Agent:")
    print("-" * 40)
    nightly = NightlyAgent()
    nightly_output = nightly.process(review_input)

    print(f"Risk Score: {nightly_output.risk_score}")
    print(f"Findings: {len(nightly_output.findings)}")
    print("\nFormatted Output:")
    print(nightly_output.to_output_string())

    # Test baseline agent
    print("\n" + "=" * 60)
    print("Baseline Agent:")
    print("-" * 40)
    baseline = BaselineAgent()
    review_input.is_baseline = True
    baseline_output = baseline.process(review_input)

    print(f"Risk Score: {baseline_output.risk_score}")
    print(f"Findings: {len(baseline_output.findings)}")


def test_ndjson_format():
    """Test NDJSON output format."""
    print("\nTesting NDJSON Format...")
    print("=" * 60)

    from src.onex_reviewer.models.finding import Finding

    finding = Finding(
        ruleset_version="0.1",
        rule_id="ONEX.NAMING.PROTOCOL_001",
        severity="error",
        repo="omnibase-spi",
        path="src/omnibase_spi/protocols/protocol_config.py",
        line=5,
        message="Protocol class does not start with 'Protocol'",
        evidence={"class_name": "ConfigProvider"},
        suggested_fix="Rename to ProtocolConfigProvider",
        fingerprint="abc12345"
    )

    ndjson_dict = finding.to_ndjson_dict()
    ndjson_str = json.dumps(ndjson_dict, ensure_ascii=True)

    print("NDJSON Output:")
    print(ndjson_str)

    # Validate it's valid JSON
    parsed = json.loads(ndjson_str)
    assert parsed["rule_id"] == "ONEX.NAMING.PROTOCOL_001"
    print("\n✅ NDJSON format valid!")


def main():
    """Run all tests."""
    print("ONEX Reviewer Test Suite")
    print("=" * 60)

    try:
        # Test rule engine
        findings = test_rule_engine()
        assert len(findings) > 0, "Should find violations in sample diff"

        # Test agents
        test_agents()

        # Test NDJSON format
        test_ndjson_format()

        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())