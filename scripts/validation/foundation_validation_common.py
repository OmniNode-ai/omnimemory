# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Shared helpers for root-level foundation validation scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def validate_expected_structure(
    base_path: Path,
    expected_structure: dict[str, str],
) -> dict[str, Any]:
    """Validate that expected files and directories exist under ``base_path``."""
    found_items = {}
    missing_items = []

    for item, description in expected_structure.items():
        item_path = base_path / item
        if item_path.exists():
            found_items[item] = description
        else:
            missing_items.append(item)

    print("  Project structure validation")
    total = len(expected_structure)
    print(f"   Found: {len(found_items)} / {total} expected items")
    if missing_items:
        items = ", ".join(missing_items[:3])
        suffix = "..." if len(missing_items) > 3 else ""
        print(f"   Missing: {items}{suffix}")

    return {
        "success": len(missing_items) == 0,
        "found_count": len(found_items),
        "total_count": total,
        "missing_items": missing_items,
    }


def print_validation_results(
    results: dict[str, dict[str, Any]],
    success_header: str,
    success_lines: list[str],
    failure_header: str,
    failure_lines: list[str],
) -> int:
    """Print common foundation validation result summary and return exit code."""
    print("\n📊 Validation Results:")
    print("=" * 30)

    passed = 0
    failed = 0

    for test_name, result in results.items():
        if result.get("success", False):
            print(f"✅ {test_name}: PASS")
            passed += 1
        else:
            print(f"❌ {test_name}: FAIL - {result.get('error', 'Unknown error')}")
            failed += 1

    print(f"\nResults: {passed} passed, {failed} failed")

    if failed == 0:
        print(f"\n{success_header}")
        for line in success_lines:
            print(line)
        return 0

    print(f"\n{failure_header.format(failed=failed)}")
    for line in failure_lines:
        print(line)
    return 1
