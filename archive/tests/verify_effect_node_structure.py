#!/usr/bin/env python3
"""Verification script for Memory Storage Effect Node structure and ONEX compliance."""

import os
import sys
from pathlib import Path

# Add src to path for direct imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def verify_file_structure():
    """Verify that all required files are present."""
    base_path = (
        Path(__file__).parent.parent
        / "src"
        / "omnimemory"
        / "nodes"
        / "node_memory_storage_effect"
    )

    required_files = [
        "__init__.py",
        "README.md",
        "v1_0_0/__init__.py",
        "v1_0_0/node.py",
        "v1_0_0/models/__init__.py",
        "v1_0_0/models/model_memory_storage_input.py",
        "v1_0_0/models/model_memory_storage_output.py",
        "v1_0_0/models/model_memory_storage_config.py",
        "v1_0_0/enums/__init__.py",
        "v1_0_0/enums/enum_memory_storage_operation_type.py",
        "v1_0_0/contracts/memory_storage_processing_subcontract.yaml",
        "v1_0_0/contracts/memory_storage_management_subcontract.yaml",
        "v1_0_0/manifests/version_manifest.yaml",
        "v1_0_0/manifests/compatibility_matrix.yaml",
    ]

    missing_files = []
    for file_path in required_files:
        full_path = base_path / file_path
        if not full_path.exists():
            missing_files.append(str(file_path))

    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    else:
        print(f"✅ All required files present ({len(required_files)} files)")
        return True


def verify_python_syntax():
    """Verify Python syntax of all Python files."""
    import py_compile

    base_path = (
        Path(__file__).parent.parent
        / "src"
        / "omnimemory"
        / "nodes"
        / "node_memory_storage_effect"
    )
    python_files = list(base_path.rglob("*.py"))

    failed_files = []
    for py_file in python_files:
        try:
            py_compile.compile(str(py_file), doraise=True)
        except py_compile.PyCompileError as e:
            failed_files.append((str(py_file), str(e)))

    if failed_files:
        print(f"❌ Syntax errors in {len(failed_files)} files:")
        for file_path, error in failed_files:
            print(f"   {file_path}: {error}")
        return False
    else:
        print(f"✅ All Python files have valid syntax ({len(python_files)} files)")
        return True


def verify_yaml_syntax():
    """Verify YAML syntax of contract and manifest files."""
    try:
        import yaml
    except ImportError:
        print("⚠️  PyYAML not available, skipping YAML validation")
        return True

    base_path = (
        Path(__file__).parent.parent
        / "src"
        / "omnimemory"
        / "nodes"
        / "node_memory_storage_effect"
    )
    yaml_files = list(base_path.rglob("*.yaml"))

    failed_files = []
    for yaml_file in yaml_files:
        try:
            with open(yaml_file, "r") as f:
                yaml.safe_load(f)
        except yaml.YAMLError as e:
            failed_files.append((str(yaml_file), str(e)))

    if failed_files:
        print(f"❌ YAML syntax errors in {len(failed_files)} files:")
        for file_path, error in failed_files:
            print(f"   {file_path}: {error}")
        return False
    else:
        print(f"✅ All YAML files have valid syntax ({len(yaml_files)} files)")
        return True


def verify_imports():
    """Verify that key imports work without external dependencies."""
    try:
        # Test enum import
        from omnimemory.nodes.node_memory_storage_effect.v1_0_0.enums.enum_memory_storage_operation_type import (
            EnumMemoryStorageOperationType,
        )

        # Check enum values
        expected_operations = [
            "STORE_MEMORY",
            "RETRIEVE_MEMORY",
            "UPDATE_MEMORY",
            "DELETE_MEMORY",
            "VECTOR_SEARCH",
            "SEMANTIC_SEARCH",
            "TEMPORAL_SEARCH",
            "BATCH_STORE",
            "BATCH_RETRIEVE",
            "HEALTH_CHECK",
            "GET_STATS",
            "OPTIMIZE_STORAGE",
        ]

        actual_operations = [op.name for op in EnumMemoryStorageOperationType]
        missing_ops = set(expected_operations) - set(actual_operations)

        if missing_ops:
            print(f"❌ Missing operation types: {missing_ops}")
            return False

        print(
            f"✅ Enum imports successful with {len(actual_operations)} operation types"
        )
        return True

    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error during imports: {e}")
        return False


def verify_onex_compliance():
    """Verify ONEX compliance patterns."""
    compliance_checks = []

    # Check main node file structure
    node_file = (
        Path(__file__).parent.parent
        / "src"
        / "omnimemory"
        / "nodes"
        / "node_memory_storage_effect"
        / "v1_0_0"
        / "node.py"
    )
    with open(node_file, "r") as f:
        content = f.read()

    # Check for ONEX compliance patterns
    onex_patterns = [
        ("NodeEffectService inheritance", "NodeEffectService" in content),
        ("ONEX effect interface", "async def effect(" in content),
        ("Typed process method", "async def process(" in content),
        ("Circuit breaker implementation", "circuit_breaker" in content.lower()),
        ("Error sanitization", "_sanitize_error_message" in content),
        ("Health check interface", "get_health_status" in content),
        ("Correlation ID validation", "_validate_correlation_id" in content),
        ("Container dependency injection", "ONEXContainer" in content),
    ]

    passed_checks = 0
    for check_name, check_result in onex_patterns:
        if check_result:
            print(f"✅ {check_name}")
            passed_checks += 1
        else:
            print(f"❌ {check_name}")

    compliance_rate = passed_checks / len(onex_patterns)
    if compliance_rate >= 0.8:
        print(
            f"✅ ONEX compliance: {compliance_rate:.1%} ({passed_checks}/{len(onex_patterns)})"
        )
        return True
    else:
        print(
            f"❌ ONEX compliance: {compliance_rate:.1%} ({passed_checks}/{len(onex_patterns)}) - Below threshold"
        )
        return False


def main():
    """Run all verification checks."""
    print("🔍 Verifying Memory Storage Effect Node Implementation")
    print("=" * 60)

    checks = [
        ("File Structure", verify_file_structure),
        ("Python Syntax", verify_python_syntax),
        ("YAML Syntax", verify_yaml_syntax),
        ("Imports", verify_imports),
        ("ONEX Compliance", verify_onex_compliance),
    ]

    results = []
    for check_name, check_func in checks:
        print(f"\n📋 {check_name}")
        print("-" * 40)
        result = check_func()
        results.append((check_name, result))

    print("\n" + "=" * 60)
    print("📊 VERIFICATION SUMMARY")
    print("=" * 60)

    passed_checks = sum(1 for _, result in results if result)
    total_checks = len(results)

    for check_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {check_name}")

    print(
        f"\nOverall: {passed_checks}/{total_checks} checks passed ({passed_checks/total_checks:.1%})"
    )

    if passed_checks == total_checks:
        print(
            "🎉 All checks passed! Memory Storage Effect Node is ready for integration."
        )
        return 0
    else:
        print("⚠️  Some checks failed. Please review the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
