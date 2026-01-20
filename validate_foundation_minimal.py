#!/usr/bin/env python3
"""
Minimal foundation validation for OmniMemory ONEX architecture.

Tests only the basic structure and contract that can be validated
without omnibase_core dependencies.
"""

import sys
import traceback
from pathlib import Path
from typing import Any

import yaml


def validate_contract_specification() -> dict[str, Any]:
    """Validate contract.yaml structure."""
    print("🔍 Testing contract specification...")

    try:
        contract_path = Path(__file__).parent / "contract.yaml"
        if not contract_path.exists():
            return {"success": False, "error": "contract.yaml not found"}

        with open(contract_path, "r") as f:
            contract = yaml.safe_load(f)

        # Validate contract structure
        required_sections = ["contract", "protocols", "schemas"]
        missing_sections = []

        for section in required_sections:
            if section not in contract:
                missing_sections.append(section)

        if missing_sections:
            return {
                "success": False,
                "error": f"Missing contract sections: {missing_sections}",
            }

        # Validate ONEX 4-node architecture (nested under contract)
        architecture = contract.get("contract", {}).get("architecture", {})
        if architecture.get("pattern") != "onex_4_node":
            pattern = architecture.get("pattern")
            return {
                "success": False,
                "error": f"Expected onex_4_node pattern, got: {pattern}",
            }

        nodes = architecture.get("nodes", {})
        expected_nodes = ["effect", "compute", "reducer", "orchestrator"]
        missing_nodes = []

        for node in expected_nodes:
            if node not in nodes:
                missing_nodes.append(node)

        if missing_nodes:
            return {"success": False, "error": f"Missing ONEX nodes: {missing_nodes}"}

        # Count protocols and data models
        protocols = contract.get("protocols", {})
        protocol_sections = [
            "memory_protocols",
            "effect_protocols",
            "compute_protocols",
            "reducer_protocols",
            "orchestrator_protocols",
        ]
        total_protocols = sum(
            len(protocols.get(section, {})) for section in protocol_sections
        )

        schemas_count = len(contract.get("schemas", {}))

        print("  Contract validation successful")
        print(f"   Architecture: {architecture.get('pattern')}")
        print(f"   Nodes: {', '.join(expected_nodes)}")
        print(f"   Protocols: {total_protocols}")
        print(f"   Schemas: {schemas_count}")

        return {
            "success": True,
            "architecture": architecture.get("pattern"),
            "nodes": expected_nodes,
            "protocols_count": total_protocols,
            "schemas_count": schemas_count,
        }

    except Exception as e:
        print(f"❌ Contract validation failed: {str(e)}")
        traceback.print_exc()
        return {"success": False, "error": str(e)}


def validate_project_structure() -> dict[str, Any]:
    """Validate overall project structure."""
    print("🔍 Testing project structure...")

    try:
        base_path = Path(__file__).parent

        # Check for expected directories and files
        expected_structure = {
            "src/omnimemory": "Main package directory",
            "src/omnimemory/__init__.py": "Package initialization",
            "src/omnimemory/protocols": "Protocol definitions",
            "src/omnimemory/protocols/__init__.py": "Protocol package",
            "src/omnimemory/protocols/base_protocols.py": "Base protocol definitions",
            "src/omnimemory/protocols/data_models.py": "Data model definitions",
            "src/omnimemory/protocols/error_models.py": "Error model definitions",
            "src/omnimemory/core": "Core implementation",
            "src/omnimemory/core/__init__.py": "Core package",
            "src/omnimemory/core/container.py": "ONEX Container",
            "src/omnimemory/core/service_providers.py": "Service providers",
            "src/omnimemory/core/base_implementations.py": "Base services",
            "contract.yaml": "ONEX contract specification",
            "pyproject.toml": "Project configuration",
            "tests": "Test directory",
            "tests/test_foundation.py": "Foundation tests",
        }

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
            "total_count": len(expected_structure),
            "missing_items": missing_items,
        }

    except Exception as e:
        print(f"❌ Project structure validation failed: {str(e)}")
        traceback.print_exc()
        return {"success": False, "error": str(e)}


def validate_file_syntax() -> dict[str, Any]:
    """Validate Python file syntax without importing."""
    print("🔍 Testing file syntax...")

    try:
        base_path = Path(__file__).parent / "src" / "omnimemory"

        python_files = []
        syntax_errors = []

        # Find all Python files
        for py_file in base_path.rglob("*.py"):
            python_files.append(py_file)

            try:
                with open(py_file, "r", encoding="utf-8") as f:
                    content = f.read()

                # Try to compile the syntax (doesn't import, just checks syntax)
                compile(content, str(py_file), "exec")

            except SyntaxError as e:
                syntax_errors.append(f"{py_file.relative_to(base_path)}: {e}")
            except Exception as e:
                # Other errors (like encoding) are also noteworthy
                syntax_errors.append(f"{py_file.relative_to(base_path)}: {e}")

        print("  File syntax validation")
        print(f"   Checked: {len(python_files)} Python files")
        if syntax_errors:
            print(f"   Syntax errors: {len(syntax_errors)}")
            for error in syntax_errors[:3]:  # Show first 3 errors
                print(f"      {error}")

        return {
            "success": len(syntax_errors) == 0,
            "files_checked": len(python_files),
            "syntax_errors": syntax_errors,
        }

    except Exception as e:
        print(f"❌ File syntax validation failed: {str(e)}")
        traceback.print_exc()
        return {"success": False, "error": str(e)}


def validate_pyproject_configuration() -> dict[str, Any]:
    """Validate pyproject.toml configuration."""
    print("🔍 Testing pyproject.toml configuration...")

    try:
        import tomllib

        pyproject_path = Path(__file__).parent / "pyproject.toml"
        if not pyproject_path.exists():
            return {"success": False, "error": "pyproject.toml not found"}

        with open(pyproject_path, "rb") as f:
            pyproject = tomllib.load(f)

        # Validate key sections
        tool_poetry = pyproject.get("tool", {}).get("poetry", {})

        required_fields = ["name", "version", "description", "authors"]
        missing_fields = []

        for field in required_fields:
            if field not in tool_poetry:
                missing_fields.append(field)

        if missing_fields:
            return {
                "success": False,
                "error": f"Missing pyproject.toml fields: {missing_fields}",
            }

        # Check dependencies
        dependencies = tool_poetry.get("dependencies", {})
        key_deps = ["python", "pydantic", "fastapi", "omnibase_spi", "omnibase_core"]
        found_deps = []

        for dep in key_deps:
            if dep in dependencies:
                found_deps.append(dep)

        print("  pyproject.toml validation successful")
        name = tool_poetry.get("name")
        version = tool_poetry.get("version")
        print(f"   Package: {name} v{version}")
        deps_total = len(dependencies)
        key_found = f"{len(found_deps)}/{len(key_deps)}"
        print(f"   Dependencies: {deps_total} total, {key_found} key deps")

        return {
            "success": True,
            "package_name": tool_poetry.get("name"),
            "version": tool_poetry.get("version"),
            "dependencies_count": len(dependencies),
            "key_deps_found": found_deps,
        }

    except Exception as e:
        print(f"❌ pyproject.toml validation failed: {str(e)}")
        traceback.print_exc()
        return {"success": False, "error": str(e)}


def main() -> int:
    """Run minimal foundation validation."""
    print("🎯 OmniMemory Minimal Foundation Validation")
    print("=" * 50)
    print("Note: Testing structure and syntax without omnibase_core imports")

    results = {}

    # Run validation tests
    results["project_structure"] = validate_project_structure()
    results["pyproject_config"] = validate_pyproject_configuration()
    results["contract"] = validate_contract_specification()
    results["file_syntax"] = validate_file_syntax()

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

    # Provide summary assessment
    if failed == 0:
        print("\n🎉 Foundation validation successful!")
        print("   ✅ Project structure is complete")
        print("   ✅ Contract specification follows ONEX 4-node pattern")
        print("   ✅ All Python files have valid syntax")
        print("   ✅ Configuration is properly set up")
        print("\n📋 Next Steps:")
        print("   1. Resolve omnibase_core dependency (Python 3.12+ requirement)")
        print("   2. Run full integration tests with omnibase_core")
        print("   3. Implement service implementations")
        print("   4. Add event bus integration framework")
        return 0
    else:
        print(f"\n⚠️  {failed} structural issues found")
        print("   Foundation architecture needs attention before proceeding")
        return 1


if __name__ == "__main__":
    sys.exit(main())
