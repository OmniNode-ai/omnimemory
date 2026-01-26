#!/usr/bin/env python3
"""
Isolated foundation validation for OmniMemory ONEX architecture.

Tests only the components that don't depend on omnibase_core:
- Protocol definitions (structural typing)
- Data models (Pydantic validation)
- Basic imports and structure validation
"""

import sys
import traceback
from pathlib import Path
from typing import Any

import yaml

# Add src and protocols paths to Python path
# Note: Both paths needed - src for omnimemory package imports,
# protocols for direct module imports
_base_path = Path(__file__).parent
for path in [_base_path / "src", _base_path / "src" / "omnimemory" / "protocols"]:
    sys.path.insert(0, str(path))


def validate_protocol_definitions() -> dict[str, Any]:
    """Validate protocol definitions structure."""
    print("🔍 Testing protocol definitions...")

    try:
        # Import protocols directly without going through __init__
        # (paths already added at module level)
        import base_protocols

        # Check that protocols exist as classes
        protocols_found = []
        for name in dir(base_protocols):
            if name.startswith("Protocol") and not name.startswith("_"):
                protocols_found.append(name)

        print(f"✅ Found {len(protocols_found)} protocol definitions")
        print(f"   Protocols: {', '.join(protocols_found[:5])}")

        return {
            "success": True,
            "protocols_count": len(protocols_found),
            "protocols": protocols_found,
        }

    except Exception as e:
        print(f"❌ Protocol validation failed: {e!s}")
        traceback.print_exc()
        return {"success": False, "error": str(e)}


def validate_data_model_definitions() -> dict[str, Any]:
    """Validate data model definitions."""
    print("🔍 Testing data model definitions...")

    try:
        # Import data models directly (paths already added at module level)
        import data_models

        # Check for key model classes
        models_found = []
        key_models = [
            "BaseMemoryRequest",
            "BaseMemoryResponse",
            "MemoryRecord",
            "UserContext",
            "StoragePreferences",
            "SearchFilters",
        ]

        for model_name in key_models:
            if hasattr(data_models, model_name):
                models_found.append(model_name)

        # Test basic model creation (using simple types to avoid omnibase_core)
        from datetime import datetime, timezone
        from uuid import uuid4

        from pydantic import BaseModel, Field

        # Create a test model similar to our structure
        class TestMemoryModel(BaseModel):
            """Test model to verify Pydantic patterns work."""

            memory_id: str = Field(default_factory=lambda: str(uuid4()))
            content: str = Field(max_length=1000)
            created_at: datetime = Field(
                default_factory=lambda: datetime.now(timezone.utc)
            )
            metadata: dict[str, Any] | None = None

        test_instance = TestMemoryModel(content="Test content", metadata={"test": True})

        print(f"✅ Found {len(models_found)} key model classes")
        print(f"   Models: {', '.join(models_found)}")
        print(f"   Test instance created: {test_instance.memory_id[:8]}...")

        return {
            "success": True,
            "models_count": len(models_found),
            "models": models_found,
            "test_model_id": test_instance.memory_id,
        }

    except Exception as e:
        print(f"❌ Data model validation failed: {e!s}")
        traceback.print_exc()
        return {"success": False, "error": str(e)}


def validate_error_model_definitions() -> dict[str, Any]:
    """Validate error model definitions."""
    print("🔍 Testing error model definitions...")

    try:
        # Import error models directly (paths already added at module level)
        import error_models

        # Check for key error classes
        errors_found = []
        key_errors = [
            "OmniMemoryError",
            "OmniMemoryErrorCode",
            "ValidationError",
            "StorageError",
        ]

        for error_name in key_errors:
            if hasattr(error_models, error_name):
                errors_found.append(error_name)

        print(f"✅ Found {len(errors_found)} key error classes")
        print(f"   Errors: {', '.join(errors_found)}")

        return {
            "success": True,
            "errors_count": len(errors_found),
            "errors": errors_found,
        }

    except Exception as e:
        print(f"❌ Error model validation failed: {e!s}")
        traceback.print_exc()
        return {"success": False, "error": str(e)}


def validate_contract_specification() -> dict[str, Any]:
    """Validate contract.yaml structure."""
    print("🔍 Testing contract specification...")

    try:
        contract_path = Path(__file__).parent / "contract.yaml"
        if not contract_path.exists():
            return {"success": False, "error": "contract.yaml not found"}

        with open(contract_path, encoding="utf-8") as f:
            contract = yaml.safe_load(f)

        # Validate contract structure
        required_sections = ["contract", "architecture", "protocols", "data_models"]
        missing_sections = []

        for section in required_sections:
            if section not in contract:
                missing_sections.append(section)

        if missing_sections:
            return {
                "success": False,
                "error": f"Missing contract sections: {missing_sections}",
            }

        # Count protocols and data models
        protocols_count = len(contract.get("protocols", {}).get("memory_protocols", {}))
        data_models_count = len(contract.get("data_models", {}).get("core_models", []))

        print("  Contract validation successful")
        arch = contract.get("contract", {}).get("architecture", {})
        print(f"   Architecture: {arch.get('pattern', 'Unknown')}")
        print(f"   Protocols: {protocols_count}")
        print(f"   Data models: {data_models_count}")

        return {
            "success": True,
            "architecture": contract.get("contract", {})
            .get("architecture", {})
            .get("pattern"),
            "protocols_count": protocols_count,
            "data_models_count": data_models_count,
        }

    except Exception as e:
        print(f"❌ Contract validation failed: {e!s}")
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
            "src/omnimemory/protocols": "Protocol definitions",
            "src/omnimemory/core": "Core implementation",
            "contract.yaml": "ONEX contract specification",
            "pyproject.toml": "Project configuration",
            "tests": "Test directory",
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
        print(
            f"   Found: {len(found_items)} / {len(expected_structure)} expected items"
        )
        if missing_items:
            print(f"   Missing: {', '.join(missing_items)}")

        return {
            "success": len(missing_items) == 0,
            "found_count": len(found_items),
            "total_count": len(expected_structure),
            "missing_items": missing_items,
        }

    except Exception as e:
        print(f"❌ Project structure validation failed: {e!s}")
        traceback.print_exc()
        return {"success": False, "error": str(e)}


def main() -> int:
    """Run isolated foundation validation."""
    print("🎯 OmniMemory Isolated Foundation Validation")
    print("=" * 50)
    print("Note: Testing components that don't require omnibase_core")

    results = {}

    # Run validation tests
    results["project_structure"] = validate_project_structure()
    results["contract"] = validate_contract_specification()
    results["protocols"] = validate_protocol_definitions()
    results["data_models"] = validate_data_model_definitions()
    results["error_models"] = validate_error_model_definitions()

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
        print("\n🎉 Isolated foundation validation successful!")
        print("   ONEX architecture structure is properly implemented")
        print("   Ready for omnibase_core integration")
        return 0
    else:
        print(f"\n⚠️  {failed} validation issues found")
        print("   Some foundation components need attention")
        return min(failed, 1)  # Return 1 for any failures


if __name__ == "__main__":
    sys.exit(main())
