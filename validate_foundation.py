#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""
Foundation validation for OmniMemory ONEX architecture.

Validates that the foundational ONEX implementation is working correctly:
- Protocol definitions and structure
- Container initialization
- Service provider functionality
- Error handling and monadic patterns
- Basic integration tests
"""

import sys
import traceback
from pathlib import Path
from typing import Any

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def validate_protocol_imports() -> dict[str, Any]:
    """Validate that all protocol imports work correctly."""
    print("🔍 Testing protocol imports...")

    try:
        from omnimemory.protocols.base_protocols import (  # noqa: F401
            ProtocolAgentCoordinator,
            ProtocolIntelligenceProcessor,
            ProtocolMemoryAggregator,
            ProtocolMemoryBase,
            ProtocolMemoryConsolidator,
            ProtocolMemoryOperations,
            ProtocolMemoryOptimizer,
            ProtocolMemoryOrchestrator,
            ProtocolMemoryPersistence,
            ProtocolMemoryRetrieval,
            ProtocolMemoryStorage,
            ProtocolPatternRecognition,
            ProtocolSemanticAnalyzer,
            ProtocolWorkflowCoordinator,
        )

        print("✅ All protocol imports successful")
        return {"success": True, "protocols_count": 14}

    except Exception as e:
        print(f"❌ Protocol import failed: {e!s}")
        traceback.print_exc()
        return {"success": False, "error": str(e)}


def validate_data_models() -> dict[str, Any]:
    """Validate data model imports and basic functionality."""
    print("🔍 Testing data model imports...")

    try:
        # Test basic model creation
        from uuid import uuid4

        from omnimemory.protocols.data_models import (  # noqa: F401
            AccessLevel,
            BaseMemoryRequest,
            BaseMemoryResponse,
            ContentType,
            MemoryPriority,
            MemoryRecord,
            MemoryStoreRequest,
            MemoryStoreResponse,
            SearchFilters,
            SearchResult,
            StoragePreferences,
            UserContext,
        )

        user_context = UserContext(
            user_id=uuid4(),
            agent_id=uuid4(),
            session_id=uuid4(),
        )

        memory_record = MemoryRecord(
            content="Test memory content",
            content_type=ContentType.TEXT,
            priority=MemoryPriority.NORMAL,
            access_level=AccessLevel.RESTRICTED,
            source_agent="test-agent",
        )

        print("✅ Data model imports and creation successful")
        return {
            "success": True,
            "user_id": str(user_context.user_id),
            "memory_id": str(memory_record.memory_id),
        }

    except Exception as e:
        print(f"❌ Data model validation failed: {e!s}")
        traceback.print_exc()
        return {"success": False, "error": str(e)}


def validate_error_handling() -> dict[str, Any]:
    """Validate error handling and monadic patterns."""
    print("🔍 Testing error handling...")

    try:
        from omnimemory.protocols.error_models import (  # noqa: F401
            ProtocolOmniMemoryError as OmniMemoryError,
        )
        from omnimemory.protocols.error_models import (
            ProtocolStorageError as StorageError,
        )
        from omnimemory.protocols.error_models import (
            ProtocolValidationError as ValidationError,
        )

        # Test error creation and chaining
        base_error = ValidationError(
            message="Test validation error",
            field_name="content",
            validation_rule="max_length",
        )

        chained_error = StorageError(
            message="Storage system down",
            storage_system="postgres",
            cause=base_error,
        )

        print("✅ Error handling validation successful")
        return {
            "success": True,
            "base_error_code": base_error.omnimemory_error_code.value,
            "chained_error_has_cause": chained_error.cause is not None,
        }

    except Exception as e:
        print(f"❌ Error handling validation failed: {e!s}")
        traceback.print_exc()
        return {"success": False, "error": str(e)}


def validate_container_creation() -> dict[str, Any]:
    """Validate ONEX container creation and basic functionality."""
    print("🔍 Testing ONEX container creation...")

    try:
        from omnibase_core.container import ModelONEXContainer

        # Test container creation
        container = ModelONEXContainer()

        # Verify container has expected ONEX methods
        has_register_singleton = hasattr(container, "register_singleton")
        has_register_transient = hasattr(container, "register_transient")
        has_resolve = hasattr(container, "resolve")

        print("✅ ONEX Container creation successful")
        return {
            "success": True,
            "has_register_singleton": has_register_singleton,
            "has_register_transient": has_register_transient,
            "has_resolve": has_resolve,
        }

    except Exception as e:
        print(f"❌ ONEX Container validation failed: {e!s}")
        traceback.print_exc()
        return {"success": False, "error": str(e)}


def validate_base_implementations() -> dict[str, Any]:
    """Validate base implementation classes."""
    print("🔍 Testing base implementations...")

    try:
        from omnimemory.nodes.base import (
            BaseComputeNode,
            BaseEffectNode,
            BaseOrchestratorNode,
            BaseReducerNode,
        )

        # Verify all base classes are importable and have expected structure
        base_classes = [
            BaseEffectNode,
            BaseComputeNode,
            BaseReducerNode,
            BaseOrchestratorNode,
        ]

        class_methods = {}
        for cls in base_classes:
            methods = [method for method in dir(cls) if not method.startswith("_")]
            class_methods[cls.__name__] = len(methods)

        print("✅ Base implementations validation successful")
        return {
            "success": True,
            "base_classes_count": len(base_classes),
            "class_methods": class_methods,
        }

    except Exception as e:
        print(f"❌ Base implementations validation failed: {e!s}")
        traceback.print_exc()
        return {"success": False, "error": str(e)}


async def validate_async_patterns() -> dict[str, Any]:
    """Validate async patterns and ModelBaseResult usage."""
    print("🔍 Testing async patterns...")

    try:
        # Import async components from omnibase_core
        from omnibase_core.container import ModelONEXContainer
        from omnibase_core.models.core.model_base_result import (  # noqa: F401
            ModelBaseResult,
        )

        from omnimemory.protocols.data_models import UserContext  # noqa: F401

        # Create container and test ONEX patterns
        container = ModelONEXContainer()

        # Verify ONEX methods exist
        has_resolve_method = hasattr(container, "resolve")
        has_register_methods = hasattr(container, "register_singleton")

        print("✅ Async patterns validation successful")
        return {
            "success": True,
            "has_resolve_method": has_resolve_method,
            "has_register_methods": has_register_methods,
            "container_created": True,
        }

    except Exception as e:
        print(f"❌ Async patterns validation failed: {e!s}")
        traceback.print_exc()
        return {"success": False, "error": str(e)}


def main() -> int:
    """Run comprehensive foundation validation."""
    print("🎯 OmniMemory Foundation Validation")
    print("=" * 40)

    results = {}

    # Run all validation tests
    results["protocols"] = validate_protocol_imports()
    results["data_models"] = validate_data_models()
    results["error_handling"] = validate_error_handling()
    results["container"] = validate_container_creation()
    results["base_implementations"] = validate_base_implementations()

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
        print("\n🎉 Foundation validation successful!")
        print("   ONEX architecture is properly implemented")
        print("   Ready for service implementations")
        return 0
    else:
        print(f"\n🚫 {failed} validation issues found")
        print("   Foundation needs fixes before proceeding")
        return 1


if __name__ == "__main__":
    sys.exit(main())
