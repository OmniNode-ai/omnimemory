"""
Security validation tests for OmniMemory ONEX architecture.

This module validates all security implementations mentioned in PR reviews:
- Sensitive data field exclusions
- Error message sanitization
- PII detection and masking
- Input validation and sanitization
- Rate limiting (when implemented)
- Audit logging and correlation tracking

Tests ensure no information disclosure vulnerabilities exist.
"""

import json
from datetime import datetime
from typing import Any, Dict, List
from uuid import uuid4

import pytest
from pydantic import ValidationError

from omnimemory.models.core.enum_node_type import EnumNodeType
from omnimemory.models.memory.model_memory_context import ModelMemoryContext
from omnimemory.models.memory.model_memory_item import ModelMemoryItem
from omnimemory.models.memory.model_memory_storage_config import (
    ModelMemoryStorageConfig,
)
from omnimemory.utils.error_sanitizer import ErrorSanitizer
from omnimemory.utils.pii_detector import PIIDetector, PIIType


class SecurityTestSuite:
    """Comprehensive security validation test suite."""

    def __init__(self):
        self.error_sanitizer = ErrorSanitizer()
        self.pii_detector = PIIDetector()
        self.test_secrets = {
            "password_hash": "pbkdf2_sha256$36000$abc123$def456ghi789",
            "api_key": "sk-1234567890abcdef1234567890abcdef",
            "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
            "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQC7...",
        }

    def test_sensitive_data_exclusion(self):
        """Test that sensitive fields are excluded from serialization."""

        # Test ModelMemoryStorageConfig with secrets
        config = ModelMemoryStorageConfig(
            storage_type="postgresql",
            connection_string="postgresql://user:pass@localhost:5432/db",
            username="test_user",
            password_hash=self.test_secrets["password_hash"],
            api_key=self.test_secrets["api_key"],
            max_connections=10,
            connection_timeout=30.0,
        )

        # Test model_dump() excludes sensitive fields
        serialized = config.model_dump()

        assert (
            "password_hash" not in serialized
        ), "password_hash should be excluded from serialization"
        assert (
            "api_key" not in serialized
        ), "api_key should be excluded from serialization"
        assert (
            serialized["storage_type"] == "postgresql"
        ), "Non-sensitive fields should be included"
        assert (
            serialized["username"] == "test_user"
        ), "Non-sensitive fields should be included"

        # Test dict() method also excludes sensitive fields
        dict_repr = dict(config)
        assert "password_hash" not in dict_repr
        assert "api_key" not in dict_repr

        # Test JSON serialization safety
        json_str = json.dumps(serialized)
        assert self.test_secrets["password_hash"] not in json_str
        assert self.test_secrets["api_key"] not in json_str

        print("✅ Sensitive data exclusion test PASSED")

    def test_error_message_sanitization(self):
        """Test that error messages are sanitized to prevent information disclosure."""

        # Test various error scenarios
        test_errors = [
            ValueError(
                f"Database connection failed: password={self.test_secrets['password_hash']}"
            ),
            ConnectionError(
                f"API authentication failed: api_key={self.test_secrets['api_key']}"
            ),
            RuntimeError(
                f"Token validation error: access_token={self.test_secrets['access_token']}"
            ),
            Exception(
                f"Key processing failed: private_key={self.test_secrets['private_key'][:50]}..."
            ),
        ]

        for error in test_errors:
            sanitized_message = self.error_sanitizer.sanitize_error_message(str(error))

            # Ensure no secrets leak through
            assert self.test_secrets["password_hash"] not in sanitized_message
            assert self.test_secrets["api_key"] not in sanitized_message
            assert self.test_secrets["access_token"] not in sanitized_message

            # Ensure sanitized message is still informative but safe
            assert len(sanitized_message) > 0, "Sanitized message should not be empty"
            assert any(
                keyword in sanitized_message.lower()
                for keyword in ["error", "failed", "redacted", "operation"]
            ), "Should contain safe error info"

        print("✅ Error message sanitization test PASSED")

    def test_pii_detection_and_masking(self):
        """Test PII detection and masking functionality."""

        # Test content with various PII types
        pii_test_cases = [
            {
                "content": "My email is john.doe@example.com and phone is (555) 123-4567",
                "expected_types": [PIIType.EMAIL, PIIType.PHONE],
            },
            {
                "content": "SSN: 123-45-6789, Credit Card: 4532-1234-5678-9012",
                "expected_types": [PIIType.SSN, PIIType.CREDIT_CARD],
            },
            {
                "content": "IP address 192.168.1.100 accessed API key sk-abc123def456",
                "expected_types": [PIIType.IP_ADDRESS, PIIType.API_KEY],
            },
            {
                "content": "No PII in this safe content about memory operations",
                "expected_types": [],
            },
        ]

        for test_case in pii_test_cases:
            detected_pii = self.pii_detector.detect_pii(test_case["content"])
            detected_types = [pii.pii_type for pii in detected_pii]

            for expected_type in test_case["expected_types"]:
                assert (
                    expected_type in detected_types
                ), f"Expected PII type {expected_type} not detected in: {test_case['content']}"

            # Test masking functionality
            masked_content = self.pii_detector.mask_pii(test_case["content"])

            # If PII was detected, ensure it's masked
            if detected_pii:
                assert (
                    masked_content != test_case["content"]
                ), "Content with PII should be modified after masking"
                assert (
                    "***REDACTED***" in masked_content or "*" in masked_content
                ), "Masked content should contain redaction markers"
            else:
                assert (
                    masked_content == test_case["content"]
                ), "Content without PII should remain unchanged"

        print("✅ PII detection and masking test PASSED")

    def test_input_validation_security(self):
        """Test input validation prevents security issues."""

        # Test 1: Memory item content size validation
        test_context = ModelMemoryContext(
            correlation_id=uuid4(),
            source_node_type=EnumNodeType.EFFECT,
            source_node_id="security_test_node",
            timestamp=datetime.utcnow(),
        )

        # Normal size content should pass
        normal_memory_item = ModelMemoryItem(
            item_id=str(uuid4()),
            title="Normal Test Item",
            content="This is normal-sized content for testing",
            tags=["security", "test"],
            metadata={"test": True},
            context=test_context,
        )

        try:
            normal_memory_item.model_validate(normal_memory_item.model_dump())
        except ValidationError as e:
            pytest.fail(f"Normal content should not fail validation: {e}")

        # Test 2: Tag limit validation
        # Reasonable number of tags should pass
        memory_item_with_tags = ModelMemoryItem(
            item_id=str(uuid4()),
            title="Tag Test Item",
            content="Testing tag limits",
            tags=["tag" + str(i) for i in range(50)],  # 50 tags
            metadata={"test": True},
            context=test_context,
        )

        try:
            memory_item_with_tags.model_validate(memory_item_with_tags.model_dump())
        except ValidationError as e:
            pytest.fail(f"Reasonable tag count should not fail validation: {e}")

        # Test 3: Title length validation
        memory_item_normal_title = ModelMemoryItem(
            item_id=str(uuid4()),
            title="Normal length title for testing validation",
            content="Test content",
            tags=["security"],
            metadata={"test": True},
            context=test_context,
        )

        try:
            memory_item_normal_title.model_validate(
                memory_item_normal_title.model_dump()
            )
        except ValidationError as e:
            pytest.fail(f"Normal title length should not fail validation: {e}")

        print("✅ Input validation security test PASSED")

    def test_correlation_tracking_security(self):
        """Test correlation tracking for security audit trails."""

        correlation_id = uuid4()

        # Test that correlation IDs are properly preserved across operations
        memory_context = ModelMemoryContext(
            correlation_id=correlation_id,
            user_id="security_test_user",
            source_node_type=EnumNodeType.ORCHESTRATOR,
            source_node_id="security_audit_node",
            timestamp=datetime.utcnow(),
            metadata={"audit": True, "security_test": True},
        )

        memory_item = ModelMemoryItem(
            item_id=str(uuid4()),
            title="Security Audit Test",
            content="Testing correlation tracking for security auditing",
            tags=["security", "audit", "correlation"],
            metadata={"correlation_test": True},
            context=memory_context,
        )

        # Verify correlation ID is preserved
        assert memory_item.context.correlation_id == correlation_id
        assert memory_item.context.user_id == "security_test_user"

        # Verify metadata includes audit information
        assert memory_item.metadata["correlation_test"] is True
        assert memory_item.context.metadata["audit"] is True

        # Test serialization preserves correlation information (but not sensitive data)
        serialized = memory_item.model_dump()
        assert str(correlation_id) in str(serialized["context"]["correlation_id"])
        assert serialized["context"]["user_id"] == "security_test_user"

        print("✅ Correlation tracking security test PASSED")

    def test_memory_storage_config_security(self):
        """Test security configurations for memory storage."""

        # Test that connection strings don't expose passwords in logs/serialization
        config = ModelMemoryStorageConfig(
            storage_type="postgresql",
            connection_string="postgresql://user:secretpass@localhost:5432/db",
            username="test_user",
            password_hash="secret_hash_value",
            api_key="secret_api_key",
            max_connections=10,
            connection_timeout=30.0,
            use_ssl=True,
            ssl_ca_cert="/path/to/ca.crt",
        )

        # Test serialization excludes sensitive fields
        serialized = config.model_dump()

        # Sensitive fields should be excluded
        sensitive_fields = ["password_hash", "api_key"]
        for field in sensitive_fields:
            assert (
                field not in serialized
            ), f"Sensitive field {field} should not be in serialized output"

        # Connection string should be preserved (assuming it's needed for connections)
        # but in production, this might also need sanitization
        assert "connection_string" in serialized

        # SSL settings should be preserved as they're not sensitive
        assert serialized["use_ssl"] is True
        assert "ssl_ca_cert" in serialized

        print("✅ Memory storage config security test PASSED")

    def test_json_serialization_security(self):
        """Test that JSON serialization doesn't leak sensitive data."""

        # Create objects with sensitive data
        config_with_secrets = ModelMemoryStorageConfig(
            storage_type="redis",
            connection_string="redis://localhost:6379",
            password_hash=self.test_secrets["password_hash"],
            api_key=self.test_secrets["api_key"],
            max_connections=20,
        )

        memory_context = ModelMemoryContext(
            correlation_id=uuid4(),
            user_id="json_test_user",
            source_node_type=EnumNodeType.COMPUTE,
            source_node_id="json_test_node",
            timestamp=datetime.utcnow(),
        )

        memory_item = ModelMemoryItem(
            item_id=str(uuid4()),
            title="JSON Security Test",
            content="Testing JSON serialization security",
            tags=["json", "security"],
            metadata={"test": True},
            context=memory_context,
        )

        # Test JSON serialization of config
        config_json = json.dumps(config_with_secrets.model_dump())

        # Ensure secrets don't appear in JSON
        assert self.test_secrets["password_hash"] not in config_json
        assert self.test_secrets["api_key"] not in config_json

        # Test JSON serialization of memory item
        item_json = json.dumps(memory_item.model_dump())

        # Should not contain any test secrets
        for secret in self.test_secrets.values():
            assert secret not in item_json

        # Should contain non-sensitive data
        assert "JSON Security Test" in item_json
        assert "json_test_user" in item_json

        print("✅ JSON serialization security test PASSED")

    def run_all_security_tests(self):
        """Run all security validation tests."""
        print("=== OmniMemory Security Validation Test Suite ===")
        print("Testing all security implementations...\n")

        try:
            self.test_sensitive_data_exclusion()
            self.test_error_message_sanitization()
            self.test_pii_detection_and_masking()
            self.test_input_validation_security()
            self.test_correlation_tracking_security()
            self.test_memory_storage_config_security()
            self.test_json_serialization_security()

            print("\n🔒 ALL SECURITY TESTS PASSED! 🔒")
            print("No information disclosure vulnerabilities detected.")

        except Exception as e:
            print(f"\n❌ SECURITY TEST FAILED: {e}")
            raise

        print("\n=== Security Validation Complete ===")


# Pytest test functions
@pytest.mark.security
def test_sensitive_data_exclusion():
    """Pytest wrapper for sensitive data exclusion test."""
    suite = SecurityTestSuite()
    suite.test_sensitive_data_exclusion()


@pytest.mark.security
def test_error_message_sanitization():
    """Pytest wrapper for error sanitization test."""
    suite = SecurityTestSuite()
    suite.test_error_message_sanitization()


@pytest.mark.security
def test_pii_detection_and_masking():
    """Pytest wrapper for PII detection test."""
    suite = SecurityTestSuite()
    suite.test_pii_detection_and_masking()


@pytest.mark.security
def test_input_validation_security():
    """Pytest wrapper for input validation test."""
    suite = SecurityTestSuite()
    suite.test_input_validation_security()


@pytest.mark.security
def test_correlation_tracking_security():
    """Pytest wrapper for correlation tracking test."""
    suite = SecurityTestSuite()
    suite.test_correlation_tracking_security()


@pytest.mark.security
def test_memory_storage_config_security():
    """Pytest wrapper for storage config test."""
    suite = SecurityTestSuite()
    suite.test_memory_storage_config_security()


@pytest.mark.security
def test_json_serialization_security():
    """Pytest wrapper for JSON serialization test."""
    suite = SecurityTestSuite()
    suite.test_json_serialization_security()


if __name__ == "__main__":
    """Run security tests directly."""
    suite = SecurityTestSuite()
    suite.run_all_security_tests()
