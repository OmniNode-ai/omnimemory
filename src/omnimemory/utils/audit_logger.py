"""
Audit logging utility for sensitive operations tracking.

Provides comprehensive audit logging for security-sensitive operations
including memory access, configuration changes, and PII detection events.
"""

import atexit
import json
import logging
import threading
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from ..models.foundation.model_audit_metadata import (
    AuditEventDetails,
    ResourceUsageMetadata,
)


class AuditEventType(str, Enum):
    """Types of auditable events."""

    MEMORY_STORE = "memory_store"
    MEMORY_RETRIEVE = "memory_retrieve"
    MEMORY_DELETE = "memory_delete"
    CONFIG_CHANGE = "config_change"
    PII_DETECTED = "pii_detected"
    PII_SANITIZED = "pii_sanitized"
    AUTH_SUCCESS = "auth_success"
    AUTH_FAILURE = "auth_failure"
    ACCESS_DENIED = "access_denied"
    SYSTEM_ERROR = "system_error"
    SECURITY_VIOLATION = "security_violation"


class AuditSeverity(str, Enum):
    """Severity levels for audit events."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AuditEvent(BaseModel):
    """Structured audit event model."""

    # Event identification
    event_id: str = Field(description="Unique event identifier")
    timestamp: datetime = Field(description="Event timestamp in UTC")
    event_type: AuditEventType = Field(description="Type of event")
    severity: AuditSeverity = Field(description="Event severity level")

    # Context information
    operation: str = Field(description="Operation being performed")
    component: str = Field(description="Component generating the event")
    user_context: str | None = Field(
        default=None, description="User context if available"
    )
    session_id: str | None = Field(default=None, description="Session identifier")

    # Event details
    message: str = Field(description="Human-readable event description")
    details: AuditEventDetails = Field(
        default_factory=lambda: AuditEventDetails(),
        description="Additional event details",
    )

    # Security context
    source_ip: str | None = Field(default=None, description="Source IP address")
    user_agent: str | None = Field(default=None, description="User agent string")

    # Performance data
    duration_ms: float | None = Field(default=None, description="Operation duration")
    resource_usage: ResourceUsageMetadata | None = Field(
        default=None, description="Resource usage metrics"
    )

    # Compliance tracking
    data_classification: str | None = Field(
        default=None, description="Data classification level"
    )
    pii_detected: bool = Field(default=False, description="Whether PII was detected")
    sanitized: bool = Field(default=False, description="Whether data was sanitized")

    model_config = ConfigDict(
        extra="forbid",
        validate_default=True,
        str_strip_whitespace=True,
        ser_json_timedelta="iso8601",
    )


class AuditLogger:
    """Advanced audit logger with structured logging and security features."""

    def __init__(
        self,
        log_file: Path | None = None,
        console_output: bool = True,
        json_format: bool = True,
    ):
        """
        Initialize audit logger.

        Args:
            log_file: Path to audit log file (None for memory-only)
            console_output: Whether to output to console
            json_format: Whether to use JSON format for logs
        """
        self.log_file = log_file
        self.console_output = console_output
        self.json_format = json_format

        # Setup Python logger
        self.logger = logging.getLogger("omnimemory.audit")
        self.logger.setLevel(logging.INFO)

        # Clear existing handlers
        self.logger.handlers = []

        # Add file handler if specified
        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)
            if json_format:
                file_handler.setFormatter(self._json_formatter())
            else:
                file_handler.setFormatter(self._text_formatter())
            self.logger.addHandler(file_handler)

        # Add console handler if specified
        if console_output:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(
                logging.WARNING
            )  # Only show warnings and above on console
            console_handler.setFormatter(self._text_formatter())
            self.logger.addHandler(console_handler)

    def _json_formatter(self) -> logging.Formatter:
        """Create JSON log formatter."""

        class JSONFormatter(logging.Formatter):
            def format(self, record: logging.LogRecord) -> str:
                log_data: dict[str, object] = {
                    "timestamp": datetime.fromtimestamp(
                        record.created, tz=timezone.utc
                    ).isoformat(),
                    "level": record.levelname,
                    "logger": record.name,
                    "message": record.getMessage(),
                }

                # Add audit event data if present
                if hasattr(record, "audit_event"):
                    log_data["audit_event"] = record.audit_event

                return json.dumps(log_data)

        return JSONFormatter()

    def _text_formatter(self) -> logging.Formatter:
        """Create human-readable log formatter."""
        return logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    def log_event(self, event: AuditEvent) -> None:
        """
        Log an audit event.

        Args:
            event: The audit event to log
        """
        # Create log record with event data
        log_level = self._severity_to_log_level(event.severity)

        # Create log message
        message = f"[{event.event_type.value}] {event.message}"

        # Create log record
        record = self.logger.makeRecord(
            name=self.logger.name,
            level=log_level,
            fn="",
            lno=0,
            msg=message,
            args=(),
            exc_info=None,
        )

        # Attach audit event data (mode="json" ensures datetime serialization)
        record.audit_event = event.model_dump(mode="json")

        # Log the event
        self.logger.handle(record)

    def _severity_to_log_level(self, severity: AuditSeverity) -> int:
        """Convert audit severity to Python log level."""
        mapping = {
            AuditSeverity.LOW: logging.INFO,
            AuditSeverity.MEDIUM: logging.WARNING,
            AuditSeverity.HIGH: logging.ERROR,
            AuditSeverity.CRITICAL: logging.CRITICAL,
        }
        return mapping.get(severity, logging.INFO)

    def log_memory_operation(
        self,
        operation_type: str,
        memory_id: str,
        success: bool,
        duration_ms: float | None = None,
        details: AuditEventDetails | None = None,
        user_context: str | None = None,
    ) -> None:
        """Log a memory operation event."""
        event_type_map = {
            "store": AuditEventType.MEMORY_STORE,
            "retrieve": AuditEventType.MEMORY_RETRIEVE,
            "delete": AuditEventType.MEMORY_DELETE,
        }

        event = AuditEvent(
            event_id=self._generate_event_id(),
            timestamp=datetime.now(timezone.utc),
            event_type=event_type_map.get(operation_type, AuditEventType.MEMORY_STORE),
            severity=AuditSeverity.LOW if success else AuditSeverity.HIGH,
            operation=f"memory_{operation_type}",
            component="memory_manager",
            message=(
                f"Memory {operation_type} "
                f"{'succeeded' if success else 'failed'} for ID: {memory_id}"
            ),
            details=details
            or AuditEventDetails(operation_type=f"memory_{operation_type}"),
            duration_ms=duration_ms,
            user_context=user_context,
        )

        self.log_event(event)

    def log_pii_detection(
        self,
        pii_types: list[str],
        content_length: int,
        sanitized: bool = False,
        details: AuditEventDetails | None = None,
    ) -> None:
        """Log PII detection event."""
        severity = AuditSeverity.HIGH if pii_types else AuditSeverity.LOW

        # Build details model, merging with any provided details
        pii_details = details or AuditEventDetails(operation_type="pii_scan")
        # Store PII-specific info in request_parameters
        pii_details = AuditEventDetails(
            operation_type=pii_details.operation_type or "pii_scan",
            resource_id=pii_details.resource_id,
            resource_type=pii_details.resource_type,
            request_parameters={
                "pii_types_detected": ",".join(pii_types) if pii_types else "",
                "content_length": str(content_length),
                "sanitized": str(sanitized),
            },
        )

        event = AuditEvent(
            event_id=self._generate_event_id(),
            timestamp=datetime.now(timezone.utc),
            event_type=(
                AuditEventType.PII_DETECTED
                if pii_types
                else AuditEventType.PII_SANITIZED
            ),
            severity=severity,
            operation="pii_scan",
            component="pii_detector",
            message=f"PII detection scan found {len(pii_types)} PII types in {content_length} chars",
            details=pii_details,
            pii_detected=bool(pii_types),
            sanitized=sanitized,
        )

        self.log_event(event)

    def log_security_violation(
        self,
        violation_type: str,
        description: str,
        source_ip: str | None = None,
        user_context: str | None = None,
        details: AuditEventDetails | None = None,
    ) -> None:
        """Log security violation event."""
        event = AuditEvent(
            event_id=self._generate_event_id(),
            timestamp=datetime.now(timezone.utc),
            event_type=AuditEventType.SECURITY_VIOLATION,
            severity=AuditSeverity.CRITICAL,
            operation="security_check",
            component="security_monitor",
            message=f"Security violation: {violation_type} - {description}",
            details=details or AuditEventDetails(operation_type="security_check"),
            source_ip=source_ip,
            user_context=user_context,
        )

        self.log_event(event)

    def log_config_change(
        self,
        config_key: str,
        old_value: str | None,
        new_value: str,
        user_context: str | None = None,
        details: AuditEventDetails | None = None,
    ) -> None:
        """Log configuration change event."""
        # Build details model with config change info
        redacted_old = (
            "***REDACTED***"
            if old_value and "secret" in config_key.lower()
            else old_value
        )
        redacted_new = "***REDACTED***" if "secret" in config_key.lower() else new_value

        config_details = AuditEventDetails(
            operation_type="config_update",
            resource_id=config_key,
            resource_type="configuration",
            old_value=redacted_old,
            new_value=redacted_new,
        )

        # Merge with any provided details
        if details:
            config_details = AuditEventDetails(
                operation_type=details.operation_type or config_details.operation_type,
                resource_id=details.resource_id or config_details.resource_id,
                resource_type=details.resource_type or config_details.resource_type,
                old_value=config_details.old_value,
                new_value=config_details.new_value,
                request_parameters=details.request_parameters,
                response_status=details.response_status,
                error_details=details.error_details,
                ip_address=details.ip_address,
                user_agent=details.user_agent,
            )

        event = AuditEvent(
            event_id=self._generate_event_id(),
            timestamp=datetime.now(timezone.utc),
            event_type=AuditEventType.CONFIG_CHANGE,
            severity=AuditSeverity.MEDIUM,
            operation="config_update",
            component="config_manager",
            message=f"Configuration changed: {config_key}",
            details=config_details,
            user_context=user_context,
        )

        self.log_event(event)

    def _generate_event_id(self) -> str:
        """Generate unique event ID."""
        import uuid

        return str(uuid.uuid4())


class _AuditLoggerState:
    """Singleton state manager for the audit logger.

    Manages the global audit logger instance using class-level attributes
    to avoid global statements. Uses a lock for thread-safe initialization.
    """

    _instance: AuditLogger | None = None
    _lock: threading.Lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> AuditLogger | None:
        """Get the current audit logger instance."""
        return cls._instance

    @classmethod
    def set_instance(cls, logger: AuditLogger | None) -> None:
        """Set the audit logger instance."""
        cls._instance = logger

    @classmethod
    def get_lock(cls) -> threading.Lock:
        """Get the singleton initialization lock."""
        return cls._lock


def get_audit_logger() -> AuditLogger:
    """Get the global audit logger instance.

    Uses double-checked locking pattern for thread-safe lazy initialization.
    """
    instance = _AuditLoggerState.get_instance()
    if instance is None:
        with _AuditLoggerState.get_lock():
            # Double-check after acquiring lock
            instance = _AuditLoggerState.get_instance()
            if instance is None:
                # Initialize with default settings
                log_file = Path("logs/audit.log")
                instance = AuditLogger(
                    log_file=log_file, console_output=True, json_format=True
                )
                _AuditLoggerState.set_instance(instance)
    return instance


def configure_audit_logger(
    log_file: Path | None = None,
    console_output: bool = True,
    json_format: bool = True,
) -> None:
    """Configure the global audit logger.

    Thread-safe configuration that replaces the existing logger instance.
    """
    with _AuditLoggerState.get_lock():
        _AuditLoggerState.set_instance(
            AuditLogger(
                log_file=log_file,
                console_output=console_output,
                json_format=json_format,
            )
        )


def _cleanup_audit_logger() -> None:
    """Clean up the audit logger on module unload.

    Properly closes all handlers to ensure log files are flushed
    and file handles are released.
    """
    instance = _AuditLoggerState.get_instance()
    if instance is not None and instance.logger is not None:
        for handler in instance.logger.handlers[:]:
            try:
                handler.flush()
                handler.close()
            except Exception:
                pass  # Ignore cleanup errors during shutdown
        instance.logger.handlers.clear()


# Register cleanup handler for the audit logger
atexit.register(_cleanup_audit_logger)
