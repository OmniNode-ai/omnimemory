# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""
PostgreSQL storage configuration model following ONEX standards.
"""

from __future__ import annotations

import re
from urllib.parse import urlparse, urlunparse

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PostgresDsn,
    field_serializer,
    field_validator,
)

_REDACTED = "***"

# PostgreSQL identifier maximum length per SQL standard
POSTGRES_IDENTIFIER_MAX_LENGTH = 63


class ModelPostgresConfig(BaseModel):
    """Configuration for PostgreSQL memory storage.

    This config defines connection parameters for PostgreSQL-based
    persistent memory storage. The DSN must be a full connection URL
    including credentials (sourced from OMNIMEMORY_DB_URL).
    """

    model_config = ConfigDict(extra="forbid")

    # Connection configuration
    dsn: PostgresDsn = Field(
        description="PostgreSQL connection URL (e.g., postgresql://user:pass@host:port/db)",
        repr=False,
    )

    # Connection pool settings
    pool_size: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Connection pool size (1-50)",
    )
    pool_timeout_seconds: int = Field(
        default=30,
        ge=1,
        le=300,
        description="Pool connection acquisition timeout in seconds",
    )
    pool_recycle_seconds: int = Field(
        default=3600,
        ge=60,
        le=86400,
        description="Connection recycle time in seconds (default 1 hour)",
    )

    # Query settings
    statement_timeout_seconds: int = Field(
        default=30,
        ge=1,
        le=300,
        description="Maximum query execution time in seconds",
    )
    lock_timeout_seconds: int = Field(
        default=10,
        ge=1,
        le=60,
        description="Maximum time to wait for locks in seconds",
    )

    # SSL configuration
    ssl_mode: str = Field(
        default="prefer",
        description=(
            "SSL mode (disable, allow, prefer, require, verify-ca, verify-full)"
        ),
    )

    # Schema configuration
    schema_name: str = Field(
        default="omnimemory",
        description="PostgreSQL schema name for memory tables",
    )

    @field_validator("ssl_mode")
    @classmethod
    def validate_ssl_mode(cls, v: str) -> str:
        """Validate SSL mode is a valid PostgreSQL SSL mode."""
        valid_modes = {
            "disable",
            "allow",
            "prefer",
            "require",
            "verify-ca",
            "verify-full",
        }
        if v not in valid_modes:
            raise ValueError(
                f"ssl_mode must be one of: {', '.join(sorted(valid_modes))}"
            )
        return v

    @field_validator("schema_name")
    @classmethod
    def validate_schema_name(cls, v: str) -> str:
        """Validate schema name is a valid PostgreSQL identifier.

        PostgreSQL schema names must:
        - Start with a letter or underscore
        - Contain only letters, digits, and underscores
        - Be at most 63 characters

        Note: PostgreSQL supports more characters with quoting, but we
        enforce this subset for safer unquoted usage.
        """
        if not v:
            raise ValueError("schema_name cannot be empty")
        if len(v) > POSTGRES_IDENTIFIER_MAX_LENGTH:
            raise ValueError(
                f"schema_name cannot exceed {POSTGRES_IDENTIFIER_MAX_LENGTH} characters"
            )
        # PostgreSQL identifier rules: start with letter/underscore
        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", v):
            raise ValueError(
                "schema_name must start with a letter or underscore and contain "
                "only letters, digits, and underscores"
            )
        return v

    def _redacted_dsn(self) -> str:
        """Return the DSN string with password replaced by a redaction marker.

        DSNs without a password component are returned unmodified since there
        is nothing sensitive to redact.
        """
        parsed = urlparse(str(self.dsn))
        if parsed.password:
            # Replace password while preserving user and other components
            netloc = parsed.hostname or ""
            if parsed.port:
                netloc = f"{netloc}:{parsed.port}"
            if parsed.username:
                netloc = f"{parsed.username}:{_REDACTED}@{netloc}"
            redacted = urlunparse(parsed._replace(netloc=netloc))
            return redacted
        # No password present — nothing to redact
        return str(self.dsn)

    @field_serializer("dsn")
    def _serialize_dsn(self, dsn: PostgresDsn, _info: object) -> str:
        """Redact password when serializing via model_dump() / model_dump_json()."""
        return self._redacted_dsn()

    def __repr__(self) -> str:
        """Redact credentials from repr to prevent password leakage in logs."""
        fields = []
        for name in self.__class__.model_fields:
            if name == "dsn":
                fields.append(f"dsn={self._redacted_dsn()!r}")
            else:
                fields.append(f"{name}={getattr(self, name)!r}")
        return f"{self.__class__.__name__}({', '.join(fields)})"

    def __str__(self) -> str:
        """Redact credentials from str to prevent password leakage in logs."""
        return self.__repr__()
