# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Tests for composable configuration models.

These tests validate that all configuration models:
- Use proper Pydantic types (HttpUrl, PostgresDsn, SecretStr)
- Never leak secrets in string representations
- Validate input correctly with fail-fast behavior
- Follow ONEX composable config pattern
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import HttpUrl, PostgresDsn, SecretStr, ValidationError

from omnimemory.models.config import (
    ModelFilesystemConfig,
    ModelMemoryServiceConfig,
    ModelPostgresConfig,
    ModelQdrantConfig,
)


class TestModelFilesystemConfig:
    """Tests for filesystem configuration model."""

    def test_valid_config_with_absolute_path(self, tmp_path: Path) -> None:
        """Test valid filesystem config with absolute path."""
        config = ModelFilesystemConfig(base_path=tmp_path)
        assert config.base_path == tmp_path
        assert config.max_file_size_bytes == 10_485_760  # default 10MB
        assert config.create_if_missing is True  # default

    def test_relative_path_rejected(self) -> None:
        """Test that relative paths are rejected."""
        with pytest.raises(ValidationError) as exc_info:
            ModelFilesystemConfig(base_path=Path("relative/path"))
        assert "absolute path" in str(exc_info.value).lower()

    def test_custom_max_file_size(self, tmp_path: Path) -> None:
        """Test custom max_file_size_bytes setting."""
        config = ModelFilesystemConfig(
            base_path=tmp_path,
            max_file_size_bytes=5_000_000,
        )
        assert config.max_file_size_bytes == 5_000_000

    def test_max_file_size_upper_bound(self, tmp_path: Path) -> None:
        """Test max_file_size_bytes upper bound (1GB)."""
        with pytest.raises(ValidationError):
            ModelFilesystemConfig(
                base_path=tmp_path,
                max_file_size_bytes=2_000_000_000,  # > 1GB
            )

    def test_max_file_size_lower_bound(self, tmp_path: Path) -> None:
        """Test max_file_size_bytes lower bound (>= 1)."""
        with pytest.raises(ValidationError):
            ModelFilesystemConfig(
                base_path=tmp_path,
                max_file_size_bytes=0,
            )

    def test_custom_allowed_extensions(self, tmp_path: Path) -> None:
        """Test custom allowed_extensions setting."""
        config = ModelFilesystemConfig(
            base_path=tmp_path,
            allowed_extensions=[".json", ".yaml"],
        )
        assert config.allowed_extensions == [".json", ".yaml"]

    def test_extensions_must_start_with_dot(self, tmp_path: Path) -> None:
        """Test that extensions must start with a dot."""
        with pytest.raises(ValidationError) as exc_info:
            ModelFilesystemConfig(
                base_path=tmp_path,
                allowed_extensions=["json", "txt"],  # missing dots
            )
        assert "start with '.'" in str(exc_info.value)

    def test_create_if_missing_flag(self, tmp_path: Path) -> None:
        """Test create_if_missing flag setting."""
        config = ModelFilesystemConfig(
            base_path=tmp_path,
            create_if_missing=False,
        )
        assert config.create_if_missing is False

    def test_compression_flag(self, tmp_path: Path) -> None:
        """Test enable_compression flag."""
        config = ModelFilesystemConfig(
            base_path=tmp_path,
            enable_compression=True,
        )
        assert config.enable_compression is True

    def test_buffer_size_bounds(self, tmp_path: Path) -> None:
        """Test buffer_size_bytes bounds (4KB to 1MB)."""
        # Valid in-range value
        config = ModelFilesystemConfig(
            base_path=tmp_path,
            buffer_size_bytes=32768,  # 32KB
        )
        assert config.buffer_size_bytes == 32768

        # Below minimum (4KB = 4096)
        with pytest.raises(ValidationError):
            ModelFilesystemConfig(
                base_path=tmp_path,
                buffer_size_bytes=2048,
            )

        # Above maximum (1MB = 1048576)
        with pytest.raises(ValidationError):
            ModelFilesystemConfig(
                base_path=tmp_path,
                buffer_size_bytes=2_000_000,
            )


class TestModelPostgresConfig:
    """Tests for PostgreSQL configuration model."""

    def test_valid_config_with_proper_dsn_type(self) -> None:
        """Test valid postgres config with proper PostgresDsn type."""
        config = ModelPostgresConfig(
            dsn="postgresql://user:pass@localhost:5432/db",
        )
        # Verify DSN is proper PostgresDsn type, not a plain string
        assert isinstance(config.dsn, PostgresDsn)
        # Access host/port via hosts() method in Pydantic v2
        hosts = config.dsn.hosts()
        assert hosts is not None
        assert len(hosts) > 0
        assert hosts[0].get("host") == "localhost"
        assert hosts[0].get("port") == 5432

    def test_invalid_dsn_rejected(self) -> None:
        """Test invalid DSN is rejected with ValidationError."""
        with pytest.raises(ValidationError):
            ModelPostgresConfig(
                dsn="not-a-valid-dsn",
            )

    def test_pool_size_bounds(self) -> None:
        """Test pool_size bounds (1-50)."""
        # Valid
        config = ModelPostgresConfig(
            dsn="postgresql://user:pass@localhost/db",
            pool_size=20,
        )
        assert config.pool_size == 20

        # Too low
        with pytest.raises(ValidationError):
            ModelPostgresConfig(
                dsn="postgresql://user:pass@localhost/db",
                pool_size=0,
            )

        # Too high
        with pytest.raises(ValidationError):
            ModelPostgresConfig(
                dsn="postgresql://user:pass@localhost/db",
                pool_size=100,
            )

    def test_ssl_mode_validation(self) -> None:
        """Test ssl_mode validates against allowed values."""
        # Valid modes
        for mode in [
            "disable",
            "allow",
            "prefer",
            "require",
            "verify-ca",
            "verify-full",
        ]:
            config = ModelPostgresConfig(
                dsn="postgresql://user:pass@localhost/db",
                ssl_mode=mode,
            )
            assert config.ssl_mode == mode

        # Invalid mode
        with pytest.raises(ValidationError) as exc_info:
            ModelPostgresConfig(
                dsn="postgresql://user:pass@localhost/db",
                ssl_mode="invalid",
            )
        assert "ssl_mode" in str(exc_info.value)

    def test_schema_name_validation_accepts_valid_identifiers(self) -> None:
        """Test schema_name accepts valid PostgreSQL identifiers."""
        valid_names = [
            "my_schema",
            "_private_schema",
            "Schema123",
            "a",
            "schema_with_numbers_123",
        ]
        for name in valid_names:
            config = ModelPostgresConfig(
                dsn="postgresql://user:pass@localhost/db",
                schema_name=name,
            )
            assert config.schema_name == name

    def test_schema_name_validation_rejects_invalid_identifiers(self) -> None:
        """Test schema_name rejects invalid PostgreSQL identifiers."""
        invalid_names = [
            "invalid-schema",  # hyphens not allowed
            "123_starts_with_number",  # cannot start with digit
            "has spaces",  # spaces not allowed
            "has.dot",  # dots not allowed
            "",  # empty not allowed
        ]
        for name in invalid_names:
            with pytest.raises(ValidationError):
                ModelPostgresConfig(
                    dsn="postgresql://user:pass@localhost/db",
                    schema_name=name,
                )

    def test_password_redacted_in_str_and_repr(self) -> None:
        """Test that DSN password is redacted in str() and repr() output."""
        config = ModelPostgresConfig(
            dsn="postgresql://dbuser:s3cret_passw0rd@localhost:5432/mydb",
        )
        config_str = str(config)
        config_repr = repr(config)

        # Password must not appear in any string representation
        assert "s3cret_passw0rd" not in config_str
        assert "s3cret_passw0rd" not in config_repr

        # Username and host should still be visible for debugging
        assert "dbuser" in config_str
        assert "localhost" in config_str

        # Redaction marker should appear
        assert "***" in config_str
        assert "***" in config_repr

    def test_password_redacted_in_model_dump(self) -> None:
        """Test that model_dump() does NOT contain the raw password in dsn."""
        config = ModelPostgresConfig(
            dsn="postgresql://dbuser:s3cret_passw0rd@localhost:5432/mydb",
        )
        dumped = config.model_dump()
        dsn_value = dumped["dsn"]

        # Raw password must never appear in serialized output
        assert "s3cret_passw0rd" not in dsn_value

        # Redaction marker should be present instead
        assert "***" in dsn_value

        # Username and host should survive redaction for debuggability
        assert "dbuser" in dsn_value
        assert "localhost" in dsn_value

    def test_password_redacted_in_model_dump_json(self) -> None:
        """Test that model_dump_json() does NOT contain the raw password."""
        config = ModelPostgresConfig(
            dsn="postgresql://dbuser:s3cret_passw0rd@localhost:5432/mydb",
        )
        json_str = config.model_dump_json()

        assert "s3cret_passw0rd" not in json_str
        assert "***" in json_str

    def test_dsn_without_password_not_mangled(self) -> None:
        """Test that a DSN without a password is not altered in repr."""
        config = ModelPostgresConfig(
            dsn="postgresql://localhost:5432/mydb",
        )
        config_repr = repr(config)
        # Should contain the DSN without any redaction marker
        assert "localhost" in config_repr


class TestModelQdrantConfig:
    """Tests for Qdrant configuration model."""

    def test_valid_config_with_proper_http_url_type(self) -> None:
        """Test valid qdrant config with proper HttpUrl type."""
        config = ModelQdrantConfig(
            url="http://localhost:6333",
            collection_name="test_collection",
        )
        # Verify URL is proper HttpUrl type, not a plain string
        assert isinstance(config.url, HttpUrl)
        # URL may have trailing slash added by Pydantic
        assert str(config.url).rstrip("/") == "http://localhost:6333"

    def test_explicit_url_is_http_url_type(self) -> None:
        """Test explicitly provided URL becomes proper HttpUrl type."""
        # Note: The default value is stored as string, but explicit values
        # are coerced to HttpUrl by Pydantic validation
        config = ModelQdrantConfig(url="http://localhost:6333")
        assert isinstance(config.url, HttpUrl)

    def test_api_key_is_optional(self) -> None:
        """Test api_key is optional and None by default."""
        config = ModelQdrantConfig(url="http://localhost:6333")
        assert config.api_key is None

    def test_api_key_uses_secret_str(self) -> None:
        """Test api_key uses SecretStr when provided."""
        config = ModelQdrantConfig(
            url="http://localhost:6333",
            api_key=SecretStr("my-api-key"),
        )
        assert isinstance(config.api_key, SecretStr)

    def test_api_key_not_leaked_in_str(self) -> None:
        """Test api_key does NOT appear in string representation."""
        config = ModelQdrantConfig(
            url="http://localhost:6333",
            api_key=SecretStr("secret-api-key-xyz"),
        )
        assert "secret-api-key-xyz" not in str(config)
        assert "secret-api-key-xyz" not in repr(config)

    def test_api_key_excluded_from_model_dump(self) -> None:
        """Test api_key is excluded from model_dump() by default."""
        config = ModelQdrantConfig(
            url="http://localhost:6333",
            api_key=SecretStr("secret-key"),
        )
        dumped = config.model_dump()
        assert "api_key" not in dumped

    def test_invalid_url_rejected(self) -> None:
        """Test invalid URL is rejected with ValidationError."""
        with pytest.raises(ValidationError):
            ModelQdrantConfig(url="not-a-url")

    def test_collection_name_validation(self) -> None:
        """Test collection_name validates as alphanumeric with underscore/hyphen."""
        # Valid names
        for name in ["my_collection", "test-collection", "collection123"]:
            config = ModelQdrantConfig(collection_name=name)
            assert config.collection_name == name

        # Invalid name (special characters)
        with pytest.raises(ValidationError):
            ModelQdrantConfig(collection_name="invalid.collection!")

    def test_vector_size_bounds(self) -> None:
        """Test vector_size bounds (1-65536)."""
        # Valid
        config = ModelQdrantConfig(vector_size=384)
        assert config.vector_size == 384

        # Too low
        with pytest.raises(ValidationError):
            ModelQdrantConfig(vector_size=0)

        # Too high
        with pytest.raises(ValidationError):
            ModelQdrantConfig(vector_size=100000)

    def test_distance_metric_validation(self) -> None:
        """Test distance_metric validates against allowed values."""
        # Valid metrics
        for metric in ["Cosine", "Euclid", "Dot"]:
            config = ModelQdrantConfig(distance_metric=metric)
            assert config.distance_metric == metric

        # Invalid metric
        with pytest.raises(ValidationError):
            ModelQdrantConfig(distance_metric="Manhattan")

    def test_score_threshold_bounds(self) -> None:
        """Test score_threshold bounds (0.0-1.0)."""
        # Valid
        config = ModelQdrantConfig(score_threshold=0.85)
        assert config.score_threshold == 0.85

        # Below minimum
        with pytest.raises(ValidationError):
            ModelQdrantConfig(score_threshold=-0.1)

        # Above maximum
        with pytest.raises(ValidationError):
            ModelQdrantConfig(score_threshold=1.5)


class TestModelMemoryServiceConfig:
    """Tests for top-level composed configuration."""

    def test_minimal_config_filesystem_only(self, tmp_path: Path) -> None:
        """Test minimal Phase 1 config with only filesystem."""
        config = ModelMemoryServiceConfig(
            filesystem=ModelFilesystemConfig(base_path=tmp_path)
        )
        assert config.filesystem is not None
        assert config.filesystem.base_path == tmp_path
        assert config.postgres is None
        assert config.qdrant is None

    def test_full_config_all_backends(self, tmp_path: Path) -> None:
        """Test full config with all backends enabled."""
        config = ModelMemoryServiceConfig(
            filesystem=ModelFilesystemConfig(base_path=tmp_path),
            postgres=ModelPostgresConfig(
                dsn="postgresql://user:pass@localhost/db",
            ),
            qdrant=ModelQdrantConfig(url="http://localhost:6333"),
        )
        assert config.filesystem is not None
        assert config.postgres is not None
        assert config.qdrant is not None

    def test_filesystem_required(self) -> None:
        """Test filesystem config is required."""
        with pytest.raises(ValidationError) as exc_info:
            ModelMemoryServiceConfig()
        assert "filesystem" in str(exc_info.value).lower()

    def test_service_level_settings(self, tmp_path: Path) -> None:
        """Test service-level settings."""
        config = ModelMemoryServiceConfig(
            filesystem=ModelFilesystemConfig(base_path=tmp_path),
            service_name="custom-memory",
            enable_metrics=False,
            enable_logging=False,
            debug_mode=True,
        )
        assert config.service_name == "custom-memory"
        assert config.enable_metrics is False
        assert config.enable_logging is False
        assert config.debug_mode is True

    def test_default_service_settings(self, tmp_path: Path) -> None:
        """Test default service-level settings."""
        config = ModelMemoryServiceConfig(
            filesystem=ModelFilesystemConfig(base_path=tmp_path)
        )
        assert config.service_name == "omnimemory"
        assert config.enable_metrics is True
        assert config.enable_logging is True
        assert config.debug_mode is False

    def test_postgres_optional(self, tmp_path: Path) -> None:
        """Test postgres is optional and can be set to None."""
        config = ModelMemoryServiceConfig(
            filesystem=ModelFilesystemConfig(base_path=tmp_path),
            postgres=None,
        )
        assert config.postgres is None

    def test_qdrant_optional(self, tmp_path: Path) -> None:
        """Test qdrant is optional and can be set to None."""
        config = ModelMemoryServiceConfig(
            filesystem=ModelFilesystemConfig(base_path=tmp_path),
            qdrant=None,
        )
        assert config.qdrant is None

    def test_secrets_not_leaked_in_composed_config(self, tmp_path: Path) -> None:
        """Test that secrets in nested configs don't leak in representations."""
        config = ModelMemoryServiceConfig(
            filesystem=ModelFilesystemConfig(base_path=tmp_path),
            postgres=ModelPostgresConfig(
                dsn="postgresql://user:pass@localhost/db",
            ),
            qdrant=ModelQdrantConfig(
                url="http://localhost:6333",
                api_key=SecretStr("qdrant_secret_key"),
            ),
        )
        config_str = str(config)
        config_repr = repr(config)

        # Secrets should not appear
        assert "qdrant_secret_key" not in config_str
        assert "qdrant_secret_key" not in config_repr

        # Postgres password in DSN must also be redacted
        assert ":pass@" not in config_str
        assert ":pass@" not in config_repr
