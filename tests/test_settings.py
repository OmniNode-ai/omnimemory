# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Tests for environment-based settings loading.

These tests validate that settings:
- Load from environment variables with OMNIMEMORY__ prefix
- Use __ as nested delimiter for structured config
- Fail fast with clear error messages when required settings are missing
- Convert to proper config models via to_config()
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from omnimemory.models.config import (
    ModelFilesystemConfig,
    ModelMemoryServiceConfig,
    ModelPostgresConfig,
    ModelQdrantConfig,
)
from omnimemory.settings import (
    FilesystemSettings,
    MemoryServiceSettings,
    PostgresSettings,
    QdrantSettings,
    load_settings,
)


def _clear_omnimemory_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear all OMNIMEMORY environment variables."""
    for key in list(os.environ.keys()):
        if key.startswith("OMNIMEMORY"):
            monkeypatch.delenv(key, raising=False)


class TestFilesystemSettings:
    """Tests for filesystem settings loading."""

    def test_loads_from_env(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test filesystem settings load from environment."""
        _clear_omnimemory_env_vars(monkeypatch)
        monkeypatch.setenv("OMNIMEMORY__FILESYSTEM__BASE_PATH", str(tmp_path))

        settings = FilesystemSettings()
        assert settings.base_path == tmp_path

    def test_loads_all_fields(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test all filesystem fields load from environment."""
        _clear_omnimemory_env_vars(monkeypatch)
        monkeypatch.setenv("OMNIMEMORY__FILESYSTEM__BASE_PATH", str(tmp_path))
        monkeypatch.setenv("OMNIMEMORY__FILESYSTEM__MAX_FILE_SIZE_BYTES", "5000000")
        monkeypatch.setenv("OMNIMEMORY__FILESYSTEM__CREATE_IF_MISSING", "false")
        monkeypatch.setenv("OMNIMEMORY__FILESYSTEM__ENABLE_COMPRESSION", "true")
        monkeypatch.setenv("OMNIMEMORY__FILESYSTEM__BUFFER_SIZE_BYTES", "32768")

        settings = FilesystemSettings()
        assert settings.base_path == tmp_path
        assert settings.max_file_size_bytes == 5000000
        assert settings.create_if_missing is False
        assert settings.enable_compression is True
        assert settings.buffer_size_bytes == 32768

    def test_to_config_converts_properly(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test to_config() produces ModelFilesystemConfig."""
        _clear_omnimemory_env_vars(monkeypatch)
        monkeypatch.setenv("OMNIMEMORY__FILESYSTEM__BASE_PATH", str(tmp_path))

        settings = FilesystemSettings()
        config = settings.to_config()

        assert isinstance(config, ModelFilesystemConfig)
        assert config.base_path == tmp_path

    def test_allowed_extensions_json_array(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test allowed_extensions supports JSON array format."""
        _clear_omnimemory_env_vars(monkeypatch)
        monkeypatch.setenv("OMNIMEMORY__FILESYSTEM__BASE_PATH", str(tmp_path))
        monkeypatch.setenv(
            "OMNIMEMORY__FILESYSTEM__ALLOWED_EXTENSIONS", '[".py", ".yaml", ".toml"]'
        )

        settings = FilesystemSettings()
        assert settings.allowed_extensions == [".py", ".yaml", ".toml"]


class TestPostgresSettings:
    """Tests for postgres settings loading."""

    def test_to_config_reads_omnimemory_db_url(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test to_config() reads OMNIMEMORY_DB_URL from environment."""
        _clear_omnimemory_env_vars(monkeypatch)
        monkeypatch.setenv("OMNIMEMORY_DB_URL", "postgresql://user:pass@localhost/db")

        settings = PostgresSettings()
        config = settings.to_config()

        assert isinstance(config, ModelPostgresConfig)
        assert str(config.dsn).startswith("postgresql://")

    def test_construction_fails_fast_without_omnimemory_db_url(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test PostgresSettings raises ValidationError when OMNIMEMORY_DB_URL is not set."""
        _clear_omnimemory_env_vars(monkeypatch)
        monkeypatch.delenv("OMNIMEMORY_DB_URL", raising=False)

        with pytest.raises(ValidationError):
            PostgresSettings()


class TestQdrantSettings:
    """Tests for qdrant settings loading."""

    def test_loads_from_env_with_defaults(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test qdrant settings load with defaults."""
        _clear_omnimemory_env_vars(monkeypatch)
        # Qdrant has all defaults, no required fields

        settings = QdrantSettings()
        assert str(settings.url).rstrip("/") == "http://localhost:6333"
        assert settings.api_key is None
        assert settings.collection_name == "omnimemory"

    def test_loads_api_key_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test qdrant api_key loads from environment."""
        _clear_omnimemory_env_vars(monkeypatch)
        monkeypatch.setenv("OMNIMEMORY__QDRANT__API_KEY", "my-api-key")

        settings = QdrantSettings()
        assert settings.api_key is not None
        assert settings.api_key.get_secret_value() == "my-api-key"

    def test_to_config_converts_properly(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test to_config() produces ModelQdrantConfig."""
        _clear_omnimemory_env_vars(monkeypatch)

        settings = QdrantSettings()
        config = settings.to_config()

        assert isinstance(config, ModelQdrantConfig)


class TestMemoryServiceSettings:
    """Tests for top-level service settings."""

    def test_load_minimal_settings(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test loading minimal Phase 1 settings."""
        _clear_omnimemory_env_vars(monkeypatch)
        monkeypatch.setenv("OMNIMEMORY__FILESYSTEM__BASE_PATH", str(tmp_path))

        settings = MemoryServiceSettings()
        assert settings.postgres_enabled is False
        assert settings.qdrant_enabled is False

    def test_to_config_produces_correct_type(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test to_config() produces ModelMemoryServiceConfig."""
        _clear_omnimemory_env_vars(monkeypatch)
        monkeypatch.setenv("OMNIMEMORY__FILESYSTEM__BASE_PATH", str(tmp_path))

        settings = MemoryServiceSettings()
        config = settings.to_config()

        assert isinstance(config, ModelMemoryServiceConfig)
        assert config.filesystem.base_path == tmp_path

    def test_postgres_enabled_requires_omnimemory_db_url(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test postgres_enabled=true requires OMNIMEMORY_DB_URL."""
        _clear_omnimemory_env_vars(monkeypatch)
        monkeypatch.setenv("OMNIMEMORY__FILESYSTEM__BASE_PATH", str(tmp_path))
        monkeypatch.setenv("OMNIMEMORY__POSTGRES_ENABLED", "true")
        monkeypatch.delenv("OMNIMEMORY_DB_URL", raising=False)

        settings = MemoryServiceSettings()
        assert settings.postgres_enabled is True

        # to_config() should fail because OMNIMEMORY_DB_URL is missing,
        # raising ValidationError from PostgresSettings construction
        with pytest.raises(ValidationError):
            settings.to_config()

    def test_postgres_enabled_with_omnimemory_db_url(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test postgres_enabled=true with OMNIMEMORY_DB_URL set."""
        _clear_omnimemory_env_vars(monkeypatch)
        monkeypatch.setenv("OMNIMEMORY__FILESYSTEM__BASE_PATH", str(tmp_path))
        monkeypatch.setenv("OMNIMEMORY__POSTGRES_ENABLED", "true")
        monkeypatch.setenv("OMNIMEMORY_DB_URL", "postgresql://user:pass@localhost/db")

        settings = MemoryServiceSettings()
        config = settings.to_config()

        assert config.postgres is not None

    def test_qdrant_enabled_loads_qdrant_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test qdrant_enabled=true loads qdrant config."""
        _clear_omnimemory_env_vars(monkeypatch)
        monkeypatch.setenv("OMNIMEMORY__FILESYSTEM__BASE_PATH", str(tmp_path))
        monkeypatch.setenv("OMNIMEMORY__QDRANT_ENABLED", "true")

        settings = MemoryServiceSettings()
        config = settings.to_config()

        assert config.qdrant is not None

    def test_service_level_settings_from_env(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test service-level settings load from environment."""
        _clear_omnimemory_env_vars(monkeypatch)
        monkeypatch.setenv("OMNIMEMORY__FILESYSTEM__BASE_PATH", str(tmp_path))
        monkeypatch.setenv("OMNIMEMORY__SERVICE_NAME", "custom-service")
        monkeypatch.setenv("OMNIMEMORY__ENABLE_METRICS", "false")
        monkeypatch.setenv("OMNIMEMORY__ENABLE_LOGGING", "false")
        monkeypatch.setenv("OMNIMEMORY__DEBUG_MODE", "true")

        settings = MemoryServiceSettings()
        assert settings.service_name == "custom-service"
        assert settings.enable_metrics is False
        assert settings.enable_logging is False
        assert settings.debug_mode is True

        # These should flow to the config
        config = settings.to_config()
        assert config.service_name == "custom-service"
        assert config.enable_metrics is False
        assert config.enable_logging is False
        assert config.debug_mode is True


class TestLoadSettings:
    """Tests for the load_settings() function."""

    def test_load_settings_returns_settings(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test load_settings returns MemoryServiceSettings."""
        _clear_omnimemory_env_vars(monkeypatch)
        monkeypatch.setenv("OMNIMEMORY__FILESYSTEM__BASE_PATH", str(tmp_path))

        settings = load_settings()
        assert isinstance(settings, MemoryServiceSettings)

    def test_missing_required_fails_fast(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test missing required settings fails immediately with ValidationError."""
        _clear_omnimemory_env_vars(monkeypatch)
        # Don't set OMNIMEMORY__FILESYSTEM__BASE_PATH

        # load_settings itself should succeed (it just loads top-level settings)
        # But to_config() will fail because FilesystemSettings requires BASE_PATH
        settings = load_settings()
        with pytest.raises(ValidationError):
            settings.to_config()

    def test_nested_delimiter_works(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test __ nested delimiter for env vars works."""
        _clear_omnimemory_env_vars(monkeypatch)
        monkeypatch.setenv("OMNIMEMORY__FILESYSTEM__BASE_PATH", str(tmp_path))
        monkeypatch.setenv("OMNIMEMORY__FILESYSTEM__MAX_FILE_SIZE_BYTES", "5000000")

        settings = load_settings()
        config = settings.to_config()

        assert config.filesystem.max_file_size_bytes == 5000000

    def test_invalid_value_raises_validation_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test invalid environment variable value raises ValidationError."""
        _clear_omnimemory_env_vars(monkeypatch)
        monkeypatch.setenv("OMNIMEMORY__FILESYSTEM__BASE_PATH", str(tmp_path))
        # Set invalid value for max_file_size_bytes (should be int)
        monkeypatch.setenv(
            "OMNIMEMORY__FILESYSTEM__MAX_FILE_SIZE_BYTES", "not-an-integer"
        )

        settings = load_settings()
        with pytest.raises(ValidationError):
            settings.to_config()

    def test_full_config_from_env(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test loading a full configuration from environment."""
        _clear_omnimemory_env_vars(monkeypatch)

        # Filesystem - required
        monkeypatch.setenv("OMNIMEMORY__FILESYSTEM__BASE_PATH", str(tmp_path))

        # Postgres - optional but enabled (uses OMNIMEMORY_DB_URL)
        monkeypatch.setenv("OMNIMEMORY__POSTGRES_ENABLED", "true")
        monkeypatch.setenv("OMNIMEMORY_DB_URL", "postgresql://user:dbpass@localhost/db")
        monkeypatch.setenv("OMNIMEMORY__POSTGRES__POOL_SIZE", "10")

        # Qdrant (optional but enabled)
        monkeypatch.setenv("OMNIMEMORY__QDRANT_ENABLED", "true")
        monkeypatch.setenv("OMNIMEMORY__QDRANT__URL", "http://qdrant.local:6333")
        monkeypatch.setenv("OMNIMEMORY__QDRANT__API_KEY", "qdrant-key")

        settings = load_settings()
        config = settings.to_config()

        # Verify full config
        assert config.filesystem.base_path == tmp_path
        assert config.postgres is not None
        assert config.postgres.pool_size == 10
        assert config.qdrant is not None
        assert str(config.qdrant.url).rstrip("/") == "http://qdrant.local:6333"
        assert config.qdrant.api_key is not None
        assert config.qdrant.api_key.get_secret_value() == "qdrant-key"
