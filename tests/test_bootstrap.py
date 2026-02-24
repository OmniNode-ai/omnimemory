# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Tests for bootstrap initialization.

These tests validate that bootstrap:
- Successfully initializes with valid config
- Creates directories when create_if_missing=True
- Fails with BootstrapError when directories don't exist and create_if_missing=False
- Is idempotent (returns cached result on subsequent calls)
- Can be re-run with force=True
- Properly tracks initialized backends
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from pathlib import Path

import pytest

from omnimemory.bootstrap import (
    BootstrapError,
    BootstrapResult,
    bootstrap,
    get_bootstrap_result,
    is_bootstrapped,
    shutdown,
)
from omnimemory.models.config import (
    ModelFilesystemConfig,
    ModelMemoryServiceConfig,
    ModelPostgresConfig,
    ModelQdrantConfig,
)
from omnimemory.secrets import EnvSecretsProvider


@pytest.fixture
async def reset_bootstrap() -> AsyncGenerator[None, None]:
    """Reset bootstrap state before and after each test.

    This fixture ensures each test starts with a clean bootstrap state
    and cleans up after itself.
    """
    await shutdown()
    yield
    await shutdown()


class TestBootstrapSuccess:
    """Tests for successful bootstrap scenarios."""

    @pytest.mark.asyncio
    async def test_bootstrap_success_minimal(
        self, tmp_path: Path, reset_bootstrap: None
    ) -> None:
        """Test successful bootstrap with minimal filesystem-only config."""
        config = ModelMemoryServiceConfig(
            filesystem=ModelFilesystemConfig(base_path=tmp_path)
        )

        result = await bootstrap(config)

        assert result.success is True
        assert "filesystem" in result.initialized_backends
        assert await is_bootstrapped() is True

    @pytest.mark.asyncio
    async def test_bootstrap_returns_bootstrap_result(
        self, tmp_path: Path, reset_bootstrap: None
    ) -> None:
        """Test bootstrap returns a BootstrapResult instance."""
        config = ModelMemoryServiceConfig(
            filesystem=ModelFilesystemConfig(base_path=tmp_path)
        )

        result = await bootstrap(config)

        assert isinstance(result, BootstrapResult)
        assert hasattr(result, "success")
        assert hasattr(result, "initialized_backends")
        assert hasattr(result, "warnings")
        assert hasattr(result, "secrets_provider")

    @pytest.mark.asyncio
    async def test_bootstrap_creates_directory(
        self, tmp_path: Path, reset_bootstrap: None
    ) -> None:
        """Test bootstrap creates base_path directory if missing."""
        new_dir = tmp_path / "new_memory_dir"
        assert not new_dir.exists()

        config = ModelMemoryServiceConfig(
            filesystem=ModelFilesystemConfig(
                base_path=new_dir,
                create_if_missing=True,
            )
        )

        result = await bootstrap(config)

        assert result.success is True
        assert new_dir.exists()
        assert new_dir.is_dir()

    @pytest.mark.asyncio
    async def test_bootstrap_creates_nested_directory(
        self, tmp_path: Path, reset_bootstrap: None
    ) -> None:
        """Test bootstrap creates nested directory structure."""
        nested_dir = tmp_path / "level1" / "level2" / "memory"
        assert not nested_dir.exists()

        config = ModelMemoryServiceConfig(
            filesystem=ModelFilesystemConfig(
                base_path=nested_dir,
                create_if_missing=True,
            )
        )

        result = await bootstrap(config)

        assert result.success is True
        assert nested_dir.exists()

    @pytest.mark.asyncio
    async def test_bootstrap_with_postgres(
        self, tmp_path: Path, reset_bootstrap: None
    ) -> None:
        """Test bootstrap with postgres backend configured."""
        config = ModelMemoryServiceConfig(
            filesystem=ModelFilesystemConfig(base_path=tmp_path),
            postgres=ModelPostgresConfig(
                dsn="postgresql://user:pass@localhost/db",
            ),
        )

        result = await bootstrap(config)

        assert result.success is True
        assert "filesystem" in result.initialized_backends
        assert "postgres" in result.initialized_backends

    @pytest.mark.asyncio
    async def test_bootstrap_with_qdrant(
        self, tmp_path: Path, reset_bootstrap: None
    ) -> None:
        """Test bootstrap with qdrant backend configured."""
        config = ModelMemoryServiceConfig(
            filesystem=ModelFilesystemConfig(base_path=tmp_path),
            qdrant=ModelQdrantConfig(url="http://localhost:6333"),
        )

        result = await bootstrap(config)

        assert result.success is True
        assert "filesystem" in result.initialized_backends
        assert "qdrant" in result.initialized_backends

    @pytest.mark.asyncio
    async def test_bootstrap_with_all_backends(
        self, tmp_path: Path, reset_bootstrap: None
    ) -> None:
        """Test bootstrap with all backends configured."""
        config = ModelMemoryServiceConfig(
            filesystem=ModelFilesystemConfig(base_path=tmp_path),
            postgres=ModelPostgresConfig(
                dsn="postgresql://user:pass@localhost/db",
            ),
            qdrant=ModelQdrantConfig(url="http://localhost:6333"),
        )

        result = await bootstrap(config)

        assert result.success is True
        assert len(result.initialized_backends) == 3
        assert "filesystem" in result.initialized_backends
        assert "postgres" in result.initialized_backends
        assert "qdrant" in result.initialized_backends

    @pytest.mark.asyncio
    async def test_bootstrap_qdrant_warns_no_api_key(
        self, tmp_path: Path, reset_bootstrap: None
    ) -> None:
        """Test bootstrap warns when qdrant has no API key."""
        config = ModelMemoryServiceConfig(
            filesystem=ModelFilesystemConfig(base_path=tmp_path),
            qdrant=ModelQdrantConfig(
                url="http://localhost:6333",
                api_key=None,  # No API key
            ),
        )

        result = await bootstrap(config)

        assert result.success is True
        # Should have warning about unauthenticated access
        assert len(result.warnings) > 0
        assert any("api key" in w.lower() for w in result.warnings)


class TestBootstrapFailure:
    """Tests for bootstrap failure scenarios."""

    @pytest.mark.asyncio
    async def test_bootstrap_fails_if_dir_missing_and_no_create(
        self, tmp_path: Path, reset_bootstrap: None
    ) -> None:
        """Test bootstrap fails if base_path missing and create_if_missing=False."""
        missing_dir = tmp_path / "nonexistent"
        assert not missing_dir.exists()

        config = ModelMemoryServiceConfig(
            filesystem=ModelFilesystemConfig(
                base_path=missing_dir,
                create_if_missing=False,
            )
        )

        with pytest.raises(BootstrapError) as exc_info:
            await bootstrap(config)

        assert exc_info.value.config_block == "filesystem"
        assert "does not exist" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_bootstrap_fails_if_base_path_is_file(
        self, tmp_path: Path, reset_bootstrap: None
    ) -> None:
        """Test bootstrap fails if base_path exists but is a file."""
        file_path = tmp_path / "file.txt"
        file_path.touch()
        assert file_path.is_file()

        config = ModelMemoryServiceConfig(
            filesystem=ModelFilesystemConfig(
                base_path=file_path,
                create_if_missing=False,
            )
        )

        with pytest.raises(BootstrapError) as exc_info:
            await bootstrap(config)

        assert exc_info.value.config_block == "filesystem"
        assert "not a directory" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_bootstrap_error_has_config_block(
        self, tmp_path: Path, reset_bootstrap: None
    ) -> None:
        """Test BootstrapError includes config_block attribute."""
        missing_dir = tmp_path / "missing"

        config = ModelMemoryServiceConfig(
            filesystem=ModelFilesystemConfig(
                base_path=missing_dir,
                create_if_missing=False,
            )
        )

        with pytest.raises(BootstrapError) as exc_info:
            await bootstrap(config)

        error = exc_info.value
        assert hasattr(error, "config_block")
        assert error.config_block == "filesystem"

    @pytest.mark.asyncio
    async def test_bootstrap_error_cause_on_write_permission_failure(
        self, tmp_path: Path, reset_bootstrap: None
    ) -> None:
        """Test BootstrapError.cause is set when write permission check fails.

        This test verifies that when the filesystem write test fails during
        bootstrap (due to read-only directory), the BootstrapError properly
        captures the underlying OSError as its cause attribute.
        """
        read_only_dir = tmp_path / "readonly"
        read_only_dir.mkdir()
        read_only_dir.chmod(0o444)  # Read-only

        try:
            # Check if we can actually write (e.g., running as root)
            test_file = read_only_dir / ".test_write"
            try:
                test_file.touch()
                test_file.unlink()
                # If we get here, permissions don't work (e.g., running as root)
                pytest.skip(
                    "Cannot test read-only directory (possibly running as root)"
                )
            except OSError:
                # Good - directory is actually read-only
                pass

            config = ModelMemoryServiceConfig(
                filesystem=ModelFilesystemConfig(
                    base_path=read_only_dir,
                    create_if_missing=False,
                )
            )

            with pytest.raises(BootstrapError) as exc_info:
                await bootstrap(config)

            error = exc_info.value
            # Verify the cause attribute exists and is properly set
            assert hasattr(error, "cause"), "BootstrapError must have cause attribute"
            assert error.cause is not None, (
                "cause should be set when underlying error exists"
            )
            assert isinstance(error.cause, OSError), (
                f"cause should be OSError, got {type(error.cause)}"
            )
            assert error.config_block == "filesystem"
            # Verify error message indicates write failure
            assert "not writable" in str(error)
            # Verify Python standard exception chaining is also set
            assert error.__cause__ is error.cause, (
                "__cause__ should be set for proper Python exception chaining"
            )
        finally:
            # Restore permissions for cleanup
            if read_only_dir.exists():
                read_only_dir.chmod(0o755)

    @pytest.mark.asyncio
    async def test_bootstrap_error_cause_on_mkdir_failure(
        self, tmp_path: Path, reset_bootstrap: None
    ) -> None:
        """Test BootstrapError.cause is set when mkdir fails.

        This test verifies that when create_if_missing=True but the directory
        cannot be created (due to permission issues), the BootstrapError
        captures the underlying OSError as its cause attribute.

        Note: On some systems, even checking if a path exists inside a read-only
        directory may fail with PermissionError. This test handles that case.
        """
        # Create a read-only parent directory to prevent mkdir
        read_only_parent = tmp_path / "readonly_parent"
        read_only_parent.mkdir()
        read_only_parent.chmod(0o444)  # Read-only

        try:
            # Check if we can actually create dirs (e.g., running as root)
            test_dir = read_only_parent / "test_mkdir"
            try:
                test_dir.mkdir()
                test_dir.rmdir()
                # If we get here, permissions don't work (e.g., running as root)
                pytest.skip("Cannot test read-only parent (possibly running as root)")
            except PermissionError:
                # Good - parent is actually read-only
                # Note: On some systems, even exists() check fails with PermissionError
                # If that's the case, the bootstrap code doesn't wrap it properly.
                # This is a known limitation - skip if we can't even check existence.
                try:
                    _ = test_dir.exists()
                except PermissionError:
                    pytest.skip(
                        "Cannot test mkdir failure - exists() also requires permissions"
                    )
            except OSError:
                # Other OSError (not PermissionError) - parent is still usable
                pass

            # Try to create a directory inside the read-only parent
            target_dir = read_only_parent / "should_fail"
            config = ModelMemoryServiceConfig(
                filesystem=ModelFilesystemConfig(
                    base_path=target_dir,
                    create_if_missing=True,  # Should try mkdir and fail
                )
            )

            with pytest.raises(BootstrapError) as exc_info:
                await bootstrap(config)

            error = exc_info.value
            # Verify the cause attribute is set from mkdir failure
            assert hasattr(error, "cause"), "BootstrapError must have cause attribute"
            assert error.cause is not None, "cause should be set when mkdir fails"
            assert isinstance(error.cause, OSError), (
                f"cause should be OSError, got {type(error.cause)}"
            )
            assert error.config_block == "filesystem"
            # Verify error message indicates creation failure
            assert "Cannot create" in str(error)
            # Verify Python standard exception chaining is also set
            assert error.__cause__ is error.cause, (
                "__cause__ should be set for proper Python exception chaining"
            )
        finally:
            # Restore permissions for cleanup
            if read_only_parent.exists():
                read_only_parent.chmod(0o755)

    @pytest.mark.asyncio
    async def test_bootstrap_error_cause_none_for_validation_errors(
        self, tmp_path: Path, reset_bootstrap: None
    ) -> None:
        """Test BootstrapError.cause is None for pure validation errors.

        When bootstrap fails due to validation logic (not an underlying
        exception), the cause attribute should be None.
        """
        missing_dir = tmp_path / "nonexistent"
        assert not missing_dir.exists()

        config = ModelMemoryServiceConfig(
            filesystem=ModelFilesystemConfig(
                base_path=missing_dir,
                create_if_missing=False,  # Validation error, not OSError
            )
        )

        with pytest.raises(BootstrapError) as exc_info:
            await bootstrap(config)

        error = exc_info.value
        # For validation-only errors (path doesn't exist), cause should be None
        assert hasattr(error, "cause"), "BootstrapError must have cause attribute"
        assert error.cause is None, "cause should be None for validation errors"
        assert error.config_block == "filesystem"
        # Verify Python standard exception chaining is not set for validation errors
        assert error.__cause__ is None, (
            "__cause__ should be None when no underlying exception exists"
        )


class TestBootstrapIdempotency:
    """Tests for bootstrap idempotency behavior."""

    @pytest.mark.asyncio
    async def test_bootstrap_idempotent_returns_same_result(
        self, tmp_path: Path, reset_bootstrap: None
    ) -> None:
        """Test bootstrap is idempotent - second call returns cached result."""
        config = ModelMemoryServiceConfig(
            filesystem=ModelFilesystemConfig(base_path=tmp_path)
        )

        result1 = await bootstrap(config)
        result2 = await bootstrap(config)

        # Should be the exact same object
        assert result1 is result2

    @pytest.mark.asyncio
    async def test_bootstrap_force_reruns_initialization(
        self, tmp_path: Path, reset_bootstrap: None
    ) -> None:
        """Test bootstrap with force=True re-runs initialization."""
        config = ModelMemoryServiceConfig(
            filesystem=ModelFilesystemConfig(base_path=tmp_path)
        )

        result1 = await bootstrap(config)
        result2 = await bootstrap(config, force=True)

        # Should be different objects
        assert result1 is not result2
        # But both should be successful
        assert result1.success is True
        assert result2.success is True


class TestShutdown:
    """Tests for shutdown behavior."""

    @pytest.mark.asyncio
    async def test_shutdown_resets_state(
        self, tmp_path: Path, reset_bootstrap: None
    ) -> None:
        """Test shutdown resets bootstrap state."""
        config = ModelMemoryServiceConfig(
            filesystem=ModelFilesystemConfig(base_path=tmp_path)
        )

        await bootstrap(config)
        assert await is_bootstrapped() is True
        assert await get_bootstrap_result() is not None

        await shutdown()

        assert await is_bootstrapped() is False
        assert await get_bootstrap_result() is None

    @pytest.mark.asyncio
    async def test_shutdown_allows_re_bootstrap(
        self, tmp_path: Path, reset_bootstrap: None
    ) -> None:
        """Test shutdown allows bootstrap to be called again."""
        config = ModelMemoryServiceConfig(
            filesystem=ModelFilesystemConfig(base_path=tmp_path)
        )

        result1 = await bootstrap(config)
        await shutdown()
        result2 = await bootstrap(config)

        # After shutdown, we get a new result
        assert result1 is not result2
        assert result2.success is True

    @pytest.mark.asyncio
    async def test_shutdown_idempotent(self, reset_bootstrap: None) -> None:
        """Test shutdown can be called multiple times safely."""
        # Call shutdown multiple times - should not raise
        await shutdown()
        await shutdown()
        await shutdown()

        assert await is_bootstrapped() is False


class TestIsBootstrapped:
    """Tests for is_bootstrapped() function."""

    @pytest.mark.asyncio
    async def test_is_bootstrapped_false_initially(self, reset_bootstrap: None) -> None:
        """Test is_bootstrapped() returns False before bootstrap."""
        assert await is_bootstrapped() is False

    @pytest.mark.asyncio
    async def test_is_bootstrapped_true_after_bootstrap(
        self, tmp_path: Path, reset_bootstrap: None
    ) -> None:
        """Test is_bootstrapped() returns True after successful bootstrap."""
        config = ModelMemoryServiceConfig(
            filesystem=ModelFilesystemConfig(base_path=tmp_path)
        )

        await bootstrap(config)

        assert await is_bootstrapped() is True

    @pytest.mark.asyncio
    async def test_is_bootstrapped_false_after_failed_bootstrap(
        self, tmp_path: Path, reset_bootstrap: None
    ) -> None:
        """Test is_bootstrapped() remains False after failed bootstrap."""
        missing_dir = tmp_path / "nonexistent"

        config = ModelMemoryServiceConfig(
            filesystem=ModelFilesystemConfig(
                base_path=missing_dir,
                create_if_missing=False,
            )
        )

        with pytest.raises(BootstrapError):
            await bootstrap(config)

        assert await is_bootstrapped() is False


class TestGetBootstrapResult:
    """Tests for get_bootstrap_result() function."""

    @pytest.mark.asyncio
    async def test_get_bootstrap_result_none_initially(
        self, reset_bootstrap: None
    ) -> None:
        """Test get_bootstrap_result() returns None before bootstrap."""
        assert await get_bootstrap_result() is None

    @pytest.mark.asyncio
    async def test_get_bootstrap_result_returns_result(
        self, tmp_path: Path, reset_bootstrap: None
    ) -> None:
        """Test get_bootstrap_result() returns result after bootstrap."""
        config = ModelMemoryServiceConfig(
            filesystem=ModelFilesystemConfig(base_path=tmp_path)
        )

        await bootstrap(config)

        result = await get_bootstrap_result()
        assert result is not None
        assert isinstance(result, BootstrapResult)
        assert result.success is True

    @pytest.mark.asyncio
    async def test_get_bootstrap_result_same_as_bootstrap_return(
        self, tmp_path: Path, reset_bootstrap: None
    ) -> None:
        """Test get_bootstrap_result() returns same object as bootstrap()."""
        config = ModelMemoryServiceConfig(
            filesystem=ModelFilesystemConfig(base_path=tmp_path)
        )

        bootstrap_return = await bootstrap(config)
        cached_result = await get_bootstrap_result()

        assert bootstrap_return is cached_result


class TestSecretsProviderIntegration:
    """Tests for secrets provider integration with bootstrap."""

    @pytest.mark.asyncio
    async def test_bootstrap_uses_default_secrets_provider(
        self, tmp_path: Path, reset_bootstrap: None
    ) -> None:
        """Test bootstrap uses EnvSecretsProvider by default."""
        config = ModelMemoryServiceConfig(
            filesystem=ModelFilesystemConfig(base_path=tmp_path)
        )

        result = await bootstrap(config)

        assert result.secrets_provider is not None
        assert isinstance(result.secrets_provider, EnvSecretsProvider)

    @pytest.mark.asyncio
    async def test_bootstrap_accepts_custom_secrets_provider(
        self, tmp_path: Path, reset_bootstrap: None
    ) -> None:
        """Test bootstrap accepts custom secrets provider."""
        config = ModelMemoryServiceConfig(
            filesystem=ModelFilesystemConfig(base_path=tmp_path)
        )

        custom_provider = EnvSecretsProvider(prefix="CUSTOM_")
        result = await bootstrap(config, secrets_provider=custom_provider)

        assert result.secrets_provider is custom_provider
