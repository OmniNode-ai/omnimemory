# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Bootstrap and initialization for OmniMemory service.

Provides explicit, idempotent initialization that validates configuration
and prepares handlers for use. Does not depend on FastAPI.

Example:
    .. code-block:: python

        import asyncio
        from omnimemory.bootstrap import bootstrap, BootstrapResult
        from omnimemory.models.config import (
            ModelFilesystemConfig,
            ModelMemoryServiceConfig,
        )
        from pathlib import Path

        async def main() -> None:
            config = ModelMemoryServiceConfig(
                filesystem=ModelFilesystemConfig(base_path=Path("/data/memory"))
            )
            result = await bootstrap(config)

            if result.success:
                print(f"Initialized backends: {result.initialized_backends}")

        asyncio.run(main())
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from omnimemory.secrets import EnvSecretsProvider

if TYPE_CHECKING:
    from omnimemory.models.config import (
        ModelFilesystemConfig,
        ModelMemoryServiceConfig,
        ModelPostgresConfig,
        ModelQdrantConfig,
    )
    from omnimemory.protocols import ProtocolSecretsProvider

logger = logging.getLogger(__name__)


class BootstrapError(Exception):
    """Error during bootstrap initialization.

    Attributes:
        config_block: Name of the configuration block that failed
            (e.g., "filesystem", "postgres").
        cause: The underlying exception that caused the failure, if any.
    """

    def __init__(
        self, message: str, config_block: str, cause: Exception | None = None
    ) -> None:
        """Initialize BootstrapError.

        Args:
            message: Human-readable description of the failure.
            config_block: Name of the configuration block that failed.
            cause: The underlying exception, if any.
        """
        self.config_block = config_block
        self.cause = cause
        super().__init__(f"Bootstrap failed [{config_block}]: {message}")
        # Set __cause__ for proper Python exception chaining (enables traceback display)
        if cause is not None:
            self.__cause__ = cause


@dataclass
class BootstrapResult:
    """Result of bootstrap initialization.

    Attributes:
        success: True if bootstrap completed successfully.
        initialized_backends: List of backend names that were initialized.
        warnings: List of non-fatal warnings encountered during initialization.
        secrets_provider: The secrets provider instance used during bootstrap.
    """

    success: bool
    initialized_backends: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    secrets_provider: ProtocolSecretsProvider | None = None


class _BootstrapState:
    """Singleton state manager for bootstrap operations.

    Manages all bootstrap state including completion status, cached results,
    and the asyncio lock for serializing operations. Uses class-level attributes
    to avoid global statements.
    """

    # State for idempotency
    _completed: bool = False
    _result: BootstrapResult | None = None
    # Lock to serialize bootstrap/shutdown operations and prevent race conditions
    _lock: asyncio.Lock | None = None
    # Track which event loop the lock was created for (to detect cross-loop reuse)
    _lock_loop: asyncio.AbstractEventLoop | None = None

    @classmethod
    def is_completed(cls) -> bool:
        """Check if bootstrap has completed."""
        return cls._completed

    @classmethod
    def set_completed(cls, value: bool) -> None:
        """Set bootstrap completion status."""
        cls._completed = value

    @classmethod
    def get_result(cls) -> BootstrapResult | None:
        """Get cached bootstrap result."""
        return cls._result

    @classmethod
    def set_result(cls, value: BootstrapResult | None) -> None:
        """Set bootstrap result."""
        cls._result = value

    @classmethod
    def get_lock(cls) -> asyncio.Lock:
        """Get or create the bootstrap lock.

        Creates the lock lazily to avoid issues with event loop not being available
        at module import time. Also handles cross-event-loop reuse by detecting when
        the current event loop differs from the one the lock was created in, and
        recreating the lock in that case.

        This is important because asyncio.Lock is bound to an event loop - using a
        lock from a different loop will raise RuntimeError. This commonly occurs
        in test environments where each test may create a new event loop.

        Returns:
            The asyncio.Lock for serializing bootstrap/shutdown operations.
        """
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop - create lock anyway, it will be bound when first used
            if cls._lock is None:
                cls._lock = asyncio.Lock()
                cls._lock_loop = None
            return cls._lock

        # Check if lock exists and was created in a different event loop
        if cls._lock is not None and cls._lock_loop is not current_loop:
            # Lock was created in a different event loop - recreate it
            logger.debug(
                "Detected event loop change, recreating bootstrap lock "
                f"(old loop: {id(cls._lock_loop)}, new loop: {id(current_loop)})"
            )
            cls._lock = None
            cls._lock_loop = None

        if cls._lock is None:
            cls._lock = asyncio.Lock()
            cls._lock_loop = current_loop

        return cls._lock

    @classmethod
    def reset(cls) -> None:
        """Reset all bootstrap state (used during shutdown)."""
        cls._completed = False
        cls._result = None


async def _validate_filesystem_config(config: ModelFilesystemConfig) -> list[str]:
    """Validate filesystem configuration.

    Validates that the base_path exists or can be created, is a directory,
    and is writable.

    Args:
        config: Filesystem configuration to validate.

    Returns:
        List of warnings (empty if all good).

    Raises:
        BootstrapError: If validation fails.
    """
    warnings: list[str] = []
    base_path = config.base_path

    # Check if path exists or can be created
    if not base_path.exists():
        if config.create_if_missing:
            try:
                base_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created filesystem base_path: {base_path}")
            except OSError as e:
                raise BootstrapError(
                    f"Cannot create base_path '{base_path}': {e}",
                    config_block="filesystem",
                    cause=e,
                )
        else:
            raise BootstrapError(
                f"base_path '{base_path}' does not exist and create_if_missing=False",
                config_block="filesystem",
            )

    # Check if path is a directory
    if not base_path.is_dir():
        raise BootstrapError(
            f"base_path '{base_path}' exists but is not a directory",
            config_block="filesystem",
        )

    # Check write permissions by creating and removing a unique temp file
    # Use tempfile to avoid overwriting/deleting any pre-existing files
    try:
        fd, temp_path = tempfile.mkstemp(
            prefix=".omnimemory_write_test_", dir=base_path
        )
        try:
            # Close the file descriptor
            os.close(fd)
        finally:
            # Clean up the temp file
            Path(temp_path).unlink()
    except OSError as e:
        raise BootstrapError(
            f"base_path '{base_path}' is not writable: {e}",
            config_block="filesystem",
            cause=e,
        )

    return warnings


async def _validate_postgres_config(config: ModelPostgresConfig) -> list[str]:
    """Validate PostgreSQL configuration.

    Note: Does not test actual connection - that happens at runtime.
    Validates that the DSN is properly formed.

    Args:
        config: PostgreSQL configuration to validate.

    Returns:
        List of warnings.

    Raises:
        BootstrapError: If validation fails.
    """
    warnings: list[str] = []

    # Extract host and port from DSN
    # In Pydantic v2, PostgresDsn uses hosts() which returns a list of dicts
    dsn = config.dsn
    hosts_info = dsn.hosts()
    if not hosts_info:
        raise BootstrapError(
            "PostgreSQL DSN must include a host", config_block="postgres"
        )
    first_host = hosts_info[0]
    host = first_host.get("host")
    if not host:
        raise BootstrapError(
            "PostgreSQL DSN must include a host", config_block="postgres"
        )
    port = first_host.get("port", 5432)

    logger.info(f"PostgreSQL configured: {host}:{port}")

    return warnings


async def _validate_qdrant_config(config: ModelQdrantConfig) -> list[str]:
    """Validate Qdrant configuration.

    Note: Does not test actual connection - that happens at runtime.
    Validates URL format and warns about missing authentication.

    Args:
        config: Qdrant configuration to validate.

    Returns:
        List of warnings.

    Raises:
        BootstrapError: If validation fails.
    """
    warnings: list[str] = []

    # URL validation is already done by Pydantic's HttpUrl
    logger.info(f"Qdrant configured: {config.url}")

    if config.api_key is None:
        warnings.append("Qdrant API key not configured - using unauthenticated access")

    return warnings


async def bootstrap(
    config: ModelMemoryServiceConfig,
    secrets_provider: ProtocolSecretsProvider | None = None,
    force: bool = False,
) -> BootstrapResult:
    """Initialize OmniMemory with the provided configuration.

    This function is idempotent - calling it multiple times with the same
    config will return the cached result unless force=True.

    Args:
        config: Memory service configuration containing backend configs.
        secrets_provider: Optional secrets provider (defaults to EnvSecretsProvider).
        force: If True, re-run bootstrap even if already completed.

    Returns:
        BootstrapResult with initialization status.

    Raises:
        BootstrapError: If initialization fails. The error includes
            config_block to indicate which configuration failed.

    Example:
        .. code-block:: python

            async def initialize_service() -> None:
                config = ModelMemoryServiceConfig(
                    filesystem=ModelFilesystemConfig(base_path=Path("/data/memory"))
                )
                result = await bootstrap(config)

                if not result.success:
                    print("Bootstrap failed")
                else:
                    print(f"Initialized: {result.initialized_backends}")
    """
    # Serialize bootstrap operations to avoid race conditions
    async with _BootstrapState.get_lock():
        # Idempotency check (inside lock to ensure thread-safety)
        if _BootstrapState.is_completed() and not force:
            logger.debug("Bootstrap already completed, returning cached result")
            cached_result = _BootstrapState.get_result()
            assert cached_result is not None
            return cached_result

        logger.info("Starting OmniMemory bootstrap...")

        # Initialize secrets provider
        if secrets_provider is None:
            secrets_provider = EnvSecretsProvider(prefix="OMNIMEMORY_")

        initialized_backends: list[str] = []
        all_warnings: list[str] = []

        # Validate filesystem (required)
        logger.info("Validating filesystem configuration...")
        warnings = await _validate_filesystem_config(config.filesystem)
        all_warnings.extend(warnings)
        initialized_backends.append("filesystem")

        # Validate postgres (optional)
        if config.postgres is not None:
            logger.info("Validating PostgreSQL configuration...")
            warnings = await _validate_postgres_config(config.postgres)
            all_warnings.extend(warnings)
            initialized_backends.append("postgres")

        # Validate qdrant (optional)
        if config.qdrant is not None:
            logger.info("Validating Qdrant configuration...")
            warnings = await _validate_qdrant_config(config.qdrant)
            all_warnings.extend(warnings)
            initialized_backends.append("qdrant")

        # Build result
        result = BootstrapResult(
            success=True,
            initialized_backends=initialized_backends,
            warnings=all_warnings,
            secrets_provider=secrets_provider,
        )

        # Cache for idempotency
        _BootstrapState.set_completed(True)
        _BootstrapState.set_result(result)

        logger.info(f"Bootstrap complete. Backends: {initialized_backends}")
        if all_warnings:
            for warning in all_warnings:
                logger.warning(f"Bootstrap warning: {warning}")

    return result


async def shutdown() -> None:
    """Cleanup resources initialized during bootstrap.

    Call this when shutting down the service to release any
    resources acquired during bootstrap. Resets the bootstrap
    state so bootstrap() can be called again.
    """
    # Serialize shutdown operations to avoid race conditions with bootstrap
    async with _BootstrapState.get_lock():
        logger.info("Shutting down OmniMemory...")

        # Reset bootstrap state
        _BootstrapState.reset()

        logger.info("Shutdown complete")


async def is_bootstrapped() -> bool:
    """Check if bootstrap has been completed.

    This function is protected by the bootstrap lock to ensure
    consistent reads when bootstrap/shutdown may be in progress.

    Returns:
        True if bootstrap() has been called successfully.
    """
    async with _BootstrapState.get_lock():
        return _BootstrapState.is_completed()


async def get_bootstrap_result() -> BootstrapResult | None:
    """Get the cached bootstrap result.

    This function is protected by the bootstrap lock to ensure
    consistent reads when bootstrap/shutdown may be in progress.

    Returns:
        BootstrapResult if bootstrap completed, None otherwise.
    """
    async with _BootstrapState.get_lock():
        return _BootstrapState.get_result()
