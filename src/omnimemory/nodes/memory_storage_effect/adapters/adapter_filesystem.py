# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""FileSystem Handler Adapter for memory storage operations.

This module provides an adapter that wraps `HandlerFileSystem` from omnibase_infra
to implement memory CRUD operations for storing, retrieving, and managing memory
snapshots as JSON files on the filesystem.

The adapter translates between the memory storage request/response models and
the underlying filesystem handler's envelope-based protocol.

Example::

    import asyncio
    from omnimemory.nodes.memory_storage_effect.handlers import (
        HandlerFileSystemAdapter,
        ModelFileSystemAdapterConfig,
    )
    from pathlib import Path

    async def example():
        config = ModelFileSystemAdapterConfig(base_path=Path("/data/memory"))
        adapter = HandlerFileSystemAdapter(config)
        await adapter.initialize()

        # Store a snapshot
        request = ModelMemoryStorageRequest(operation="store", snapshot=my_snapshot)
        response = await adapter.execute(request)

    asyncio.run(example())

.. versionadded:: 0.1.0
    Initial implementation for OMN-1384.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from pathlib import Path

from pydantic import ValidationError

# omnibase_infra is a dev dependency - make imports conditional
# to allow test collection and provide clear error messages
_OMNIBASE_INFRA_AVAILABLE = False
_OMNIBASE_INFRA_IMPORT_ERROR: str | None = None

try:
    from omnibase_infra.errors.error_infra import InfraConnectionError
    from omnibase_infra.handlers.handler_filesystem import HandlerFileSystem

    _OMNIBASE_INFRA_AVAILABLE = True
except ImportError as e:
    _OMNIBASE_INFRA_IMPORT_ERROR = str(e)

    # Provide stub types for type checking and to allow module to load
    class InfraConnectionError(Exception):  # type: ignore[no-redef]
        """Stub for InfraConnectionError when omnibase_infra is not installed."""

    class HandlerFileSystem:  # type: ignore[no-redef]
        """Stub for HandlerFileSystem when omnibase_infra is not installed."""

        def __init__(self) -> None:
            raise ImportError(
                f"omnibase_infra is required for HandlerFileSystemAdapter. "
                f"Install it with: poetry install --with dev. "
                f"Original error: {_OMNIBASE_INFRA_IMPORT_ERROR}"
            )


from omnimemory.models.adapters import ModelFileSystemAdapterConfig

from ..models import (
    ModelMemoryStorageRequest,
    ModelMemoryStorageResponse,
)

logger = logging.getLogger(__name__)

__all__ = [
    "HandlerFileSystemAdapter",
    "ModelFileSystemAdapterConfig",
]


def _is_not_found_infra_error(error: InfraConnectionError) -> bool:
    """Check if an InfraConnectionError indicates a "not found" condition.

    This helper uses a prioritized detection strategy to identify "not found"
    errors from the filesystem handler:

    Detection Priority:
        1. Check __cause__ for FileNotFoundError (standard Python exception)
        2. Check error_code for RESOURCE_NOT_FOUND variants
        3. Fall back to message pattern matching (fragile, last resort)

    The filesystem handler (omnibase_infra) currently uses SERVICE_UNAVAILABLE
    error code for all filesystem errors. This function provides forward
    compatibility by checking structured attributes before falling back to
    message parsing.

    Args:
        error: The InfraConnectionError to check.

    Returns:
        True if the error indicates a resource was not found, False otherwise.

    Note:
        Priority 1 and 2 are preferred as they use structured data. Priority 3
        (message matching) is fragile and depends on message format stability
        in omnibase_infra. This is documented as a single point of maintenance.
    """
    # Priority 1: Check the underlying cause chain for FileNotFoundError
    cause: BaseException | None = error.__cause__
    while cause is not None:
        if isinstance(cause, FileNotFoundError):
            return True
        cause = cause.__cause__

    # Priority 2: Check error_code attribute for RESOURCE_NOT_FOUND variants
    # InfraConnectionError inherits error_code from ModelOnexError
    error_code = getattr(error, "error_code", None)
    if error_code is not None:
        # Check both enum value and string representation
        error_code_str = str(error_code).upper()
        if any(
            pattern in error_code_str
            for pattern in ("RESOURCE_NOT_FOUND", "NOT_FOUND", "FILE_NOT_FOUND")
        ):
            return True

    # Priority 3: Fall back to message pattern matching (fragile, last resort)
    # Handler message patterns: "File not found: {name}", "Directory not found: {name}"
    message = str(error).lower()
    return "not found:" in message


def _is_permission_denied_infra_error(error: InfraConnectionError) -> bool:
    """Check if an InfraConnectionError indicates a permission denied condition.

    This helper uses a prioritized detection strategy to identify permission
    errors from the filesystem handler:

    Detection Priority:
        1. Check __cause__ for PermissionError (standard Python exception)
        2. Check error_code for PERMISSION_DENIED variants
        3. Fall back to message pattern matching (fragile, last resort)

    The filesystem handler (omnibase_infra) currently uses SERVICE_UNAVAILABLE
    error code for all filesystem errors. This function provides forward
    compatibility by checking structured attributes before falling back to
    message parsing.

    Args:
        error: The InfraConnectionError to check.

    Returns:
        True if the error indicates permission was denied, False otherwise.

    Note:
        Priority 1 and 2 are preferred as they use structured data. Priority 3
        (message matching) is fragile and depends on message format stability
        in omnibase_infra. This is documented as a single point of maintenance.
    """
    # Priority 1: Check the underlying cause chain for PermissionError
    cause: BaseException | None = error.__cause__
    while cause is not None:
        if isinstance(cause, PermissionError):
            return True
        cause = cause.__cause__

    # Priority 2: Check error_code attribute for permission variants
    # InfraConnectionError inherits error_code from ModelOnexError
    error_code = getattr(error, "error_code", None)
    if error_code is not None:
        # Check both enum value and string representation
        error_code_str = str(error_code).upper()
        if any(
            pattern in error_code_str
            for pattern in ("PERMISSION_DENIED", "ACCESS_DENIED", "FORBIDDEN")
        ):
            return True

    # Priority 3: Fall back to message pattern matching (fragile, last resort)
    # Handler message patterns: "permission denied", "access denied"
    message = str(error).lower()
    return "permission denied" in message or "access denied" in message


class HandlerFileSystemAdapter:
    """Adapter that wraps HandlerFileSystem for memory storage operations.

    This adapter provides a high-level interface for memory CRUD operations
    on top of the low-level filesystem handler. It manages:

    - Directory structure creation and management
    - Snapshot serialization/deserialization to/from JSON
    - Translation between memory models and filesystem envelopes
    - Error handling and status reporting

    Directory structure:
        <base_path>/
        +-- snapshots/
            +-- <snapshot_id>.json
            +-- <snapshot_id>.json
            +-- ...

    Attributes:
        config: The adapter configuration.
        handler: The underlying HandlerFileSystem instance.

    Example::

        async def example():
            config = ModelFileSystemAdapterConfig(base_path=Path("/data"))
            adapter = HandlerFileSystemAdapter(config)
            await adapter.initialize()

            # Store operation
            store_req = ModelMemoryStorageRequest(
                operation="store",
                snapshot=snapshot
            )
            response = await adapter.execute(store_req)
            assert response.status == "success"
    """

    def __init__(self, config: ModelFileSystemAdapterConfig) -> None:
        """Initialize the adapter with configuration.

        Args:
            config: The adapter configuration specifying paths and limits.
        """
        self._config = config
        self._handler = HandlerFileSystem()
        self._snapshots_path = config.base_path / config.snapshots_dir
        self._initialized = False
        self._init_lock = asyncio.Lock()

    @property
    def config(self) -> ModelFileSystemAdapterConfig:
        """Get the adapter configuration."""
        return self._config

    @property
    def handler(self) -> HandlerFileSystem:
        """Get the underlying filesystem handler."""
        return self._handler

    @property
    def snapshots_path(self) -> Path:
        """Get the path where snapshots are stored."""
        return self._snapshots_path

    @property
    def is_initialized(self) -> bool:
        """Check if the adapter has been initialized."""
        return self._initialized

    def _validate_snapshot_id(self, snapshot_id: str) -> str | None:
        """Validate snapshot_id for path traversal attacks.

        This method checks that the snapshot_id does not contain characters
        or sequences that could be used to escape the snapshots directory.

        Args:
            snapshot_id: The snapshot identifier to validate.

        Returns:
            None if valid, error message string if invalid.

        Example:
            >>> error = self._validate_snapshot_id("my-snapshot-123")
            >>> assert error is None  # Valid
            >>> error = self._validate_snapshot_id("../../../etc/passwd")
            >>> assert error is not None  # Invalid - path traversal
        """
        if not snapshot_id:
            return "snapshot_id cannot be empty"
        if "/" in snapshot_id or "\\" in snapshot_id:
            return "snapshot_id contains invalid path characters"
        if ".." in snapshot_id:
            return "snapshot_id contains path traversal sequence"
        # Verify resolved path is within snapshots directory
        file_path = (self._snapshots_path / f"{snapshot_id}.json").resolve()
        if not str(file_path).startswith(str(self._snapshots_path.resolve())):
            return "snapshot_id resolves outside allowed directory"
        return None

    async def initialize(self) -> None:
        """Initialize the handler and create directories.

        This method must be called before any execute operations. It:
        1. Initializes the underlying filesystem handler with allowed paths
        2. Creates the snapshots directory if it doesn't exist

        Thread-safe: Uses asyncio.Lock to prevent concurrent initialization.

        Raises:
            RuntimeError: If initialization fails or directory creation fails.
            PermissionError: If permission denied creating directories.
            OSError: If filesystem operation fails.
        """
        # Fast path: already initialized (no lock needed for read)
        if self._initialized:
            return

        async with self._init_lock:
            # Double-check after acquiring lock (another call may have completed)
            if self._initialized:
                return

            try:
                # Initialize underlying handler with allowed paths
                await self._handler.initialize(
                    {
                        "allowed_paths": [str(self._config.base_path)],
                        "max_read_size": self._config.max_file_size,
                        "max_write_size": self._config.max_file_size,
                    }
                )

                # Create snapshots directory using the handler
                envelope = self._build_envelope(
                    operation="filesystem.ensure_directory",
                    payload={"path": str(self._snapshots_path)},
                )
                result = await self._handler.execute(envelope)

                # Verify directory creation succeeded
                if result.result is None or result.result.get("status") != "success":
                    error_msg = "Directory creation returned no result"
                    if result.result:
                        error_msg = result.result.get(
                            "error", "Directory creation failed"
                        )
                    raise RuntimeError(
                        f"Failed to create snapshots directory "
                        f"'{self._snapshots_path}': {error_msg}"
                    )

                # Defense-in-depth: Verify directory actually exists on filesystem
                # This guards against handler bugs or edge cases where success is
                # reported but the directory was not actually created
                if not self._snapshots_path.exists():
                    raise RuntimeError(
                        f"Directory creation reported success but "
                        f"'{self._snapshots_path}' does not exist on filesystem"
                    )
                if not self._snapshots_path.is_dir():
                    raise RuntimeError(
                        f"Path '{self._snapshots_path}' exists but is not a directory"
                    )

                self._initialized = True
                logger.info(
                    "FileSystem adapter initialized with snapshots path: %s",
                    self._snapshots_path,
                )

            except PermissionError as e:
                logger.error(
                    "Permission denied initializing adapter at %s: %s",
                    self._snapshots_path,
                    e,
                )
                raise
            except TimeoutError as e:
                # TimeoutError must be caught before OSError (it's a subclass)
                logger.error(
                    "Timeout during initialization at %s: %s",
                    self._snapshots_path,
                    e,
                )
                raise RuntimeError(f"Initialization timed out: {e}") from e
            except OSError as e:
                logger.error(
                    "OS error initializing adapter at %s: %s",
                    self._snapshots_path,
                    e,
                )
                raise
            except RuntimeError:
                # Re-raise RuntimeError without wrapping
                raise
            except InfraConnectionError as e:
                logger.error(
                    "Infrastructure error initializing adapter at %s: %s",
                    self._snapshots_path,
                    e,
                )
                raise RuntimeError(f"Initialization failed: {e}") from e
            except Exception as e:
                # Safety net for truly unexpected errors - log at ERROR level
                logger.error(
                    "Unexpected error during initialization: %s (type: %s)",
                    e,
                    type(e).__name__,
                )
                raise RuntimeError(f"Initialization failed: {e}") from e

    async def execute(
        self, request: ModelMemoryStorageRequest
    ) -> ModelMemoryStorageResponse:
        """Execute a memory storage operation.

        This method routes the request to the appropriate handler method
        based on the operation type.

        Args:
            request: The storage operation request.

        Returns:
            Response with operation status and any retrieved data.

        Raises:
            RuntimeError: If the adapter is not initialized.
        """
        if not self._initialized:
            await self.initialize()

        match request.operation:
            case "store":
                return await self._store(request)
            case "retrieve":
                return await self._retrieve(request)
            case "delete":
                return await self._delete(request)
            case "update":
                return await self._update(request)
            case "list":
                return await self._list(request)
            case _:
                return ModelMemoryStorageResponse(
                    status="error",
                    error_message=f"Unknown operation: {request.operation}",
                )

    async def _store(
        self, request: ModelMemoryStorageRequest
    ) -> ModelMemoryStorageResponse:
        """Store a memory snapshot.

        Args:
            request: The store request containing the snapshot.

        Returns:
            Response with success status and stored snapshot, or error.
        """
        if request.snapshot is None:
            return ModelMemoryStorageResponse(
                status="error",
                error_message="snapshot is required for store operation",
            )

        snapshot_id = str(request.snapshot.snapshot_id)

        # Validate snapshot_id for path traversal attacks
        validation_error = self._validate_snapshot_id(snapshot_id)
        if validation_error:
            logger.warning(
                "Invalid snapshot_id rejected in store operation: %s - %s",
                snapshot_id,
                validation_error,
            )
            return ModelMemoryStorageResponse(
                status="error",
                error_message=f"Invalid snapshot_id: {validation_error}",
            )

        # Serialize snapshot and check file size limit
        content = request.snapshot.model_dump_json(indent=2)
        content_bytes = content.encode("utf-8")
        if len(content_bytes) > self._config.max_file_size:
            logger.warning(
                "Snapshot %s exceeds max file size: %d > %d bytes",
                snapshot_id,
                len(content_bytes),
                self._config.max_file_size,
            )
            return ModelMemoryStorageResponse(
                status="error",
                error_message=(
                    f"Snapshot exceeds maximum file size of "
                    f"{self._config.max_file_size} bytes"
                ),
            )

        file_path = self._snapshots_path / f"{snapshot_id}.json"

        envelope = self._build_envelope(
            operation="filesystem.write_file",
            payload={
                "path": str(file_path),
                "content": content,
            },
        )

        try:
            result = await self._handler.execute(envelope)
            if result.result and result.result.get("status") == "success":
                return ModelMemoryStorageResponse(
                    status="success",
                    snapshot=request.snapshot,
                )
            # Extract specific error from result if available
            error_msg = "Write operation failed"
            if result.result:
                error_msg = result.result.get("error", error_msg)
            return ModelMemoryStorageResponse(
                status="error",
                error_message=error_msg,
            )
        except PermissionError as e:
            logger.warning("Permission denied storing snapshot %s: %s", snapshot_id, e)
            return ModelMemoryStorageResponse(
                status="permission_denied",
                error_message=f"Permission denied writing to {file_path}: {e}",
            )
        except TimeoutError as e:
            # TimeoutError must be caught before OSError (it's a subclass)
            logger.warning("Timeout storing snapshot %s: %s", snapshot_id, e)
            return ModelMemoryStorageResponse(
                status="error",
                error_message=f"Store operation timed out: {e}",
            )
        except OSError as e:
            logger.warning("I/O error storing snapshot %s: %s", snapshot_id, e)
            return ModelMemoryStorageResponse(
                status="error",
                error_message=f"I/O error storing snapshot: {e}",
            )
        except InfraConnectionError as e:
            # Check for permission denied first
            if _is_permission_denied_infra_error(e):
                logger.warning(
                    "Permission denied storing snapshot %s: %s",
                    snapshot_id,
                    e,
                )
                return ModelMemoryStorageResponse(
                    status="permission_denied",
                    error_message=f"Permission denied writing to {file_path}: {e}",
                )
            # Handler raises InfraConnectionError for various filesystem issues
            logger.warning(
                "Infrastructure error storing snapshot %s: %s",
                snapshot_id,
                e,
            )
            return ModelMemoryStorageResponse(
                status="error",
                error_message=f"Store failed: {e}",
            )
        except UnicodeEncodeError as e:
            logger.warning(
                "Unicode encoding error storing snapshot %s: %s",
                snapshot_id,
                e,
            )
            return ModelMemoryStorageResponse(
                status="error",
                error_message=f"Snapshot contains invalid characters: {e}",
            )
        except Exception as e:
            # Safety net for truly unexpected errors - log at ERROR level
            logger.error(
                "Unexpected error storing snapshot %s: %s (type: %s)",
                snapshot_id,
                e,
                type(e).__name__,
            )
            return ModelMemoryStorageResponse(
                status="error",
                error_message=f"Store failed: {e}",
            )

    async def _retrieve(
        self, request: ModelMemoryStorageRequest
    ) -> ModelMemoryStorageResponse:
        """Retrieve a memory snapshot by ID.

        Args:
            request: The retrieve request containing the snapshot ID.

        Returns:
            Response with success status and retrieved snapshot, or not_found.
        """
        if request.snapshot_id is None:
            return ModelMemoryStorageResponse(
                status="error",
                error_message="snapshot_id is required for retrieve operation",
            )

        # Validate snapshot_id for path traversal attacks
        validation_error = self._validate_snapshot_id(request.snapshot_id)
        if validation_error:
            logger.warning(
                "Invalid snapshot_id rejected in retrieve operation: %s - %s",
                request.snapshot_id,
                validation_error,
            )
            return ModelMemoryStorageResponse(
                status="error",
                error_message=f"Invalid snapshot_id: {validation_error}",
            )

        file_path = self._snapshots_path / f"{request.snapshot_id}.json"

        envelope = self._build_envelope(
            operation="filesystem.read_file",
            payload={"path": str(file_path)},
        )

        try:
            result = await self._handler.execute(envelope)
            if result.result and result.result.get("status") == "success":
                payload = result.result.get("payload", {})
                content = payload.get("content", "")

                # Import here to avoid circular imports
                from omnibase_core.models.omnimemory import ModelMemorySnapshot

                snapshot = ModelMemorySnapshot.model_validate_json(content)
                return ModelMemoryStorageResponse(
                    status="success",
                    snapshot=snapshot,
                )
            # Check for specific not_found status in result
            if result.result and result.result.get("status") == "not_found":
                return ModelMemoryStorageResponse(
                    status="not_found",
                    error_message=f"Snapshot {request.snapshot_id} not found",
                )
            # Other non-success statuses are errors
            error_msg = "Read operation failed"
            if result.result:
                error_msg = result.result.get("error", error_msg)
            return ModelMemoryStorageResponse(
                status="error",
                error_message=error_msg,
            )
        except FileNotFoundError:
            return ModelMemoryStorageResponse(
                status="not_found",
                error_message=f"Snapshot {request.snapshot_id} not found",
            )
        except PermissionError as e:
            logger.warning(
                "Permission denied retrieving snapshot %s: %s",
                request.snapshot_id,
                e,
            )
            return ModelMemoryStorageResponse(
                status="permission_denied",
                error_message=f"Permission denied reading {file_path}: {e}",
            )
        except ValidationError as e:
            logger.warning(
                "Invalid snapshot data for %s: %s",
                request.snapshot_id,
                e,
            )
            return ModelMemoryStorageResponse(
                status="error",
                error_message=f"Invalid snapshot data format: {e}",
            )
        except TimeoutError as e:
            # TimeoutError must be caught before OSError (it's a subclass)
            logger.warning(
                "Timeout retrieving snapshot %s: %s",
                request.snapshot_id,
                e,
            )
            return ModelMemoryStorageResponse(
                status="error",
                error_message=f"Retrieve operation timed out: {e}",
            )
        except OSError as e:
            logger.warning(
                "I/O error retrieving snapshot %s: %s",
                request.snapshot_id,
                e,
            )
            return ModelMemoryStorageResponse(
                status="error",
                error_message=f"I/O error retrieving snapshot: {e}",
            )
        except InfraConnectionError as e:
            # Handler raises InfraConnectionError for file not found
            # Use helper to detect "not found" condition
            # (see _is_not_found_infra_error docs)
            if _is_not_found_infra_error(e):
                return ModelMemoryStorageResponse(
                    status="not_found",
                    error_message=f"Snapshot {request.snapshot_id} not found",
                )
            # Check for permission denied
            if _is_permission_denied_infra_error(e):
                logger.warning(
                    "Permission denied retrieving snapshot %s: %s",
                    request.snapshot_id,
                    e,
                )
                return ModelMemoryStorageResponse(
                    status="permission_denied",
                    error_message=f"Permission denied reading {file_path}: {e}",
                )
            logger.warning(
                "Infrastructure error retrieving snapshot %s: %s",
                request.snapshot_id,
                e,
            )
            return ModelMemoryStorageResponse(
                status="error",
                error_message=f"Retrieve failed: {e}",
            )
        except json.JSONDecodeError as e:
            logger.warning(
                "JSON decode error for snapshot %s: %s",
                request.snapshot_id,
                e,
            )
            return ModelMemoryStorageResponse(
                status="error",
                error_message=f"Snapshot file contains invalid JSON: {e}",
            )
        except UnicodeDecodeError as e:
            logger.warning(
                "Unicode decode error for snapshot %s: %s",
                request.snapshot_id,
                e,
            )
            return ModelMemoryStorageResponse(
                status="error",
                error_message=f"Snapshot file contains invalid encoding: {e}",
            )
        except Exception as e:
            # Safety net for truly unexpected errors - log at ERROR level
            logger.error(
                "Unexpected error retrieving snapshot %s: %s (type: %s)",
                request.snapshot_id,
                e,
                type(e).__name__,
            )
            return ModelMemoryStorageResponse(
                status="error",
                error_message=f"Retrieve failed: {e}",
            )

    async def _delete(
        self, request: ModelMemoryStorageRequest
    ) -> ModelMemoryStorageResponse:
        """Delete a memory snapshot by ID.

        Args:
            request: The delete request containing the snapshot ID.

        Returns:
            Response with success status, not_found, permission_denied, or error.
        """
        if request.snapshot_id is None:
            return ModelMemoryStorageResponse(
                status="error",
                error_message="snapshot_id is required for delete operation",
            )

        # Validate snapshot_id for path traversal attacks
        validation_error = self._validate_snapshot_id(request.snapshot_id)
        if validation_error:
            logger.warning(
                "Invalid snapshot_id rejected in delete operation: %s - %s",
                request.snapshot_id,
                validation_error,
            )
            return ModelMemoryStorageResponse(
                status="error",
                error_message=f"Invalid snapshot_id: {validation_error}",
            )

        file_path = self._snapshots_path / f"{request.snapshot_id}.json"

        envelope = self._build_envelope(
            operation="filesystem.delete_file",
            payload={"path": str(file_path)},
        )

        try:
            result = await self._handler.execute(envelope)
            if result.result and result.result.get("status") == "success":
                return ModelMemoryStorageResponse(status="success")

            # Differentiate between error types from result
            if result.result:
                result_status = result.result.get("status", "error")
                error_msg = result.result.get("error", "Delete operation failed")

                # Handle specific status codes from handler
                if result_status == "not_found":
                    return ModelMemoryStorageResponse(
                        status="not_found",
                        error_message=f"Snapshot {request.snapshot_id} not found",
                    )
                if result_status == "permission_denied":
                    return ModelMemoryStorageResponse(
                        status="permission_denied",
                        error_message=f"Permission denied deleting {file_path}",
                    )
                # Generic error with message from result
                return ModelMemoryStorageResponse(
                    status="error",
                    error_message=error_msg,
                )

            # No result at all is an error
            return ModelMemoryStorageResponse(
                status="error",
                error_message="Delete operation returned no result",
            )
        except FileNotFoundError:
            return ModelMemoryStorageResponse(
                status="not_found",
                error_message=f"Snapshot {request.snapshot_id} not found",
            )
        except PermissionError as e:
            logger.warning(
                "Permission denied deleting snapshot %s: %s",
                request.snapshot_id,
                e,
            )
            return ModelMemoryStorageResponse(
                status="permission_denied",
                error_message=f"Permission denied deleting {file_path}: {e}",
            )
        except IsADirectoryError as e:
            logger.warning(
                "Cannot delete %s - is a directory: %s",
                request.snapshot_id,
                e,
            )
            return ModelMemoryStorageResponse(
                status="error",
                error_message=f"Cannot delete: {file_path} is a directory",
            )
        except TimeoutError as e:
            # TimeoutError must be caught before OSError (it's a subclass)
            logger.warning(
                "Timeout deleting snapshot %s: %s",
                request.snapshot_id,
                e,
            )
            return ModelMemoryStorageResponse(
                status="error",
                error_message=f"Delete operation timed out: {e}",
            )
        except OSError as e:
            logger.warning(
                "I/O error deleting snapshot %s: %s",
                request.snapshot_id,
                e,
            )
            return ModelMemoryStorageResponse(
                status="error",
                error_message=f"I/O error deleting snapshot: {e}",
            )
        except InfraConnectionError as e:
            # Handler raises InfraConnectionError for various filesystem errors
            # Use helpers to detect specific conditions (see helper function docs)
            if _is_not_found_infra_error(e):
                return ModelMemoryStorageResponse(
                    status="not_found",
                    error_message=f"Snapshot {request.snapshot_id} not found",
                )
            if _is_permission_denied_infra_error(e):
                logger.warning(
                    "Permission denied deleting snapshot %s: %s",
                    request.snapshot_id,
                    e,
                )
                return ModelMemoryStorageResponse(
                    status="permission_denied",
                    error_message=f"Permission denied deleting {file_path}: {e}",
                )
            logger.warning(
                "Infrastructure error deleting snapshot %s: %s",
                request.snapshot_id,
                e,
            )
            return ModelMemoryStorageResponse(
                status="error",
                error_message=f"Delete failed: {e}",
            )
        except Exception as e:
            # Safety net for truly unexpected errors - log at ERROR level
            logger.error(
                "Unexpected error deleting snapshot %s: %s (type: %s)",
                request.snapshot_id,
                e,
                type(e).__name__,
            )
            return ModelMemoryStorageResponse(
                status="error",
                error_message=f"Delete failed: {e}",
            )

    async def _update(
        self, request: ModelMemoryStorageRequest
    ) -> ModelMemoryStorageResponse:
        """Update a memory snapshot (store with existing ID).

        For filesystem storage, update is equivalent to store since
        writing to the same path overwrites the existing file.

        Args:
            request: The update request containing the snapshot.

        Returns:
            Response with success status and updated snapshot, or error.
        """
        # For FileSystem, update is the same as store
        return await self._store(request)

    async def _list(
        self, request: ModelMemoryStorageRequest
    ) -> ModelMemoryStorageResponse:
        """List all snapshot IDs, optionally filtered by tags.

        Note: Tag filtering is not implemented for filesystem storage.
        All snapshot IDs are returned.

        Args:
            request: The list request (tags filter not currently used).

        Returns:
            Response with list of snapshot IDs, or error if listing fails.

        Note:
            Unlike some operations, this method distinguishes between an empty
            directory (success with empty list) and a failure to list the
            directory (error status).
        """
        envelope = self._build_envelope(
            operation="filesystem.list_directory",
            payload={"path": str(self._snapshots_path)},
        )

        try:
            result = await self._handler.execute(envelope)
            if result.result and result.result.get("status") == "success":
                payload = result.result.get("payload", {})
                entries = payload.get("entries", [])
                # Extract snapshot IDs from filenames
                snapshot_ids = [
                    Path(entry.get("name", "")).stem
                    for entry in entries
                    if entry.get("name", "").endswith(".json")
                ]
                return ModelMemoryStorageResponse(
                    status="success",
                    snapshot_ids=snapshot_ids,
                )

            # IMPORTANT: Non-success status is an ERROR, not empty list
            # This ensures we distinguish "empty directory" from "failed to list"
            if result.result:
                result_status = result.result.get("status", "error")
                error_msg = result.result.get("error", "List operation failed")

                # Handle specific statuses
                if result_status == "not_found":
                    return ModelMemoryStorageResponse(
                        status="error",
                        error_message=(
                            f"Snapshots directory not found: {self._snapshots_path}"
                        ),
                    )
                if result_status == "permission_denied":
                    return ModelMemoryStorageResponse(
                        status="permission_denied",
                        error_message=(
                            f"Permission denied listing {self._snapshots_path}"
                        ),
                    )
                return ModelMemoryStorageResponse(
                    status="error",
                    error_message=error_msg,
                )

            # No result at all is an error
            return ModelMemoryStorageResponse(
                status="error",
                error_message="List operation returned no result",
            )
        except FileNotFoundError:
            return ModelMemoryStorageResponse(
                status="error",
                error_message=f"Snapshots directory not found: {self._snapshots_path}",
            )
        except PermissionError as e:
            logger.warning(
                "Permission denied listing snapshots directory %s: %s",
                self._snapshots_path,
                e,
            )
            return ModelMemoryStorageResponse(
                status="permission_denied",
                error_message=f"Permission denied listing {self._snapshots_path}: {e}",
            )
        except NotADirectoryError as e:
            logger.warning(
                "Cannot list %s - not a directory: %s",
                self._snapshots_path,
                e,
            )
            return ModelMemoryStorageResponse(
                status="error",
                error_message=f"Cannot list: {self._snapshots_path} is not a directory",
            )
        except TimeoutError as e:
            # TimeoutError must be caught before OSError (it's a subclass)
            logger.warning(
                "Timeout listing snapshots in %s: %s",
                self._snapshots_path,
                e,
            )
            return ModelMemoryStorageResponse(
                status="error",
                error_message=f"List operation timed out: {e}",
            )
        except OSError as e:
            logger.warning(
                "I/O error listing snapshots directory %s: %s",
                self._snapshots_path,
                e,
            )
            return ModelMemoryStorageResponse(
                status="error",
                error_message=f"I/O error listing snapshots: {e}",
            )
        except InfraConnectionError as e:
            # Handler raises InfraConnectionError for directory not found
            # Use helper to detect "not found" condition
            # (see _is_not_found_infra_error docs)
            if _is_not_found_infra_error(e):
                return ModelMemoryStorageResponse(
                    status="error",
                    error_message=(
                        f"Snapshots directory not found: {self._snapshots_path}"
                    ),
                )
            logger.warning(
                "Infrastructure error listing snapshots in %s: %s",
                self._snapshots_path,
                e,
            )
            return ModelMemoryStorageResponse(
                status="error",
                error_message=f"List failed: {e}",
            )
        except Exception as e:
            # Safety net for truly unexpected errors - log at ERROR level
            logger.error(
                "Unexpected error listing snapshots in %s: %s (type: %s)",
                self._snapshots_path,
                e,
                type(e).__name__,
            )
            return ModelMemoryStorageResponse(
                status="error",
                error_message=f"List failed: {e}",
            )

    def _build_envelope(
        self,
        operation: str,
        payload: dict[str, object],
    ) -> dict[str, object]:
        """Build an envelope for the filesystem handler.

        Args:
            operation: The filesystem operation (e.g., "filesystem.write_file").
            payload: The operation-specific payload.

        Returns:
            A properly formatted envelope for the handler.
        """
        return {
            "envelope_id": str(uuid.uuid4()),
            "correlation_id": str(uuid.uuid4()),
            "operation": operation,
            "payload": payload,
        }

    async def shutdown(self) -> None:
        """Shutdown the handler and release resources."""
        if self._initialized:
            await self._handler.shutdown()
            self._initialized = False
