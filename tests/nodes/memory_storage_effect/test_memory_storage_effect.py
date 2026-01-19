# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Comprehensive integration tests for memory_storage_effect node.

This module tests all CRUD operations for the memory storage effect node,
including the store/retrieve round-trip, error handling, and edge cases.

Test Categories:
    - Store Operations: Test storing memory snapshots to filesystem
    - Retrieve Operations: Test fetching snapshots by ID
    - Round-Trip: Test full store/retrieve cycle with data integrity
    - Delete Operations: Test removing snapshots
    - List Operations: Test listing stored snapshot IDs
    - Update Operations: Test modifying existing snapshots
    - Error Handling: Test validation and not_found scenarios

Usage:
    pytest tests/nodes/memory_storage_effect/ -v
    pytest tests/nodes/memory_storage_effect/test_memory_storage_effect.py -v -k "store"

.. versionadded:: 0.1.0
    Initial implementation for OMN-1384.
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest
from omnibase_core.enums.enum_subject_type import EnumSubjectType
from omnibase_core.models.omnimemory import (
    ModelCostLedger,
    ModelMemorySnapshot,
    ModelSubjectRef,
)

from omnimemory.nodes.memory_storage_effect import (
    HandlerFileSystemAdapter,
    HandlerFileSystemAdapterConfig,
    ModelMemoryStorageRequest,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def adapter(tmp_path: Path) -> HandlerFileSystemAdapter:
    """Create adapter with temporary directory.

    Args:
        tmp_path: Pytest fixture providing temporary directory path.

    Returns:
        Configured HandlerFileSystemAdapter instance.
    """
    config = HandlerFileSystemAdapterConfig(base_path=tmp_path)
    adapter = HandlerFileSystemAdapter(config)
    return adapter


@pytest.fixture
def sample_snapshot() -> ModelMemorySnapshot:
    """Create a sample memory snapshot for testing.

    Returns:
        A valid ModelMemorySnapshot instance with minimal required fields.
    """
    subject = ModelSubjectRef(
        subject_type=EnumSubjectType.AGENT,
        subject_id=uuid4(),
    )
    ledger = ModelCostLedger(budget_total=100.0)
    return ModelMemorySnapshot(
        snapshot_id=uuid4(),
        subject=subject,
        cost_ledger=ledger,
        schema_version="1.0.0",
    )


@pytest.fixture
def sample_snapshot_with_id() -> ModelMemorySnapshot:
    """Create a sample memory snapshot with enriched identifier metadata fields.

    This fixture creates a snapshot with additional optional identifier-related
    fields populated: namespace, subject_key, and tags. The "with_id" suffix
    indicates that all optional identifier and metadata fields are populated,
    NOT that the snapshot_id is deterministic or fixed.

    Note:
        The snapshot_id is randomly generated using uuid4() for uniqueness.
        Each test invocation receives a different snapshot_id.
        Use this fixture when testing scenarios that require optional metadata.

    Returns:
        A valid ModelMemorySnapshot instance with optional metadata populated.
    """
    subject = ModelSubjectRef(
        subject_type=EnumSubjectType.AGENT,
        subject_id=uuid4(),
        namespace="test",
        subject_key="test-agent",
    )
    ledger = ModelCostLedger(budget_total=50.0)
    return ModelMemorySnapshot(
        snapshot_id=uuid4(),
        subject=subject,
        cost_ledger=ledger,
        schema_version="1.0.0",
        tags=("test", "sample"),
    )


def create_unique_snapshot(
    version: int = 1,
    tags: tuple[str, ...] = (),
) -> ModelMemorySnapshot:
    """Create a unique memory snapshot with specified attributes.

    Args:
        version: Version number for the snapshot.
        tags: Optional tuple of tags.

    Returns:
        A new ModelMemorySnapshot instance with unique IDs.
    """
    subject = ModelSubjectRef(
        subject_type=EnumSubjectType.AGENT,
        subject_id=uuid4(),
    )
    ledger = ModelCostLedger(budget_total=100.0)
    return ModelMemorySnapshot(
        snapshot_id=uuid4(),
        version=version,
        subject=subject,
        cost_ledger=ledger,
        schema_version="1.0.0",
        tags=tags,
    )


# =============================================================================
# Store Operation Tests
# =============================================================================


class TestStoreOperation:
    """Tests for the store operation."""

    @pytest.mark.asyncio
    async def test_store_snapshot_success(
        self,
        adapter: HandlerFileSystemAdapter,
        sample_snapshot: ModelMemorySnapshot,
    ) -> None:
        """Test storing a snapshot returns success response.

        Given: A valid memory snapshot
        When: Executing a store operation
        Then: Response status is 'success' and snapshot is returned
        """
        request = ModelMemoryStorageRequest(
            operation="store",
            snapshot=sample_snapshot,
        )

        response = await adapter.execute(request)

        assert (
            response.status == "success"
        ), f"Expected status 'success', got '{response.status}'"
        assert response.snapshot is not None, "Expected snapshot in response"
        assert (
            response.snapshot.snapshot_id == sample_snapshot.snapshot_id
        ), "Snapshot ID mismatch"
        assert (
            response.error_message is None
        ), f"Unexpected error: {response.error_message}"

    @pytest.mark.asyncio
    async def test_store_creates_file(
        self,
        adapter: HandlerFileSystemAdapter,
        sample_snapshot: ModelMemorySnapshot,
    ) -> None:
        """Test that store operation creates a file on disk.

        Given: A valid memory snapshot
        When: Executing a store operation
        Then: A JSON file is created at the expected path
        """
        request = ModelMemoryStorageRequest(
            operation="store",
            snapshot=sample_snapshot,
        )

        await adapter.execute(request)

        expected_file = adapter.snapshots_path / f"{sample_snapshot.snapshot_id}.json"
        assert expected_file.exists(), f"Expected file {expected_file} was not created"

    def test_store_without_snapshot_raises_validation_error(
        self,
    ) -> None:
        """Test that store operation without snapshot raises ValidationError.

        Given: Attempting to create a store request with no snapshot
        When: Constructing the ModelMemoryStorageRequest
        Then: Pydantic raises ValidationError with appropriate message
        """
        from pydantic import ValidationError

        with pytest.raises(ValidationError) as exc_info:
            ModelMemoryStorageRequest(
                operation="store",
                snapshot=None,
            )

        error_message = str(exc_info.value)
        assert (
            "store" in error_message.lower()
        ), f"Error should mention 'store': {error_message}"
        assert (
            "snapshot" in error_message.lower()
        ), f"Error should mention 'snapshot': {error_message}"

    @pytest.mark.asyncio
    async def test_store_with_metadata(
        self,
        adapter: HandlerFileSystemAdapter,
        sample_snapshot: ModelMemorySnapshot,
    ) -> None:
        """Test storing a snapshot with metadata succeeds.

        Given: A valid memory snapshot and metadata
        When: Executing a store operation with metadata
        Then: Response status is 'success'
        """
        request = ModelMemoryStorageRequest(
            operation="store",
            snapshot=sample_snapshot,
            metadata={"source": "test", "priority": "high"},
        )

        response = await adapter.execute(request)

        assert (
            response.status == "success"
        ), f"Expected status 'success', got '{response.status}'"

    @pytest.mark.asyncio
    async def test_store_with_tags(
        self,
        adapter: HandlerFileSystemAdapter,
        sample_snapshot: ModelMemorySnapshot,
    ) -> None:
        """Test storing a snapshot with tags succeeds.

        Given: A valid memory snapshot and tags
        When: Executing a store operation with tags
        Then: Response status is 'success'
        """
        request = ModelMemoryStorageRequest(
            operation="store",
            snapshot=sample_snapshot,
            tags=["important", "test-case"],
        )

        response = await adapter.execute(request)

        assert (
            response.status == "success"
        ), f"Expected status 'success', got '{response.status}'"


# =============================================================================
# Retrieve Operation Tests
# =============================================================================


class TestRetrieveOperation:
    """Tests for the retrieve operation."""

    @pytest.mark.asyncio
    async def test_retrieve_stored_snapshot(
        self,
        adapter: HandlerFileSystemAdapter,
        sample_snapshot: ModelMemorySnapshot,
    ) -> None:
        """Test retrieving a previously stored snapshot.

        Given: A stored memory snapshot
        When: Executing a retrieve operation with the snapshot ID
        Then: Response contains the correct snapshot data
        """
        # Store first
        store_request = ModelMemoryStorageRequest(
            operation="store",
            snapshot=sample_snapshot,
        )
        await adapter.execute(store_request)

        # Retrieve
        retrieve_request = ModelMemoryStorageRequest(
            operation="retrieve",
            snapshot_id=str(sample_snapshot.snapshot_id),
        )
        response = await adapter.execute(retrieve_request)

        assert (
            response.status == "success"
        ), f"Expected status 'success', got '{response.status}'"
        assert response.snapshot is not None, "Expected snapshot in response"
        assert (
            response.snapshot.snapshot_id == sample_snapshot.snapshot_id
        ), "Snapshot ID mismatch"

    @pytest.mark.asyncio
    async def test_retrieve_not_found(
        self,
        adapter: HandlerFileSystemAdapter,
    ) -> None:
        """Test retrieving a non-existent snapshot returns not_found.

        Given: A snapshot ID that doesn't exist in storage
        When: Executing a retrieve operation
        Then: Response status is 'not_found'
        """
        request = ModelMemoryStorageRequest(
            operation="retrieve",
            snapshot_id="non-existent-snapshot-id-12345",
        )

        response = await adapter.execute(request)

        assert (
            response.status == "not_found"
        ), f"Expected status 'not_found', got '{response.status}'"
        assert response.snapshot is None, "Should not return a snapshot"

    def test_retrieve_without_snapshot_id_raises_validation_error(
        self,
    ) -> None:
        """Test that retrieve operation without snapshot_id raises ValidationError.

        Given: Attempting to create a retrieve request with no snapshot_id
        When: Constructing the ModelMemoryStorageRequest
        Then: Pydantic raises ValidationError with appropriate message
        """
        from pydantic import ValidationError

        with pytest.raises(ValidationError) as exc_info:
            ModelMemoryStorageRequest(
                operation="retrieve",
                snapshot_id=None,
            )

        error_message = str(exc_info.value)
        assert (
            "retrieve" in error_message.lower()
        ), f"Error should mention 'retrieve': {error_message}"
        assert (
            "snapshot_id" in error_message.lower()
        ), f"Error should mention 'snapshot_id': {error_message}"


# =============================================================================
# Store-Retrieve Round-Trip Tests
# =============================================================================


class TestStoreRetrieveRoundTrip:
    """Tests for full store/retrieve cycle with data integrity verification."""

    @pytest.mark.asyncio
    async def test_round_trip_basic(
        self,
        adapter: HandlerFileSystemAdapter,
        sample_snapshot: ModelMemorySnapshot,
    ) -> None:
        """Test basic store-retrieve round-trip preserves data.

        Given: A memory snapshot
        When: Storing and then retrieving the snapshot
        Then: Retrieved snapshot matches the original
        """
        # Store
        store_request = ModelMemoryStorageRequest(
            operation="store",
            snapshot=sample_snapshot,
        )
        store_response = await adapter.execute(store_request)
        assert store_response.status == "success"

        # Retrieve
        retrieve_request = ModelMemoryStorageRequest(
            operation="retrieve",
            snapshot_id=str(sample_snapshot.snapshot_id),
        )
        retrieve_response = await adapter.execute(retrieve_request)

        assert retrieve_response.status == "success"
        retrieved = retrieve_response.snapshot
        assert retrieved is not None

        # Verify key fields match
        assert retrieved.snapshot_id == sample_snapshot.snapshot_id
        assert retrieved.version == sample_snapshot.version
        assert retrieved.schema_version == sample_snapshot.schema_version
        assert retrieved.subject.subject_type == sample_snapshot.subject.subject_type
        assert retrieved.cost_ledger.budget_total == (
            sample_snapshot.cost_ledger.budget_total
        )

    @pytest.mark.asyncio
    async def test_round_trip_with_tags(
        self,
        adapter: HandlerFileSystemAdapter,
    ) -> None:
        """Test round-trip preserves tags.

        Given: A snapshot with tags
        When: Storing and retrieving
        Then: Tags are preserved
        """
        snapshot = create_unique_snapshot(tags=("tag1", "tag2", "tag3"))

        # Store
        await adapter.execute(
            ModelMemoryStorageRequest(operation="store", snapshot=snapshot)
        )

        # Retrieve
        response = await adapter.execute(
            ModelMemoryStorageRequest(
                operation="retrieve",
                snapshot_id=str(snapshot.snapshot_id),
            )
        )

        assert response.status == "success"
        assert response.snapshot is not None
        assert response.snapshot.tags == ("tag1", "tag2", "tag3")

    @pytest.mark.asyncio
    async def test_round_trip_multiple_snapshots(
        self,
        adapter: HandlerFileSystemAdapter,
    ) -> None:
        """Test storing and retrieving multiple snapshots.

        Given: Multiple unique snapshots
        When: Storing all and retrieving each
        Then: Each retrieved snapshot matches its original
        """
        snapshots = [create_unique_snapshot(version=i) for i in range(1, 4)]

        # Store all
        for snapshot in snapshots:
            response = await adapter.execute(
                ModelMemoryStorageRequest(operation="store", snapshot=snapshot)
            )
            assert response.status == "success"

        # Retrieve all and verify
        for original in snapshots:
            response = await adapter.execute(
                ModelMemoryStorageRequest(
                    operation="retrieve",
                    snapshot_id=str(original.snapshot_id),
                )
            )
            assert response.status == "success"
            assert response.snapshot is not None
            assert response.snapshot.snapshot_id == original.snapshot_id
            assert response.snapshot.version == original.version


# =============================================================================
# Delete Operation Tests
# =============================================================================


class TestDeleteOperation:
    """Tests for the delete operation."""

    @pytest.mark.asyncio
    async def test_delete_stored_snapshot(
        self,
        adapter: HandlerFileSystemAdapter,
        sample_snapshot: ModelMemorySnapshot,
    ) -> None:
        """Test deleting a previously stored snapshot.

        Given: A stored memory snapshot
        When: Executing a delete operation
        Then: Response status is 'success'
        """
        # Store first
        await adapter.execute(
            ModelMemoryStorageRequest(operation="store", snapshot=sample_snapshot)
        )

        # Delete
        response = await adapter.execute(
            ModelMemoryStorageRequest(
                operation="delete",
                snapshot_id=str(sample_snapshot.snapshot_id),
            )
        )

        assert (
            response.status == "success"
        ), f"Expected status 'success', got '{response.status}'"

    @pytest.mark.asyncio
    async def test_delete_removes_file(
        self,
        adapter: HandlerFileSystemAdapter,
        sample_snapshot: ModelMemorySnapshot,
    ) -> None:
        """Test that delete operation removes the file from disk.

        Given: A stored snapshot file
        When: Executing a delete operation
        Then: The file no longer exists
        """
        # Store
        await adapter.execute(
            ModelMemoryStorageRequest(operation="store", snapshot=sample_snapshot)
        )

        expected_file = adapter.snapshots_path / f"{sample_snapshot.snapshot_id}.json"
        assert expected_file.exists(), "File should exist after store"

        # Delete
        await adapter.execute(
            ModelMemoryStorageRequest(
                operation="delete",
                snapshot_id=str(sample_snapshot.snapshot_id),
            )
        )

        assert not expected_file.exists(), "File should not exist after delete"

    @pytest.mark.asyncio
    async def test_delete_then_retrieve_returns_not_found(
        self,
        adapter: HandlerFileSystemAdapter,
        sample_snapshot: ModelMemorySnapshot,
    ) -> None:
        """Test that deleted snapshot cannot be retrieved.

        Given: A stored and then deleted snapshot
        When: Attempting to retrieve
        Then: Response status is 'not_found'
        """
        # Store
        await adapter.execute(
            ModelMemoryStorageRequest(operation="store", snapshot=sample_snapshot)
        )

        # Delete
        await adapter.execute(
            ModelMemoryStorageRequest(
                operation="delete",
                snapshot_id=str(sample_snapshot.snapshot_id),
            )
        )

        # Attempt retrieve
        response = await adapter.execute(
            ModelMemoryStorageRequest(
                operation="retrieve",
                snapshot_id=str(sample_snapshot.snapshot_id),
            )
        )

        assert response.status == "not_found"

    @pytest.mark.asyncio
    async def test_delete_not_found(
        self,
        adapter: HandlerFileSystemAdapter,
    ) -> None:
        """Test deleting a non-existent snapshot returns not_found.

        Given: A snapshot ID that doesn't exist
        When: Executing a delete operation
        Then: Response status is 'not_found'
        """
        response = await adapter.execute(
            ModelMemoryStorageRequest(
                operation="delete",
                snapshot_id="non-existent-id-67890",
            )
        )

        assert (
            response.status == "not_found"
        ), f"Expected status 'not_found', got '{response.status}'"

    def test_delete_without_snapshot_id_raises_validation_error(
        self,
    ) -> None:
        """Test that delete operation without snapshot_id raises ValidationError.

        Given: Attempting to create a delete request with no snapshot_id
        When: Constructing the ModelMemoryStorageRequest
        Then: Pydantic raises ValidationError with appropriate message
        """
        from pydantic import ValidationError

        with pytest.raises(ValidationError) as exc_info:
            ModelMemoryStorageRequest(
                operation="delete",
                snapshot_id=None,
            )

        error_message = str(exc_info.value)
        assert (
            "delete" in error_message.lower()
        ), f"Error should mention 'delete': {error_message}"
        assert (
            "snapshot_id" in error_message.lower()
        ), f"Error should mention 'snapshot_id': {error_message}"


# =============================================================================
# List Operation Tests
# =============================================================================


class TestListOperation:
    """Tests for the list operation."""

    @pytest.mark.asyncio
    async def test_list_empty_storage(
        self,
        adapter: HandlerFileSystemAdapter,
    ) -> None:
        """Test listing when no snapshots are stored.

        Given: Empty storage
        When: Executing a list operation
        Then: Response returns empty list
        """
        response = await adapter.execute(ModelMemoryStorageRequest(operation="list"))

        assert response.status == "success"
        assert response.snapshot_ids is not None
        assert len(response.snapshot_ids) == 0

    @pytest.mark.asyncio
    async def test_list_single_snapshot(
        self,
        adapter: HandlerFileSystemAdapter,
        sample_snapshot: ModelMemorySnapshot,
    ) -> None:
        """Test listing with one stored snapshot.

        Given: One stored snapshot
        When: Executing a list operation
        Then: Response contains the snapshot ID
        """
        await adapter.execute(
            ModelMemoryStorageRequest(operation="store", snapshot=sample_snapshot)
        )

        response = await adapter.execute(ModelMemoryStorageRequest(operation="list"))

        assert response.status == "success"
        assert response.snapshot_ids is not None
        assert len(response.snapshot_ids) == 1
        assert str(sample_snapshot.snapshot_id) in response.snapshot_ids

    @pytest.mark.asyncio
    async def test_list_multiple_snapshots(
        self,
        adapter: HandlerFileSystemAdapter,
    ) -> None:
        """Test listing with multiple stored snapshots.

        Given: Multiple stored snapshots
        When: Executing a list operation
        Then: Response contains all snapshot IDs
        """
        snapshots = [create_unique_snapshot() for _ in range(5)]

        for snapshot in snapshots:
            await adapter.execute(
                ModelMemoryStorageRequest(operation="store", snapshot=snapshot)
            )

        response = await adapter.execute(ModelMemoryStorageRequest(operation="list"))

        assert response.status == "success"
        assert response.snapshot_ids is not None
        assert len(response.snapshot_ids) == 5

        # Verify all IDs are present
        returned_ids = set(response.snapshot_ids)
        for snapshot in snapshots:
            assert str(snapshot.snapshot_id) in returned_ids

    @pytest.mark.asyncio
    async def test_list_after_delete(
        self,
        adapter: HandlerFileSystemAdapter,
    ) -> None:
        """Test that list reflects deletions.

        Given: Multiple snapshots, one deleted
        When: Executing a list operation
        Then: Deleted snapshot ID is not in the list
        """
        snapshots = [create_unique_snapshot() for _ in range(3)]

        for snapshot in snapshots:
            await adapter.execute(
                ModelMemoryStorageRequest(operation="store", snapshot=snapshot)
            )

        # Delete the second one
        await adapter.execute(
            ModelMemoryStorageRequest(
                operation="delete",
                snapshot_id=str(snapshots[1].snapshot_id),
            )
        )

        response = await adapter.execute(ModelMemoryStorageRequest(operation="list"))

        assert response.status == "success"
        assert response.snapshot_ids is not None
        assert len(response.snapshot_ids) == 2
        assert str(snapshots[1].snapshot_id) not in response.snapshot_ids


# =============================================================================
# Update Operation Tests
# =============================================================================


class TestUpdateOperation:
    """Tests for the update operation."""

    @pytest.mark.asyncio
    async def test_update_existing_snapshot(
        self,
        adapter: HandlerFileSystemAdapter,
        sample_snapshot: ModelMemorySnapshot,
    ) -> None:
        """Test updating an existing snapshot.

        Given: A stored snapshot
        When: Updating with new data (same snapshot_id)
        Then: Retrieve returns the updated data
        """
        # Store original
        await adapter.execute(
            ModelMemoryStorageRequest(operation="store", snapshot=sample_snapshot)
        )

        # Create updated version with same ID but different data
        updated_snapshot = sample_snapshot.model_copy(
            update={"version": 2, "tags": ("updated",)}
        )

        # Update
        response = await adapter.execute(
            ModelMemoryStorageRequest(
                operation="update",
                snapshot=updated_snapshot,
            )
        )

        assert response.status == "success"

        # Retrieve and verify update
        retrieve_response = await adapter.execute(
            ModelMemoryStorageRequest(
                operation="retrieve",
                snapshot_id=str(sample_snapshot.snapshot_id),
            )
        )

        assert retrieve_response.status == "success"
        assert retrieve_response.snapshot is not None
        assert retrieve_response.snapshot.version == 2
        assert retrieve_response.snapshot.tags == ("updated",)

    def test_update_without_snapshot_raises_validation_error(
        self,
    ) -> None:
        """Test that update operation without snapshot raises ValidationError.

        Given: Attempting to create an update request with no snapshot
        When: Constructing the ModelMemoryStorageRequest
        Then: Pydantic raises ValidationError with appropriate message
        """
        from pydantic import ValidationError

        with pytest.raises(ValidationError) as exc_info:
            ModelMemoryStorageRequest(
                operation="update",
                snapshot=None,
            )

        error_message = str(exc_info.value)
        assert (
            "update" in error_message.lower()
        ), f"Error should mention 'update': {error_message}"
        assert (
            "snapshot" in error_message.lower()
        ), f"Error should mention 'snapshot': {error_message}"

    @pytest.mark.asyncio
    async def test_update_preserves_snapshot_id(
        self,
        adapter: HandlerFileSystemAdapter,
        sample_snapshot: ModelMemorySnapshot,
    ) -> None:
        """Test that update preserves the snapshot ID.

        Given: A stored snapshot
        When: Updating with modified version
        Then: The snapshot ID remains the same
        """
        # Store original
        await adapter.execute(
            ModelMemoryStorageRequest(operation="store", snapshot=sample_snapshot)
        )

        # Update with new version
        updated = sample_snapshot.model_copy(update={"version": 5})
        await adapter.execute(
            ModelMemoryStorageRequest(operation="update", snapshot=updated)
        )

        # List should still show only one snapshot
        list_response = await adapter.execute(
            ModelMemoryStorageRequest(operation="list")
        )

        assert list_response.snapshot_ids is not None
        assert len(list_response.snapshot_ids) == 1


# =============================================================================
# Adapter Lifecycle Tests
# =============================================================================


class TestAdapterLifecycle:
    """Tests for adapter initialization and shutdown."""

    @pytest.mark.asyncio
    async def test_adapter_auto_initializes(
        self,
        tmp_path: Path,
    ) -> None:
        """Test that adapter initializes automatically on first execute.

        Given: An uninitialized adapter
        When: Executing an operation
        Then: Adapter initializes automatically
        """
        config = HandlerFileSystemAdapterConfig(base_path=tmp_path)
        adapter = HandlerFileSystemAdapter(config)

        assert not adapter.is_initialized

        response = await adapter.execute(ModelMemoryStorageRequest(operation="list"))

        assert response.status == "success"
        assert adapter.is_initialized

    @pytest.mark.asyncio
    async def test_adapter_creates_snapshots_directory(
        self,
        tmp_path: Path,
    ) -> None:
        """Test that adapter creates snapshots directory on initialization.

        Given: A fresh adapter
        When: Initializing or executing first operation
        Then: Snapshots directory is created
        """
        config = HandlerFileSystemAdapterConfig(base_path=tmp_path)
        adapter = HandlerFileSystemAdapter(config)

        await adapter.initialize()

        assert adapter.snapshots_path.exists()
        assert adapter.snapshots_path.is_dir()

    @pytest.mark.asyncio
    async def test_adapter_shutdown(
        self,
        adapter: HandlerFileSystemAdapter,
    ) -> None:
        """Test that adapter shutdown works correctly.

        Given: An initialized adapter
        When: Calling shutdown
        Then: Adapter is no longer initialized
        """
        # Initialize by executing an operation
        await adapter.execute(ModelMemoryStorageRequest(operation="list"))
        assert adapter.is_initialized

        await adapter.shutdown()

        assert not adapter.is_initialized

    @pytest.mark.asyncio
    async def test_adapter_reinitializes_after_shutdown(
        self,
        adapter: HandlerFileSystemAdapter,
        sample_snapshot: ModelMemorySnapshot,
    ) -> None:
        """Test that adapter can reinitialize after shutdown.

        Given: An adapter that was shutdown
        When: Executing a new operation
        Then: Adapter reinitializes and operates correctly
        """
        # Initialize, store, shutdown
        await adapter.execute(
            ModelMemoryStorageRequest(operation="store", snapshot=sample_snapshot)
        )
        await adapter.shutdown()

        # Should reinitialize on next execute
        response = await adapter.execute(ModelMemoryStorageRequest(operation="list"))

        assert response.status == "success"
        assert adapter.is_initialized

    @pytest.mark.asyncio
    async def test_concurrent_initialization_is_safe(
        self,
        tmp_path: Path,
    ) -> None:
        """Test that concurrent initialization calls are handled safely.

        Given: An uninitialized adapter
        When: Multiple coroutines call initialize() concurrently
        Then: Only one initialization occurs, all calls succeed
        """
        import asyncio

        config = HandlerFileSystemAdapterConfig(base_path=tmp_path)
        adapter = HandlerFileSystemAdapter(config)
        initialization_count = 0
        original_initialize = adapter._handler.initialize

        async def counting_initialize(*args: object, **kwargs: object) -> object:
            nonlocal initialization_count
            initialization_count += 1
            return await original_initialize(*args, **kwargs)

        # Monkey-patch to count actual handler initializations
        adapter._handler.initialize = counting_initialize  # type: ignore[method-assign]

        # Launch multiple concurrent initialization calls
        tasks = [adapter.initialize() for _ in range(10)]
        await asyncio.gather(*tasks)

        # Verify only one actual initialization occurred
        assert adapter.is_initialized
        assert (
            initialization_count == 1
        ), f"Expected exactly 1 initialization, got {initialization_count}"
        assert adapter.snapshots_path.exists()

    @pytest.mark.asyncio
    async def test_concurrent_execute_with_initialization(
        self,
        tmp_path: Path,
    ) -> None:
        """Test that concurrent execute calls with auto-init are safe.

        Given: An uninitialized adapter
        When: Multiple coroutines call execute() concurrently
        Then: Initialization happens once, all operations succeed
        """
        import asyncio

        config = HandlerFileSystemAdapterConfig(base_path=tmp_path)
        adapter = HandlerFileSystemAdapter(config)

        # Launch multiple concurrent execute calls that will trigger init
        snapshots = [create_unique_snapshot() for _ in range(5)]
        tasks = [
            adapter.execute(ModelMemoryStorageRequest(operation="store", snapshot=s))
            for s in snapshots
        ]
        results = await asyncio.gather(*tasks)

        # All operations should succeed
        assert all(r.status == "success" for r in results)
        assert adapter.is_initialized

        # Verify all snapshots were stored
        list_response = await adapter.execute(
            ModelMemoryStorageRequest(operation="list")
        )
        assert list_response.status == "success"
        assert list_response.snapshot_ids is not None
        assert len(list_response.snapshot_ids) == 5


# =============================================================================
# Configuration Tests
# =============================================================================


class TestAdapterConfiguration:
    """Tests for adapter configuration options."""

    @pytest.mark.asyncio
    async def test_custom_snapshots_dir(
        self,
        tmp_path: Path,
    ) -> None:
        """Test using a custom snapshots directory name.

        Given: Configuration with custom snapshots_dir
        When: Storing a snapshot
        Then: File is created in the custom directory
        """
        config = HandlerFileSystemAdapterConfig(
            base_path=tmp_path,
            snapshots_dir="custom_memories",
        )
        adapter = HandlerFileSystemAdapter(config)
        snapshot = create_unique_snapshot()

        await adapter.execute(
            ModelMemoryStorageRequest(operation="store", snapshot=snapshot)
        )

        expected_dir = tmp_path / "custom_memories"
        expected_file = expected_dir / f"{snapshot.snapshot_id}.json"

        assert expected_dir.exists()
        assert expected_file.exists()

    def test_config_has_properties(self, tmp_path: Path) -> None:
        """Test that adapter exposes configuration properties.

        Given: A configured adapter
        When: Accessing properties
        Then: Properties return expected values
        """
        config = HandlerFileSystemAdapterConfig(
            base_path=tmp_path,
            snapshots_dir="memories",
            max_file_size=5 * 1024 * 1024,  # 5MB
        )
        adapter = HandlerFileSystemAdapter(config)

        assert adapter.config.base_path == tmp_path
        assert adapter.config.snapshots_dir == "memories"
        assert adapter.config.max_file_size == 5 * 1024 * 1024
        assert adapter.snapshots_path == tmp_path / "memories"


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_unknown_operation_returns_error(
        self,
        adapter: HandlerFileSystemAdapter,
    ) -> None:
        """Test that unknown operations return an error.

        Given: A request with an invalid operation
        When: Executing the request
        Then: Response status is 'error'

        Note: This test requires constructing an invalid request manually
        since Pydantic validation would normally prevent this.
        """
        # Create a request and modify the operation after construction
        # This simulates receiving an unknown operation type
        # We use type: ignore to bypass Pydantic's type checking
        try:
            request = ModelMemoryStorageRequest(
                operation="unknown_operation",  # type: ignore[arg-type]
            )
            # If Pydantic allows it, execute should handle it
            response = await adapter.execute(request)
            # The handler should return an error for unknown operations
            assert response.status == "error"
        except ValueError:
            # Pydantic validation correctly rejects invalid operation
            pass

    @pytest.mark.asyncio
    async def test_store_overwrite_existing(
        self,
        adapter: HandlerFileSystemAdapter,
        sample_snapshot: ModelMemorySnapshot,
    ) -> None:
        """Test that storing with same ID overwrites existing.

        Given: A stored snapshot
        When: Storing another snapshot with the same ID
        Then: The file is overwritten with new content
        """
        # Store original
        await adapter.execute(
            ModelMemoryStorageRequest(operation="store", snapshot=sample_snapshot)
        )

        # Store with same ID but different version
        updated = sample_snapshot.model_copy(update={"version": 99})
        await adapter.execute(
            ModelMemoryStorageRequest(operation="store", snapshot=updated)
        )

        # Retrieve should return updated version
        response = await adapter.execute(
            ModelMemoryStorageRequest(
                operation="retrieve",
                snapshot_id=str(sample_snapshot.snapshot_id),
            )
        )

        assert response.snapshot is not None
        assert response.snapshot.version == 99

    @pytest.mark.asyncio
    async def test_concurrent_operations(
        self,
        adapter: HandlerFileSystemAdapter,
    ) -> None:
        """Test handling multiple operations without issues.

        Given: Multiple snapshots
        When: Performing various operations
        Then: All operations complete successfully
        """
        import asyncio

        snapshots = [create_unique_snapshot() for _ in range(5)]

        # Store all concurrently
        store_tasks = [
            adapter.execute(ModelMemoryStorageRequest(operation="store", snapshot=s))
            for s in snapshots
        ]
        store_results = await asyncio.gather(*store_tasks)

        assert all(r.status == "success" for r in store_results)

        # Retrieve all concurrently
        retrieve_tasks = [
            adapter.execute(
                ModelMemoryStorageRequest(
                    operation="retrieve",
                    snapshot_id=str(s.snapshot_id),
                )
            )
            for s in snapshots
        ]
        retrieve_results = await asyncio.gather(*retrieve_tasks)

        assert all(r.status == "success" for r in retrieve_results)
        assert all(r.snapshot is not None for r in retrieve_results)


# =============================================================================
# Per-Operation Field Validation Tests
# =============================================================================


class TestPerOperationValidation:
    """Tests for per-operation field validation on ModelMemoryStorageRequest.

    These tests verify that the model_validator correctly enforces field
    requirements based on the operation type:

    - store: requires snapshot
    - retrieve: requires snapshot_id
    - delete: requires snapshot_id
    - update: requires snapshot
    - list: no required fields (optional filters)
    """

    def test_store_requires_snapshot(self) -> None:
        """Test that store operation requires snapshot field."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError) as exc_info:
            ModelMemoryStorageRequest(operation="store")

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert "snapshot" in str(errors[0])

    def test_store_with_snapshot_valid(
        self,
        sample_snapshot: ModelMemorySnapshot,
    ) -> None:
        """Test that store operation with snapshot is valid."""
        request = ModelMemoryStorageRequest(
            operation="store",
            snapshot=sample_snapshot,
        )
        assert request.operation == "store"
        assert request.snapshot is not None

    def test_retrieve_requires_snapshot_id(self) -> None:
        """Test that retrieve operation requires snapshot_id field."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError) as exc_info:
            ModelMemoryStorageRequest(operation="retrieve")

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert "snapshot_id" in str(errors[0])

    def test_retrieve_with_snapshot_id_valid(self) -> None:
        """Test that retrieve operation with snapshot_id is valid."""
        request = ModelMemoryStorageRequest(
            operation="retrieve",
            snapshot_id="test-id-123",
        )
        assert request.operation == "retrieve"
        assert request.snapshot_id == "test-id-123"

    def test_delete_requires_snapshot_id(self) -> None:
        """Test that delete operation requires snapshot_id field."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError) as exc_info:
            ModelMemoryStorageRequest(operation="delete")

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert "snapshot_id" in str(errors[0])

    def test_delete_with_snapshot_id_valid(self) -> None:
        """Test that delete operation with snapshot_id is valid."""
        request = ModelMemoryStorageRequest(
            operation="delete",
            snapshot_id="test-id-456",
        )
        assert request.operation == "delete"
        assert request.snapshot_id == "test-id-456"

    def test_update_requires_snapshot(self) -> None:
        """Test that update operation requires snapshot field."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError) as exc_info:
            ModelMemoryStorageRequest(operation="update")

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert "snapshot" in str(errors[0])

    def test_update_with_snapshot_valid(
        self,
        sample_snapshot: ModelMemorySnapshot,
    ) -> None:
        """Test that update operation with snapshot is valid."""
        request = ModelMemoryStorageRequest(
            operation="update",
            snapshot=sample_snapshot,
        )
        assert request.operation == "update"
        assert request.snapshot is not None

    def test_list_no_required_fields(self) -> None:
        """Test that list operation has no required fields."""
        request = ModelMemoryStorageRequest(operation="list")
        assert request.operation == "list"
        assert request.snapshot is None
        assert request.snapshot_id is None

    def test_list_with_optional_filters(self) -> None:
        """Test that list operation accepts optional metadata and tags."""
        request = ModelMemoryStorageRequest(
            operation="list",
            metadata={"source": "test"},
            tags=["important", "decision"],
        )
        assert request.operation == "list"
        assert request.metadata == {"source": "test"}
        assert request.tags == ["important", "decision"]

    def test_store_with_extra_snapshot_id_valid(
        self,
        sample_snapshot: ModelMemorySnapshot,
    ) -> None:
        """Test that store with both snapshot and snapshot_id is valid.

        The snapshot_id field can be provided for documentation purposes
        even though the snapshot contains its own ID.
        """
        request = ModelMemoryStorageRequest(
            operation="store",
            snapshot=sample_snapshot,
            snapshot_id="extra-id",
        )
        assert request.operation == "store"
        assert request.snapshot is not None
        assert request.snapshot_id == "extra-id"

    def test_retrieve_with_extra_fields_valid(self) -> None:
        """Test that retrieve with extra metadata/tags is valid.

        Extra fields don't invalidate the request, they're just ignored
        or used for logging/filtering purposes.
        """
        request = ModelMemoryStorageRequest(
            operation="retrieve",
            snapshot_id="test-id",
            metadata={"reason": "audit"},
            tags=["priority"],
        )
        assert request.operation == "retrieve"
        assert request.snapshot_id == "test-id"
        assert request.metadata is not None
        assert request.tags is not None

    def test_invalid_operation_rejected(self) -> None:
        """Test that invalid operation values are rejected by Pydantic."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ModelMemoryStorageRequest(
                operation="invalid_op",  # type: ignore[arg-type]
            )


# =============================================================================
# Security Validation Tests
# =============================================================================


class TestSecurityValidation:
    """Tests for security validations."""

    @pytest.mark.asyncio
    async def test_retrieve_path_traversal_rejected(
        self,
        adapter: HandlerFileSystemAdapter,
    ) -> None:
        """Test that path traversal in snapshot_id is rejected.

        Given: A malicious snapshot_id with path traversal
        When: Executing a retrieve operation
        Then: Response status is 'error' with appropriate message
        """
        malicious_ids = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32",
            "valid/../../../etc/passwd",
            "snapshot_id/../../secret",
        ]
        for malicious_id in malicious_ids:
            response = await adapter.execute(
                ModelMemoryStorageRequest(
                    operation="retrieve",
                    snapshot_id=malicious_id,
                )
            )
            assert (
                response.status == "error"
            ), f"Path traversal should be rejected: {malicious_id}"
            assert (
                "path" in response.error_message.lower()
                or "invalid" in response.error_message.lower()
            )

    @pytest.mark.asyncio
    async def test_delete_path_traversal_rejected(
        self,
        adapter: HandlerFileSystemAdapter,
    ) -> None:
        """Test that path traversal in snapshot_id is rejected for delete.

        Given: A malicious snapshot_id with path traversal
        When: Executing a delete operation
        Then: Response status is 'error' with appropriate message
        """
        response = await adapter.execute(
            ModelMemoryStorageRequest(
                operation="delete",
                snapshot_id="../../../etc/passwd",
            )
        )
        assert response.status == "error"

    @pytest.mark.asyncio
    async def test_store_file_size_limit_enforced(
        self,
        tmp_path: Path,
    ) -> None:
        """Test that file size limit is enforced during store.

        Given: A snapshot that exceeds the configured file size limit
        When: Executing a store operation
        Then: Response status is 'error' with file size message
        """
        # Create adapter with very small file size limit
        config = HandlerFileSystemAdapterConfig(
            base_path=tmp_path,
            max_file_size=100,  # Very small limit
        )
        adapter = HandlerFileSystemAdapter(config)

        # Create a snapshot (will serialize to more than 100 bytes)
        snapshot = create_unique_snapshot()

        response = await adapter.execute(
            ModelMemoryStorageRequest(
                operation="store",
                snapshot=snapshot,
            )
        )

        assert response.status == "error"
        assert "size" in response.error_message.lower()


# =============================================================================
# Performance Benchmark Tests
# =============================================================================


class TestPerformanceBenchmarks:
    """Performance benchmark tests for memory storage operations.

    These tests verify that all CRUD operations complete within acceptable
    time thresholds as specified in the contract.yaml SLA.

    Contract Reference:
        contract.yaml line 128 specifies:
            performance:
                max_response_time_ms: 100

    CI Configuration:
        Tests use a CI-safe threshold of 500ms to account for variable
        CI environment performance. For local strict benchmarks, the
        contract SLA of 100ms should be enforced.

    Note:
        These tests are marked with @pytest.mark.benchmark for optional
        filtering. Run with `pytest -m benchmark` to run only benchmarks,
        or `pytest -m "not benchmark"` to exclude them.
    """

    # CI-safe threshold in seconds (500ms)
    # Contract SLA is 100ms (see contract.yaml line 128)
    CI_THRESHOLD_SECONDS: float = 0.5
    CONTRACT_SLA_SECONDS: float = 0.1  # 100ms from contract.yaml

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_store_performance(
        self,
        adapter: HandlerFileSystemAdapter,
        sample_snapshot: ModelMemorySnapshot,
    ) -> None:
        """Benchmark: verify store operation completes within threshold.

        Contract Reference:
            performance.max_response_time_ms: 100 (contract.yaml line 128)

        Given: A valid memory snapshot
        When: Executing a store operation
        Then: Operation completes within CI threshold (500ms)

        Note:
            CI uses 500ms threshold for stability. Local strict benchmarks
            should verify against the 100ms contract SLA.
        """
        import time

        request = ModelMemoryStorageRequest(
            operation="store",
            snapshot=sample_snapshot,
        )

        start_time = time.perf_counter()
        response = await adapter.execute(request)
        elapsed_time = time.perf_counter() - start_time

        # Verify operation succeeded
        assert (
            response.status == "success"
        ), f"Store operation failed: {response.error_message}"

        # Verify performance threshold (CI-safe: 500ms)
        # Contract SLA is 100ms - see contract.yaml line 128
        assert elapsed_time < self.CI_THRESHOLD_SECONDS, (
            f"Store operation took {elapsed_time:.3f}s, "
            f"exceeds CI threshold of {self.CI_THRESHOLD_SECONDS}s "
            f"(contract SLA: {self.CONTRACT_SLA_SECONDS}s)"
        )

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_retrieve_performance(
        self,
        adapter: HandlerFileSystemAdapter,
        sample_snapshot: ModelMemorySnapshot,
    ) -> None:
        """Benchmark: verify retrieve operation completes within threshold.

        Contract Reference:
            performance.max_response_time_ms: 100 (contract.yaml line 128)

        Given: A previously stored memory snapshot
        When: Executing a retrieve operation
        Then: Operation completes within CI threshold (500ms)

        Note:
            CI uses 500ms threshold for stability. Local strict benchmarks
            should verify against the 100ms contract SLA.
        """
        import time

        # Setup: store the snapshot first
        await adapter.execute(
            ModelMemoryStorageRequest(operation="store", snapshot=sample_snapshot)
        )

        request = ModelMemoryStorageRequest(
            operation="retrieve",
            snapshot_id=str(sample_snapshot.snapshot_id),
        )

        start_time = time.perf_counter()
        response = await adapter.execute(request)
        elapsed_time = time.perf_counter() - start_time

        # Verify operation succeeded
        assert (
            response.status == "success"
        ), f"Retrieve operation failed: {response.error_message}"
        assert response.snapshot is not None, "Expected snapshot in response"

        # Verify performance threshold (CI-safe: 500ms)
        # Contract SLA is 100ms - see contract.yaml line 128
        assert elapsed_time < self.CI_THRESHOLD_SECONDS, (
            f"Retrieve operation took {elapsed_time:.3f}s, "
            f"exceeds CI threshold of {self.CI_THRESHOLD_SECONDS}s "
            f"(contract SLA: {self.CONTRACT_SLA_SECONDS}s)"
        )

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_delete_performance(
        self,
        adapter: HandlerFileSystemAdapter,
        sample_snapshot: ModelMemorySnapshot,
    ) -> None:
        """Benchmark: verify delete operation completes within threshold.

        Contract Reference:
            performance.max_response_time_ms: 100 (contract.yaml line 128)

        Given: A previously stored memory snapshot
        When: Executing a delete operation
        Then: Operation completes within CI threshold (500ms)

        Note:
            CI uses 500ms threshold for stability. Local strict benchmarks
            should verify against the 100ms contract SLA.
        """
        import time

        # Setup: store the snapshot first
        await adapter.execute(
            ModelMemoryStorageRequest(operation="store", snapshot=sample_snapshot)
        )

        request = ModelMemoryStorageRequest(
            operation="delete",
            snapshot_id=str(sample_snapshot.snapshot_id),
        )

        start_time = time.perf_counter()
        response = await adapter.execute(request)
        elapsed_time = time.perf_counter() - start_time

        # Verify operation succeeded
        assert (
            response.status == "success"
        ), f"Delete operation failed: {response.error_message}"

        # Verify performance threshold (CI-safe: 500ms)
        # Contract SLA is 100ms - see contract.yaml line 128
        assert elapsed_time < self.CI_THRESHOLD_SECONDS, (
            f"Delete operation took {elapsed_time:.3f}s, "
            f"exceeds CI threshold of {self.CI_THRESHOLD_SECONDS}s "
            f"(contract SLA: {self.CONTRACT_SLA_SECONDS}s)"
        )

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_list_performance(
        self,
        adapter: HandlerFileSystemAdapter,
    ) -> None:
        """Benchmark: verify list operation completes within threshold.

        Contract Reference:
            performance.max_response_time_ms: 100 (contract.yaml line 128)

        Given: Multiple stored memory snapshots (10 snapshots)
        When: Executing a list operation
        Then: Operation completes within CI threshold (500ms)

        Note:
            CI uses 500ms threshold for stability. Local strict benchmarks
            should verify against the 100ms contract SLA.
        """
        import time

        # Setup: store multiple snapshots
        snapshots = [create_unique_snapshot() for _ in range(10)]
        for snapshot in snapshots:
            await adapter.execute(
                ModelMemoryStorageRequest(operation="store", snapshot=snapshot)
            )

        request = ModelMemoryStorageRequest(operation="list")

        start_time = time.perf_counter()
        response = await adapter.execute(request)
        elapsed_time = time.perf_counter() - start_time

        # Verify operation succeeded
        assert (
            response.status == "success"
        ), f"List operation failed: {response.error_message}"
        assert response.snapshot_ids is not None
        assert len(response.snapshot_ids) == 10

        # Verify performance threshold (CI-safe: 500ms)
        # Contract SLA is 100ms - see contract.yaml line 128
        assert elapsed_time < self.CI_THRESHOLD_SECONDS, (
            f"List operation took {elapsed_time:.3f}s, "
            f"exceeds CI threshold of {self.CI_THRESHOLD_SECONDS}s "
            f"(contract SLA: {self.CONTRACT_SLA_SECONDS}s)"
        )

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_update_performance(
        self,
        adapter: HandlerFileSystemAdapter,
        sample_snapshot: ModelMemorySnapshot,
    ) -> None:
        """Benchmark: verify update operation completes within threshold.

        Contract Reference:
            performance.max_response_time_ms: 100 (contract.yaml line 128)

        Given: A previously stored memory snapshot
        When: Executing an update operation with modified data
        Then: Operation completes within CI threshold (500ms)

        Note:
            CI uses 500ms threshold for stability. Local strict benchmarks
            should verify against the 100ms contract SLA.
        """
        import time

        # Setup: store the snapshot first
        await adapter.execute(
            ModelMemoryStorageRequest(operation="store", snapshot=sample_snapshot)
        )

        # Create updated version
        updated_snapshot = sample_snapshot.model_copy(
            update={"version": 2, "tags": ("updated", "benchmark")}
        )

        request = ModelMemoryStorageRequest(
            operation="update",
            snapshot=updated_snapshot,
        )

        start_time = time.perf_counter()
        response = await adapter.execute(request)
        elapsed_time = time.perf_counter() - start_time

        # Verify operation succeeded
        assert (
            response.status == "success"
        ), f"Update operation failed: {response.error_message}"

        # Verify performance threshold (CI-safe: 500ms)
        # Contract SLA is 100ms - see contract.yaml line 128
        assert elapsed_time < self.CI_THRESHOLD_SECONDS, (
            f"Update operation took {elapsed_time:.3f}s, "
            f"exceeds CI threshold of {self.CI_THRESHOLD_SECONDS}s "
            f"(contract SLA: {self.CONTRACT_SLA_SECONDS}s)"
        )

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_round_trip_performance(
        self,
        adapter: HandlerFileSystemAdapter,
        sample_snapshot: ModelMemorySnapshot,
    ) -> None:
        """Benchmark: verify complete store-retrieve round-trip performance.

        Contract Reference:
            performance.max_response_time_ms: 100 (contract.yaml line 128)

        Given: A valid memory snapshot
        When: Executing store followed by retrieve operations
        Then: Combined operations complete within 2x CI threshold (1000ms)

        Note:
            This test verifies the full round-trip performance, which is
            useful for measuring realistic usage patterns. The threshold
            is 2x the single operation threshold.
        """
        import time

        store_request = ModelMemoryStorageRequest(
            operation="store",
            snapshot=sample_snapshot,
        )

        retrieve_request = ModelMemoryStorageRequest(
            operation="retrieve",
            snapshot_id=str(sample_snapshot.snapshot_id),
        )

        start_time = time.perf_counter()
        store_response = await adapter.execute(store_request)
        retrieve_response = await adapter.execute(retrieve_request)
        elapsed_time = time.perf_counter() - start_time

        # Verify operations succeeded
        assert (
            store_response.status == "success"
        ), f"Store operation failed: {store_response.error_message}"
        assert (
            retrieve_response.status == "success"
        ), f"Retrieve operation failed: {retrieve_response.error_message}"
        assert retrieve_response.snapshot is not None

        # Verify performance threshold (2x CI threshold for round-trip)
        # Contract SLA is 100ms per operation - see contract.yaml line 128
        round_trip_threshold = self.CI_THRESHOLD_SECONDS * 2
        assert elapsed_time < round_trip_threshold, (
            f"Round-trip took {elapsed_time:.3f}s, "
            f"exceeds threshold of {round_trip_threshold}s "
            f"(contract SLA: {self.CONTRACT_SLA_SECONDS * 2}s for 2 ops)"
        )
