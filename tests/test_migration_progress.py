"""Tests for migration progress tracking model.

These tests validate error type extraction edge cases and migration tracking.
"""

from __future__ import annotations

import pytest

from omnimemory.models.foundation.model_migration_progress import (
    MigrationProgressMetrics,
    MigrationProgressTracker,
)


def _create_tracker(name: str = "test") -> MigrationProgressTracker:
    """Helper to create a tracker instance with required fields."""
    return MigrationProgressTracker(
        name=name,
        metrics=MigrationProgressMetrics(total_files=0),
    )


class TestErrorTypeExtraction:
    """Tests for _extract_error_type method edge cases."""

    @pytest.fixture
    def tracker(self) -> MigrationProgressTracker:
        """Create a tracker instance for testing."""
        return _create_tracker()

    def test_simple_error_type(self, tracker: MigrationProgressTracker) -> None:
        """Test extracting simple error type like 'ValueError: message'."""
        result = tracker._extract_error_type("ValueError: invalid literal")
        assert result == "ValueError"

    def test_module_qualified_error_type(
        self, tracker: MigrationProgressTracker
    ) -> None:
        """Test module-qualified error like 'requests.exceptions.HTTPError'."""
        result = tracker._extract_error_type(
            "requests.exceptions.HTTPError: 404 Not Found"
        )
        assert result == "HTTPError"

    def test_deeply_nested_module(self, tracker: MigrationProgressTracker) -> None:
        """Test extracting deeply nested module error."""
        result = tracker._extract_error_type(
            "some.deeply.nested.module.CustomError: message"
        )
        assert result == "CustomError"

    def test_builtins_qualified_error(self, tracker: MigrationProgressTracker) -> None:
        """Test extracting builtins-qualified error like 'builtins.ValueError'."""
        result = tracker._extract_error_type("builtins.ValueError: message")
        assert result == "ValueError"

    def test_chained_exception_format(self, tracker: MigrationProgressTracker) -> None:
        """Test extracting error type from chained exceptions."""
        # Should extract the first error type
        result = tracker._extract_error_type("ValueError: IOError: actual message")
        assert result == "ValueError"

    def test_custom_exception_name(self, tracker: MigrationProgressTracker) -> None:
        """Test extracting custom exception without Error/Exception suffix."""
        result = tracker._extract_error_type("ConnectionRefused: server unavailable")
        assert result == "ConnectionRefused"

    def test_empty_message(self, tracker: MigrationProgressTracker) -> None:
        """Test empty message returns UnknownError."""
        result = tracker._extract_error_type("")
        assert result == "UnknownError"

    def test_no_colon_separator(self, tracker: MigrationProgressTracker) -> None:
        """Test message without colon separator returns UnknownError."""
        result = tracker._extract_error_type("Something went wrong")
        assert result == "UnknownError"

    def test_lowercase_start(self, tracker: MigrationProgressTracker) -> None:
        """Test message starting with lowercase returns UnknownError."""
        result = tracker._extract_error_type("error: something failed")
        assert result == "UnknownError"

    def test_invalid_identifier_characters(
        self, tracker: MigrationProgressTracker
    ) -> None:
        """Test message with invalid identifier returns UnknownError."""
        result = tracker._extract_error_type("Error-Type: something failed")
        assert result == "UnknownError"

    def test_invalid_module_path(self, tracker: MigrationProgressTracker) -> None:
        """Test invalid module path returns UnknownError."""
        result = tracker._extract_error_type("invalid..path.Error: message")
        assert result == "UnknownError"

    def test_path_with_invalid_identifier(
        self, tracker: MigrationProgressTracker
    ) -> None:
        """Test module path with invalid identifier part returns UnknownError."""
        result = tracker._extract_error_type("module.123invalid.Error: message")
        assert result == "UnknownError"

    def test_colon_without_space(self, tracker: MigrationProgressTracker) -> None:
        """Test colon without space (like file paths) returns UnknownError."""
        # "C:\\path" looks like it has a colon but not ": "
        result = tracker._extract_error_type("Error at path C:\\folder\\file.txt")
        assert result == "UnknownError"

    def test_very_long_potential_type(self, tracker: MigrationProgressTracker) -> None:
        """Test very long string before colon returns UnknownError."""
        long_prefix = "A" * 150
        result = tracker._extract_error_type(f"{long_prefix}: message")
        assert result == "UnknownError"

    def test_whitespace_handling(self, tracker: MigrationProgressTracker) -> None:
        """Test whitespace around error type is handled correctly."""
        result = tracker._extract_error_type("  ValueError : message")
        # Leading space in type should still work since we strip
        assert result == "ValueError"

    def test_only_colon_space(self, tracker: MigrationProgressTracker) -> None:
        """Test message that is just ': message' returns UnknownError."""
        result = tracker._extract_error_type(": message")
        assert result == "UnknownError"


class TestMigrationProgressTrackerErrorTracking:
    """Integration tests for error tracking in migration progress."""

    def test_complete_file_with_error_tracks_type(self) -> None:
        """Test that completing a file with error properly tracks error type."""
        tracker = _create_tracker()
        tracker.add_file("test.py", file_size=100)
        tracker.start_file_processing("test.py")
        tracker.complete_file_processing(
            "test.py", success=False, error_message="ValueError: invalid input"
        )

        assert "ValueError" in tracker.error_summary
        assert tracker.error_summary["ValueError"] == 1

    def test_multiple_errors_of_same_type(self) -> None:
        """Test that multiple errors of same type are counted correctly."""
        tracker = _create_tracker()

        for i in range(3):
            tracker.add_file(f"test{i}.py", file_size=100)
            tracker.start_file_processing(f"test{i}.py")
            tracker.complete_file_processing(
                f"test{i}.py", success=False, error_message="IOError: disk full"
            )

        assert tracker.error_summary.get("IOError") == 3

    def test_different_error_types_tracked_separately(self) -> None:
        """Test that different error types are tracked separately."""
        tracker = _create_tracker()

        errors = [
            "ValueError: bad value",
            "IOError: disk error",
            "TypeError: wrong type",
        ]

        for i, error_msg in enumerate(errors):
            tracker.add_file(f"test{i}.py", file_size=100)
            tracker.start_file_processing(f"test{i}.py")
            tracker.complete_file_processing(
                f"test{i}.py", success=False, error_message=error_msg
            )

        assert tracker.error_summary.get("ValueError") == 1
        assert tracker.error_summary.get("IOError") == 1
        assert tracker.error_summary.get("TypeError") == 1
