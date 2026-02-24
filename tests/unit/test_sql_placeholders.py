# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Unit tests for _sql_placeholders() function.

These tests verify that the _sql_placeholders() function ONLY produces safe
parameterized SQL placeholders ($1, $2, $3, ...) and never includes user data.
This is critical for SQL injection prevention.

The function is decorated with @lru_cache, so tests must account for caching
behavior when testing edge cases.
"""

from __future__ import annotations

import re

import pytest

from omnimemory.handlers.handler_subscription import _sql_placeholders


class TestSqlPlaceholdersBasicFunctionality:
    """Tests for basic functionality of _sql_placeholders()."""

    def test_generates_three_placeholders(self) -> None:
        """Test that _sql_placeholders(3) returns '$1, $2, $3'."""
        result = _sql_placeholders(3)
        assert result == "$1, $2, $3"

    def test_generates_placeholders_with_custom_start(self) -> None:
        """Test that _sql_placeholders(2, start=5) returns '$5, $6'."""
        result = _sql_placeholders(2, start=5)
        assert result == "$5, $6"

    def test_generates_single_placeholder(self) -> None:
        """Test that _sql_placeholders(1) returns '$1'."""
        result = _sql_placeholders(1)
        assert result == "$1"

    def test_generates_single_placeholder_with_start(self) -> None:
        """Test that _sql_placeholders(1, start=10) returns '$10'."""
        result = _sql_placeholders(1, start=10)
        assert result == "$10"

    def test_generates_many_placeholders(self) -> None:
        """Test generating many placeholders (performance edge case)."""
        result = _sql_placeholders(100)
        placeholders = result.split(", ")
        assert len(placeholders) == 100
        assert placeholders[0] == "$1"
        assert placeholders[99] == "$100"


class TestSqlPlaceholdersEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_zero_count_returns_empty_string(self) -> None:
        """Test that _sql_placeholders(0) returns ''."""
        result = _sql_placeholders(0)
        assert result == ""

    def test_negative_count_returns_empty_string(self) -> None:
        """Test that _sql_placeholders(-1) returns ''."""
        result = _sql_placeholders(-1)
        assert result == ""

    def test_large_negative_count_returns_empty_string(self) -> None:
        """Test that _sql_placeholders(-100) returns ''."""
        result = _sql_placeholders(-100)
        assert result == ""

    def test_start_zero_raises_value_error(self) -> None:
        """Test that _sql_placeholders(1, start=0) raises ValueError."""
        with pytest.raises(ValueError, match="start must be >= 1"):
            _sql_placeholders(1, start=0)

    def test_negative_start_raises_value_error(self) -> None:
        """Test that _sql_placeholders(1, start=-1) raises ValueError."""
        with pytest.raises(ValueError, match="start must be >= 1"):
            _sql_placeholders(1, start=-1)

    def test_count_exceeds_max_raises_value_error(self) -> None:
        """Test that _sql_placeholders(10001) raises ValueError."""
        with pytest.raises(ValueError, match="count exceeds maximum"):
            _sql_placeholders(10001)

    def test_boundary_count_at_max_succeeds(self) -> None:
        """Test that _sql_placeholders(10000) succeeds at the boundary."""
        result = _sql_placeholders(10000)
        placeholders = result.split(", ")
        assert len(placeholders) == 10000
        assert placeholders[0] == "$1"
        assert placeholders[-1] == "$10000"

    def test_zero_count_with_custom_start_returns_empty(self) -> None:
        """Test that _sql_placeholders(0, start=5) returns '' (count takes precedence)."""
        result = _sql_placeholders(0, start=5)
        assert result == ""


class TestSqlPlaceholdersSafetyValidation:
    """Tests to verify the function ONLY produces safe placeholder patterns.

    These tests ensure the output never contains anything that could be
    SQL injection vectors - only $N patterns separated by ', '.
    """

    # Pattern: Either empty string OR $N followed by optional (, $N)*
    SAFE_PLACEHOLDER_PATTERN = re.compile(r"^(\$\d+(, \$\d+)*)?$")

    def test_output_matches_safe_pattern_single(self) -> None:
        """Test single placeholder matches safe pattern."""
        result = _sql_placeholders(1)
        assert self.SAFE_PLACEHOLDER_PATTERN.match(result), f"Unsafe output: {result}"

    def test_output_matches_safe_pattern_multiple(self) -> None:
        """Test multiple placeholders match safe pattern."""
        result = _sql_placeholders(5)
        assert self.SAFE_PLACEHOLDER_PATTERN.match(result), f"Unsafe output: {result}"

    def test_output_matches_safe_pattern_empty(self) -> None:
        """Test empty result matches safe pattern."""
        result = _sql_placeholders(0)
        assert self.SAFE_PLACEHOLDER_PATTERN.match(result), f"Unsafe output: {result}"

    def test_output_matches_safe_pattern_with_start(self) -> None:
        """Test placeholders with custom start match safe pattern."""
        result = _sql_placeholders(3, start=10)
        assert self.SAFE_PLACEHOLDER_PATTERN.match(result), f"Unsafe output: {result}"

    def test_output_matches_safe_pattern_large_count(self) -> None:
        """Test large placeholder count matches safe pattern."""
        result = _sql_placeholders(500)
        assert self.SAFE_PLACEHOLDER_PATTERN.match(result), f"Unsafe output: {result}"

    @pytest.mark.parametrize("count", [1, 2, 3, 5, 10, 50, 100])
    def test_output_matches_safe_pattern_parametrized(self, count: int) -> None:
        """Test various counts all match safe pattern."""
        result = _sql_placeholders(count)
        assert self.SAFE_PLACEHOLDER_PATTERN.match(result), f"Unsafe output: {result}"

    @pytest.mark.parametrize("start", [1, 5, 10, 50, 100, 999])
    def test_output_matches_safe_pattern_various_starts(self, start: int) -> None:
        """Test various start values all produce safe patterns."""
        result = _sql_placeholders(3, start=start)
        assert self.SAFE_PLACEHOLDER_PATTERN.match(result), f"Unsafe output: {result}"

    def test_output_contains_only_expected_characters(self) -> None:
        """Test output contains only $, digits, comma, and space."""
        result = _sql_placeholders(100)
        # Allowed characters: $, 0-9, comma, space
        allowed_chars = set("$0123456789, ")
        result_chars = set(result)
        unexpected = result_chars - allowed_chars
        assert not unexpected, f"Unexpected characters in output: {unexpected}"

    def test_no_sql_keywords_in_output(self) -> None:
        """Test output contains no SQL keywords."""
        result = _sql_placeholders(100)
        sql_keywords = [
            "SELECT",
            "INSERT",
            "UPDATE",
            "DELETE",
            "DROP",
            "UNION",
            "OR",
            "AND",
            "--",
            ";",
            "'",
            '"',
        ]
        result_upper = result.upper()
        for keyword in sql_keywords:
            assert keyword not in result_upper, (
                f"SQL keyword '{keyword}' found in output"
            )

    def test_no_parentheses_in_output(self) -> None:
        """Test output contains no parentheses (could be function calls)."""
        result = _sql_placeholders(100)
        assert "(" not in result, "Opening parenthesis found in output"
        assert ")" not in result, "Closing parenthesis found in output"


class TestSqlPlaceholdersSequentialValidity:
    """Tests to verify placeholder numbers are sequential and correct."""

    def test_placeholders_are_sequential(self) -> None:
        """Test that placeholder numbers are sequential starting from start."""
        result = _sql_placeholders(5, start=3)
        assert result == "$3, $4, $5, $6, $7"

    def test_placeholders_sequence_correctness(self) -> None:
        """Test that each placeholder has the correct sequential number."""
        count = 10
        start = 7
        result = _sql_placeholders(count, start=start)
        placeholders = result.split(", ")

        for i, placeholder in enumerate(placeholders):
            expected_num = start + i
            assert placeholder == f"${expected_num}", (
                f"Expected ${expected_num} at position {i}, got {placeholder}"
            )

    def test_no_duplicate_placeholders(self) -> None:
        """Test that no placeholder numbers are duplicated."""
        result = _sql_placeholders(100)
        placeholders = result.split(", ")
        assert len(placeholders) == len(set(placeholders)), (
            "Duplicate placeholders found"
        )

    def test_no_gaps_in_sequence(self) -> None:
        """Test that there are no gaps in the placeholder sequence."""
        result = _sql_placeholders(50, start=10)
        placeholders = result.split(", ")
        numbers = [int(p[1:]) for p in placeholders]  # Extract numbers after $

        for i in range(len(numbers) - 1):
            assert numbers[i + 1] == numbers[i] + 1, (
                f"Gap in sequence between {numbers[i]} and {numbers[i + 1]}"
            )
