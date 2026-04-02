# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for persona classification enums."""

import pytest

from omnimemory.enums import EnumMemoryType, EnumPreferredTone, EnumTechnicalLevel


@pytest.mark.unit
class TestEnumMemoryType:
    def test_member_count(self) -> None:
        assert len(EnumMemoryType) == 4

    def test_string_values(self) -> None:
        assert EnumMemoryType.FACTUAL == "factual"
        assert EnumMemoryType.PREFERENCE == "preference"
        assert EnumMemoryType.SENSITIVE == "sensitive"
        assert EnumMemoryType.EPHEMERAL == "ephemeral"

    def test_str_subclass(self) -> None:
        assert isinstance(EnumMemoryType.FACTUAL, str)


@pytest.mark.unit
class TestEnumTechnicalLevel:
    def test_member_count(self) -> None:
        assert len(EnumTechnicalLevel) == 4

    def test_string_values(self) -> None:
        assert EnumTechnicalLevel.BEGINNER == "beginner"
        assert EnumTechnicalLevel.INTERMEDIATE == "intermediate"
        assert EnumTechnicalLevel.ADVANCED == "advanced"
        assert EnumTechnicalLevel.EXPERT == "expert"

    def test_str_subclass(self) -> None:
        assert isinstance(EnumTechnicalLevel.BEGINNER, str)

    def test_ordering_is_progressive(self) -> None:
        levels = list(EnumTechnicalLevel)
        assert levels == [
            EnumTechnicalLevel.BEGINNER,
            EnumTechnicalLevel.INTERMEDIATE,
            EnumTechnicalLevel.ADVANCED,
            EnumTechnicalLevel.EXPERT,
        ]


@pytest.mark.unit
class TestEnumPreferredTone:
    def test_member_count(self) -> None:
        assert len(EnumPreferredTone) == 4

    def test_string_values(self) -> None:
        assert EnumPreferredTone.EXPLANATORY == "explanatory"
        assert EnumPreferredTone.CONCISE == "concise"
        assert EnumPreferredTone.FORMAL == "formal"
        assert EnumPreferredTone.CASUAL == "casual"

    def test_str_subclass(self) -> None:
        assert isinstance(EnumPreferredTone.EXPLANATORY, str)
