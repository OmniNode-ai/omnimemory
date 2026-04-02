# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for persona builder compute node classification logic."""

from datetime import datetime, timezone

import pytest

from omnimemory.enums import EnumPreferredTone, EnumTechnicalLevel
from omnimemory.models.persona import ModelPersonaSignal, ModelUserPersonaV1
from omnimemory.nodes.node_persona_builder_compute import (
    ModelPersonaClassifyRequest,
    classify_persona,
)


def _signal(
    signal_type: str,
    inferred_value: str,
    confidence: float = 0.85,
    user_id: str = "test-user",
    session_id: str = "sess-001",
) -> ModelPersonaSignal:
    """Helper to create a persona signal."""
    return ModelPersonaSignal(
        user_id=user_id,
        session_id=session_id,
        signal_type=signal_type,
        evidence=f"Test evidence for {signal_type}={inferred_value}",
        inferred_value=inferred_value,
        confidence=confidence,
        emitted_at=datetime.now(tz=timezone.utc),
    )


def _existing_profile(
    technical_level: EnumTechnicalLevel = EnumTechnicalLevel.INTERMEDIATE,
    session_count: int = 5,
    persona_version: int = 1,
) -> ModelUserPersonaV1:
    """Helper to create an existing persona."""
    return ModelUserPersonaV1(
        user_id="test-user",
        technical_level=technical_level,
        session_count=session_count,
        persona_version=persona_version,
        created_at=datetime.now(tz=timezone.utc),
    )


@pytest.mark.unit
class TestClassifyPersona:
    def test_beginner_from_signals(self) -> None:
        request = ModelPersonaClassifyRequest(
            user_id="test-user",
            signals=[
                _signal("technical_level", "beginner", confidence=0.85),
                _signal("preferred_tone", "explanatory", confidence=0.78),
            ],
        )
        result = classify_persona(request)
        assert result.status == "success"
        assert result.persona is not None
        assert result.persona.technical_level == EnumTechnicalLevel.BEGINNER
        assert result.persona.preferred_tone == EnumPreferredTone.EXPLANATORY

    def test_expert_from_signals(self) -> None:
        request = ModelPersonaClassifyRequest(
            user_id="test-user",
            signals=[
                _signal("technical_level", "expert", confidence=0.92),
                _signal("preferred_tone", "concise", confidence=0.88),
            ],
        )
        result = classify_persona(request)
        assert result.status == "success"
        assert result.persona is not None
        assert result.persona.technical_level == EnumTechnicalLevel.EXPERT
        assert result.persona.preferred_tone == EnumPreferredTone.CONCISE

    def test_insufficient_data_no_signals(self) -> None:
        request = ModelPersonaClassifyRequest(
            user_id="test-user",
            signals=[],
        )
        result = classify_persona(request)
        assert result.status == "insufficient_data"
        assert result.persona is None

    def test_no_signals_with_existing_returns_existing(self) -> None:
        existing = _existing_profile()
        request = ModelPersonaClassifyRequest(
            user_id="test-user",
            signals=[],
            existing_profile=existing,
        )
        result = classify_persona(request)
        assert result.status == "success"
        assert result.persona == existing

    def test_vocabulary_ema_slow_drift(self) -> None:
        existing = _existing_profile()
        assert existing.vocabulary_complexity == 0.5

        request = ModelPersonaClassifyRequest(
            user_id="test-user",
            signals=[_signal("vocabulary", "0.9", confidence=0.8)],
            existing_profile=existing,
        )
        result = classify_persona(request)
        assert result.status == "success"
        assert result.persona is not None
        # EMA: 0.2 * 0.9 + 0.8 * 0.5 = 0.58
        assert abs(result.persona.vocabulary_complexity - 0.58) < 0.01

    def test_domain_familiarity_increments(self) -> None:
        request = ModelPersonaClassifyRequest(
            user_id="test-user",
            signals=[
                _signal("domain_familiarity", "omnimemory"),
                _signal("domain_familiarity", "omnimemory"),
                _signal("domain_familiarity", "omniclaude"),
            ],
        )
        result = classify_persona(request)
        assert result.status == "success"
        assert result.persona is not None
        assert result.persona.domain_familiarity["omnimemory"] == pytest.approx(0.2)
        assert result.persona.domain_familiarity["omniclaude"] == pytest.approx(0.1)

    def test_domain_familiarity_caps_at_one(self) -> None:
        # Create profile with domain already at 0.95
        existing_with_domain = ModelUserPersonaV1(
            user_id="test-user",
            domain_familiarity={"omnimemory": 0.95},
            session_count=5,
            persona_version=1,
            created_at=datetime.now(tz=timezone.utc),
        )
        request = ModelPersonaClassifyRequest(
            user_id="test-user",
            signals=[_signal("domain_familiarity", "omnimemory")],
            existing_profile=existing_with_domain,
        )
        result = classify_persona(request)
        assert result.persona is not None
        assert result.persona.domain_familiarity["omnimemory"] == 1.0

    def test_technical_level_conservative_with_existing(self) -> None:
        """Technical level should not shift on low-confidence signals."""
        existing = _existing_profile(
            technical_level=EnumTechnicalLevel.INTERMEDIATE,
            session_count=5,
        )
        request = ModelPersonaClassifyRequest(
            user_id="test-user",
            signals=[
                _signal("technical_level", "expert", confidence=0.5),  # Below threshold
            ],
            existing_profile=existing,
        )
        result = classify_persona(request)
        assert result.persona is not None
        # Low confidence signal should not shift level
        assert result.persona.technical_level == EnumTechnicalLevel.INTERMEDIATE

    def test_tone_mode_selection(self) -> None:
        """Tone should be the mode of recent signals."""
        request = ModelPersonaClassifyRequest(
            user_id="test-user",
            signals=[
                _signal("preferred_tone", "concise"),
                _signal("preferred_tone", "concise"),
                _signal("preferred_tone", "explanatory"),
            ],
        )
        result = classify_persona(request)
        assert result.persona is not None
        assert result.persona.preferred_tone == EnumPreferredTone.CONCISE

    def test_session_count_increments(self) -> None:
        existing = _existing_profile(session_count=5)
        request = ModelPersonaClassifyRequest(
            user_id="test-user",
            signals=[_signal("technical_level", "intermediate")],
            existing_profile=existing,
        )
        result = classify_persona(request)
        assert result.persona is not None
        assert result.persona.session_count == 6

    def test_persona_version_increments(self) -> None:
        existing = _existing_profile(persona_version=3)
        request = ModelPersonaClassifyRequest(
            user_id="test-user",
            signals=[_signal("technical_level", "intermediate")],
            existing_profile=existing,
        )
        result = classify_persona(request)
        assert result.persona is not None
        assert result.persona.persona_version == 4

    def test_signals_processed_count(self) -> None:
        request = ModelPersonaClassifyRequest(
            user_id="test-user",
            signals=[
                _signal("technical_level", "beginner"),
                _signal("vocabulary", "0.3"),
                _signal("preferred_tone", "explanatory"),
            ],
        )
        result = classify_persona(request)
        assert result.signals_processed == 3
