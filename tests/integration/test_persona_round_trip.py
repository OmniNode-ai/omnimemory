# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Persona round-trip integration test.

Verifies the full pipeline: signal creation -> classification -> profile build
-> context formatting for both beginner and expert personas.

This test does NOT require infrastructure (Postgres, Kafka). It exercises the
pure compute path: ModelPersonaSignal -> classify_persona -> ModelUserPersonaV1
-> format as injectable context.
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest

from omnimemory.enums import EnumPreferredTone, EnumTechnicalLevel
from omnimemory.models.persona import ModelPersonaSignal, ModelUserPersonaV1
from omnimemory.nodes.node_persona_builder_compute.handlers.handler_persona_classify import (
    classify_persona,
)
from omnimemory.nodes.node_persona_builder_compute.models import (
    ModelPersonaClassifyRequest,
)


def _make_signal(
    user_id: str,
    signal_type: str,
    inferred_value: str,
    confidence: float,
    evidence: str = "test evidence",
    session_id: str | None = None,
) -> ModelPersonaSignal:
    return ModelPersonaSignal(
        signal_id=uuid4(),
        user_id=user_id,
        session_id=session_id or f"sess-{uuid4().hex[:8]}",
        signal_type=signal_type,
        evidence=evidence,
        inferred_value=inferred_value,
        confidence=confidence,
        emitted_at=datetime.now(tz=timezone.utc),
    )


def _format_persona_context(persona: dict[str, object] | None) -> str:
    """Inline formatter matching the omniclaude persona_context_client pattern.

    This is a local copy for integration testing without cross-repo imports.
    The canonical implementation lives in omniclaude.
    """
    if not persona:
        return ""

    lines = ["## User Persona", ""]
    tech = str(persona.get("technical_level", "intermediate"))
    tone = str(persona.get("preferred_tone", "explanatory"))
    vocab_raw = persona.get("vocabulary_complexity", 0.5)
    vocab = float(vocab_raw) if isinstance(vocab_raw, (int, float)) else 0.5

    lines.append(f"- **Technical level:** {tech}")
    lines.append(f"- **Preferred tone:** {tone}")

    if vocab > 0.7:
        vocab_label = "advanced"
    elif vocab > 0.3:
        vocab_label = "standard"
    else:
        vocab_label = "simple"
    lines.append(f"- **Vocabulary:** {vocab_label}")

    domain_familiarity = persona.get("domain_familiarity", {})
    if isinstance(domain_familiarity, dict) and domain_familiarity:
        top = sorted(
            domain_familiarity.items(),
            key=lambda x: float(x[1]) if isinstance(x[1], (int, float)) else 0.0,
            reverse=True,
        )[:3]
        lines.append(
            f"- **Top domains:** {', '.join(f'{k} ({float(v):.0%})' for k, v in top)}"
        )

    lines.append("")
    lines.append(
        f"_Adapt output to this user's level. {tech} users prefer {tone} responses._"
    )

    return "\n".join(lines)


@pytest.mark.integration
class TestPersonaRoundTrip:
    def test_beginner_persona_flow(self) -> None:
        """Beginner user: short prompts, asks what things mean, few tools."""
        signals = [
            _make_signal(
                user_id="test-user",
                signal_type="technical_level",
                evidence="Asked 'what does frozen=True mean?'",
                inferred_value="beginner",
                confidence=0.85,
            ),
            _make_signal(
                user_id="test-user",
                signal_type="preferred_tone",
                evidence="Used 'explain' 3 times in prompt",
                inferred_value="explanatory",
                confidence=0.78,
            ),
            _make_signal(
                user_id="test-user",
                signal_type="vocabulary",
                evidence="Average word length 4.2, no technical terms",
                inferred_value="0.2",
                confidence=0.70,
            ),
        ]

        request = ModelPersonaClassifyRequest(
            user_id="test-user",
            signals=signals,
            existing_profile=None,
        )
        result = classify_persona(request)
        assert result.status == "success"
        assert result.persona is not None

        profile = result.persona
        assert profile.technical_level == EnumTechnicalLevel.BEGINNER
        assert profile.preferred_tone == EnumPreferredTone.EXPLANATORY
        assert profile.session_count == 1
        assert profile.persona_version == 1

        # Format as injectable context
        context = _format_persona_context(profile.model_dump())
        assert "## User Persona" in context
        assert "beginner" in context.lower()
        assert "explanatory" in context.lower()
        assert "Adapt output" in context

    def test_expert_persona_flow(self) -> None:
        """Expert user: terse prompts, architectural language, high recovery."""
        signals = [
            _make_signal(
                user_id="test-expert",
                signal_type="technical_level",
                evidence="Used terms: contract.yaml, handler pattern, MixinConsumerHealth",
                inferred_value="expert",
                confidence=0.92,
            ),
            _make_signal(
                user_id="test-expert",
                signal_type="preferred_tone",
                evidence="Average prompt length 12 words, imperative form",
                inferred_value="concise",
                confidence=0.88,
            ),
            _make_signal(
                user_id="test-expert",
                signal_type="vocabulary",
                evidence="High technical term density, code-heavy prompts",
                inferred_value="0.9",
                confidence=0.85,
            ),
            _make_signal(
                user_id="test-expert",
                signal_type="domain_familiarity",
                evidence="Working in omnibase_core repo",
                inferred_value="omnibase_core",
                confidence=0.95,
            ),
        ]

        request = ModelPersonaClassifyRequest(
            user_id="test-expert",
            signals=signals,
            existing_profile=None,
        )
        result = classify_persona(request)
        assert result.status == "success"
        assert result.persona is not None

        profile = result.persona
        assert profile.technical_level == EnumTechnicalLevel.EXPERT
        assert profile.preferred_tone == EnumPreferredTone.CONCISE
        assert profile.session_count == 1

        # Format as injectable context
        context = _format_persona_context(profile.model_dump())
        assert "expert" in context.lower()
        assert "concise" in context.lower()

    def test_incremental_update_preserves_history(self) -> None:
        """Second session incrementally updates an existing profile."""
        now = datetime.now(tz=timezone.utc)
        existing = ModelUserPersonaV1(
            user_id="test-user",
            technical_level=EnumTechnicalLevel.INTERMEDIATE,
            vocabulary_complexity=0.5,
            preferred_tone=EnumPreferredTone.EXPLANATORY,
            domain_familiarity={"omniclaude": 0.3},
            session_count=5,
            persona_version=5,
            created_at=now,
            rebuilt_from_signals=10,
        )

        new_signals = [
            _make_signal(
                user_id="test-user",
                signal_type="domain_familiarity",
                evidence="Working in omnimemory repo",
                inferred_value="omnimemory",
                confidence=0.9,
            ),
        ]

        request = ModelPersonaClassifyRequest(
            user_id="test-user",
            signals=new_signals,
            existing_profile=existing,
        )
        result = classify_persona(request)
        assert result.status == "success"
        assert result.persona is not None

        profile = result.persona
        assert profile.session_count == 6
        assert profile.persona_version == 6
        assert "omnimemory" in profile.domain_familiarity
        # Previous domain preserved
        assert "omniclaude" in profile.domain_familiarity

    def test_empty_signals_returns_existing(self) -> None:
        """No signals with existing profile returns the existing profile."""
        now = datetime.now(tz=timezone.utc)
        existing = ModelUserPersonaV1(
            user_id="test-user",
            technical_level=EnumTechnicalLevel.ADVANCED,
            vocabulary_complexity=0.8,
            preferred_tone=EnumPreferredTone.CONCISE,
            domain_familiarity={},
            session_count=10,
            persona_version=10,
            created_at=now,
            rebuilt_from_signals=20,
        )

        request = ModelPersonaClassifyRequest(
            user_id="test-user",
            signals=[],
            existing_profile=existing,
        )
        result = classify_persona(request)
        assert result.status == "success"
        assert result.persona is existing
        assert result.signals_processed == 0

    def test_context_format_with_domains(self) -> None:
        """Context formatting includes top 3 domains only."""
        now = datetime.now(tz=timezone.utc)
        persona = ModelUserPersonaV1(
            user_id="test-user",
            technical_level=EnumTechnicalLevel.ADVANCED,
            vocabulary_complexity=0.75,
            preferred_tone=EnumPreferredTone.FORMAL,
            domain_familiarity={
                "omnibase_core": 0.9,
                "omniclaude": 0.7,
                "omnimemory": 0.5,
                "omnidash": 0.2,
            },
            session_count=15,
            persona_version=15,
            created_at=now,
            rebuilt_from_signals=30,
        )

        context = _format_persona_context(persona.model_dump())
        assert "omnibase_core" in context
        assert "omniclaude" in context
        assert "omnimemory" in context
        # 4th domain should be excluded (top 3 only)
        assert "omnidash" not in context
        assert "advanced" in context.lower()
