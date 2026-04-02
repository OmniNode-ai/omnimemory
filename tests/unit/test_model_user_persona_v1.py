# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for ModelUserPersonaV1 and ModelPersonaSignal."""

from datetime import datetime, timezone
from uuid import uuid4

import pytest
from pydantic import ValidationError

from omnimemory.enums import EnumPreferredTone, EnumTechnicalLevel
from omnimemory.models.persona import ModelPersonaSignal, ModelUserPersonaV1


@pytest.mark.unit
class TestModelUserPersonaV1:
    def test_create_with_defaults(self) -> None:
        persona = ModelUserPersonaV1(
            user_id="test-user",
            created_at=datetime.now(tz=timezone.utc),
        )
        assert persona.technical_level == EnumTechnicalLevel.INTERMEDIATE
        assert persona.vocabulary_complexity == 0.5
        assert persona.preferred_tone == EnumPreferredTone.EXPLANATORY
        assert persona.domain_familiarity == {}
        assert persona.session_count == 0
        assert persona.persona_version == 1
        assert persona.rebuilt_from_signals == 0
        assert persona.agent_id is None

    def test_create_with_all_fields(self) -> None:
        now = datetime.now(tz=timezone.utc)
        persona = ModelUserPersonaV1(
            user_id="test-user",
            agent_id="caia-001",
            technical_level=EnumTechnicalLevel.EXPERT,
            vocabulary_complexity=0.9,
            preferred_tone=EnumPreferredTone.CONCISE,
            domain_familiarity={"omnimemory": 0.8, "omniclaude": 0.6},
            session_count=25,
            persona_version=3,
            created_at=now,
            rebuilt_from_signals=42,
        )
        assert persona.technical_level == EnumTechnicalLevel.EXPERT
        assert persona.vocabulary_complexity == 0.9
        assert persona.domain_familiarity["omnimemory"] == 0.8
        assert persona.session_count == 25

    def test_frozen_immutability(self) -> None:
        persona = ModelUserPersonaV1(
            user_id="test-user",
            created_at=datetime.now(tz=timezone.utc),
        )
        with pytest.raises(ValidationError):
            persona.technical_level = EnumTechnicalLevel.EXPERT  # type: ignore[misc]

    def test_persona_version_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            ModelUserPersonaV1(
                user_id="test-user",
                persona_version=0,
                created_at=datetime.now(tz=timezone.utc),
            )

    def test_vocabulary_complexity_bounds(self) -> None:
        with pytest.raises(ValidationError):
            ModelUserPersonaV1(
                user_id="test-user",
                vocabulary_complexity=1.5,
                created_at=datetime.now(tz=timezone.utc),
            )
        with pytest.raises(ValidationError):
            ModelUserPersonaV1(
                user_id="test-user",
                vocabulary_complexity=-0.1,
                created_at=datetime.now(tz=timezone.utc),
            )

    def test_extra_fields_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            ModelUserPersonaV1(
                user_id="test-user",
                created_at=datetime.now(tz=timezone.utc),
                unknown_field="bad",  # type: ignore[call-arg]
            )

    def test_serialization_round_trip(self) -> None:
        now = datetime.now(tz=timezone.utc)
        persona = ModelUserPersonaV1(
            user_id="test-user",
            agent_id="caia-001",
            technical_level=EnumTechnicalLevel.ADVANCED,
            vocabulary_complexity=0.7,
            preferred_tone=EnumPreferredTone.FORMAL,
            domain_familiarity={"omnibase_core": 0.5},
            session_count=10,
            persona_version=2,
            created_at=now,
            rebuilt_from_signals=15,
        )
        data = persona.model_dump()
        restored = ModelUserPersonaV1.model_validate(data)
        assert restored == persona


@pytest.mark.unit
class TestModelPersonaSignal:
    def test_create_signal(self) -> None:
        now = datetime.now(tz=timezone.utc)
        signal = ModelPersonaSignal(
            user_id="test-user",
            session_id="sess-001",
            signal_type="technical_level",
            evidence="Used terms: contract.yaml, handler pattern",
            inferred_value="expert",
            confidence=0.92,
            emitted_at=now,
        )
        assert signal.user_id == "test-user"
        assert signal.signal_type == "technical_level"
        assert signal.confidence == 0.92
        assert signal.signal_id is not None

    def test_frozen_immutability(self) -> None:
        signal = ModelPersonaSignal(
            user_id="test-user",
            session_id="sess-001",
            signal_type="tone",
            evidence="Short imperative prompts",
            inferred_value="concise",
            confidence=0.8,
            emitted_at=datetime.now(tz=timezone.utc),
        )
        with pytest.raises(ValidationError):
            signal.confidence = 0.5  # type: ignore[misc]

    def test_confidence_bounds(self) -> None:
        with pytest.raises(ValidationError):
            ModelPersonaSignal(
                user_id="test-user",
                session_id="sess-001",
                signal_type="tone",
                evidence="test",
                inferred_value="concise",
                confidence=1.5,
                emitted_at=datetime.now(tz=timezone.utc),
            )

    def test_evidence_max_length(self) -> None:
        with pytest.raises(ValidationError):
            ModelPersonaSignal(
                user_id="test-user",
                session_id="sess-001",
                signal_type="tone",
                evidence="x" * 501,
                inferred_value="concise",
                confidence=0.5,
                emitted_at=datetime.now(tz=timezone.utc),
            )

    def test_custom_signal_id(self) -> None:
        custom_id = uuid4()
        signal = ModelPersonaSignal(
            signal_id=custom_id,
            user_id="test-user",
            session_id="sess-001",
            signal_type="vocabulary",
            evidence="High jargon density",
            inferred_value="0.85",
            confidence=0.7,
            emitted_at=datetime.now(tz=timezone.utc),
        )
        assert signal.signal_id == custom_id

    def test_serialization_round_trip(self) -> None:
        now = datetime.now(tz=timezone.utc)
        signal = ModelPersonaSignal(
            user_id="test-user",
            session_id="sess-001",
            signal_type="technical_level",
            evidence="Asked 'what does frozen=True mean?'",
            inferred_value="beginner",
            confidence=0.85,
            emitted_at=now,
        )
        data = signal.model_dump()
        restored = ModelPersonaSignal.model_validate(data)
        assert restored == signal
