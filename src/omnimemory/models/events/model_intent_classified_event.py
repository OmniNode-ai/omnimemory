# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

from datetime import datetime
from typing import Literal
from uuid import UUID

from omnibase_core.enums.intelligence import EnumIntentClass
from pydantic import BaseModel, ConfigDict, Field, model_validator

# Alias map: legacy ``intent_category`` values that don't directly match
# ``EnumIntentClass`` string values. Applied when normalizing legacy wire payloads.
# Values not in this map are tried directly against ``EnumIntentClass``; unknown
# values fall back to ``_FALLBACK_INTENT_CLASS``.
_INTENT_CLASS_ALIASES: dict[str, str] = {
    "feat": "feature",
    "feature_request": "feature",
    "debugging": "bugfix",
    "bug": "bugfix",
    "refactoring": "refactor",
    "code_generation": "feature",
    "testing": "feature",
    "architecture": "feature",
    "api_design": "feature",
    "devops": "configuration",
    "database": "migration",
    "quality_assessment": "analysis",
    "semantic_analysis": "analysis",
    "pattern_learning": "analysis",
    "code_review": "analysis",
    "explanation": "analysis",
    "help": "analysis",
    "clarify": "analysis",
    "feedback": "analysis",
    "unknown": "analysis",
}

# Fallback class when intent value cannot be resolved to any EnumIntentClass member
_FALLBACK_INTENT_CLASS: EnumIntentClass = EnumIntentClass.ANALYSIS


def _normalize_to_intent_class(raw: str) -> EnumIntentClass:
    """Normalize a raw intent string to the nearest ``EnumIntentClass``.

    Applies the alias map first, then attempts a direct ``EnumIntentClass``
    lookup. Falls back to ``EnumIntentClass.ANALYSIS`` for unknown values.

    Args:
        raw: Casefold-stripped intent string from the wire payload.

    Returns:
        Resolved ``EnumIntentClass`` member.
    """
    normalized = _INTENT_CLASS_ALIASES.get(raw, raw)
    try:
        return EnumIntentClass(normalized)
    except ValueError:
        return _FALLBACK_INTENT_CLASS


class ModelIntentClassifiedEvent(BaseModel):
    """Incoming event from omniintelligence intent classifier.

    Handles both the legacy wire format (``intent_category: str``) emitted by
    older omniintelligence versions and the canonical format (``intent_class:
    EnumIntentClass``) introduced in OMN-3248.

    The ``normalize_intent_field`` model_validator runs before field validation
    and ensures exactly one of the following is true after normalization:
    - ``intent_class`` is present as a valid ``EnumIntentClass`` string value.

    Resolution rules:
    1. Both ``intent_class`` and ``intent_category`` present → canonical
       ``intent_class`` wins; ``intent_category`` is discarded.
    2. Only ``intent_category`` present → normalized via alias map, mapped to
       the nearest ``EnumIntentClass`` value, and stored as ``intent_class``.
    3. Only ``intent_class`` present → used as-is.

    Handler rule: internal code uses ``event.intent_class`` (Enum);
    emitted payloads use ``event.intent_class.value`` (string). No mixing.

    Note: Uses ``extra="ignore"`` to allow forward compatibility — if
    omniintelligence adds new fields, this consumer won't reject them.

    TODO(OMN-future): Consider migrating to omnibase_core.models.events
    once cross-repo event schemas are standardized. This is logically
    an omniintelligence-owned event; omnimemory is a consumer.
    """

    model_config = ConfigDict(frozen=True, extra="ignore")

    event_type: Literal["IntentClassified"] = Field(
        "IntentClassified", description="Event type discriminator for message routing"
    )
    session_id: str = Field(..., min_length=1, description="Session identifier")
    correlation_id: UUID | None = Field(
        default=None,
        description="Correlation ID for tracing. Optional to match upstream ModelPatternLifecycleEvent (OMN-2841).",
    )
    intent_class: EnumIntentClass = Field(
        ...,
        description=(
            "Canonical typed intent class (OMN-3248). Populated directly from the "
            "``intent_class`` wire field (new format) or normalized from the legacy "
            "``intent_category`` wire field. See ``normalize_intent_field`` for rules."
        ),
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Classification confidence score"
    )
    keywords: tuple[str, ...] = Field(
        default=(),
        description="Extracted keywords (forward-compatible with OMN-1626)",
    )
    emitted_at: datetime = Field(
        ...,
        description=(
            "Timestamp when the event was emitted (UTC). "
            "Aligns with omniintelligence ModelIntentClassifiedEnvelope.emitted_at."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_intent_field(cls, values: dict) -> dict:  # type: ignore[type-arg]
        """Normalize the intent field from legacy ``intent_category`` to canonical ``intent_class``.

        Applied before field validation so that both wire formats produce a valid
        ``intent_class: EnumIntentClass`` value.

        Resolution rules:
        1. Both ``intent_class`` and ``intent_category`` present: canonical
           ``intent_class`` wins; ``intent_category`` is discarded to prevent
           silent corruption.
        2. Only ``intent_category`` present: normalized via ``_INTENT_CLASS_ALIASES``,
           then resolved against ``EnumIntentClass``; falls back to ANALYSIS.
        3. Only ``intent_class`` present: passed through unchanged (Pydantic validates).

        Args:
            values: Raw incoming dict from the wire payload.

        Returns:
            Mutated dict with ``intent_category`` removed and ``intent_class``
            set to a string value accepted by ``EnumIntentClass``.
        """
        has_class = "intent_class" in values
        has_category = "intent_category" in values

        if has_class and has_category:
            # Canonical wins — discard legacy to prevent silent corruption
            values.pop("intent_category")
            return values

        if has_category and not has_class:
            raw = str(values.pop("intent_category")).casefold().strip()
            if not raw:
                raise ValueError(
                    "intent_category must not be empty; "
                    "provide a non-empty string or use intent_class directly"
                )
            values["intent_class"] = _normalize_to_intent_class(raw).value

        return values

    @property
    def intent_category(self) -> str:
        """Backward-compatible accessor returning the string value of ``intent_class``.

        Allows existing code that reads ``event.intent_category`` to continue
        working without modification. New code should read ``event.intent_class``
        (Enum) directly. Emitted payloads should use ``event.intent_class.value``.
        """
        return self.intent_class.value
