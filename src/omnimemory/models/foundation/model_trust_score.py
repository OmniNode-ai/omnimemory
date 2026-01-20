"""
Trust score model with time decay following ONEX standards.
"""

import math
from datetime import datetime, timezone
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    ValidationInfo,
    field_validator,
    model_validator,
)

from omnimemory.enums import EnumDecayFunction, EnumTrustLevel


def ensure_timezone_aware(
    dt: datetime | None, field_name: str = "datetime"
) -> datetime | None:
    """
    Ensure a datetime is timezone-aware, converting naive datetimes to UTC.

    Args:
        dt: The datetime to validate
        field_name: Name of the field for error messages

    Returns:
        Timezone-aware datetime or None

    Raises:
        TypeError: If the value is not a datetime or None
    """
    if dt is None:
        return None
    if not isinstance(dt, datetime):
        raise TypeError(f"{field_name} must be a datetime, got {type(dt).__name__}")
    if dt.tzinfo is None:
        # Naive datetime - assume it represents UTC time and attach timezone
        return dt.replace(tzinfo=timezone.utc)
    return dt


class ModelTrustScore(BaseModel):
    """Trust score with time-based decay and validation."""

    model_config = ConfigDict(frozen=False)

    base_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Base trust score without time decay",
    )
    current_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Current trust score with time decay applied",
    )
    trust_level: EnumTrustLevel = Field(
        description="Categorical trust level",
    )

    # Time decay configuration
    decay_function: EnumDecayFunction = Field(
        default=EnumDecayFunction.EXPONENTIAL,
        description="Type of time decay function to apply",
    )
    decay_rate: float = Field(
        default=0.01,
        ge=0.0,
        le=1.0,
        description="Rate of trust decay (0=no decay, 1=fast decay)",
    )
    half_life_days: int = Field(
        default=30,
        ge=1,
        le=3650,
        description="Days for trust score to decay to half value",
    )

    # Temporal information
    initial_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the trust score was initially established",
    )
    last_updated: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the trust score was last updated",
    )
    last_verified: datetime | None = Field(
        default=None,
        description="When the trust was last externally verified",
    )

    # Metadata
    source_node_id: UUID | None = Field(
        default=None,
        description="Node that established this trust score",
    )
    verification_count: int = Field(
        default=0,
        ge=0,
        description="Number of times this trust has been verified",
    )
    violation_count: int = Field(
        default=0,
        ge=0,
        description="Number of trust violations recorded",
    )

    # Performance optimization caching (using PrivateAttr for underscore names)
    _cached_score: float | None = PrivateAttr(default=None)
    _cache_timestamp: datetime | None = PrivateAttr(default=None)
    _cache_ttl_seconds: int = PrivateAttr(default=300)

    @model_validator(mode="after")
    def validate_timezone_aware(self) -> "ModelTrustScore":
        """Ensure all datetime values are timezone-aware to prevent comparison errors.

        Uses model_validator(mode='after') to run after all fields are parsed,
        ensuring datetime fields are already coerced from strings/other formats.
        """
        # Guard against naive datetimes and convert to UTC
        result = ensure_timezone_aware(self.initial_timestamp, "initial_timestamp")
        if result is not None:
            self.initial_timestamp = result

        result = ensure_timezone_aware(self.last_updated, "last_updated")
        if result is not None:
            self.last_updated = result

        self.last_verified = ensure_timezone_aware(self.last_verified, "last_verified")

        return self

    @field_validator("trust_level")
    @classmethod
    def validate_trust_level_matches_score(
        cls, v: EnumTrustLevel, info: ValidationInfo
    ) -> EnumTrustLevel:
        """Ensure trust level matches base score."""
        if info.data and "current_score" in info.data:
            score = info.data["current_score"]
            expected_level = cls._score_to_level(score)
            if v != expected_level:
                raise ValueError(
                    f"Trust level {v} doesn't match score {score}, expected {expected_level}"
                )
        return v

    @staticmethod
    def _score_to_level(score: float) -> EnumTrustLevel:
        """Convert numeric score to trust level."""
        if score >= 0.9:
            return EnumTrustLevel.VERIFIED
        elif score >= 0.7:
            return EnumTrustLevel.HIGH
        elif score >= 0.5:
            return EnumTrustLevel.MEDIUM
        elif score >= 0.2:
            return EnumTrustLevel.LOW
        else:
            return EnumTrustLevel.UNTRUSTED

    def calculate_current_score(
        self, as_of: datetime | None = None, force_recalculate: bool = False
    ) -> float:
        """Calculate current trust score with time decay and caching for performance."""
        if as_of is None:
            validated_as_of = datetime.now(timezone.utc)
        else:
            # Ensure provided datetime is timezone-aware to prevent TypeError
            result = ensure_timezone_aware(as_of, "as_of")
            # ensure_timezone_aware only returns None if input is None, which we checked
            assert (
                result is not None
            ), "as_of was not None but ensure_timezone_aware returned None"
            validated_as_of = result

        # Check cache validity if not forcing recalculation
        if (
            not force_recalculate
            and self._is_cache_valid(validated_as_of)
            and self._cached_score is not None
        ):
            return self._cached_score

        if self.decay_function == EnumDecayFunction.NONE:
            score = self.base_score
            self._update_cache(score, validated_as_of)
            return score

        # Calculate time elapsed
        time_elapsed = validated_as_of - self.last_updated
        days_elapsed = time_elapsed.total_seconds() / 86400  # Convert to days

        if days_elapsed <= 0:
            score = self.base_score
            self._update_cache(score, validated_as_of)
            return score

        # Apply decay function
        if self.decay_function == EnumDecayFunction.LINEAR:
            decay_factor = max(0, 1 - (days_elapsed * self.decay_rate))
        elif self.decay_function == EnumDecayFunction.EXPONENTIAL:
            decay_factor = math.exp(-days_elapsed / self.half_life_days * math.log(2))
        elif self.decay_function == EnumDecayFunction.LOGARITHMIC:
            decay_factor = max(0, 1 - (math.log(1 + days_elapsed) * self.decay_rate))
        else:
            decay_factor = 1.0

        decayed_score = self.base_score * decay_factor
        score = max(0.0, min(1.0, decayed_score))

        # Cache the calculated score
        self._update_cache(score, validated_as_of)
        return score

    def _is_cache_valid(self, as_of: datetime) -> bool:
        """Check if cached score is still valid."""
        if self._cached_score is None or self._cache_timestamp is None:
            return False

        # Ensure as_of is timezone-aware for safe comparison
        validated_as_of = ensure_timezone_aware(as_of, "as_of")
        if validated_as_of is None:
            return False

        cache_age = (validated_as_of - self._cache_timestamp).total_seconds()
        return cache_age < self._cache_ttl_seconds

    def _update_cache(self, score: float, timestamp: datetime) -> None:
        """Update cached score and timestamp."""
        self._cached_score = score
        # Ensure cache timestamp is timezone-aware for safe comparisons
        self._cache_timestamp = ensure_timezone_aware(timestamp, "timestamp")

    def invalidate_cache(self) -> None:
        """Manually invalidate the score cache."""
        self._cached_score = None
        self._cache_timestamp = None

    def update_score(self, new_base_score: float, verified: bool = False) -> None:
        """Update the trust score and invalidate cache."""
        # Invalidate cache since base parameters changed
        self.invalidate_cache()

        self.base_score = new_base_score
        self.current_score = self.calculate_current_score()
        self.trust_level = self._score_to_level(self.current_score)
        self.last_updated = datetime.now(timezone.utc)

        if verified:
            self.last_verified = datetime.now(timezone.utc)
            self.verification_count += 1

    def record_violation(self, penalty: float = 0.1) -> None:
        """Record a trust violation with penalty."""
        self.violation_count += 1
        penalty_factor = min(penalty * self.violation_count, 0.5)  # Max 50% penalty
        self.base_score = max(0.0, self.base_score - penalty_factor)
        self.current_score = self.calculate_current_score()
        self.trust_level = self._score_to_level(self.current_score)
        self.last_updated = datetime.now(timezone.utc)

    def refresh_current_score(self) -> None:
        """Refresh the current score based on time decay."""
        self.current_score = self.calculate_current_score()
        self.trust_level = self._score_to_level(self.current_score)

    @classmethod
    def create_from_float(
        cls, score: float, source_node_id: UUID | None = None
    ) -> "ModelTrustScore":
        """Create trust score model from legacy float value."""
        trust_level = cls._score_to_level(score)
        return cls(
            base_score=score,
            current_score=score,
            trust_level=trust_level,
            source_node_id=source_node_id,
        )
