"""
Trust score model with time decay following ONEX standards.
"""

import math
from datetime import datetime, timedelta
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field, field_validator

from omnimemory.enums import EnumTrustLevel, EnumDecayFunction


class ModelTrustScore(BaseModel):
    """Trust score with time-based decay and validation."""
    
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
        default_factory=datetime.utcnow,
        description="When the trust score was initially established",
    )
    last_updated: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the trust score was last updated",
    )
    last_verified: Optional[datetime] = Field(
        default=None,
        description="When the trust was last externally verified",
    )
    
    # Metadata
    source_node_id: Optional[UUID] = Field(
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
    
    @field_validator('trust_level')
    @classmethod
    def validate_trust_level_matches_score(cls, v, info):
        """Ensure trust level matches base score."""
        if 'current_score' in info.data:
            score = info.data['current_score']
            expected_level = cls._score_to_level(score)
            if v != expected_level:
                raise ValueError(f"Trust level {v} doesn't match score {score}, expected {expected_level}")
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
    
    def calculate_current_score(self, as_of: Optional[datetime] = None) -> float:
        """Calculate current trust score with time decay."""
        if as_of is None:
            as_of = datetime.utcnow()
            
        if self.decay_function == EnumDecayFunction.NONE:
            return self.base_score
            
        # Calculate time elapsed
        time_elapsed = as_of - self.last_updated
        days_elapsed = time_elapsed.total_seconds() / 86400  # Convert to days
        
        if days_elapsed <= 0:
            return self.base_score
            
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
        return max(0.0, min(1.0, decayed_score))
    
    def update_score(self, new_base_score: float, verified: bool = False) -> None:
        """Update the trust score."""
        self.base_score = new_base_score
        self.current_score = self.calculate_current_score()
        self.trust_level = self._score_to_level(self.current_score)
        self.last_updated = datetime.utcnow()
        
        if verified:
            self.last_verified = datetime.utcnow()
            self.verification_count += 1
    
    def record_violation(self, penalty: float = 0.1) -> None:
        """Record a trust violation with penalty."""
        self.violation_count += 1
        penalty_factor = min(penalty * self.violation_count, 0.5)  # Max 50% penalty
        self.base_score = max(0.0, self.base_score - penalty_factor)
        self.current_score = self.calculate_current_score()
        self.trust_level = self._score_to_level(self.current_score)
        self.last_updated = datetime.utcnow()
    
    def refresh_current_score(self) -> None:
        """Refresh the current score based on time decay."""
        self.current_score = self.calculate_current_score()
        self.trust_level = self._score_to_level(self.current_score)
    
    @classmethod
    def create_from_float(cls, score: float, source_node_id: Optional[UUID] = None) -> "ModelTrustScore":
        """Create trust score model from legacy float value."""
        trust_level = cls._score_to_level(score)
        return cls(
            base_score=score,
            current_score=score,
            trust_level=trust_level,
            source_node_id=source_node_id,
        )