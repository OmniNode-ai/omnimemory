"""ONEX finding model."""

from typing import Dict, Literal
from pydantic import BaseModel, Field


class Finding(BaseModel):
    """Represents a single ONEX compliance finding."""

    ruleset_version: str = Field(description="Version of the ruleset used")
    rule_id: str = Field(description="Unique identifier for the rule")
    severity: Literal["error", "warning"] = Field(description="Severity level of the finding")
    repo: str = Field(description="Repository name")
    path: str = Field(description="File path where violation occurred")
    line: int = Field(description="Line number of the violation")
    message: str = Field(description="Description of the violation")
    evidence: Dict[str, str] = Field(description="Evidence supporting the finding")
    suggested_fix: str = Field(description="Recommended fix for the violation")
    fingerprint: str = Field(description="Unique fingerprint for this finding")

    def to_ndjson_dict(self) -> Dict[str, str | int | Dict[str, str]]:
        """Convert to NDJSON-compatible dictionary."""
        return {
            "ruleset_version": self.ruleset_version,
            "rule_id": self.rule_id,
            "severity": self.severity,
            "repo": self.repo,
            "path": self.path,
            "line": self.line,
            "message": self.message,
            "evidence": self.evidence,
            "suggested_fix": self.suggested_fix,
            "fingerprint": self.fingerprint
        }