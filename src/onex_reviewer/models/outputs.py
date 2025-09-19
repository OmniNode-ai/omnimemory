"""Review output models."""

from typing import List
from pydantic import BaseModel, Field
from .finding import Finding


class ReviewOutput(BaseModel):
    """Output from review agents."""

    findings: List[Finding] = Field(description="List of compliance findings")
    summary: str = Field(description="Markdown summary of findings")
    coverage_notes: str = Field(default="", description="Notes about coverage limitations")
    risk_score: int = Field(description="Overall risk score 0-100")

    def to_output_string(self) -> str:
        """Format as agent output with separator."""
        ndjson_lines = [finding.to_ndjson_dict() for finding in self.findings]
        ndjson_str = "\n".join([str(line).replace("'", '"') for line in ndjson_lines])

        return f"{ndjson_str}\n---ONEX-SEP---\n{self.summary}"