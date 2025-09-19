"""Base agent for ONEX reviewers."""

from abc import ABC, abstractmethod
from typing import List, Tuple
import json

from ..models.finding import Finding
from ..models.inputs import ReviewInput
from ..models.outputs import ReviewOutput
from ..rules.engine import RuleEngine


class BaseAgent(ABC):
    """Base class for review agents."""

    def __init__(self):
        self.rule_engine = RuleEngine()

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Get the system prompt for the agent."""
        pass

    @abstractmethod
    def format_user_prompt(self, review_input: ReviewInput) -> str:
        """Format the user prompt with review input."""
        pass

    def process(self, review_input: ReviewInput) -> ReviewOutput:
        """Process review input and generate findings."""
        # Apply rules to generate findings
        findings = self.rule_engine.apply_rules(review_input)

        # Calculate risk score
        risk_score = self._calculate_risk_score(findings)

        # Generate summary
        summary = self._generate_summary(findings, risk_score, review_input)

        return ReviewOutput(
            findings=findings,
            summary=summary,
            risk_score=risk_score,
            coverage_notes=self._get_coverage_notes(review_input)
        )

    def _calculate_risk_score(self, findings: List[Finding]) -> int:
        """Calculate overall risk score based on findings."""
        if not findings:
            return 0

        error_count = sum(1 for f in findings if f.severity == "error")
        warning_count = sum(1 for f in findings if f.severity == "warning")

        # Weight errors more heavily than warnings
        score = min(100, error_count * 20 + warning_count * 5)
        return score

    def _generate_summary(self, findings: List[Finding], risk_score: int, review_input: ReviewInput) -> str:
        """Generate markdown summary of findings."""
        # Count findings by category
        categories = {}
        for finding in findings:
            category = finding.rule_id.split(".")[1].lower()
            if category not in categories:
                categories[category] = {"error": 0, "warning": 0}
            categories[category][finding.severity] += 1

        # Build summary
        lines = []
        lines.append("## Executive summary")
        lines.append(f"Risk score: {risk_score}")

        # Category counts
        category_summary = []
        for cat, counts in categories.items():
            parts = []
            if counts["error"] > 0:
                parts.append(f"{counts['error']} {cat} error{'s' if counts['error'] > 1 else ''}")
            if counts["warning"] > 0:
                parts.append(f"{counts['warning']} {cat} warning{'s' if counts['warning'] > 1 else ''}")
            if parts:
                category_summary.append(", ".join(parts))

        if category_summary:
            lines.append(f"- {', '.join(category_summary)}")
        else:
            lines.append("- No violations found")

        # Top violations
        lines.append("## Top violations")
        if findings:
            # Show up to 3 most severe violations
            severe_findings = sorted(findings, key=lambda f: (0 if f.severity == "error" else 1, f.rule_id))[:3]
            for finding in severe_findings:
                lines.append(f"- {finding.message} in {finding.path}:{finding.line}")
        else:
            lines.append("- None")

        # Waiver issues
        lines.append("## Waiver issues")
        waiver_findings = [f for f in findings if "WAIVER" in f.rule_id]
        if waiver_findings:
            for finding in waiver_findings[:2]:
                lines.append(f"- {finding.message} in {finding.path}:{finding.line}")
        else:
            lines.append("- None")

        # Next actions
        lines.append("## Next actions")
        if findings:
            actions = set()
            for finding in findings:
                if "NAMING" in finding.rule_id:
                    actions.add("Rename classes to follow conventions")
                elif "BOUNDARY" in finding.rule_id:
                    actions.add("Remove forbidden imports")
                elif "TYPE" in finding.rule_id:
                    actions.add("Add missing type annotations")
                elif "SPI" in finding.rule_id:
                    actions.add("Fix SPI compliance issues")
                elif "WAIVER" in finding.rule_id:
                    actions.add("Update expired or malformed waivers")

            lines.append(f"- {', '.join(sorted(actions))}")
        else:
            lines.append("- No actions required")

        # Coverage
        lines.append("## Coverage")
        if review_input.is_baseline and review_input.shard_index:
            lines.append(f"- Processing shard {review_input.shard_index}")
        else:
            # Count files in diff
            file_count = len(set(line.split("\t")[1] for line in review_input.git_names.split("\n") if "\t" in line))
            lines.append(f"- Reviewed {file_count} files")

        return "\n".join(lines)

    def _get_coverage_notes(self, review_input: ReviewInput) -> str:
        """Get coverage notes for the review."""
        if len(review_input.git_diff) > 500000:
            return "Diff truncated due to size limit"
        return ""