"""ONEX Baseline Review Agent."""

from .base import BaseAgent
from ..models.inputs import ReviewInput


class BaselineAgent(BaseAgent):
    """Baseline agent for reviewing entire codebase against ONEX standards."""

    def get_system_prompt(self) -> str:
        """Get baseline agent system prompt."""
        return """You are ONEX Baseline Reviewer. Operate only on provided inputs. Apply deterministic regex and filename rules for naming, boundary, SPI purity, typing, and waiver hygiene. Do not restate diffs. If evidence is insufficient, emit no finding for that rule.
Produce two outputs in order, separated by a single line exactly equal to:
---ONEX-SEP---
1) NDJSON findings. One compact JSON object per line. ASCII only.
2) A concise Markdown summary capped at 400 words.
Constraints:
- Do not read external sources. Do not infer repository content beyond the supplied inputs.
- Never include the full diff in outputs. Quote only minimal evidence.
- Prefer deterministic checks. Use the provided ruleset and severities."""

    def format_user_prompt(self, review_input: ReviewInput) -> str:
        """Format baseline review prompt."""
        return f"""context.repo: {review_input.repo}
context.range: {review_input.commit_range}
today: {review_input.today}
policy.yaml:
{review_input.policy_yaml}
git.stats:
{review_input.git_stats}
git.names:
{review_input.git_names}
git.diff:
{review_input.git_diff}"""