"""ONEX Nightly Review Agent."""

from .base import BaseAgent
from ..models.inputs import ReviewInput


class NightlyAgent(BaseAgent):
    """Nightly agent for reviewing incremental changes."""

    def get_system_prompt(self) -> str:
        """Get nightly agent system prompt."""
        return """You are ONEX Nightly Reviewer. Operate only on provided inputs. Apply deterministic regex and filename rules to detect drift against naming, boundary, SPI purity, typing, and waiver hygiene policies. Do not restate diffs.
Produce NDJSON findings then a Markdown summary, separated by:
---ONEX-SEP---
Constraints identical to the Baseline Reviewer."""

    def format_user_prompt(self, review_input: ReviewInput) -> str:
        """Format nightly review prompt."""
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