"""Review input models."""

from pydantic import BaseModel, Field
from typing import Optional


class ReviewInput(BaseModel):
    """Input data for review agents."""

    repo: str = Field(description="Repository name")
    commit_range: str = Field(description="Git commit range to review")
    today: str = Field(description="Current date in YYYY-MM-DD format")
    policy_yaml: str = Field(description="Policy configuration YAML content")
    git_stats: str = Field(description="Git diff statistics")
    git_names: str = Field(description="Git file names and statuses")
    git_diff: str = Field(description="Git unified diff content")
    is_baseline: bool = Field(default=False, description="Whether this is a baseline review")
    shard_index: Optional[int] = Field(default=None, description="Index of current shard if sharded")