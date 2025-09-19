"""ONEX rule definitions."""

import re
from typing import Dict, List, Pattern, Literal
from pydantic import BaseModel, Field


RULESET_VERSION = "0.1"


class RuleDefinition(BaseModel):
    """Definition of a single ONEX rule."""

    rule_id: str = Field(description="Unique rule identifier")
    severity: Literal["error", "warning"] = Field(description="Severity level")
    category: str = Field(description="Rule category")
    description: str = Field(description="Rule description")
    pattern: Pattern[str] | None = Field(default=None, description="Regex pattern for detection")
    file_pattern: Pattern[str] | None = Field(default=None, description="File path pattern")


RULE_DEFINITIONS: List[RuleDefinition] = [
    # Naming rules
    RuleDefinition(
        rule_id="ONEX.NAMING.PROTOCOL_001",
        severity="error",
        category="naming",
        description="Protocol class does not start with 'Protocol'",
        file_pattern=re.compile(r"protocol_.*\.py$"),
        pattern=re.compile(r"^\+class\s+(?!Protocol)([A-Z][A-Za-z0-9_]*)\s*\(", re.MULTILINE)
    ),
    RuleDefinition(
        rule_id="ONEX.NAMING.MODEL_001",
        severity="warning",
        category="naming",
        description="Model class does not start with 'Model'",
        file_pattern=re.compile(r"model_.*\.py$"),
        pattern=re.compile(r"^\+class\s+(?!Model)([A-Z][A-Za-z0-9_]*)\s*\(", re.MULTILINE)
    ),
    RuleDefinition(
        rule_id="ONEX.NAMING.ENUM_001",
        severity="warning",
        category="naming",
        description="Enum class does not start with 'Enum'",
        file_pattern=re.compile(r"enum_.*\.py$"),
        pattern=re.compile(r"^\+class\s+(?!Enum)([A-Z][A-Za-z0-9_]*)\s*\(", re.MULTILINE)
    ),
    RuleDefinition(
        rule_id="ONEX.NAMING.NODE_001",
        severity="warning",
        category="naming",
        description="Node class does not start with 'Node'",
        file_pattern=re.compile(r"node_.*\.py$"),
        pattern=re.compile(r"^\+class\s+(?!Node)([A-Z][A-Za-z0-9_]*)\s*\(", re.MULTILINE)
    ),

    # Typing hygiene
    RuleDefinition(
        rule_id="ONEX.TYPE.UNANNOTATED_DEF_001",
        severity="warning",
        category="typing",
        description="Function definition lacks type annotations",
        pattern=re.compile(r"^\+def\s+[a-zA-Z_][a-zA-Z0-9_]*\([^)]*\)(?:\s*:)?(?!\s*->)", re.MULTILINE)
    ),
    RuleDefinition(
        rule_id="ONEX.TYPE.ANY_001",
        severity="warning",
        category="typing",
        description="Use of 'Any' type in non-test code",
        pattern=re.compile(r"^\+.*\bAny\b", re.MULTILINE)
    ),
    RuleDefinition(
        rule_id="ONEX.TYPE.OPTIONAL_ASSERT_001",
        severity="warning",
        category="typing",
        description="Optional[T] immediately forced non-null",
        pattern=re.compile(r"^\+.*\bOptional\[.*\].*\n.*assert.*is\s+not\s+None", re.MULTILINE)
    ),
]


# Regex patterns for various checks
REGEX_PATTERNS = {
    "CLASS_HEADER": re.compile(r"^\+class\s+([A-Z][A-Za-z0-9_]*)\s*\(", re.MULTILINE),
    "PROTOCOL_DECORATOR": re.compile(r"^\+@runtime_checkable\s*$", re.MULTILINE),
    "IMPORT_LINE": re.compile(r"^\+(from|import)\s+[^#\n]+", re.MULTILINE),
    "DEF_HEADER": re.compile(r"^\+def\s+[a-zA-Z_][a-zA-Z0-9_]*\(.*\)\s*:", re.MULTILINE),
    "ANY_TOKEN": re.compile(r"^\+.*\bAny\b", re.MULTILINE),
    "OPTIONAL_TOKEN": re.compile(r"^\+.*\bOptional\[.*\]", re.MULTILINE),
    "ASSERT_NOT_NONE": re.compile(r"^\+.*assert\s+[^=]+\s+is\s+not\s+None", re.MULTILINE),
    "WAIVER_LINE": re.compile(r"^\+.*#\s*onex:ignore\s+([A-Z._0-9]+)", re.MULTILINE),
    "WAIVER_REASON": re.compile(r"reason=([^#\s]+)"),
    "WAIVER_EXPIRES": re.compile(r"expires=(\d{4}-\d{2}-\d{2})"),
}