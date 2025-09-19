"""ONEX rule engine implementation."""

import hashlib
import re
import yaml
from datetime import datetime
from typing import List, Dict, Optional, Set, Tuple
from pathlib import Path

from ..models.finding import Finding
from ..models.inputs import ReviewInput
from .definitions import RULE_DEFINITIONS, REGEX_PATTERNS, RULESET_VERSION


class RuleEngine:
    """Engine for applying ONEX rules to code diffs."""

    def __init__(self):
        self.rules = {rule.rule_id: rule for rule in RULE_DEFINITIONS}
        self.patterns = REGEX_PATTERNS

    def apply_rules(self, review_input: ReviewInput) -> List[Finding]:
        """Apply all rules to the review input."""
        findings = []

        # Parse policy to get forbidden imports
        policy = yaml.safe_load(review_input.policy_yaml) if review_input.policy_yaml else {}
        repo_config = policy.get("repos", {}).get(review_input.repo, {})
        forbidden_patterns = [
            re.compile(pattern) for pattern in repo_config.get("forbids", [])
        ]

        # Process diff hunks
        diff_hunks = self._parse_diff(review_input.git_diff)

        for hunk in diff_hunks:
            file_path = hunk["path"]
            content = hunk["content"]

            # Apply naming rules
            findings.extend(self._check_naming_rules(file_path, content, review_input.repo))

            # Apply boundary rules (forbidden imports)
            findings.extend(self._check_boundary_rules(
                file_path, content, forbidden_patterns, review_input.repo
            ))

            # Apply SPI purity rules
            findings.extend(self._check_spi_rules(file_path, content, review_input.repo))

            # Apply typing hygiene rules
            findings.extend(self._check_typing_rules(file_path, content, review_input.repo))

            # Apply waiver hygiene rules
            findings.extend(self._check_waiver_rules(
                file_path, content, review_input.today, review_input.repo
            ))

        return findings

    def _parse_diff(self, diff_content: str) -> List[Dict[str, str]]:
        """Parse unified diff into structured hunks."""
        hunks = []
        current_file = None
        current_content = []

        for line in diff_content.split("\n"):
            if line.startswith("diff --git"):
                if current_file:
                    hunks.append({
                        "path": current_file,
                        "content": "\n".join(current_content)
                    })
                # Extract file path
                match = re.match(r"diff --git a/(.*) b/", line)
                current_file = match.group(1) if match else None
                current_content = []
            elif current_file:
                current_content.append(line)

        # Add last hunk
        if current_file:
            hunks.append({
                "path": current_file,
                "content": "\n".join(current_content)
            })

        return hunks

    def _check_naming_rules(self, file_path: str, content: str, repo: str) -> List[Finding]:
        """Check naming convention rules."""
        findings = []

        for rule in self.rules.values():
            if rule.category != "naming":
                continue

            if rule.file_pattern and not rule.file_pattern.match(file_path):
                continue

            if rule.pattern:
                for match in rule.pattern.finditer(content):
                    line_num = content[:match.start()].count("\n") + 1
                    class_name = match.group(1) if match.groups() else ""

                    findings.append(Finding(
                        ruleset_version=RULESET_VERSION,
                        rule_id=rule.rule_id,
                        severity=rule.severity,
                        repo=repo,
                        path=file_path,
                        line=line_num,
                        message=rule.description,
                        evidence={"class_name": class_name},
                        suggested_fix=f"Rename to {rule.rule_id.split('.')[-1].split('_')[0]}{class_name}",
                        fingerprint=self._generate_fingerprint(
                            file_path, line_num, rule.rule_id, class_name
                        )
                    ))

        return findings

    def _check_boundary_rules(
        self, file_path: str, content: str, forbidden_patterns: List[re.Pattern], repo: str
    ) -> List[Finding]:
        """Check boundary violation rules."""
        findings = []

        import_pattern = self.patterns["IMPORT_LINE"]
        for match in import_pattern.finditer(content):
            import_line = match.group(0)
            line_num = content[:match.start()].count("\n") + 1

            for forbidden in forbidden_patterns:
                if forbidden.match(import_line.lstrip("+")):
                    findings.append(Finding(
                        ruleset_version=RULESET_VERSION,
                        rule_id="ONEX.BOUNDARY.FORBIDDEN_IMPORT_001",
                        severity="error",
                        repo=repo,
                        path=file_path,
                        line=line_num,
                        message="Forbidden import detected",
                        evidence={"import_line": import_line.lstrip("+")},
                        suggested_fix="Remove forbidden import",
                        fingerprint=self._generate_fingerprint(
                            file_path, line_num, "ONEX.BOUNDARY.FORBIDDEN_IMPORT_001", import_line
                        )
                    ))

        return findings

    def _check_spi_rules(self, file_path: str, content: str, repo: str) -> List[Finding]:
        """Check SPI purity rules."""
        findings = []

        # Check for runtime_checkable on Protocol classes
        if "src/omnibase_spi/" in file_path:
            class_pattern = self.patterns["CLASS_HEADER"]
            lines = content.split("\n")

            for match in class_pattern.finditer(content):
                if "Protocol" in match.group(0):
                    line_num = content[:match.start()].count("\n") + 1
                    # Check for @runtime_checkable within 5 lines before
                    start_check = max(0, line_num - 6)
                    context = "\n".join(lines[start_check:line_num])

                    if "@runtime_checkable" not in context:
                        findings.append(Finding(
                            ruleset_version=RULESET_VERSION,
                            rule_id="ONEX.SPI.RUNTIMECHECKABLE_001",
                            severity="error",
                            repo=repo,
                            path=file_path,
                            line=line_num,
                            message="Protocol class without @runtime_checkable",
                            evidence={"class_definition": match.group(0).lstrip("+")},
                            suggested_fix="Add @runtime_checkable decorator",
                            fingerprint=self._generate_fingerprint(
                                file_path, line_num, "ONEX.SPI.RUNTIMECHECKABLE_001", match.group(0)
                            )
                        ))

            # Check for forbidden libraries in SPI
            forbidden_spi = ["os", "pathlib", "sqlite3", "requests", "httpx", "socket", "open("]
            for forbidden in forbidden_spi:
                escaped = re.escape(forbidden)
                pattern = re.compile(rf"^\+.*(import\s+{escaped}|from\s+{escaped}|{escaped})", re.MULTILINE)
                for match in pattern.finditer(content):
                    line_num = content[:match.start()].count("\n") + 1
                    findings.append(Finding(
                        ruleset_version=RULESET_VERSION,
                        rule_id="ONEX.SPI.FORBIDDEN_LIB_001",
                        severity="error",
                        repo=repo,
                        path=file_path,
                        line=line_num,
                        message=f"Forbidden library '{forbidden}' in SPI",
                        evidence={"line": match.group(0).lstrip("+")},
                        suggested_fix=f"Remove usage of '{forbidden}'",
                        fingerprint=self._generate_fingerprint(
                            file_path, line_num, "ONEX.SPI.FORBIDDEN_LIB_001", match.group(0)
                        )
                    ))

        return findings

    def _check_typing_rules(self, file_path: str, content: str, repo: str) -> List[Finding]:
        """Check typing hygiene rules."""
        findings = []

        # Skip test files
        if "test" in file_path.lower():
            return findings

        for rule in self.rules.values():
            if rule.category != "typing":
                continue

            if rule.pattern:
                for match in rule.pattern.finditer(content):
                    line_num = content[:match.start()].count("\n") + 1

                    findings.append(Finding(
                        ruleset_version=RULESET_VERSION,
                        rule_id=rule.rule_id,
                        severity=rule.severity,
                        repo=repo,
                        path=file_path,
                        line=line_num,
                        message=rule.description,
                        evidence={"line": match.group(0).lstrip("+")},
                        suggested_fix="Add type annotations",
                        fingerprint=self._generate_fingerprint(
                            file_path, line_num, rule.rule_id, match.group(0)
                        )
                    ))

        return findings

    def _check_waiver_rules(self, file_path: str, content: str, today: str, repo: str) -> List[Finding]:
        """Check waiver hygiene rules."""
        findings = []

        waiver_pattern = self.patterns["WAIVER_LINE"]
        reason_pattern = self.patterns["WAIVER_REASON"]
        expires_pattern = self.patterns["WAIVER_EXPIRES"]

        for match in waiver_pattern.finditer(content):
            waiver_line = match.group(0)
            line_num = content[:match.start()].count("\n") + 1

            # Check for malformed waiver
            if not reason_pattern.search(waiver_line) or not expires_pattern.search(waiver_line):
                findings.append(Finding(
                    ruleset_version=RULESET_VERSION,
                    rule_id="ONEX.WAIVER.MALFORMED_001",
                    severity="warning",
                    repo=repo,
                    path=file_path,
                    line=line_num,
                    message="Malformed waiver missing reason or expires",
                    evidence={"waiver_line": waiver_line.lstrip("+")},
                    suggested_fix="Add reason= and expires= in ISO format",
                    fingerprint=self._generate_fingerprint(
                        file_path, line_num, "ONEX.WAIVER.MALFORMED_001", waiver_line
                    )
                ))

            # Check for expired waiver
            expires_match = expires_pattern.search(waiver_line)
            if expires_match:
                expires_date = expires_match.group(1)
                if expires_date < today:
                    findings.append(Finding(
                        ruleset_version=RULESET_VERSION,
                        rule_id="ONEX.WAIVER.EXPIRED_001",
                        severity="error",
                        repo=repo,
                        path=file_path,
                        line=line_num,
                        message="Expired waiver",
                        evidence={"expires": expires_date, "today": today},
                        suggested_fix="Update or remove expired waiver",
                        fingerprint=self._generate_fingerprint(
                            file_path, line_num, "ONEX.WAIVER.EXPIRED_001", expires_date
                        )
                    ))

        return findings

    def _generate_fingerprint(self, path: str, line: int, rule_id: str, evidence: str) -> str:
        """Generate unique fingerprint for a finding."""
        content = f"{path}:{line}:{rule_id}:{evidence}"
        return hashlib.md5(content.encode()).hexdigest()[:8]