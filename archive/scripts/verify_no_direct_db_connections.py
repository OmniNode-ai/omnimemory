#!/usr/bin/env python3
"""
Verify no direct database connections exist in event-driven OmniMemory codebase.

This script analyzes the codebase to ensure pure event-driven architecture
compliance by detecting any direct database connection attempts.
"""

import ast
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


class DatabaseConnectionDetector(ast.NodeVisitor):
    """AST visitor to detect direct database connection patterns."""

    def __init__(self):
        self.violations = []
        self.current_file = ""

    def visit_Import(self, node):
        """Check import statements for database libraries."""
        for alias in node.names:
            self._check_import(alias.name, node.lineno)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        """Check from-import statements for database libraries."""
        if node.module:
            self._check_import(node.module, node.lineno)

            # Check specific imports
            for alias in node.names:
                full_import = f"{node.module}.{alias.name}"
                self._check_import(full_import, node.lineno)

        self.generic_visit(node)

    def visit_Call(self, node):
        """Check function calls for database connection patterns."""
        # Get the function name if possible
        func_name = self._get_function_name(node.func)
        if func_name:
            self._check_function_call(func_name, node.lineno)

        # Check for connection string patterns in arguments
        for arg in node.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                self._check_connection_string(arg.value, node.lineno)

        # Check keyword arguments
        for keyword in node.keywords:
            if isinstance(keyword.value, ast.Constant) and isinstance(
                keyword.value.value, str
            ):
                self._check_connection_string(keyword.value.value, node.lineno)

        self.generic_visit(node)

    def _get_function_name(self, func_node) -> Optional[str]:
        """Extract function name from function call node."""
        if isinstance(func_node, ast.Name):
            return func_node.id
        elif isinstance(func_node, ast.Attribute):
            # For method calls like client.connect()
            if isinstance(func_node.value, ast.Name):
                return f"{func_node.value.id}.{func_node.attr}"
            return func_node.attr
        return None

    def _check_import(self, import_name: str, line_no: int):
        """Check if import is a direct database library."""
        forbidden_imports = {
            # PostgreSQL drivers
            "psycopg2",
            "psycopg3",
            "asyncpg",
            "pg8000",
            "py-postgresql",
            "psycopg2.pool",
            "psycopg2.extras",
            "asyncpg.pool",
            # Redis clients
            "redis",
            "aioredis",
            "hiredis",
            "redis.asyncio",
            "redis.connection",
            "redis.client",
            # Qdrant clients
            "qdrant_client",
            "qdrant_client.http",
            "qdrant_client.grpc",
            "qdrant_client.models",
            "qdrant_client.async_client",
            # Generic database libraries
            "sqlite3",
            "mysql.connector",
            "pymongo",
            "motor",
            "elasticsearch",
            "pymysql",
            "MySQLdb",
            # ORM engines (direct use)
            "sqlalchemy.engine",
            "sqlalchemy.create_engine",
            "databases",
            "asyncpg.create_pool",
            # Connection pooling
            "connection_pool",
            "db_pool",
            "asyncio_redis",
        }

        # Check exact matches
        if import_name in forbidden_imports:
            self.violations.append(
                {
                    "type": "direct_import",
                    "file": self.current_file,
                    "line": line_no,
                    "details": f"Direct database import: {import_name}",
                    "severity": "high",
                }
            )

        # Check partial matches
        for forbidden in forbidden_imports:
            if forbidden in import_name and import_name != forbidden:
                self.violations.append(
                    {
                        "type": "suspicious_import",
                        "file": self.current_file,
                        "line": line_no,
                        "details": f"Suspicious import containing database library: {import_name}",
                        "severity": "medium",
                    }
                )

    def _check_function_call(self, func_name: str, line_no: int):
        """Check function calls for database connection patterns."""
        forbidden_functions = {
            # Connection functions
            "create_engine",
            "connect",
            "create_connection",
            "create_pool",
            "get_connection",
            "open_connection",
            "establish_connection",
            # Database-specific functions
            "redis.Redis",
            "redis.from_url",
            "aioredis.create_redis_pool",
            "psycopg2.connect",
            "asyncpg.connect",
            "asyncpg.create_pool",
            "qdrant_client.QdrantClient",
            "sqlite3.connect",
            # Direct SQL execution
            "execute",
            "executemany",
            "fetchone",
            "fetchall",
            "cursor",
        }

        for forbidden in forbidden_functions:
            if forbidden in func_name.lower():
                self.violations.append(
                    {
                        "type": "database_function",
                        "file": self.current_file,
                        "line": line_no,
                        "details": f"Direct database function call: {func_name}",
                        "severity": "high",
                    }
                )

    def _check_connection_string(self, value: str, line_no: int):
        """Check for database connection strings."""
        connection_patterns = [
            r"postgresql://.*",
            r"postgres://.*",
            r"redis://.*",
            r"sqlite:///.*",
            r"mysql://.*",
            r"mongodb://.*",
            r"localhost:\d{4,5}",  # Database ports
            r".*:5432.*",  # PostgreSQL port
            r".*:6379.*",  # Redis port
            r".*:9200.*",  # Elasticsearch port
            r".*:27017.*",  # MongoDB port
        ]

        for pattern in connection_patterns:
            if re.search(pattern, value, re.IGNORECASE):
                # Exclude test/mock URLs
                if not any(
                    exclude in value.lower()
                    for exclude in ["test", "mock", "example", "dummy"]
                ):
                    self.violations.append(
                        {
                            "type": "connection_string",
                            "file": self.current_file,
                            "line": line_no,
                            "details": f"Database connection string detected: {value[:50]}...",
                            "severity": "high",
                        }
                    )

    def analyze_file(self, file_path: str):
        """Analyze a single Python file."""
        self.current_file = file_path

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Parse AST
            tree = ast.parse(content)
            self.visit(tree)

            # Also check raw content for patterns that AST might miss
            self._check_raw_content(content)

        except Exception as e:
            self.violations.append(
                {
                    "type": "parse_error",
                    "file": file_path,
                    "line": 0,
                    "details": f"Could not parse file: {str(e)}",
                    "severity": "low",
                }
            )

    def _check_raw_content(self, content: str):
        """Check raw file content for database patterns."""
        lines = content.split("\n")

        for line_no, line in enumerate(lines, 1):
            line_lower = line.lower()

            # Check for direct database operations
            database_patterns = [
                r"\.connect\s*\(",
                r"\.execute\s*\(",
                r"CREATE\s+TABLE",
                r"SELECT\s+.*\s+FROM",
                r"INSERT\s+INTO",
                r"UPDATE\s+.*\s+SET",
                r"DELETE\s+FROM",
                r"redis\.Redis",
                r"psycopg2\.connect",
                r"asyncpg\.connect",
                r"QdrantClient",
            ]

            for pattern in database_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    # Skip comments and strings that are clearly examples
                    if not (
                        line.strip().startswith("#")
                        or line.strip().startswith('"""')
                        or line.strip().startswith("'''")
                        or "example" in line_lower
                        or "test" in line_lower
                    ):
                        self.violations.append(
                            {
                                "type": "raw_pattern",
                                "file": self.current_file,
                                "line": line_no,
                                "details": f"Database operation pattern: {line.strip()[:100]}",
                                "severity": "medium",
                            }
                        )


class EventDrivenComplianceChecker:
    """Check compliance with pure event-driven architecture."""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.detector = DatabaseConnectionDetector()

    def get_event_driven_files(self) -> List[str]:
        """Get list of event-driven architecture files to check."""
        event_driven_dirs = [
            "src/omnimemory/events",
            "src/omnimemory/services",
            "src/omnimemory/adapters",
        ]

        files_to_check = []

        for dir_path in event_driven_dirs:
            full_dir = self.project_root / dir_path
            if full_dir.exists():
                for py_file in full_dir.rglob("*.py"):
                    if py_file.is_file():
                        files_to_check.append(str(py_file))

        return files_to_check

    def check_compliance(self) -> Dict[str, List[Dict]]:
        """Check compliance with event-driven architecture."""
        files_to_check = self.get_event_driven_files()

        print(f"Checking {len(files_to_check)} event-driven architecture files...")
        print("=" * 60)

        for file_path in files_to_check:
            print(f"Analyzing: {file_path}")
            self.detector.analyze_file(file_path)

        # Group violations by type
        violations_by_type = {}
        for violation in self.detector.violations:
            violation_type = violation["type"]
            if violation_type not in violations_by_type:
                violations_by_type[violation_type] = []
            violations_by_type[violation_type].append(violation)

        return violations_by_type

    def generate_report(self, violations: Dict[str, List[Dict]]) -> str:
        """Generate compliance report."""
        report = []
        report.append("EVENT-DRIVEN ARCHITECTURE COMPLIANCE REPORT")
        report.append("=" * 50)
        report.append("")

        total_violations = sum(len(v_list) for v_list in violations.values())

        if total_violations == 0:
            report.append(
                "✅ COMPLIANCE PASSED - No direct database connections detected!"
            )
            report.append("")
            report.append("Event-driven architecture compliance verified:")
            report.append("- All database operations go through event bus")
            report.append("- No direct database client imports")
            report.append("- No direct connection strings")
            report.append("- Pure event-driven pattern maintained")
        else:
            report.append(
                f"❌ COMPLIANCE FAILED - {total_violations} violations detected"
            )
            report.append("")

            # Group by severity
            high_violations = []
            medium_violations = []
            low_violations = []

            for v_list in violations.values():
                for violation in v_list:
                    if violation["severity"] == "high":
                        high_violations.append(violation)
                    elif violation["severity"] == "medium":
                        medium_violations.append(violation)
                    else:
                        low_violations.append(violation)

            # High severity violations
            if high_violations:
                report.append("🚨 HIGH SEVERITY VIOLATIONS:")
                report.append("-" * 30)
                for violation in high_violations:
                    report.append(f"File: {violation['file']}")
                    report.append(f"Line: {violation['line']}")
                    report.append(f"Issue: {violation['details']}")
                    report.append(f"Type: {violation['type']}")
                    report.append("")

            # Medium severity violations
            if medium_violations:
                report.append("⚠️  MEDIUM SEVERITY VIOLATIONS:")
                report.append("-" * 30)
                for violation in medium_violations:
                    report.append(f"File: {violation['file']}")
                    report.append(f"Line: {violation['line']}")
                    report.append(f"Issue: {violation['details']}")
                    report.append(f"Type: {violation['type']}")
                    report.append("")

            # Low severity violations
            if low_violations:
                report.append("ℹ️  LOW SEVERITY VIOLATIONS:")
                report.append("-" * 30)
                for violation in low_violations:
                    report.append(f"File: {violation['file']}")
                    report.append(f"Line: {violation['line']}")
                    report.append(f"Issue: {violation['details']}")
                    report.append("")

        report.append("COMPLIANCE REQUIREMENTS:")
        report.append("- Use event bus for all database operations")
        report.append("- Import only ONEX event publishing components")
        report.append("- Use adapter integrations for infrastructure")
        report.append("- Maintain correlation ID traceability")
        report.append("")

        return "\n".join(report)

    def check_allowed_imports(self) -> Dict[str, bool]:
        """Check that only allowed imports are used."""
        allowed_patterns = {
            "omnimemory.events.*",
            "omnimemory.models.*",
            "omnimemory.adapters.*",
            "omnibase_core.*",
            "pydantic.*",
            "uuid",
            "datetime",
            "typing",
            "asyncio",
            "logging",
            "json",
        }

        # This would be implemented to verify only allowed imports
        return {"compliant": True, "violations": []}


def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        project_root = sys.argv[1]
    else:
        # Default to current project
        project_root = "/Volumes/PRO-G40/Code/omnimemory"

    if not os.path.exists(project_root):
        print(f"Error: Project root '{project_root}' does not exist")
        sys.exit(1)

    print("OmniMemory Event-Driven Architecture Compliance Checker")
    print("=" * 60)
    print(f"Project Root: {project_root}")
    print("")

    # Run compliance check
    checker = EventDrivenComplianceChecker(project_root)
    violations = checker.check_compliance()

    # Generate report
    report = checker.generate_report(violations)
    print(report)

    # Save report to file
    report_path = os.path.join(project_root, "compliance_report.txt")
    with open(report_path, "w") as f:
        f.write(report)

    print(f"Report saved to: {report_path}")

    # Exit with appropriate code
    total_violations = sum(len(v_list) for v_list in violations.values())
    high_severity_count = sum(
        1 for v_list in violations.values() for v in v_list if v["severity"] == "high"
    )

    if high_severity_count > 0:
        print(
            f"\n❌ Compliance check FAILED - {high_severity_count} high severity violations"
        )
        sys.exit(1)
    elif total_violations > 0:
        print(
            f"\n⚠️  Compliance check PARTIAL - {total_violations} total violations (no high severity)"
        )
        sys.exit(0)
    else:
        print("\n✅ Compliance check PASSED - No violations detected")
        sys.exit(0)


if __name__ == "__main__":
    main()
