# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""I/O Audit CLI for ONEX Node Purity Enforcement.

Detects forbidden I/O patterns in ONEX compute nodes through AST-based static
analysis. Enforces the "pure compute / no I/O" architectural invariant.

Usage:
    python -m omnimemory.audit
    python -m omnimemory.audit src/omnimemory/nodes
    python -m omnimemory.audit --whitelist custom_whitelist.yaml
    python -m omnimemory.audit --verbose
    python -m omnimemory.audit --json

Exit Codes:
    0 - No I/O violations found
    1 - Violations found
    2 - Error
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

from omnimemory.audit.io_audit import (
    DEFAULT_WHITELIST_PATH,
    IO_AUDIT_TARGETS,
    REMEDIATION_HINTS,
    EnumIOAuditRule,
    ModelAuditResult,
    ModelIOAuditViolation,
    run_audit,
)

JSON_INDENT_SPACES = 2


def _format_text_output(
    result: ModelAuditResult,
    verbose: bool = False,
) -> str:
    """Format audit result as human-readable text."""
    lines: list[str] = []

    if result.is_clean:
        lines.append(f"No I/O violations found. ({result.files_scanned} files scanned)")
    else:
        violations_by_file: dict[Path, list[ModelIOAuditViolation]] = {}
        for v in result.violations:
            if v.file not in violations_by_file:
                violations_by_file[v.file] = []
            violations_by_file[v.file].append(v)

        for file_path, file_violations in sorted(violations_by_file.items()):
            lines.append(f"{file_path}:")

            rules_in_file: set[EnumIOAuditRule] = set()

            for v in sorted(file_violations, key=lambda x: x.line):
                lines.append(f"  Line {v.line}: [{v.rule.value}] {v.message}")
                rules_in_file.add(v.rule)

            hints = [
                REMEDIATION_HINTS[rule]
                for rule in sorted(rules_in_file, key=lambda r: r.value)
                if rule in REMEDIATION_HINTS
            ]
            if hints:
                lines.append(f"  -> Hints: {'; '.join(hints)}")

        file_count = len(violations_by_file)
        lines.append("")
        lines.append(
            f"Summary: {len(result.violations)} violation(s) in {file_count} file(s) "
            f"({result.files_scanned} files scanned)"
        )

    if verbose and not result.is_clean:
        lines.append("")
        lines.append("Use --whitelist to specify allowed exceptions.")

    return "\n".join(lines)


def _format_json_output(result: ModelAuditResult) -> str:
    """Format audit result as JSON."""
    output: dict[str, object] = {
        "violations": [
            {
                "file": str(v.file),
                "line": v.line,
                "column": v.column,
                "rule": v.rule.value,
                "message": v.message,
            }
            for v in result.violations
        ],
        "files_scanned": result.files_scanned,
        "is_clean": result.is_clean,
    }
    return json.dumps(output, indent=JSON_INDENT_SPACES)


def main(args: list[str] | None = None) -> int:
    """CLI entry point for I/O audit."""
    parser = argparse.ArgumentParser(
        description="ONEX node I/O audit - detect forbidden I/O patterns in compute nodes",
        prog="python -m omnimemory.audit",
    )

    parser.add_argument(
        "targets",
        nargs="*",
        default=None,
        help=f"Directories to scan (default: {', '.join(IO_AUDIT_TARGETS)})",
    )

    parser.add_argument(
        "--whitelist",
        "-w",
        type=Path,
        default=Path(DEFAULT_WHITELIST_PATH),
        metavar="PATH",
        help=f"Path to whitelist YAML file (default: {DEFAULT_WHITELIST_PATH})",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Output in JSON format for CI integration",
    )

    parsed_args = parser.parse_args(args)

    try:
        targets = parsed_args.targets if parsed_args.targets else IO_AUDIT_TARGETS

        result = run_audit(
            targets=targets,
            whitelist_path=parsed_args.whitelist,
        )

        if parsed_args.json:
            output = _format_json_output(result)
        else:
            output = _format_text_output(result, verbose=parsed_args.verbose)

        print(output)

        return 0 if result.is_clean else 1

    except FileNotFoundError as e:
        error_msg = f"Error: {e}"
        if parsed_args.json:
            print(json.dumps({"error": error_msg}, indent=JSON_INDENT_SPACES))
        else:
            print(error_msg, file=sys.stderr)
        return 2

    except Exception as e:
        error_msg = f"Unexpected error: {e}"
        if parsed_args.json:
            print(json.dumps({"error": error_msg}, indent=JSON_INDENT_SPACES))
        else:
            print(error_msg, file=sys.stderr)
            if parsed_args.verbose:
                traceback.print_exc(file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
