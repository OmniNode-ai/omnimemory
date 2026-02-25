# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Contract Linter CLI for ONEX Node Contract Validation.

Validates YAML contract files against the ONEX schema using the
ProtocolContractValidator stub. Provides structured error output
with field paths for easy debugging and CI/CD integration.

Usage:
    python -m omnimemory.tools.contract_linter path/to/contract.yaml
    python -m omnimemory.tools.contract_linter file1.yaml file2.yaml
    python -m omnimemory.tools.contract_linter path/to/contract.yaml --json
    python -m omnimemory.tools.contract_linter path/to/contract.yaml --verbose

Reference: omniintelligence PR #70 (contract validation)
Added for OMN-2218: Phase 7 CI Infrastructure Alignment
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from omnimemory.tools.stubs.contract_validator import ProtocolContractValidator

if TYPE_CHECKING:
    from collections.abc import Sequence

# Valid ONEX node types (case-insensitive lookup)
VALID_NODE_TYPES: frozenset[str] = frozenset(
    {
        "compute",
        "effect",
        "reducer",
        "orchestrator",
        "compute_generic",
        "effect_generic",
        "reducer_generic",
        "orchestrator_generic",
    }
)

# JSON output indentation
JSON_INDENT_SPACES = 2


def _detect_contract_type(data: dict[str, object]) -> str:
    """Detect the type of contract from YAML content.

    Returns:
        Contract type string: 'fsm_subcontract', 'workflow', 'node_contract',
        'subcontract', or 'unknown'
    """
    if "state_machine_name" in data or "states" in data:
        return "fsm_subcontract"

    if "workflow_type" in data or (
        "subcontract_name" in data and "max_concurrent_workflows" in data
    ):
        return "workflow"

    if "node_type" in data:
        return "node_contract"

    if "operations" in data:
        return "subcontract"

    return "unknown"


def validate_contract(file_path: str | Path) -> dict[str, object]:
    """Validate a single contract file.

    Args:
        file_path: Path to the YAML contract file.

    Returns:
        Dict with file_path, is_valid, errors, and contract_type.
    """
    path = Path(file_path)
    errors: list[str] = []

    if not path.exists():
        return {
            "file_path": str(path),
            "is_valid": False,
            "errors": [f"File not found: {path}"],
            "contract_type": None,
        }

    if path.is_dir():
        return {
            "file_path": str(path),
            "is_valid": False,
            "errors": [f"Path is a directory, not a file: {path}"],
            "contract_type": None,
        }

    try:
        content = path.read_text(encoding="utf-8")
    except OSError as e:
        return {
            "file_path": str(path),
            "is_valid": False,
            "errors": [f"Error reading file: {e}"],
            "contract_type": None,
        }

    if not content.strip():
        return {
            "file_path": str(path),
            "is_valid": False,
            "errors": ["File is empty"],
            "contract_type": None,
        }

    try:
        data = yaml.safe_load(content)
    except yaml.YAMLError as e:
        return {
            "file_path": str(path),
            "is_valid": False,
            "errors": [f"Invalid YAML syntax: {e}"],
            "contract_type": None,
        }

    if data is None:
        return {
            "file_path": str(path),
            "is_valid": False,
            "errors": ["File contains no YAML content (only comments or empty)"],
            "contract_type": None,
        }

    if not isinstance(data, dict):
        return {
            "file_path": str(path),
            "is_valid": False,
            "errors": [f"Contract must be a YAML mapping, got {type(data).__name__}"],
            "contract_type": None,
        }

    contract_type = _detect_contract_type(data)

    if contract_type == "node_contract":
        node_type = data.get("node_type")
        node_type_lower = node_type.lower() if isinstance(node_type, str) else None

        if node_type_lower and node_type_lower not in VALID_NODE_TYPES:
            valid_types = ", ".join(sorted(VALID_NODE_TYPES))
            errors.append(
                f"Invalid node_type: '{node_type}'. Must be one of: {valid_types}"
            )
        else:
            validator = ProtocolContractValidator()
            result = validator.validate_contract_file(
                path,
                contract_type=node_type_lower,
            )
            errors.extend(result.violations)
            return {
                "file_path": str(path),
                "is_valid": result.is_valid,
                "errors": errors,
                "contract_type": contract_type,
            }
    elif contract_type == "unknown":
        errors.append(
            "Unable to detect contract type. Expected one of: "
            "node contract (with node_type), FSM subcontract "
            "(with state_machine_name/states), workflow (with workflow_type), "
            "or subcontract (with operations)"
        )

    return {
        "file_path": str(path),
        "is_valid": len(errors) == 0,
        "errors": errors,
        "contract_type": contract_type,
    }


def validate_batch(
    file_paths: Sequence[str | Path],
) -> list[dict[str, object]]:
    """Validate multiple contract files.

    Args:
        file_paths: Sequence of paths to contract files.

    Returns:
        List of validation results.
    """
    return [validate_contract(fp) for fp in file_paths]


def _format_text_output(
    results: list[dict[str, object]],
    verbose: bool = False,
) -> str:
    """Format validation results as human-readable text."""
    lines: list[str] = []

    for result in results:
        is_valid = result.get("is_valid", False)
        status = "PASS" if is_valid else "FAIL"
        file_path = result.get("file_path", "unknown")

        if is_valid:
            lines.append(f"[{status}] {file_path}")
        elif verbose:
            lines.append(f"[{status}] {file_path}")
            errors_raw = result.get("errors", [])
            errors_list = list(errors_raw) if isinstance(errors_raw, list) else []
            for error in errors_list:
                lines.append(f"  - {error}")
        else:
            errors_raw = result.get("errors", [])
            errors_list = list(errors_raw) if isinstance(errors_raw, list) else []
            error_count = len(errors_list)
            lines.append(f"[{status}] {file_path} ({error_count} error(s))")

    total = len(results)
    valid = sum(1 for r in results if r.get("is_valid", False))
    lines.append("")
    lines.append(f"Summary: {valid}/{total} contracts passed")

    return "\n".join(lines)


def _format_json_output(
    results: list[dict[str, object]],
) -> str:
    """Format validation results as JSON."""
    return json.dumps(
        {
            "results": results,
            "summary": {
                "total_count": len(results),
                "valid_count": sum(1 for r in results if r.get("is_valid", False)),
                "invalid_count": sum(
                    1 for r in results if not r.get("is_valid", False)
                ),
            },
        },
        indent=JSON_INDENT_SPACES,
    )


def main(args: list[str] | None = None) -> int:
    """CLI entry point for contract linter.

    Returns:
        Exit code:
            0 - All contracts passed validation
            1 - One or more contracts failed validation
            2 - Input/usage error
    """
    parser = argparse.ArgumentParser(
        description="Validate ONEX contract YAML files",
        prog="contract_linter",
    )

    parser.add_argument(
        "contracts",
        nargs="*",
        help="Path(s) to contract YAML file(s)",
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed error messages",
    )

    parsed_args = parser.parse_args(args)

    if not parsed_args.contracts:
        parser.print_help()
        return 2

    results = validate_batch(parsed_args.contracts)

    if parsed_args.json:
        output = _format_json_output(results)
    else:
        output = _format_text_output(results, verbose=parsed_args.verbose)

    print(output)

    has_errors = any(not r.get("is_valid", False) for r in results)
    if has_errors:
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
