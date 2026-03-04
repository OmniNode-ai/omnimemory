# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Stub implementation of ProtocolContractValidator.

that validates ONEX node contract YAML files against basic structural rules.

This is a temporary implementation until the actual omnibase_core module
provides the ProtocolContractValidator class.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class ProtocolContractValidatorResult:
    """Result of contract validation.

    Attributes:
        is_valid: Whether the contract passed all validation checks.
        violations: List of violation messages describing validation failures.
    """

    is_valid: bool = True
    violations: list[str] = field(default_factory=list)


class ProtocolContractValidator:
    """Stub implementation of ProtocolContractValidator for ONEX node contracts.

    This validator performs basic structural validation of YAML contract files.
    It checks for required fields based on the contract_type (compute, effect,
    reducer, orchestrator).
    """

    COMMON_REQUIRED_FIELDS: tuple[str, ...] = (
        "name",
        "node_type",
        "description",
        "input_model",
        "output_model",
    )

    VERSION_FIELDS: tuple[str, ...] = (
        "contract_version",
        "node_version",
    )

    NODE_CONTRACT_REQUIRED_VERSION_FIELDS: tuple[str, ...] = (
        "contract_version",
        "node_version",
    )

    NODE_TYPE_REQUIRED_FIELDS: dict[str, tuple[str, ...]] = {
        "compute": (),
        "effect": (),
        "reducer": (),
        "orchestrator": (),
        "compute_generic": (),
        "effect_generic": (),
        "reducer_generic": (),
        "orchestrator_generic": (),
    }

    def __init__(self) -> None:
        """Initialize the contract validator."""

    def validate_contract_file(
        self,
        path: Path | str,
        *,
        contract_type: str | None = None,
    ) -> ProtocolContractValidatorResult:
        """Validate a contract YAML file.

        Args:
            path: Path to the YAML contract file.
            contract_type: Expected contract type (compute, effect, reducer, orchestrator).

        Returns:
            ProtocolContractValidatorResult with validation status and any violations.
        """
        path = Path(path) if isinstance(path, str) else path
        violations: list[str] = []

        try:
            content = path.read_text(encoding="utf-8")
            data = yaml.safe_load(content)
        except FileNotFoundError:
            return ProtocolContractValidatorResult(
                is_valid=False,
                violations=[f"file: Contract file not found: {path}"],
            )
        except yaml.YAMLError as e:
            return ProtocolContractValidatorResult(
                is_valid=False,
                violations=[f"yaml: Invalid YAML syntax: {e}"],
            )
        except OSError as e:
            return ProtocolContractValidatorResult(
                is_valid=False,
                violations=[f"file: Error reading contract file: {e}"],
            )

        if data is None:
            return ProtocolContractValidatorResult(
                is_valid=False,
                violations=["file: Contract file is empty or contains only comments"],
            )

        if not isinstance(data, dict):
            return ProtocolContractValidatorResult(
                is_valid=False,
                violations=[
                    f"structure: Contract must be a YAML mapping, got {type(data).__name__}"
                ],
            )

        violations.extend(
            self._validate_required_fields(data, self.COMMON_REQUIRED_FIELDS)
        )

        node_type = data.get("node_type")
        is_node_contract = node_type is not None and isinstance(node_type, str)

        violations.extend(
            self._validate_version_fields(data, is_node_contract=is_node_contract)
        )

        if (
            contract_type
            and is_node_contract
            and isinstance(node_type, str)
            and node_type.lower() != contract_type.lower()
        ):
            violations.append(
                f"node_type: Expected '{contract_type}', got '{node_type}'"
            )

        effective_type = contract_type or (
            node_type.lower()
            if is_node_contract and isinstance(node_type, str)
            else None
        )
        if effective_type and effective_type in self.NODE_TYPE_REQUIRED_FIELDS:
            violations.extend(
                self._validate_required_fields(
                    data, self.NODE_TYPE_REQUIRED_FIELDS[effective_type]
                )
            )

        if "name" in data:
            violations.extend(self._validate_name(data["name"]))

        return ProtocolContractValidatorResult(
            is_valid=len(violations) == 0,
            violations=violations,
        )

    def _validate_required_fields(
        self,
        data: dict[str, object],
        required_fields: tuple[str, ...],
    ) -> list[str]:
        """Check for missing required fields."""
        violations: list[str] = []
        for field_name in required_fields:
            if field_name not in data:
                violations.append(f"{field_name}: Missing required field")
            elif data[field_name] is None:
                violations.append(f"{field_name}: Field cannot be null")
        return violations

    def _validate_version_fields(
        self, data: dict[str, object], is_node_contract: bool = True
    ) -> list[str]:
        """Validate version fields are present and valid."""
        violations: list[str] = []

        version_fields_present = [f for f in self.VERSION_FIELDS if f in data]

        if is_node_contract:
            for required_field in self.NODE_CONTRACT_REQUIRED_VERSION_FIELDS:
                if required_field not in data:
                    violations.append(
                        f"{required_field}: Missing required field for node contract."
                    )
            for f in version_fields_present:
                violations.extend(self._validate_version_structure(data[f], f))
        elif not version_fields_present:
            violations.append(
                f"version: Missing required version field. "
                f"Expected one of: {', '.join(self.VERSION_FIELDS)}"
            )
        else:
            for f in version_fields_present:
                violations.extend(self._validate_version_structure(data[f], f))

        return violations

    def _validate_version_structure(
        self, version: object, field_name: str = "version"
    ) -> list[str]:
        """Validate version field structure."""
        violations: list[str] = []

        if version is None:
            violations.append(f"{field_name}: Version field cannot be null")
            return violations

        if isinstance(version, str):
            violations.append(
                f"{field_name}: Expected object with major/minor/patch fields, got string '{version}'."
            )
        elif isinstance(version, dict):
            for component in ("major", "minor", "patch"):
                if component not in version:
                    violations.append(
                        f"{field_name}.{component}: Missing required field."
                    )
                else:
                    value = version[component]
                    if value is None:
                        violations.append(
                            f"{field_name}.{component}: Version component cannot be null"
                        )
                    elif not isinstance(value, int):
                        violations.append(
                            f"{field_name}.{component}: Expected non-negative integer, "
                            f"got {type(value).__name__}"
                        )
                    elif value < 0:
                        violations.append(
                            f"{field_name}.{component}: Version component must be non-negative"
                        )
        else:
            violations.append(
                f"{field_name}: Expected object with major/minor/patch, "
                f"got {type(version).__name__}"
            )

        return violations

    def _validate_name(self, name: object) -> list[str]:
        """Validate name field."""
        violations: list[str] = []
        if not isinstance(name, str):
            violations.append(f"name: Expected string, got {type(name).__name__}")
        elif not name.strip():
            violations.append("name: Cannot be empty or whitespace only")
        return violations
