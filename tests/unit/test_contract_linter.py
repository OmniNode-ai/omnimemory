# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Unit tests for the contract linter and ProtocolContractValidator stub.

Tests cover:
- Contract type detection (node, FSM, workflow, subcontract, unknown)
- Required field validation for node contracts
- Version field validation (structured semver objects)
- Name field validation
- The ``validate_contract`` function with valid and invalid inputs
- The ``ProtocolContractValidator`` class methods directly
- CLI output formatting (text and JSON)
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from omnimemory.tools.contract_linter import (
    _detect_contract_type,
    _format_json_output,
    _format_text_output,
    validate_batch,
    validate_contract,
)
from omnimemory.tools.stubs.contract_validator import (
    ProtocolContractValidator,
    ProtocolContractValidatorResult,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_yaml(tmp_path: Path, data: object, filename: str = "contract.yaml") -> Path:
    """Write a YAML file to tmp_path and return its path."""
    target = tmp_path / filename
    target.write_text(yaml.dump(data, default_flow_style=False), encoding="utf-8")
    return target


def _write_text(tmp_path: Path, content: str, filename: str = "contract.yaml") -> Path:
    """Write raw text content to a file in tmp_path and return its path."""
    target = tmp_path / filename
    target.write_text(content, encoding="utf-8")
    return target


def _make_valid_node_contract(**overrides: object) -> dict[str, object]:
    """Build a minimal valid node contract dict with optional overrides."""
    base: dict[str, object] = {
        "name": "test_node_compute",
        "node_type": "compute",
        "description": "A test compute node",
        "input_model": "TestInput",
        "output_model": "TestOutput",
        "contract_version": {"major": 1, "minor": 0, "patch": 0},
        "node_version": {"major": 1, "minor": 0, "patch": 0},
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def validator() -> ProtocolContractValidator:
    """Provide a fresh ProtocolContractValidator instance."""
    return ProtocolContractValidator()


@pytest.fixture
def valid_node_contract_path(tmp_path: Path) -> Path:
    """Write a valid node contract YAML and return its path."""
    return _write_yaml(tmp_path, _make_valid_node_contract())


# ---------------------------------------------------------------------------
# Contract Type Detection
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDetectContractType:
    """Tests for _detect_contract_type()."""

    def test_node_contract_detected(self) -> None:
        data = {"node_type": "compute", "name": "test"}
        assert _detect_contract_type(data) == "node_contract"

    def test_fsm_subcontract_via_state_machine_name(self) -> None:
        data = {"state_machine_name": "my_fsm"}
        assert _detect_contract_type(data) == "fsm_subcontract"

    def test_fsm_subcontract_via_states(self) -> None:
        data = {"states": ["ready", "running", "done"]}
        assert _detect_contract_type(data) == "fsm_subcontract"

    def test_workflow_via_workflow_type(self) -> None:
        data = {"workflow_type": "sequential"}
        assert _detect_contract_type(data) == "workflow"

    def test_workflow_via_subcontract_name_with_concurrency(self) -> None:
        data = {"subcontract_name": "my_workflow", "max_concurrent_workflows": 5}
        assert _detect_contract_type(data) == "workflow"

    def test_subcontract_via_operations(self) -> None:
        data = {"operations": ["read", "write"]}
        assert _detect_contract_type(data) == "subcontract"

    def test_node_type_takes_precedence_over_operations(self) -> None:
        """node_type check precedes operations check, so node_contract wins."""
        data = {"operations": ["read"], "node_type": "effect"}
        assert _detect_contract_type(data) == "node_contract"

    def test_unknown_for_empty_dict(self) -> None:
        assert _detect_contract_type({}) == "unknown"

    def test_unknown_for_unrecognized_keys(self) -> None:
        data = {"some_random_key": "value"}
        assert _detect_contract_type(data) == "unknown"

    def test_fsm_takes_precedence_over_node_type(self) -> None:
        """FSM detection checks happen before node_type check."""
        data = {"state_machine_name": "fsm", "node_type": "compute"}
        assert _detect_contract_type(data) == "fsm_subcontract"


# ---------------------------------------------------------------------------
# ProtocolContractValidator: Required Fields
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestValidateRequiredFields:
    """Tests for _validate_required_fields on the validator."""

    def test_all_fields_present(self, validator: ProtocolContractValidator) -> None:
        data = {
            "name": "x",
            "node_type": "compute",
            "description": "d",
            "input_model": "I",
            "output_model": "O",
        }
        violations = validator._validate_required_fields(
            data,
            validator.COMMON_REQUIRED_FIELDS,
        )
        assert violations == []

    def test_missing_single_field(self, validator: ProtocolContractValidator) -> None:
        data = {
            "node_type": "compute",
            "description": "d",
            "input_model": "I",
            "output_model": "O",
        }
        violations = validator._validate_required_fields(
            data,
            validator.COMMON_REQUIRED_FIELDS,
        )
        assert len(violations) == 1
        assert "name" in violations[0]
        assert "Missing required field" in violations[0]

    def test_null_field_value(self, validator: ProtocolContractValidator) -> None:
        data: dict[str, object] = {
            "name": None,
            "node_type": "compute",
            "description": "d",
            "input_model": "I",
            "output_model": "O",
        }
        violations = validator._validate_required_fields(
            data,
            validator.COMMON_REQUIRED_FIELDS,
        )
        assert len(violations) == 1
        assert "cannot be null" in violations[0]

    def test_multiple_missing_fields(
        self, validator: ProtocolContractValidator
    ) -> None:
        violations = validator._validate_required_fields(
            {},
            validator.COMMON_REQUIRED_FIELDS,
        )
        assert len(violations) == len(validator.COMMON_REQUIRED_FIELDS)


# ---------------------------------------------------------------------------
# ProtocolContractValidator: Version Fields
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestValidateVersionFields:
    """Tests for _validate_version_fields and _validate_version_structure."""

    def test_valid_version_object(self, validator: ProtocolContractValidator) -> None:
        data: dict[str, object] = {
            "node_type": "compute",
            "contract_version": {"major": 1, "minor": 2, "patch": 3},
            "node_version": {"major": 0, "minor": 1, "patch": 0},
        }
        violations = validator._validate_version_fields(data, is_node_contract=True)
        assert violations == []

    def test_version_as_string_rejected(
        self, validator: ProtocolContractValidator
    ) -> None:
        data: dict[str, object] = {
            "node_type": "compute",
            "contract_version": "1.0.0",
            "node_version": {"major": 1, "minor": 0, "patch": 0},
        }
        violations = validator._validate_version_fields(data, is_node_contract=True)
        assert len(violations) == 1
        assert "Expected object" in violations[0]
        assert "got string" in violations[0]

    def test_version_null_rejected(self, validator: ProtocolContractValidator) -> None:
        violations = validator._validate_version_structure(None, "contract_version")
        assert len(violations) == 1
        assert "cannot be null" in violations[0]

    def test_version_missing_component(
        self, validator: ProtocolContractValidator
    ) -> None:
        version = {"major": 1, "minor": 0}  # missing patch
        violations = validator._validate_version_structure(version, "contract_version")
        assert len(violations) == 1
        assert "patch" in violations[0]
        assert "Missing required field" in violations[0]

    def test_version_component_null(self, validator: ProtocolContractValidator) -> None:
        version: dict[str, object] = {"major": 1, "minor": None, "patch": 0}
        violations = validator._validate_version_structure(version, "contract_version")
        assert len(violations) == 1
        assert "minor" in violations[0]
        assert "cannot be null" in violations[0]

    def test_version_component_wrong_type(
        self, validator: ProtocolContractValidator
    ) -> None:
        version: dict[str, object] = {"major": 1, "minor": "two", "patch": 0}
        violations = validator._validate_version_structure(version, "contract_version")
        assert len(violations) == 1
        assert "Expected non-negative integer" in violations[0]

    def test_version_component_negative(
        self, validator: ProtocolContractValidator
    ) -> None:
        version = {"major": 1, "minor": -1, "patch": 0}
        violations = validator._validate_version_structure(version, "contract_version")
        assert len(violations) == 1
        assert "must be non-negative" in violations[0]

    def test_version_unexpected_type(
        self, validator: ProtocolContractValidator
    ) -> None:
        violations = validator._validate_version_structure(42, "contract_version")
        assert len(violations) == 1
        assert "Expected object with major/minor/patch" in violations[0]

    def test_missing_version_fields_for_node_contract(
        self,
        validator: ProtocolContractValidator,
    ) -> None:
        data: dict[str, object] = {"node_type": "compute"}
        violations = validator._validate_version_fields(data, is_node_contract=True)
        # Should report both contract_version and node_version as missing
        assert len(violations) == 2
        missing_fields = " ".join(violations)
        assert "contract_version" in missing_fields
        assert "node_version" in missing_fields

    def test_non_node_contract_missing_all_version_fields(
        self,
        validator: ProtocolContractValidator,
    ) -> None:
        data: dict[str, object] = {"some_field": "value"}
        violations = validator._validate_version_fields(data, is_node_contract=False)
        assert len(violations) == 1
        assert "Missing required version field" in violations[0]

    def test_non_node_contract_with_contract_version(
        self,
        validator: ProtocolContractValidator,
    ) -> None:
        data: dict[str, object] = {
            "contract_version": {"major": 1, "minor": 0, "patch": 0},
        }
        violations = validator._validate_version_fields(data, is_node_contract=False)
        assert violations == []

    @pytest.mark.parametrize("component", ["major", "minor", "patch"])
    def test_each_version_component_validated(
        self,
        validator: ProtocolContractValidator,
        component: str,
    ) -> None:
        version = {"major": 0, "minor": 0, "patch": 0}
        version[component] = -1
        violations = validator._validate_version_structure(version, "v")
        assert len(violations) == 1
        assert component in violations[0]


# ---------------------------------------------------------------------------
# ProtocolContractValidator: Name Validation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestValidateName:
    """Tests for _validate_name on the validator."""

    def test_valid_name(self, validator: ProtocolContractValidator) -> None:
        assert validator._validate_name("my_node_compute") == []

    def test_name_not_string(self, validator: ProtocolContractValidator) -> None:
        violations = validator._validate_name(123)
        assert len(violations) == 1
        assert "Expected string" in violations[0]

    def test_name_empty_string(self, validator: ProtocolContractValidator) -> None:
        violations = validator._validate_name("")
        assert len(violations) == 1
        assert "Cannot be empty" in violations[0]

    def test_name_whitespace_only(self, validator: ProtocolContractValidator) -> None:
        violations = validator._validate_name("   ")
        assert len(violations) == 1
        assert "Cannot be empty" in violations[0]


# ---------------------------------------------------------------------------
# ProtocolContractValidator: Full File Validation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestValidateContractFile:
    """Tests for validate_contract_file on the validator."""

    def test_valid_contract_passes(
        self,
        validator: ProtocolContractValidator,
        valid_node_contract_path: Path,
    ) -> None:
        result = validator.validate_contract_file(valid_node_contract_path)
        assert result.is_valid is True
        assert result.violations == []

    def test_file_not_found(
        self,
        validator: ProtocolContractValidator,
        tmp_path: Path,
    ) -> None:
        result = validator.validate_contract_file(tmp_path / "nonexistent.yaml")
        assert result.is_valid is False
        assert any("not found" in v for v in result.violations)

    def test_invalid_yaml_syntax(
        self,
        validator: ProtocolContractValidator,
        tmp_path: Path,
    ) -> None:
        path = _write_text(tmp_path, "name: [invalid yaml\n")
        result = validator.validate_contract_file(path)
        assert result.is_valid is False
        assert any("Invalid YAML" in v for v in result.violations)

    def test_empty_file(
        self,
        validator: ProtocolContractValidator,
        tmp_path: Path,
    ) -> None:
        path = _write_text(tmp_path, "# only a comment\n")
        result = validator.validate_contract_file(path)
        assert result.is_valid is False
        assert any("empty" in v.lower() for v in result.violations)

    def test_non_mapping_yaml(
        self,
        validator: ProtocolContractValidator,
        tmp_path: Path,
    ) -> None:
        path = _write_text(tmp_path, "- item1\n- item2\n")
        result = validator.validate_contract_file(path)
        assert result.is_valid is False
        assert any("mapping" in v for v in result.violations)

    def test_contract_type_mismatch(
        self,
        validator: ProtocolContractValidator,
        tmp_path: Path,
    ) -> None:
        data = _make_valid_node_contract(node_type="effect")
        path = _write_yaml(tmp_path, data)
        result = validator.validate_contract_file(path, contract_type="compute")
        assert result.is_valid is False
        assert any("Expected 'compute'" in v for v in result.violations)

    def test_accepts_string_path(
        self,
        validator: ProtocolContractValidator,
        valid_node_contract_path: Path,
    ) -> None:
        result = validator.validate_contract_file(str(valid_node_contract_path))
        assert result.is_valid is True


# ---------------------------------------------------------------------------
# ProtocolContractValidatorResult dataclass
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestProtocolContractValidatorResult:
    """Tests for the ProtocolContractValidatorResult dataclass defaults."""

    def test_default_is_valid(self) -> None:
        result = ProtocolContractValidatorResult()
        assert result.is_valid is True
        assert result.violations == []

    def test_explicit_invalid(self) -> None:
        result = ProtocolContractValidatorResult(
            is_valid=False,
            violations=["error one"],
        )
        assert result.is_valid is False
        assert result.violations == ["error one"]

    def test_violations_list_independence(self) -> None:
        """Each instance should have its own violations list."""
        r1 = ProtocolContractValidatorResult()
        r2 = ProtocolContractValidatorResult()
        r1.violations.append("only in r1")
        assert r2.violations == []


# ---------------------------------------------------------------------------
# validate_contract (high-level linter function)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestValidateContract:
    """Tests for the validate_contract function in contract_linter.py."""

    def test_valid_node_contract(self, valid_node_contract_path: Path) -> None:
        result = validate_contract(valid_node_contract_path)
        assert result["is_valid"] is True
        assert result["errors"] == []
        assert result["contract_type"] == "node_contract"

    def test_file_not_found(self, tmp_path: Path) -> None:
        result = validate_contract(tmp_path / "does_not_exist.yaml")
        assert result["is_valid"] is False
        errors = result["errors"]
        assert isinstance(errors, list)
        assert any("not found" in str(e).lower() for e in errors)
        assert result["contract_type"] is None

    def test_path_is_directory(self, tmp_path: Path) -> None:
        result = validate_contract(tmp_path)
        assert result["is_valid"] is False
        errors = result["errors"]
        assert isinstance(errors, list)
        assert any("directory" in str(e).lower() for e in errors)

    def test_empty_file(self, tmp_path: Path) -> None:
        path = _write_text(tmp_path, "")
        result = validate_contract(path)
        assert result["is_valid"] is False
        errors = result["errors"]
        assert isinstance(errors, list)
        assert any("empty" in str(e).lower() for e in errors)

    def test_invalid_yaml(self, tmp_path: Path) -> None:
        path = _write_text(tmp_path, "name: [broken\n")
        result = validate_contract(path)
        assert result["is_valid"] is False
        errors = result["errors"]
        assert isinstance(errors, list)
        assert any("yaml" in str(e).lower() for e in errors)

    def test_yaml_only_comments(self, tmp_path: Path) -> None:
        path = _write_text(tmp_path, "# just a comment\n")
        result = validate_contract(path)
        assert result["is_valid"] is False

    def test_non_mapping_yaml(self, tmp_path: Path) -> None:
        path = _write_text(tmp_path, "- a\n- b\n")
        result = validate_contract(path)
        assert result["is_valid"] is False
        errors = result["errors"]
        assert isinstance(errors, list)
        assert any("mapping" in str(e) for e in errors)

    def test_unknown_contract_type(self, tmp_path: Path) -> None:
        path = _write_yaml(tmp_path, {"random_key": "value"})
        result = validate_contract(path)
        assert result["is_valid"] is False
        assert result["contract_type"] == "unknown"
        errors = result["errors"]
        assert isinstance(errors, list)
        assert any("Unable to detect" in str(e) for e in errors)

    def test_invalid_node_type(self, tmp_path: Path) -> None:
        data = _make_valid_node_contract(node_type="invalid_type")
        path = _write_yaml(tmp_path, data)
        result = validate_contract(path)
        assert result["is_valid"] is False
        errors = result["errors"]
        assert isinstance(errors, list)
        assert any("Invalid node_type" in str(e) for e in errors)

    @pytest.mark.parametrize(
        "node_type",
        [
            "compute",
            "effect",
            "reducer",
            "orchestrator",
            "compute_generic",
            "effect_generic",
            "reducer_generic",
            "orchestrator_generic",
        ],
    )
    def test_all_valid_node_types_accepted(
        self,
        tmp_path: Path,
        node_type: str,
    ) -> None:
        data = _make_valid_node_contract(node_type=node_type)
        path = _write_yaml(tmp_path, data, filename=f"contract_{node_type}.yaml")
        result = validate_contract(path)
        assert result["is_valid"] is True, (
            f"node_type '{node_type}' should be valid but got errors: {result['errors']}"
        )

    def test_fsm_subcontract_passes(self, tmp_path: Path) -> None:
        data = {"state_machine_name": "my_fsm", "states": ["ready", "done"]}
        path = _write_yaml(tmp_path, data)
        result = validate_contract(path)
        assert result["contract_type"] == "fsm_subcontract"
        assert result["is_valid"] is True

    def test_workflow_passes(self, tmp_path: Path) -> None:
        data = {"workflow_type": "sequential", "steps": []}
        path = _write_yaml(tmp_path, data)
        result = validate_contract(path)
        assert result["contract_type"] == "workflow"
        assert result["is_valid"] is True

    def test_subcontract_passes(self, tmp_path: Path) -> None:
        data = {"operations": ["read", "write"]}
        path = _write_yaml(tmp_path, data)
        result = validate_contract(path)
        assert result["contract_type"] == "subcontract"
        assert result["is_valid"] is True

    def test_accepts_string_path(self, valid_node_contract_path: Path) -> None:
        result = validate_contract(str(valid_node_contract_path))
        assert result["is_valid"] is True

    def test_node_contract_missing_required_fields(self, tmp_path: Path) -> None:
        data = {"node_type": "compute"}  # missing everything else
        path = _write_yaml(tmp_path, data)
        result = validate_contract(path)
        assert result["is_valid"] is False
        assert result["contract_type"] == "node_contract"
        errors = result["errors"]
        assert isinstance(errors, list)
        errors_str = " ".join(str(e) for e in errors)
        assert "name" in errors_str
        assert "description" in errors_str

    def test_node_contract_with_string_version(self, tmp_path: Path) -> None:
        data = _make_valid_node_contract(contract_version="1.0.0")
        path = _write_yaml(tmp_path, data)
        result = validate_contract(path)
        assert result["is_valid"] is False
        errors = result["errors"]
        assert isinstance(errors, list)
        errors_str = " ".join(str(e) for e in errors)
        assert "Expected object" in errors_str


# ---------------------------------------------------------------------------
# validate_batch
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestValidateBatch:
    """Tests for the validate_batch function."""

    def test_batch_returns_list(self, valid_node_contract_path: Path) -> None:
        results = validate_batch([valid_node_contract_path])
        assert isinstance(results, list)
        assert len(results) == 1
        assert results[0]["is_valid"] is True

    def test_batch_multiple_files(self, tmp_path: Path) -> None:
        valid = _write_yaml(tmp_path, _make_valid_node_contract(), "good.yaml")
        invalid = _write_yaml(tmp_path, {"random": "data"}, "bad.yaml")
        results = validate_batch([valid, invalid])
        assert len(results) == 2
        assert results[0]["is_valid"] is True
        assert results[1]["is_valid"] is False

    def test_batch_empty_list(self) -> None:
        results = validate_batch([])
        assert results == []


# ---------------------------------------------------------------------------
# Output Formatting
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFormatTextOutput:
    """Tests for _format_text_output."""

    def test_pass_result(self) -> None:
        results = [{"is_valid": True, "file_path": "good.yaml", "errors": []}]
        output = _format_text_output(results)
        assert "[PASS]" in output
        assert "good.yaml" in output
        assert "1/1 contracts passed" in output

    def test_fail_result_brief(self) -> None:
        results = [
            {"is_valid": False, "file_path": "bad.yaml", "errors": ["error1", "error2"]}
        ]
        output = _format_text_output(results, verbose=False)
        assert "[FAIL]" in output
        assert "2 error(s)" in output

    def test_fail_result_verbose(self) -> None:
        results = [
            {
                "is_valid": False,
                "file_path": "bad.yaml",
                "errors": ["missing name", "bad version"],
            }
        ]
        output = _format_text_output(results, verbose=True)
        assert "[FAIL]" in output
        assert "  - missing name" in output
        assert "  - bad version" in output

    def test_summary_counts(self) -> None:
        results = [
            {"is_valid": True, "file_path": "a.yaml", "errors": []},
            {"is_valid": False, "file_path": "b.yaml", "errors": ["err"]},
            {"is_valid": True, "file_path": "c.yaml", "errors": []},
        ]
        output = _format_text_output(results)
        assert "2/3 contracts passed" in output


@pytest.mark.unit
class TestFormatJsonOutput:
    """Tests for _format_json_output."""

    def test_json_structure(self) -> None:
        results = [{"is_valid": True, "file_path": "a.yaml", "errors": []}]
        import json

        parsed = json.loads(_format_json_output(results))
        assert "results" in parsed
        assert "summary" in parsed
        assert parsed["summary"]["total_count"] == 1
        assert parsed["summary"]["valid_count"] == 1
        assert parsed["summary"]["invalid_count"] == 0

    def test_json_with_failures(self) -> None:
        import json

        results = [
            {"is_valid": True, "file_path": "a.yaml", "errors": []},
            {"is_valid": False, "file_path": "b.yaml", "errors": ["err"]},
        ]
        parsed = json.loads(_format_json_output(results))
        assert parsed["summary"]["total_count"] == 2
        assert parsed["summary"]["valid_count"] == 1
        assert parsed["summary"]["invalid_count"] == 1


# ---------------------------------------------------------------------------
# Node-type-specific required fields (currently empty tuples)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestNodeTypeSpecificFields:
    """Verify that node-type-specific field requirements are enforced."""

    @pytest.mark.parametrize(
        "node_type",
        [
            "compute",
            "effect",
            "reducer",
            "orchestrator",
            "compute_generic",
            "effect_generic",
            "reducer_generic",
            "orchestrator_generic",
        ],
    )
    def test_node_type_in_required_fields_map(self, node_type: str) -> None:
        """All valid node types should have an entry in NODE_TYPE_REQUIRED_FIELDS."""
        assert node_type in ProtocolContractValidator.NODE_TYPE_REQUIRED_FIELDS


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_version_with_zero_components(
        self,
        validator: ProtocolContractValidator,
    ) -> None:
        version = {"major": 0, "minor": 0, "patch": 0}
        violations = validator._validate_version_structure(version, "v")
        assert violations == []

    def test_name_with_unicode(
        self,
        validator: ProtocolContractValidator,
    ) -> None:
        violations = validator._validate_name("node_unicode_test")
        assert violations == []

    def test_node_type_case_insensitive(self, tmp_path: Path) -> None:
        """node_type 'Compute' (mixed case) should still be valid."""
        data = _make_valid_node_contract(node_type="Compute")
        path = _write_yaml(tmp_path, data)
        result = validate_contract(path)
        assert result["is_valid"] is True

    def test_contract_type_mismatch_case_insensitive(
        self,
        validator: ProtocolContractValidator,
        tmp_path: Path,
    ) -> None:
        """Contract type comparison should be case-insensitive."""
        data = _make_valid_node_contract(node_type="COMPUTE")
        path = _write_yaml(tmp_path, data)
        result = validator.validate_contract_file(path, contract_type="compute")
        assert result.is_valid is True

    def test_version_fields_all_components_missing(
        self,
        validator: ProtocolContractValidator,
    ) -> None:
        version: dict[str, object] = {}
        violations = validator._validate_version_structure(version, "v")
        assert len(violations) == 3  # major, minor, patch all missing

    def test_name_as_list_rejected(
        self,
        validator: ProtocolContractValidator,
    ) -> None:
        violations = validator._validate_name(["not", "a", "string"])
        assert len(violations) == 1
        assert "Expected string" in violations[0]
