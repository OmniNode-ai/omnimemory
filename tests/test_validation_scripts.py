# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Comprehensive unit tests for ONEX validation scripts.

These tests ensure that the validation scripts in scripts/validation/
correctly detect code quality violations and allow valid patterns.
Since these scripts are critical for code quality enforcement, bugs
in them could silently allow violations.

Test Coverage:
- validate_secrets.py: Secret detection with skip pattern handling
- validate_pydantic_patterns.py: Pydantic model_config validation
- validate_naming.py: ONEX naming conventions (ModelXxx, EnumXxx, etc.)
- validate_enum_casing.py: UPPER_SNAKE_CASE enforcement
- validate_single_class_per_file.py: Single class per file rule
- validate_no_backward_compatibility.py: Anti-pattern detection
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

# Project root directory (for subprocess cwd in tests)
PROJECT_ROOT = Path(__file__).parent.parent

# Add scripts/validation to path for imports
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "validation"))

from validate_enum_casing import validate_file as validate_enum_casing
from validate_http_imports import validate_file as validate_http_imports
from validate_naming import validate_file as validate_naming
from validate_no_backward_compatibility import (
    validate_file as validate_no_backward_compat,
)
from validate_pydantic_patterns import validate_file as validate_pydantic_patterns
from validate_secrets import is_test_file
from validate_secrets import validate_file as validate_secrets
from validate_single_class_per_file import (
    count_classes,
)
from validate_single_class_per_file import (
    validate_file as validate_single_class,
)

if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture
def temp_py_file() -> Generator[Path, None, None]:
    """Create a temporary Python file for testing."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        yield Path(f.name)
    # Cleanup
    Path(f.name).unlink(missing_ok=True)


def write_temp_file(content: str, suffix: str = ".py") -> Path:
    """Write content to a temporary file and return the path."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=suffix, delete=False, encoding="utf-8"
    ) as f:
        f.write(content)
        return Path(f.name)


def cleanup_temp_file(path: Path) -> None:
    """Clean up a temporary file."""
    path.unlink(missing_ok=True)


# =============================================================================
# TEST: validate_secrets.py
# =============================================================================


class TestValidateSecrets:
    """Tests for validate_secrets.py - secret detection."""

    def test_detects_hardcoded_api_key(self) -> None:
        """Test detection of hardcoded API keys."""
        code = '''api_key = "sk-1234567890abcdefghij"'''
        path = write_temp_file(code)
        try:
            violations = validate_secrets(path)
            assert len(violations) == 1
            assert "API key" in violations[0].message
        finally:
            cleanup_temp_file(path)

    def test_detects_hardcoded_password(self) -> None:
        """Test detection of hardcoded passwords."""
        code = '''password = "supersecret"'''
        path = write_temp_file(code)
        try:
            violations = validate_secrets(path)
            assert len(violations) == 1
            assert "password" in violations[0].message.lower()
        finally:
            cleanup_temp_file(path)

    def test_detects_hardcoded_secret_key(self) -> None:
        """Test detection of hardcoded secret keys."""
        code = '''secret_key = "my-super-secret-key-12345"'''
        path = write_temp_file(code)
        try:
            violations = validate_secrets(path)
            assert len(violations) == 1
            assert "secret key" in violations[0].message.lower()
        finally:
            cleanup_temp_file(path)

    def test_detects_hardcoded_auth_token(self) -> None:
        """Test detection of hardcoded auth tokens."""
        code = '''auth_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"'''
        path = write_temp_file(code)
        try:
            violations = validate_secrets(path)
            assert len(violations) == 1
            assert "auth token" in violations[0].message.lower()
        finally:
            cleanup_temp_file(path)

    def test_detects_bearer_token(self) -> None:
        """Test detection of hardcoded bearer tokens."""
        code = """headers = {"Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ"}"""
        path = write_temp_file(code)
        try:
            violations = validate_secrets(path)
            assert len(violations) == 1
            assert "bearer token" in violations[0].message.lower()
        finally:
            cleanup_temp_file(path)

    def test_skip_os_getenv_without_default(self) -> None:
        """Test skip pattern for os.getenv without default."""
        code = """api_key = os.getenv("API_KEY")"""
        path = write_temp_file(code)
        try:
            violations = validate_secrets(path)
            assert len(violations) == 0
        finally:
            cleanup_temp_file(path)

    def test_skip_os_environ_dict_access(self) -> None:
        """Test skip pattern for os.environ[] dict access."""
        code = """api_key = os.environ["API_KEY"]"""
        path = write_temp_file(code)
        try:
            violations = validate_secrets(path)
            assert len(violations) == 0
        finally:
            cleanup_temp_file(path)

    def test_skip_os_getenv_with_none_default(self) -> None:
        """Test skip pattern for os.getenv with None default."""
        code = """api_key = os.getenv("API_KEY", None)"""
        path = write_temp_file(code)
        try:
            violations = validate_secrets(path)
            assert len(violations) == 0
        finally:
            cleanup_temp_file(path)

    def test_skip_os_getenv_with_empty_string_default(self) -> None:
        """Test skip pattern for os.getenv with empty string default."""
        code = """api_key = os.getenv("API_KEY", "")"""
        path = write_temp_file(code)
        try:
            violations = validate_secrets(path)
            assert len(violations) == 0
        finally:
            cleanup_temp_file(path)

    def test_skip_pydantic_field_with_env(self) -> None:
        """Test skip pattern for Pydantic Field with env= parameter."""
        code = """api_key: str = Field(..., env="API_KEY")"""
        path = write_temp_file(code)
        try:
            violations = validate_secrets(path)
            assert len(violations) == 0
        finally:
            cleanup_temp_file(path)

    def test_skip_pydantic_field_with_default_factory(self) -> None:
        """Test skip pattern for Pydantic Field with default_factory."""
        code = """api_key: str = Field(default_factory=get_api_key)"""
        path = write_temp_file(code)
        try:
            violations = validate_secrets(path)
            assert len(violations) == 0
        finally:
            cleanup_temp_file(path)

    def test_skip_nosec_comment(self) -> None:
        """Test line-level skip for # nosec comment."""
        code = """api_key = "sk-1234567890abcdefghij"  # nosec"""
        path = write_temp_file(code)
        try:
            violations = validate_secrets(path)
            assert len(violations) == 0
        finally:
            cleanup_temp_file(path)

    def test_skip_not_a_secret_comment(self) -> None:
        """Test line-level skip for 'not a secret' comment."""
        code = """api_key = "sk-1234567890abcdefghij"  # not a real secret"""
        path = write_temp_file(code)
        try:
            violations = validate_secrets(path)
            assert len(violations) == 0
        finally:
            cleanup_temp_file(path)

    def test_skip_example_placeholder_comment(self) -> None:
        """Test line-level skip for example/placeholder comments."""
        code = """api_key = "sk-1234567890abcdefghij"  # example api_key"""
        path = write_temp_file(code)
        try:
            violations = validate_secrets(path)
            assert len(violations) == 0
        finally:
            cleanup_temp_file(path)

    def test_skip_placeholder_your_api_key(self) -> None:
        """Test skip pattern for 'your-api-key' placeholder."""
        code = '''api_key = "your-api-key"'''
        path = write_temp_file(code)
        try:
            violations = validate_secrets(path)
            assert len(violations) == 0
        finally:
            cleanup_temp_file(path)

    def test_skip_placeholder_xxx(self) -> None:
        """Test skip pattern for 'xxx' placeholder strings."""
        code = '''api_key = "xxxxxxxx"'''
        path = write_temp_file(code)
        try:
            violations = validate_secrets(path)
            assert len(violations) == 0
        finally:
            cleanup_temp_file(path)

    def test_skip_placeholder_changeme(self) -> None:
        """Test skip pattern for 'CHANGEME' placeholder."""
        code = '''password = "CHANGEME"'''
        path = write_temp_file(code)
        try:
            violations = validate_secrets(path)
            assert len(violations) == 0
        finally:
            cleanup_temp_file(path)

    def test_skip_placeholder_angle_brackets(self) -> None:
        """Test skip pattern for '<API_KEY>' style placeholders."""
        code = '''api_key = "<YOUR_API_KEY>"'''
        path = write_temp_file(code)
        try:
            violations = validate_secrets(path)
            assert len(violations) == 0
        finally:
            cleanup_temp_file(path)

    def test_overlap_detection_rejects_non_overlapping_skip(self) -> None:
        """Test that skip pattern must overlap with secret match.

        If os.getenv("VAR") appears BEFORE a secret on the same line,
        the skip pattern should NOT mask the actual secret.
        """
        # os.getenv("SAFE") followed by separate hardcoded secret
        code = '''config = os.getenv("LOG_LEVEL"); api_key = "sk-1234567890abcdef"'''
        path = write_temp_file(code)
        try:
            violations = validate_secrets(path)
            # Should detect the hardcoded secret despite os.getenv on same line
            assert len(violations) == 1
            assert "API key" in violations[0].message
        finally:
            cleanup_temp_file(path)

    def test_is_test_file_function(self) -> None:
        """Test is_test_file() helper function."""
        assert is_test_file(Path("test_something.py")) is True
        assert is_test_file(Path("something_test.py")) is True
        assert is_test_file(Path("/path/to/tests/file.py")) is True
        assert is_test_file(Path("/path/to/test/file.py")) is True
        assert is_test_file(Path("regular_module.py")) is False

    def test_test_file_skipped(self) -> None:
        """Test that test files are skipped entirely."""
        code = '''api_key = "sk-1234567890abcdefghij"'''
        # Create file in a tests/ directory pattern
        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = Path(tmpdir) / "tests"
            test_dir.mkdir()
            test_file = test_dir / "test_secrets.py"
            test_file.write_text(code)
            violations = validate_secrets(test_file)
            assert len(violations) == 0

    def test_clean_file_passes(self) -> None:
        """Test that clean files pass without violations."""
        code = '''
import os

def get_config():
    """Get configuration from environment."""
    return {
        "api_key": os.getenv("API_KEY"),
        "database": os.environ["DATABASE_URL"],
        "debug": True,
    }
'''
        path = write_temp_file(code)
        try:
            violations = validate_secrets(path)
            assert len(violations) == 0
        finally:
            cleanup_temp_file(path)

    def test_detects_secret_in_env_default(self) -> None:
        """Test detection of hardcoded secrets in getenv defaults."""
        code = """api_key = os.getenv("API_KEY", "sk-1234567890abcdefghij")"""
        path = write_temp_file(code)
        try:
            violations = validate_secrets(path)
            assert len(violations) == 1
            assert "secret" in violations[0].message.lower()
        finally:
            cleanup_temp_file(path)


# =============================================================================
# TEST: validate_pydantic_patterns.py
# =============================================================================


class TestValidatePydanticPatterns:
    """Tests for validate_pydantic_patterns.py - Pydantic model validation."""

    def test_detects_missing_model_config(self) -> None:
        """Test detection of Pydantic model without model_config."""
        code = """
from pydantic import BaseModel

class User(BaseModel):
    name: str
    email: str
"""
        path = write_temp_file(code)
        try:
            violations = validate_pydantic_patterns(path)
            assert len(violations) == 1
            assert "Missing model_config" in violations[0].message
        finally:
            cleanup_temp_file(path)

    def test_detects_empty_config_dict(self) -> None:
        """Test detection of empty ConfigDict()."""
        code = """
from pydantic import BaseModel, ConfigDict

class User(BaseModel):
    model_config = ConfigDict()
    name: str
"""
        path = write_temp_file(code)
        try:
            violations = validate_pydantic_patterns(path)
            assert len(violations) == 1
            assert "Empty ConfigDict" in violations[0].message
        finally:
            cleanup_temp_file(path)

    def test_detects_bare_dict_model_config(self) -> None:
        """Test detection of bare dict {} for model_config."""
        code = """
from pydantic import BaseModel

class User(BaseModel):
    model_config = {}
    name: str
"""
        path = write_temp_file(code)
        try:
            violations = validate_pydantic_patterns(path)
            assert len(violations) == 1
            assert "Empty dict" in violations[0].message
        finally:
            cleanup_temp_file(path)

    def test_detects_non_empty_bare_dict(self) -> None:
        """Test detection of non-empty bare dict for model_config."""
        code = """
from pydantic import BaseModel

class User(BaseModel):
    model_config = {"frozen": True}
    name: str
"""
        path = write_temp_file(code)
        try:
            violations = validate_pydantic_patterns(path)
            assert len(violations) == 1
            assert "Bare dict" in violations[0].message
        finally:
            cleanup_temp_file(path)

    def test_valid_model_config_passes(self) -> None:
        """Test that valid model_config = ConfigDict(...) passes."""
        code = """
from pydantic import BaseModel, ConfigDict

class User(BaseModel):
    model_config = ConfigDict(frozen=True)
    name: str
"""
        path = write_temp_file(code)
        try:
            violations = validate_pydantic_patterns(path)
            assert len(violations) == 0
        finally:
            cleanup_temp_file(path)

    def test_inherited_model_config_recognized(self) -> None:
        """Test that inherited model_config from parent class is recognized."""
        code = """
from pydantic import BaseModel, ConfigDict

class BaseUser(BaseModel):
    model_config = ConfigDict(frozen=True)

class AdminUser(BaseUser):
    is_admin: bool = True
"""
        path = write_temp_file(code)
        try:
            violations = validate_pydantic_patterns(path)
            # AdminUser inherits model_config from BaseUser, so no violation
            assert len(violations) == 0
        finally:
            cleanup_temp_file(path)

    def test_transitive_pydantic_inheritance(self) -> None:
        """Test transitive Pydantic model inheritance is recognized."""
        code = """
from pydantic import BaseModel, ConfigDict

class BaseEntity(BaseModel):
    model_config = ConfigDict(frozen=True)
    id: int

class User(BaseEntity):
    name: str

class AdminUser(User):
    permissions: list
"""
        path = write_temp_file(code)
        try:
            violations = validate_pydantic_patterns(path)
            # All classes inherit model_config transitively
            assert len(violations) == 0
        finally:
            cleanup_temp_file(path)

    def test_annotation_only_model_config_detected(self) -> None:
        """Test detection of annotation-only model_config without value.

        Note: Annotation-only model_config results in two violations:
        1. "Annotation-only model_config" - the annotation has no value
        2. "Missing model_config" - annotation without value doesn't count as having config
        """
        code = """
from pydantic import BaseModel, ConfigDict

class User(BaseModel):
    model_config: ConfigDict
    name: str
"""
        path = write_temp_file(code)
        try:
            violations = validate_pydantic_patterns(path)
            # Annotation-only triggers both "Annotation-only" and "Missing model_config"
            assert len(violations) == 2
            messages = [v.message for v in violations]
            assert any("Annotation-only" in m for m in messages)
            assert any("Missing model_config" in m for m in messages)
        finally:
            cleanup_temp_file(path)

    def test_non_pydantic_class_ignored(self) -> None:
        """Test that non-Pydantic classes are ignored."""
        code = """
class RegularClass:
    def __init__(self):
        self.name = "test"
"""
        path = write_temp_file(code)
        try:
            violations = validate_pydantic_patterns(path)
            assert len(violations) == 0
        finally:
            cleanup_temp_file(path)

    def test_settings_config_dict_recognized(self) -> None:
        """Test that SettingsConfigDict is recognized for BaseSettings."""
        code = """
from pydantic_settings import BaseSettings, SettingsConfigDict

class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="APP_")
    debug: bool = False
"""
        path = write_temp_file(code)
        try:
            violations = validate_pydantic_patterns(path)
            assert len(violations) == 0
        finally:
            cleanup_temp_file(path)

    def test_aliased_config_dict_import(self) -> None:
        """Test that aliased ConfigDict import is recognized."""
        code = """
from pydantic import BaseModel
from pydantic import ConfigDict as CD

class User(BaseModel):
    model_config = CD(frozen=True)
    name: str
"""
        path = write_temp_file(code)
        try:
            violations = validate_pydantic_patterns(path)
            assert len(violations) == 0
        finally:
            cleanup_temp_file(path)

    def test_skip_utils_directory(self) -> None:
        """Test that files in utils/ directory are skipped."""
        code = """
from pydantic import BaseModel

class UtilityModel(BaseModel):
    value: str
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            utils_dir = Path(tmpdir) / "utils"
            utils_dir.mkdir()
            util_file = utils_dir / "helpers.py"
            util_file.write_text(code)
            violations = validate_pydantic_patterns(util_file)
            assert len(violations) == 0

    def test_skip_tests_directory(self) -> None:
        """Test that files in tests/ directory are skipped."""
        code = """
from pydantic import BaseModel

class TestModel(BaseModel):
    value: str
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tests_dir = Path(tmpdir) / "tests"
            tests_dir.mkdir()
            test_file = tests_dir / "fixtures.py"
            test_file.write_text(code)
            violations = validate_pydantic_patterns(test_file)
            assert len(violations) == 0


# =============================================================================
# TEST: validate_naming.py
# =============================================================================


class TestValidateNaming:
    """Tests for validate_naming.py - ONEX naming conventions."""

    def test_model_prefix_enforced(self) -> None:
        """Test ModelXxx naming is enforced for BaseModel subclasses."""
        code = """
from pydantic import BaseModel

class User(BaseModel):
    name: str
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            models_dir = Path(tmpdir) / "models"
            models_dir.mkdir()
            model_file = models_dir / "model_user.py"
            model_file.write_text(code)
            violations = validate_naming(model_file)
            assert len(violations) == 1
            assert (
                "ModelXxx" in violations[0].message
                or "ONEX naming" in violations[0].message
            )

    def test_valid_model_name_passes(self) -> None:
        """Test that valid ModelXxx naming passes."""
        code = """
from pydantic import BaseModel

class ModelUser(BaseModel):
    name: str
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            models_dir = Path(tmpdir) / "models"
            models_dir.mkdir()
            model_file = models_dir / "model_user.py"
            model_file.write_text(code)
            violations = validate_naming(model_file)
            assert len(violations) == 0

    def test_enum_prefix_enforced(self) -> None:
        """Test EnumXxx naming is enforced for Enum subclasses."""
        code = """
from enum import Enum

class Status(Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            enums_dir = Path(tmpdir) / "enums"
            enums_dir.mkdir()
            enum_file = enums_dir / "enum_status.py"
            enum_file.write_text(code)
            violations = validate_naming(enum_file)
            assert len(violations) == 1
            assert (
                "EnumXxx" in violations[0].message
                or "ONEX naming" in violations[0].message
            )

    def test_valid_enum_name_passes(self) -> None:
        """Test that valid EnumXxx naming passes."""
        code = """
from enum import Enum

class EnumStatus(Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            enums_dir = Path(tmpdir) / "enums"
            enums_dir.mkdir()
            enum_file = enums_dir / "enum_status.py"
            enum_file.write_text(code)
            violations = validate_naming(enum_file)
            assert len(violations) == 0

    def test_protocol_prefix_enforced(self) -> None:
        """Test ProtocolXxx naming is enforced for Protocol classes."""
        code = """
from typing import Protocol

class Repository(Protocol):
    def get(self, id: str) -> dict: ...
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            protocols_dir = Path(tmpdir) / "protocols"
            protocols_dir.mkdir()
            protocol_file = protocols_dir / "protocol_repository.py"
            protocol_file.write_text(code)
            violations = validate_naming(protocol_file)
            assert len(violations) == 1
            assert (
                "ProtocolXxx" in violations[0].message
                or "ONEX naming" in violations[0].message
            )

    def test_valid_protocol_name_passes(self) -> None:
        """Test that valid ProtocolXxx naming passes."""
        code = """
from typing import Protocol

class ProtocolRepository(Protocol):
    def get(self, id: str) -> dict: ...
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            protocols_dir = Path(tmpdir) / "protocols"
            protocols_dir.mkdir()
            protocol_file = protocols_dir / "protocol_repository.py"
            protocol_file.write_text(code)
            violations = validate_naming(protocol_file)
            assert len(violations) == 0

    def test_settings_suffix_enforced(self) -> None:
        """Test XxxSettings suffix pattern for BaseSettings."""
        code = """
from pydantic_settings import BaseSettings

class DatabaseConfig(BaseSettings):
    host: str
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Settings don't require typed directory, test in regular dir
            settings_file = Path(tmpdir) / "config.py"
            settings_file.write_text(code)
            violations = validate_naming(settings_file)
            assert len(violations) == 1
            assert "Settings" in violations[0].message

    def test_valid_settings_suffix_passes(self) -> None:
        """Test that valid XxxSettings naming passes."""
        code = """
from pydantic_settings import BaseSettings

class DatabaseSettings(BaseSettings):
    host: str
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            settings_file = Path(tmpdir) / "settings.py"
            settings_file.write_text(code)
            violations = validate_naming(settings_file)
            assert len(violations) == 0

    def test_file_naming_in_models_directory(self) -> None:
        """Test file naming enforcement in models/ directory."""
        code = """
from pydantic import BaseModel

class ModelUser(BaseModel):
    name: str
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            models_dir = Path(tmpdir) / "models"
            models_dir.mkdir()
            # Wrong file naming - should be model_xxx.py
            bad_file = models_dir / "user_model.py"
            bad_file.write_text(code)
            violations = validate_naming(bad_file)
            # Should have file naming violation
            file_violations = [v for v in violations if "model_xxx.py" in v.message]
            assert len(file_violations) == 1

    def test_relaxed_naming_in_utils_directory(self) -> None:
        """Test that utils/ directory has relaxed class prefix naming."""
        code = """
from pydantic import BaseModel

class ConnectionMetadata(BaseModel):
    host: str
    port: int
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            utils_dir = Path(tmpdir) / "utils"
            utils_dir.mkdir()
            util_file = utils_dir / "connection.py"
            util_file.write_text(code)
            violations = validate_naming(util_file)
            # utils/ has relaxed class prefix naming
            assert len(violations) == 0

    def test_skip_init_files(self) -> None:
        """Test that __init__.py files are skipped."""
        code = """
from pydantic import BaseModel

class User(BaseModel):
    name: str
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            pkg_dir = Path(tmpdir) / "models"
            pkg_dir.mkdir()
            init_file = pkg_dir / "__init__.py"
            init_file.write_text(code)
            violations = validate_naming(init_file)
            assert len(violations) == 0

    def test_skip_tests_directory(self) -> None:
        """Test that tests/ directory is skipped."""
        code = """
from pydantic import BaseModel

class TestUser(BaseModel):
    name: str
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tests_dir = Path(tmpdir) / "tests"
            tests_dir.mkdir()
            test_file = tests_dir / "fixtures.py"
            test_file.write_text(code)
            violations = validate_naming(test_file)
            assert len(violations) == 0

    def test_private_classes_skipped(self) -> None:
        """Test that private classes (underscore prefix) are skipped."""
        code = """
from pydantic import BaseModel

class _PrivateHelper(BaseModel):
    value: str
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            models_dir = Path(tmpdir) / "models"
            models_dir.mkdir()
            model_file = models_dir / "model_internal.py"
            model_file.write_text(code)
            violations = validate_naming(model_file)
            # Private class should not trigger naming violation
            class_violations = [v for v in violations if "_PrivateHelper" in v.message]
            assert len(class_violations) == 0

    def test_str_enum_naming(self) -> None:
        """Test EnumXxx naming for StrEnum subclasses."""
        code = """
from enum import StrEnum

class Status(StrEnum):
    ACTIVE = "active"
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            enums_dir = Path(tmpdir) / "enums"
            enums_dir.mkdir()
            enum_file = enums_dir / "enum_status.py"
            enum_file.write_text(code)
            violations = validate_naming(enum_file)
            assert len(violations) == 1
            assert (
                "EnumXxx" in violations[0].message
                or "ONEX naming" in violations[0].message
            )


# =============================================================================
# TEST: validate_enum_casing.py
# =============================================================================


class TestValidateEnumCasing:
    """Tests for validate_enum_casing.py - UPPER_SNAKE_CASE enforcement."""

    def test_detects_lowercase_enum_member(self) -> None:
        """Test detection of lowercase enum member names."""
        code = """
from enum import Enum

class Status(Enum):
    active = "active"
    inactive = "inactive"
"""
        path = write_temp_file(code)
        try:
            violations = validate_enum_casing(path)
            assert len(violations) == 2
            assert all("UPPER_SNAKE_CASE" in v.message for v in violations)
        finally:
            cleanup_temp_file(path)

    def test_detects_mixed_case_enum_member(self) -> None:
        """Test detection of mixed case enum member names."""
        code = """
from enum import Enum

class Status(Enum):
    ActiveStatus = "active"
    InactiveStatus = "inactive"
"""
        path = write_temp_file(code)
        try:
            violations = validate_enum_casing(path)
            assert len(violations) == 2
            assert all("UPPER_SNAKE_CASE" in v.message for v in violations)
        finally:
            cleanup_temp_file(path)

    def test_valid_upper_snake_case_passes(self) -> None:
        """Test that valid UPPER_SNAKE_CASE passes."""
        code = """
from enum import Enum

class Status(Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    IN_PROGRESS = "in_progress"
"""
        path = write_temp_file(code)
        try:
            violations = validate_enum_casing(path)
            assert len(violations) == 0
        finally:
            cleanup_temp_file(path)

    def test_valid_upper_snake_case_with_numbers(self) -> None:
        """Test that UPPER_SNAKE_CASE with numbers passes."""
        code = """
from enum import Enum

class Priority(Enum):
    P1_CRITICAL = 1
    P2_HIGH = 2
    LEVEL_10 = 10
"""
        path = write_temp_file(code)
        try:
            violations = validate_enum_casing(path)
            assert len(violations) == 0
        finally:
            cleanup_temp_file(path)

    def test_str_enum_casing(self) -> None:
        """Test casing enforcement for StrEnum."""
        code = """
from enum import StrEnum

class Color(StrEnum):
    red = "red"
    green = "green"
"""
        path = write_temp_file(code)
        try:
            violations = validate_enum_casing(path)
            assert len(violations) == 2
        finally:
            cleanup_temp_file(path)

    def test_int_enum_casing(self) -> None:
        """Test casing enforcement for IntEnum."""
        code = """
from enum import IntEnum

class Level(IntEnum):
    low = 1
    medium = 2
    high = 3
"""
        path = write_temp_file(code)
        try:
            violations = validate_enum_casing(path)
            assert len(violations) == 3
        finally:
            cleanup_temp_file(path)

    def test_int_flag_casing(self) -> None:
        """Test casing enforcement for IntFlag."""
        code = """
from enum import IntFlag

class Permission(IntFlag):
    read = 1
    write = 2
    execute = 4
"""
        path = write_temp_file(code)
        try:
            violations = validate_enum_casing(path)
            assert len(violations) == 3
        finally:
            cleanup_temp_file(path)

    def test_private_members_skipped(self) -> None:
        """Test that private/dunder members are skipped."""
        code = """
from enum import Enum

class Status(Enum):
    _internal = "internal"
    __special = "special"
    ACTIVE = "active"
"""
        path = write_temp_file(code)
        try:
            violations = validate_enum_casing(path)
            # Only public members are checked
            assert len(violations) == 0
        finally:
            cleanup_temp_file(path)

    def test_annotated_assignment_casing(self) -> None:
        """Test casing enforcement for annotated enum assignments."""
        code = """
from enum import Enum

class Status(Enum):
    active: str = "active"
"""
        path = write_temp_file(code)
        try:
            violations = validate_enum_casing(path)
            assert len(violations) == 1
            assert "active" in violations[0].message
        finally:
            cleanup_temp_file(path)

    def test_non_enum_class_ignored(self) -> None:
        """Test that non-enum classes are ignored."""
        code = """
class RegularClass:
    active = "active"
    inactive = "inactive"
"""
        path = write_temp_file(code)
        try:
            violations = validate_enum_casing(path)
            assert len(violations) == 0
        finally:
            cleanup_temp_file(path)


# =============================================================================
# TEST: validate_single_class_per_file.py
# =============================================================================


class TestValidateSingleClassPerFile:
    """Tests for validate_single_class_per_file.py - single class rule."""

    def test_single_class_passes(self) -> None:
        """Test that single class per file passes."""
        code = """
from pydantic import BaseModel

class User(BaseModel):
    name: str
    email: str
"""
        path = write_temp_file(code)
        try:
            violations = validate_single_class(path)
            assert len(violations) == 0
        finally:
            cleanup_temp_file(path)

    def test_multiple_non_enum_classes_flagged(self) -> None:
        """Test that multiple non-enum classes are flagged."""
        code = """
from pydantic import BaseModel

class User(BaseModel):
    name: str

class Profile(BaseModel):
    bio: str

class Settings(BaseModel):
    theme: str
"""
        path = write_temp_file(code)
        try:
            violations = validate_single_class(path)
            assert len(violations) == 1
            assert "Multiple non-enum classes" in violations[0].message
            assert "3" in violations[0].message  # Should show count
        finally:
            cleanup_temp_file(path)

    def test_multiple_enums_allowed(self) -> None:
        """Test that multiple enums in one file are allowed."""
        code = """
from enum import Enum

class Status(Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"

class Priority(Enum):
    LOW = 1
    HIGH = 2

class Color(Enum):
    RED = "red"
    BLUE = "blue"
"""
        path = write_temp_file(code)
        try:
            violations = validate_single_class(path)
            assert len(violations) == 0
        finally:
            cleanup_temp_file(path)

    def test_one_class_with_multiple_enums_allowed(self) -> None:
        """Test that one non-enum class with multiple enums is allowed."""
        code = """
from enum import Enum
from pydantic import BaseModel

class Status(Enum):
    ACTIVE = "active"

class Priority(Enum):
    HIGH = 1

class User(BaseModel):
    name: str
    status: Status
    priority: Priority
"""
        path = write_temp_file(code)
        try:
            violations = validate_single_class(path)
            # 1 non-enum class + multiple enums = OK
            assert len(violations) == 0
        finally:
            cleanup_temp_file(path)

    def test_init_file_skipped(self) -> None:
        """Test that __init__.py files are skipped."""
        code = """
class ClassA:
    pass

class ClassB:
    pass
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            init_file = Path(tmpdir) / "__init__.py"
            init_file.write_text(code)
            violations = validate_single_class(init_file)
            assert len(violations) == 0

    def test_base_py_exemption(self) -> None:
        """Test that base.py files are exempt."""
        code = """
class BaseModel:
    pass

class BaseService:
    pass
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_file = Path(tmpdir) / "base.py"
            base_file.write_text(code)
            violations = validate_single_class(base_file)
            assert len(violations) == 0

    def test_core_directory_exemption(self) -> None:
        """Test that files in core/ directory are exempt."""
        code = """
class TypeA:
    pass

class TypeB:
    pass
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            core_dir = Path(tmpdir) / "core"
            core_dir.mkdir()
            core_file = core_dir / "types.py"
            core_file.write_text(code)
            violations = validate_single_class(core_file)
            assert len(violations) == 0

    def test_foundation_directory_exemption(self) -> None:
        """Test that files in foundation/ directory are exempt."""
        code = """
class FoundationA:
    pass

class FoundationB:
    pass
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            foundation_dir = Path(tmpdir) / "foundation"
            foundation_dir.mkdir()
            foundation_file = foundation_dir / "base_types.py"
            foundation_file.write_text(code)
            violations = validate_single_class(foundation_file)
            assert len(violations) == 0

    def test_protocols_directory_exemption(self) -> None:
        """Test that files in protocols/ directory are exempt."""
        code = """
from typing import Protocol

class ProtocolA(Protocol):
    pass

class ProtocolB(Protocol):
    pass
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            protocols_dir = Path(tmpdir) / "protocols"
            protocols_dir.mkdir()
            protocol_file = protocols_dir / "interfaces.py"
            protocol_file.write_text(code)
            violations = validate_single_class(protocol_file)
            assert len(violations) == 0

    def test_count_classes_function(self) -> None:
        """Test the count_classes helper function."""
        code = """
from enum import Enum

class RegularClass:
    pass

class AnotherClass:
    pass

class MyEnum(Enum):
    VALUE = 1
"""
        path = write_temp_file(code)
        try:
            non_enum_count, non_enum_names, enum_count, enum_names = count_classes(path)
            assert non_enum_count == 2
            assert set(non_enum_names) == {"RegularClass", "AnotherClass"}
            assert enum_count == 1
            assert enum_names == ["MyEnum"]
        finally:
            cleanup_temp_file(path)

    def test_violation_message_includes_class_names(self) -> None:
        """Test that violation message includes the class names."""
        code = """
class UserService:
    pass

class ProfileService:
    pass
"""
        path = write_temp_file(code)
        try:
            violations = validate_single_class(path)
            assert len(violations) == 1
            assert "UserService" in violations[0].message
            assert "ProfileService" in violations[0].message
        finally:
            cleanup_temp_file(path)


# =============================================================================
# TEST: validate_no_backward_compatibility.py
# =============================================================================


class TestValidateNoBackwardCompatibility:
    """Tests for validate_no_backward_compatibility.py - anti-pattern detection."""

    def test_detects_backward_compat_comment(self) -> None:
        """Test detection of '# backward compat' comments."""
        code = """
def old_function():
    # backward compat
    return new_function()
"""
        path = write_temp_file(code)
        try:
            violations = validate_no_backward_compat(path)
            assert len(violations) == 1
            assert "Backward compatibility comment" in violations[0].message
        finally:
            cleanup_temp_file(path)

    def test_detects_backwards_compatibility_comment(self) -> None:
        """Test detection of '# backwards compatibility' comment variant."""
        code = """
# backwards compatibility with old API
old_name = new_name
"""
        path = write_temp_file(code)
        try:
            violations = validate_no_backward_compat(path)
            assert len(violations) == 1
            assert "Backward compatibility comment" in violations[0].message
        finally:
            cleanup_temp_file(path)

    def test_detects_deprecated_decorator(self) -> None:
        """Test detection of @deprecated decorator."""
        code = """
@deprecated
def old_function():
    pass
"""
        path = write_temp_file(code)
        try:
            violations = validate_no_backward_compat(path)
            assert len(violations) == 1
            assert "Deprecated decorator" in violations[0].message
        finally:
            cleanup_temp_file(path)

    def test_detects_deprecated_comment(self) -> None:
        """Test detection of '# deprecated' comment."""
        code = """
# deprecated
def old_function():
    pass
"""
        path = write_temp_file(code)
        try:
            violations = validate_no_backward_compat(path)
            assert len(violations) == 1
            assert "Deprecated comment" in violations[0].message
        finally:
            cleanup_temp_file(path)

    def test_detects_legacy_comment(self) -> None:
        """Test detection of '# legacy' comment.

        Note: The regex uses a negative lookahead (?!\\s*[_a-z]) to avoid
        matching when followed by underscore or lowercase (e.g., # legacy_helper).
        Use "# legacy" or "# legacy:" to trigger detection.
        """
        code = """
# legacy
def old_handler():
    pass
"""
        path = write_temp_file(code)
        try:
            violations = validate_no_backward_compat(path)
            assert len(violations) == 1
            assert "Legacy comment" in violations[0].message
        finally:
            cleanup_temp_file(path)

    def test_detects_alias_assignment(self) -> None:
        """Test detection of alias assignments."""
        code = """
old_function = new_function  # alias
"""
        path = write_temp_file(code)
        try:
            violations = validate_no_backward_compat(path)
            assert len(violations) == 1
            assert "Alias assignment" in violations[0].message
        finally:
            cleanup_temp_file(path)

    def test_detects_todo_remove_deprecated(  # TODO_FORMAT_EXEMPT: test fixture
        self,
    ) -> None:
        """Test detection of TODO to remove deprecated code."""
        code = """
# TODO: remove deprecated function in next version
def old_function():
    pass
"""
        path = write_temp_file(code)
        try:
            violations = validate_no_backward_compat(path)
            assert len(violations) == 1
            assert "TODO to remove deprecated" in violations[0].message
        finally:
            cleanup_temp_file(path)

    def test_clean_file_passes(self) -> None:
        """Test that clean files pass without violations."""
        code = '''
def modern_function():
    """A modern function with no backward compatibility cruft."""
    return do_something()


class ModernClass:
    """A clean, modern implementation."""

    def process(self):
        return self._internal_process()
'''
        path = write_temp_file(code)
        try:
            violations = validate_no_backward_compat(path)
            assert len(violations) == 0
        finally:
            cleanup_temp_file(path)

    def test_skip_omnibase_core_import(self) -> None:
        """Test that imports from omnibase_core are skipped."""
        code = """
from omnibase_core import deprecated_module  # This is fine
"""
        path = write_temp_file(code)
        try:
            violations = validate_no_backward_compat(path)
            assert len(violations) == 0
        finally:
            cleanup_temp_file(path)

    def test_does_not_match_deprecated_in_variable_name(self) -> None:
        """Test that 'deprecated' in variable names doesn't trigger.

        The pattern uses word boundaries to avoid matching variable names
        like 'deprecated_feature' or 'is_deprecated'.
        """
        code = '''
def check_deprecated_features():
    """Check for deprecated features."""
    deprecated_count = 0
    return deprecated_count
'''
        path = write_temp_file(code)
        try:
            violations = validate_no_backward_compat(path)
            assert len(violations) == 0
        finally:
            cleanup_temp_file(path)

    def test_does_not_match_legacy_in_variable_name(self) -> None:
        """Test that 'legacy' in variable names doesn't trigger."""
        code = '''
def handle_legacy_data():
    """Handle legacy data format."""
    legacy_count = process_legacy_records()
    return legacy_count
'''
        path = write_temp_file(code)
        try:
            violations = validate_no_backward_compat(path)
            assert len(violations) == 0
        finally:
            cleanup_temp_file(path)

    def test_case_insensitive_detection(self) -> None:
        """Test that detection is case insensitive.

        Note: The regex uses negative lookahead to avoid matching when followed
        by lowercase letters. Using @Deprecated decorator and backwards compat
        comment tests case insensitivity reliably.
        """
        code = """
@Deprecated
def old_function():
    pass

# backwards compatibility
old_name = new_name
"""
        path = write_temp_file(code)
        try:
            violations = validate_no_backward_compat(path)
            assert len(violations) == 2
        finally:
            cleanup_temp_file(path)

    def test_one_violation_per_line(self) -> None:
        """Test that only one violation is reported per line."""
        # Line that matches multiple patterns
        code = """
@deprecated  # legacy backward compat
def old_function():
    pass
"""
        path = write_temp_file(code)
        try:
            violations = validate_no_backward_compat(path)
            # Even though multiple patterns match, only one violation per line
            line_2_violations = [v for v in violations if v.line == 2]
            assert len(line_2_violations) == 1
        finally:
            cleanup_temp_file(path)


# =============================================================================
# TEST: validate_http_imports.py
# =============================================================================


class TestValidateHttpImports:
    """Tests for validate_http_imports.py - HTTP import boundary enforcement."""

    def test_detects_direct_httpx_import(self) -> None:
        """Test detection of direct 'import httpx'."""
        code = """import httpx"""
        path = write_temp_file(code)
        try:
            violations = validate_http_imports(path)
            assert len(violations) == 1
            assert "import httpx" in violations[0].message.lower()
        finally:
            cleanup_temp_file(path)

    def test_detects_from_httpx_import(self) -> None:
        """Test detection of 'from httpx import ...'."""
        code = """from httpx import AsyncClient"""
        path = write_temp_file(code)
        try:
            violations = validate_http_imports(path)
            assert len(violations) == 1
            assert "from httpx" in violations[0].message.lower()
        finally:
            cleanup_temp_file(path)

    def test_detects_direct_requests_import(self) -> None:
        """Test detection of direct 'import requests'."""
        code = """import requests"""
        path = write_temp_file(code)
        try:
            violations = validate_http_imports(path)
            assert len(violations) == 1
            assert "import requests" in violations[0].message.lower()
        finally:
            cleanup_temp_file(path)

    def test_detects_direct_aiohttp_import(self) -> None:
        """Test detection of direct 'import aiohttp'."""
        code = """import aiohttp"""
        path = write_temp_file(code)
        try:
            violations = validate_http_imports(path)
            assert len(violations) == 1
            assert "import aiohttp" in violations[0].message.lower()
        finally:
            cleanup_temp_file(path)

    def test_allows_type_checking_imports(self) -> None:
        """Test that imports in TYPE_CHECKING blocks are allowed."""
        code = """
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import httpx
    from requests import Session
"""
        path = write_temp_file(code)
        try:
            violations = validate_http_imports(path)
            assert len(violations) == 0
        finally:
            cleanup_temp_file(path)

    def test_allows_explicit_exemption_annotation(self) -> None:
        """Test that explicit omnimemory-http-exempt annotation skips the line."""
        code = (
            """import httpx  # omnimemory-http-exempt: Testing HTTP client directly"""
        )
        path = write_temp_file(code)
        try:
            violations = validate_http_imports(path)
            assert len(violations) == 0
        finally:
            cleanup_temp_file(path)

    def test_allows_exemption_annotation_case_insensitive(self) -> None:
        """Test that exemption annotation works case-insensitively."""
        code = (
            """import httpx  # OMNIMEMORY-HTTP-EXEMPT: Testing HTTP client directly"""
        )
        path = write_temp_file(code)
        try:
            violations = validate_http_imports(path)
            assert len(violations) == 0
        finally:
            cleanup_temp_file(path)

    def test_does_not_match_generic_adapter_comment(self) -> None:
        """Test that generic 'use adapter' comments do NOT skip violations.

        This is the key test ensuring the pattern is specific enough.
        Comments like '# We use adapter pattern here' should NOT be exemptions.
        """
        code = """# We use adapter pattern here for this implementation
import httpx"""
        path = write_temp_file(code)
        try:
            violations = validate_http_imports(path)
            # Should still detect the violation - generic adapter comments don't count
            assert len(violations) == 1
            assert "httpx" in violations[0].message.lower()
        finally:
            cleanup_temp_file(path)

    def test_does_not_match_using_adapter_comment(self) -> None:
        """Test that 'using adapter' comments do NOT skip violations."""
        code = """# This function will be using adapter for X
import requests"""
        path = write_temp_file(code)
        try:
            violations = validate_http_imports(path)
            # Should still detect the violation
            assert len(violations) == 1
            assert "requests" in violations[0].message.lower()
        finally:
            cleanup_temp_file(path)

    def test_allows_adapters_directory(self) -> None:
        """Test that files in handlers/adapters/ directory are allowed."""
        code = """import httpx
from requests import Session"""
        with tempfile.TemporaryDirectory() as tmpdir:
            adapters_dir = Path(tmpdir) / "handlers" / "adapters"
            adapters_dir.mkdir(parents=True)
            adapter_file = adapters_dir / "http_client.py"
            adapter_file.write_text(code)
            violations = validate_http_imports(adapter_file)
            assert len(violations) == 0

    def test_allows_tests_directory(self) -> None:
        """Test that files in tests/ directory are allowed."""
        code = """import httpx"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tests_dir = Path(tmpdir) / "tests"
            tests_dir.mkdir()
            test_file = tests_dir / "test_http.py"
            test_file.write_text(code)
            violations = validate_http_imports(test_file)
            assert len(violations) == 0

    def test_clean_file_passes(self) -> None:
        """Test that clean files without HTTP imports pass."""
        code = """
from typing import Protocol

class HttpClientProtocol(Protocol):
    async def get(self, url: str) -> dict: ...
"""
        path = write_temp_file(code)
        try:
            violations = validate_http_imports(path)
            assert len(violations) == 0
        finally:
            cleanup_temp_file(path)

    def test_handles_missing_file(self) -> None:
        """Test that missing files are handled gracefully."""
        path = Path("/nonexistent/file.py")
        violations = validate_http_imports(path)
        assert violations == []


# =============================================================================
# ADDITIONAL COVERAGE TESTS
# =============================================================================


class TestValidateSecretsAdditional:
    """Additional tests for validate_secrets.py edge cases and error handling."""

    def test_handles_unicode_decode_error(self) -> None:
        """Test that files with encoding issues are handled gracefully."""
        # Create a file with invalid UTF-8
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".py", delete=False) as f:
            f.write(b"\xff\xfe")  # Invalid UTF-8 bytes
            path = Path(f.name)
        try:
            violations = validate_secrets(path)
            # Should return empty list, not crash
            assert violations == []
        finally:
            path.unlink(missing_ok=True)

    def test_handles_missing_file(self) -> None:
        """Test that missing files are handled gracefully."""
        path = Path("/nonexistent/file.py")
        violations = validate_secrets(path)
        # Should return empty list, not crash
        assert violations == []

    def test_verbose_mode_logging(self) -> None:
        """Test that verbose mode works (doesn't crash)."""
        code = """api_key = os.getenv("API_KEY")"""
        path = write_temp_file(code)
        try:
            # Should not crash with verbose=True
            violations = validate_secrets(path, verbose=True)
            assert len(violations) == 0
        finally:
            cleanup_temp_file(path)

    def test_verbose_mode_with_skip(self) -> None:
        """Test verbose mode logs skip patterns."""
        code = '''api_key = "your-api-key"'''
        path = write_temp_file(code)
        try:
            # Should not crash with verbose=True
            violations = validate_secrets(path, verbose=True)
            assert len(violations) == 0
        finally:
            cleanup_temp_file(path)

    def test_multiple_secrets_on_different_lines(self) -> None:
        """Test detection of multiple secrets across lines."""
        code = """
api_key = "sk-1234567890abcdef"
password = "supersecret"
secret_key = "my-secret-key-12345"
"""
        path = write_temp_file(code)
        try:
            violations = validate_secrets(path)
            assert len(violations) == 3
        finally:
            cleanup_temp_file(path)

    def test_private_key_detection(self) -> None:
        """Test detection of hardcoded private keys."""
        code = '''private_key = "my-private-key-value-here-1234"'''
        path = write_temp_file(code)
        try:
            violations = validate_secrets(path)
            assert len(violations) == 1
            assert "private key" in violations[0].message.lower()
        finally:
            cleanup_temp_file(path)


class TestValidatePydanticPatternsAdditional:
    """Additional tests for validate_pydantic_patterns.py edge cases."""

    def test_handles_syntax_error_file(self) -> None:
        """Test that files with syntax errors are handled gracefully."""
        code = """
def broken_function(
    # Incomplete syntax
"""
        path = write_temp_file(code)
        try:
            violations = validate_pydantic_patterns(path)
            # Should return empty list for syntax error files
            assert violations == []
        finally:
            cleanup_temp_file(path)

    def test_handles_missing_file(self) -> None:
        """Test that missing files are handled gracefully."""
        path = Path("/nonexistent/file.py")
        violations = validate_pydantic_patterns(path)
        assert violations == []

    def test_generic_model_recognized(self) -> None:
        """Test that GenericModel subclasses are recognized."""
        code = """
from pydantic import BaseModel, ConfigDict
from typing import Generic, TypeVar

T = TypeVar("T")

class ModelContainer(BaseModel, Generic[T]):
    model_config = ConfigDict(frozen=True)
    value: T
"""
        path = write_temp_file(code)
        try:
            violations = validate_pydantic_patterns(path)
            assert len(violations) == 0
        finally:
            cleanup_temp_file(path)

    def test_non_config_dict_callable_detected(self) -> None:
        """Test detection of non-ConfigDict callable for model_config."""
        code = """
from pydantic import BaseModel

class User(BaseModel):
    model_config = SomeOtherCallable()
    name: str
"""
        path = write_temp_file(code)
        try:
            violations = validate_pydantic_patterns(path)
            assert len(violations) >= 1
            # Should detect invalid callable
            messages = [v.message for v in violations]
            assert any("SomeOtherCallable" in m or "ConfigDict" in m for m in messages)
        finally:
            cleanup_temp_file(path)

    def test_pydantic_config_dict_via_attribute(self) -> None:
        """Test ConfigDict via pydantic.ConfigDict attribute access."""
        code = """
import pydantic

class ModelUser(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(frozen=True)
    name: str
"""
        path = write_temp_file(code)
        try:
            violations = validate_pydantic_patterns(path)
            assert len(violations) == 0
        finally:
            cleanup_temp_file(path)

    def test_classvar_annotation_ignored(self) -> None:
        """Test that ClassVar annotations are ignored as fields."""
        code = """
from pydantic import BaseModel, ConfigDict
from typing import ClassVar

class ModelUser(BaseModel):
    model_config = ConfigDict(frozen=True)
    name: str
    _internal: ClassVar[int] = 0
"""
        path = write_temp_file(code)
        try:
            violations = validate_pydantic_patterns(path)
            assert len(violations) == 0
        finally:
            cleanup_temp_file(path)

    def test_circular_inheritance_handling(self) -> None:
        """Test handling of potentially circular inheritance chains."""
        # Note: This is contrived - Python wouldn't allow actual circular inheritance
        # but tests the resolution limit protection
        code = """
from pydantic import BaseModel, ConfigDict

class ModelA(BaseModel):
    model_config = ConfigDict(frozen=True)

class ModelB(ModelA):
    pass

class ModelC(ModelB):
    pass

class ModelD(ModelC):
    pass
"""
        path = write_temp_file(code)
        try:
            violations = validate_pydantic_patterns(path)
            # All classes should resolve properly
            assert len(violations) == 0
        finally:
            cleanup_temp_file(path)


class TestValidateNamingAdditional:
    """Additional tests for validate_naming.py edge cases."""

    def test_handles_syntax_error_file(self) -> None:
        """Test that files with syntax errors are handled gracefully."""
        code = """
class Incomplete(
"""
        path = write_temp_file(code)
        try:
            violations = validate_naming(path)
            assert violations == []
        finally:
            cleanup_temp_file(path)

    def test_subscript_base_class_handled(self) -> None:
        """Test handling of Generic[T] base class syntax."""
        code = """
from typing import Generic, TypeVar
from pydantic import BaseModel

T = TypeVar("T")

class ModelGeneric(BaseModel, Generic[T]):
    value: T
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            models_dir = Path(tmpdir) / "models"
            models_dir.mkdir()
            model_file = models_dir / "model_generic.py"
            model_file.write_text(code)
            violations = validate_naming(model_file)
            # Should not crash on subscript base classes
            # ModelGeneric follows ONEX naming
            class_violations = [v for v in violations if "ModelGeneric" in v.message]
            assert len(class_violations) == 0

    def test_node_naming_enforcement(self) -> None:
        """Test NodeXxx naming for node base classes."""
        code = '''
class MyEffectNode:
    """Effect node without proper naming."""
    pass

class NodeMyEffect:
    """Properly named node."""
    pass
'''
        with tempfile.TemporaryDirectory() as tmpdir:
            nodes_dir = Path(tmpdir) / "nodes"
            nodes_dir.mkdir()
            node_file = nodes_dir / "node_effect.py"
            node_file.write_text(code)
            violations = validate_naming(node_file)
            # MyEffectNode violates naming
            assert any("MyEffectNode" in v.message for v in violations)

    def test_service_naming_enforcement(self) -> None:
        """Test ServiceXxx naming for service classes."""
        code = """
class BaseService:
    pass

class UserManager(BaseService):
    pass
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            services_dir = Path(tmpdir) / "services"
            services_dir.mkdir()
            service_file = services_dir / "service_user.py"
            service_file.write_text(code)
            violations = validate_naming(service_file)
            # UserManager violates ServiceXxx naming
            assert any("UserManager" in v.message for v in violations)

    def test_handler_naming_enforcement(self) -> None:
        """Test HandlerXxx naming for handler classes."""
        code = """
class RequestHandler:
    pass

class UserRequestProcessor(RequestHandler):
    pass
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            handlers_dir = Path(tmpdir) / "handlers"
            handlers_dir.mkdir()
            handler_file = handlers_dir / "handler_user.py"
            handler_file.write_text(code)
            violations = validate_naming(handler_file)
            # UserRequestProcessor violates HandlerXxx naming
            assert any("UserRequestProcessor" in v.message for v in violations)

    def test_conftest_file_skipped(self) -> None:
        """Test that conftest.py files are skipped."""
        code = """
from pydantic import BaseModel

class TestFixture(BaseModel):
    pass
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            conftest_file = Path(tmpdir) / "conftest.py"
            conftest_file.write_text(code)
            violations = validate_naming(conftest_file)
            assert len(violations) == 0

    def test_compat_directory_skipped(self) -> None:
        """Test that compat/ directory is skipped."""
        code = """
class OldStyleClass:
    pass
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            compat_dir = Path(tmpdir) / "compat"
            compat_dir.mkdir()
            compat_file = compat_dir / "legacy.py"
            compat_file.write_text(code)
            violations = validate_naming(compat_file)
            assert len(violations) == 0


class TestValidateEnumCasingAdditional:
    """Additional tests for validate_enum_casing.py edge cases."""

    def test_handles_syntax_error_file(self) -> None:
        """Test that files with syntax errors are handled gracefully."""
        code = """
class BrokenEnum(Enum
"""
        path = write_temp_file(code)
        try:
            violations = validate_enum_casing(path)
            assert violations == []
        finally:
            cleanup_temp_file(path)

    def test_flag_enum_casing(self) -> None:
        """Test casing enforcement for Flag enums."""
        code = """
from enum import Flag

class Permissions(Flag):
    read = 1
    WRITE = 2
"""
        path = write_temp_file(code)
        try:
            violations = validate_enum_casing(path)
            # 'read' violates, 'WRITE' is fine
            assert len(violations) == 1
            assert "read" in violations[0].message
        finally:
            cleanup_temp_file(path)

    def test_enum_via_attribute_access(self) -> None:
        """Test enum detection via attribute access (enum.Enum)."""
        code = """
import enum

class Status(enum.Enum):
    active = "active"
"""
        path = write_temp_file(code)
        try:
            violations = validate_enum_casing(path)
            assert len(violations) == 1
        finally:
            cleanup_temp_file(path)


class TestValidateSingleClassAdditional:
    """Additional tests for validate_single_class_per_file.py edge cases."""

    def test_handles_syntax_error_file(self) -> None:
        """Test that files with syntax errors are handled gracefully."""
        code = """
class Incomplete(
"""
        path = write_temp_file(code)
        try:
            non_enum, non_enum_names, enum_count, enum_names = count_classes(path)
            assert non_enum == 0
            assert enum_count == 0
        finally:
            cleanup_temp_file(path)

    def test_data_models_py_exemption(self) -> None:
        """Test that data_models.py files are exempt."""
        code = """
class DataType1:
    pass

class DataType2:
    pass
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_file = Path(tmpdir) / "data_models.py"
            data_file.write_text(code)
            violations = validate_single_class(data_file)
            assert len(violations) == 0

    def test_error_models_py_exemption(self) -> None:
        """Test that error_models.py files are exempt."""
        code = """
class Error1:
    pass

class Error2:
    pass
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            error_file = Path(tmpdir) / "error_models.py"
            error_file.write_text(code)
            violations = validate_single_class(error_file)
            assert len(violations) == 0

    def test_nested_exempt_directory(self) -> None:
        """Test that nested exempt directories work."""
        code = """
class TypeA:
    pass

class TypeB:
    pass
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            nested_dir = Path(tmpdir) / "src" / "foundation" / "types"
            nested_dir.mkdir(parents=True)
            type_file = nested_dir / "primitives.py"
            type_file.write_text(code)
            violations = validate_single_class(type_file)
            # Should be exempt due to 'foundation' in path
            assert len(violations) == 0

    def test_str_enum_counted_as_enum(self) -> None:
        """Test that StrEnum classes are counted as enums."""
        code = """
from enum import StrEnum

class Status1(StrEnum):
    A = "a"

class Status2(StrEnum):
    B = "b"
"""
        path = write_temp_file(code)
        try:
            violations = validate_single_class(path)
            # Multiple StrEnums are allowed
            assert len(violations) == 0
        finally:
            cleanup_temp_file(path)


class TestValidateNoBackwardCompatAdditional:
    """Additional tests for validate_no_backward_compatibility.py edge cases."""

    def test_handles_missing_file(self) -> None:
        """Test that missing files are handled gracefully."""
        path = Path("/nonexistent/file.py")
        violations = validate_no_backward_compat(path)
        assert violations == []

    def test_delete_deprecated_todo(  # TODO_FORMAT_EXEMPT: test fixture
        self,
    ) -> None:
        """Test detection of TODO: delete deprecated."""
        code = """
# TODO: delete deprecated code
"""
        path = write_temp_file(code)
        try:
            violations = validate_no_backward_compat(path)
            assert len(violations) == 1
            assert "TODO" in violations[0].message
        finally:
            cleanup_temp_file(path)

    def test_deprecated_decorator_with_parens(self) -> None:
        """Test @deprecated() with parentheses."""
        code = """
@deprecated("Use new_function instead")
def old_function():
    pass
"""
        path = write_temp_file(code)
        try:
            violations = validate_no_backward_compat(path)
            assert len(violations) == 1
            assert "Deprecated decorator" in violations[0].message
        finally:
            cleanup_temp_file(path)

    def test_handles_unicode_error(self) -> None:
        """Test that files with encoding issues are handled gracefully."""
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".py", delete=False) as f:
            f.write(b"\xff\xfe")  # Invalid UTF-8 bytes
            path = Path(f.name)
        try:
            violations = validate_no_backward_compat(path)
            assert violations == []
        finally:
            path.unlink(missing_ok=True)


class TestValidateSingleClassMainAdditional:
    """Additional main() tests for validate_single_class_per_file.py."""

    def test_main_with_violations(self) -> None:
        """Test main() with files that have violations."""
        import subprocess

        code = """class A:
    pass

class B:
    pass

class C:
    pass
"""
        path = write_temp_file(code)
        try:
            result = subprocess.run(
                [
                    "python",
                    "scripts/validation/validate_single_class_per_file.py",
                    str(path),
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            assert result.returncode == 1
            assert "single-class-per-file" in result.stdout.lower()
        finally:
            cleanup_temp_file(path)

    def test_main_with_directory(self) -> None:
        """Test main() scanning a directory."""
        import subprocess

        code = """class SingleClass:
    pass
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "module.py"
            test_file.write_text(code)
            result = subprocess.run(
                [
                    "python",
                    "scripts/validation/validate_single_class_per_file.py",
                    tmpdir,
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            assert result.returncode == 0


class TestValidateEnumCasingMainAdditional:
    """Additional main() tests for validate_enum_casing.py."""

    def test_main_with_violations(self) -> None:
        """Test main() with enum casing violations."""
        import subprocess

        code = """from enum import Enum

class Status(Enum):
    active = "active"
    inactive = "inactive"
"""
        path = write_temp_file(code)
        try:
            result = subprocess.run(
                ["python", "scripts/validation/validate_enum_casing.py", str(path)],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            assert result.returncode == 1
            assert "enum casing" in result.stdout.lower()
        finally:
            cleanup_temp_file(path)

    def test_main_with_directory(self) -> None:
        """Test main() scanning a directory."""
        import subprocess

        code = """from enum import Enum

class Status(Enum):
    ACTIVE = "active"
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "enums.py"
            test_file.write_text(code)
            result = subprocess.run(
                ["python", "scripts/validation/validate_enum_casing.py", tmpdir],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            assert result.returncode == 0


class TestValidateNamingMainAdditional:
    """Additional main() tests for validate_naming.py."""

    def test_main_with_violations(self) -> None:
        """Test main() with naming violations."""
        import subprocess

        code = """from pydantic import BaseModel

class User(BaseModel):
    name: str
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            models_dir = Path(tmpdir) / "models"
            models_dir.mkdir()
            model_file = models_dir / "model_user.py"
            model_file.write_text(code)
            result = subprocess.run(
                ["python", "scripts/validation/validate_naming.py", tmpdir],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            assert result.returncode == 1
            assert "naming" in result.stdout.lower()


class TestValidatePydanticPatternsMainAdditional:
    """Additional main() tests for validate_pydantic_patterns.py."""

    def test_main_with_violations(self) -> None:
        """Test main() with Pydantic pattern violations."""
        import subprocess

        code = """from pydantic import BaseModel

class User(BaseModel):
    name: str
"""
        path = write_temp_file(code)
        try:
            result = subprocess.run(
                [
                    "python",
                    "scripts/validation/validate_pydantic_patterns.py",
                    str(path),
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            assert result.returncode == 1
            assert (
                "pydantic" in result.stdout.lower()
                or "model_config" in result.stdout.lower()
            )
        finally:
            cleanup_temp_file(path)

    def test_main_with_directory(self) -> None:
        """Test main() scanning a directory."""
        import subprocess

        code = """from pydantic import BaseModel, ConfigDict

class ModelUser(BaseModel):
    model_config = ConfigDict(frozen=True)
    name: str
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "models.py"
            test_file.write_text(code)
            result = subprocess.run(
                ["python", "scripts/validation/validate_pydantic_patterns.py", tmpdir],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            assert result.returncode == 0


# =============================================================================
# CLI MAIN FUNCTION TESTS
# =============================================================================


class TestValidateSecretsMain:
    """Tests for validate_secrets.py main() function."""

    def test_main_no_args_returns_error(self) -> None:
        """Test main() with no arguments returns error code."""
        import subprocess

        result = subprocess.run(
            ["python", "scripts/validation/validate_secrets.py"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 1
        assert "Usage:" in result.stdout

    def test_main_with_clean_file(self) -> None:
        """Test main() with a clean file returns success."""
        import subprocess

        code = """import os\napi_key = os.getenv("API_KEY")\n"""
        path = write_temp_file(code)
        try:
            result = subprocess.run(
                ["python", "scripts/validation/validate_secrets.py", str(path)],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            assert result.returncode == 0
        finally:
            cleanup_temp_file(path)

    def test_main_with_violation(self) -> None:
        """Test main() with violations returns error code."""
        import subprocess

        code = """api_key = "sk-1234567890abcdef"\n"""
        path = write_temp_file(code)
        try:
            result = subprocess.run(
                ["python", "scripts/validation/validate_secrets.py", str(path)],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            assert result.returncode == 1
            assert "potential secret" in result.stdout.lower()
        finally:
            cleanup_temp_file(path)

    def test_main_with_verbose_flag(self) -> None:
        """Test main() with --verbose flag."""
        import subprocess

        code = """api_key = os.getenv("API_KEY")\n"""
        path = write_temp_file(code)
        try:
            result = subprocess.run(
                [
                    "python",
                    "scripts/validation/validate_secrets.py",
                    "--verbose",
                    str(path),
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            assert result.returncode == 0
        finally:
            cleanup_temp_file(path)


class TestValidatePydanticPatternsMain:
    """Tests for validate_pydantic_patterns.py main() function."""

    def test_main_no_args_returns_error(self) -> None:
        """Test main() with no arguments returns error code."""
        import subprocess

        result = subprocess.run(
            ["python", "scripts/validation/validate_pydantic_patterns.py"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 1
        assert "Usage:" in result.stdout

    def test_main_with_clean_file(self) -> None:
        """Test main() with a valid Pydantic model."""
        import subprocess

        code = """from pydantic import BaseModel, ConfigDict

class ModelUser(BaseModel):
    model_config = ConfigDict(frozen=True)
    name: str
"""
        path = write_temp_file(code)
        try:
            result = subprocess.run(
                [
                    "python",
                    "scripts/validation/validate_pydantic_patterns.py",
                    str(path),
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            assert result.returncode == 0
        finally:
            cleanup_temp_file(path)


class TestValidateNamingMain:
    """Tests for validate_naming.py main() function."""

    def test_main_no_args_returns_error(self) -> None:
        """Test main() with no arguments returns error code."""
        import subprocess

        result = subprocess.run(
            ["python", "scripts/validation/validate_naming.py"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 1
        assert "Usage:" in result.stdout

    def test_main_with_invalid_directory(self) -> None:
        """Test main() with non-existent directory."""
        import subprocess

        result = subprocess.run(
            ["python", "scripts/validation/validate_naming.py", "/nonexistent/dir"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 1
        assert (
            "not found" in result.stdout.lower()
            or "not a directory" in result.stdout.lower()
        )


class TestValidateEnumCasingMain:
    """Tests for validate_enum_casing.py main() function."""

    def test_main_no_args_returns_error(self) -> None:
        """Test main() with no arguments returns error code."""
        import subprocess

        result = subprocess.run(
            ["python", "scripts/validation/validate_enum_casing.py"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 1
        assert "Usage:" in result.stdout

    def test_main_with_valid_file(self) -> None:
        """Test main() with valid enum."""
        import subprocess

        code = """from enum import Enum

class Status(Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
"""
        path = write_temp_file(code)
        try:
            result = subprocess.run(
                ["python", "scripts/validation/validate_enum_casing.py", str(path)],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            assert result.returncode == 0
        finally:
            cleanup_temp_file(path)


class TestValidateSingleClassMain:
    """Tests for validate_single_class_per_file.py main() function."""

    def test_main_no_args_returns_error(self) -> None:
        """Test main() with no arguments returns error code."""
        import subprocess

        result = subprocess.run(
            ["python", "scripts/validation/validate_single_class_per_file.py"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 1
        assert "Usage:" in result.stdout


class TestValidateNoBackwardCompatMain:
    """Tests for validate_no_backward_compatibility.py main() function."""

    def test_main_with_invalid_directory(self) -> None:
        """Test main() with non-existent directory."""
        import subprocess

        result = subprocess.run(
            [
                "python",
                "scripts/validation/validate_no_backward_compatibility.py",
                "-d",
                "/nonexistent",
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 1
        assert (
            "not found" in result.stdout.lower()
            or "not a directory" in result.stdout.lower()
        )

    def test_main_with_clean_directory(self) -> None:
        """Test main() with a clean temporary directory."""
        import subprocess

        code = """def modern_function():
    return 42
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "module.py"
            test_file.write_text(code)
            result = subprocess.run(
                [
                    "python",
                    "scripts/validation/validate_no_backward_compatibility.py",
                    "-d",
                    tmpdir,
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            assert result.returncode == 0


class TestValidateHttpImportsMain:
    """Tests for validate_http_imports.py main() function."""

    def test_main_with_clean_file(self) -> None:
        """Test main() with a clean file without HTTP imports."""
        import subprocess

        code = """from typing import Protocol

class HttpClientProtocol(Protocol):
    async def get(self, url: str) -> dict: ...
"""
        path = write_temp_file(code)
        try:
            result = subprocess.run(
                ["python", "scripts/validation/validate_http_imports.py", str(path)],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            assert result.returncode == 0
        finally:
            cleanup_temp_file(path)

    def test_main_with_violation(self) -> None:
        """Test main() with HTTP import violation."""
        import subprocess

        code = """import httpx"""
        path = write_temp_file(code)
        try:
            result = subprocess.run(
                ["python", "scripts/validation/validate_http_imports.py", str(path)],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            assert result.returncode == 1
            assert "http import boundary" in result.stdout.lower()
        finally:
            cleanup_temp_file(path)

    def test_main_with_exemption_annotation(self) -> None:
        """Test main() with exemption annotation passes."""
        import subprocess

        code = """import httpx  # omnimemory-http-exempt: Test utility"""
        path = write_temp_file(code)
        try:
            result = subprocess.run(
                ["python", "scripts/validation/validate_http_imports.py", str(path)],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            assert result.returncode == 0
        finally:
            cleanup_temp_file(path)

    def test_main_with_directory(self) -> None:
        """Test main() scanning a directory."""
        import subprocess

        code = """from typing import Dict

def process_data(data: Dict) -> None:
    pass
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "module.py"
            test_file.write_text(code)
            result = subprocess.run(
                ["python", "scripts/validation/validate_http_imports.py", tmpdir],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            assert result.returncode == 0


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestIntegration:
    """Integration tests for validation script combinations."""

    def test_compliant_pydantic_model_passes_all(self) -> None:
        """Test that a fully compliant Pydantic model passes all validators."""
        code = '''
from enum import Enum
from pydantic import BaseModel, ConfigDict, Field


class EnumStatus(Enum):
    """Status enum with UPPER_SNAKE_CASE members."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    IN_PROGRESS = "in_progress"


class ModelUser(BaseModel):
    """User model following ONEX conventions."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    name: str = Field(..., description="User's full name")
    email: str = Field(..., description="User's email address")
    status: EnumStatus = Field(default=EnumStatus.ACTIVE, description="User status")
'''
        with tempfile.TemporaryDirectory() as tmpdir:
            models_dir = Path(tmpdir) / "models"
            models_dir.mkdir()
            model_file = models_dir / "model_user.py"
            model_file.write_text(code)

            # Should pass all validators
            secret_violations = validate_secrets(model_file)
            pydantic_violations = validate_pydantic_patterns(model_file)
            enum_violations = validate_enum_casing(model_file)
            single_class_violations = validate_single_class(model_file)
            compat_violations = validate_no_backward_compat(model_file)

            # One non-enum class + one enum = allowed
            assert len(secret_violations) == 0
            assert len(pydantic_violations) == 0
            assert len(enum_violations) == 0
            assert len(single_class_violations) == 0
            assert len(compat_violations) == 0

    def test_non_compliant_model_caught_by_multiple_validators(self) -> None:
        """Test that non-compliant code is caught by appropriate validators."""
        code = """
from enum import Enum
from pydantic import BaseModel

# backward compat with old API
api_key = "sk-1234567890abcdef"

class Status(Enum):
    active = "active"
    inactive = "inactive"

class User(BaseModel):
    name: str
"""
        path = write_temp_file(code)
        try:
            secret_violations = validate_secrets(path)
            pydantic_violations = validate_pydantic_patterns(path)
            enum_violations = validate_enum_casing(path)
            compat_violations = validate_no_backward_compat(path)

            # Each type of violation should be caught
            assert len(secret_violations) >= 1  # Hardcoded API key
            assert len(pydantic_violations) >= 1  # Missing model_config
            assert len(enum_violations) >= 1  # Lowercase enum members
            assert len(compat_violations) >= 1  # Backward compat comment
        finally:
            cleanup_temp_file(path)
