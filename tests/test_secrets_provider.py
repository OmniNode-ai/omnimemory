# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Tests for secrets provider.

These tests validate that the secrets provider:
- Implements ProtocolSecretsProvider correctly
- Returns SecretStr for all secret values
- Supports environment variable prefix for namespace isolation
- Raises KeyError for missing secrets
- Provides has_secret() for existence checking
- Provides get_secret_or_default() for optional secrets
"""

from __future__ import annotations

import pytest
from pydantic import SecretStr

from omnimemory.protocols import ProtocolSecretsProvider
from omnimemory.secrets import EnvSecretsProvider


class TestEnvSecretsProviderProtocol:
    """Tests for protocol compliance."""

    @pytest.mark.asyncio
    async def test_implements_protocol(self) -> None:
        """Test EnvSecretsProvider implements ProtocolSecretsProvider."""
        provider = EnvSecretsProvider()
        # runtime_checkable protocol check
        assert isinstance(provider, ProtocolSecretsProvider)

    @pytest.mark.asyncio
    async def test_has_required_methods(self) -> None:
        """Test EnvSecretsProvider has all required protocol methods."""
        provider = EnvSecretsProvider()
        assert hasattr(provider, "get_secret")
        assert hasattr(provider, "get_secret_or_default")
        assert hasattr(provider, "has_secret")
        assert callable(provider.get_secret)
        assert callable(provider.get_secret_or_default)
        assert callable(provider.has_secret)


class TestGetSecret:
    """Tests for get_secret() method."""

    @pytest.mark.asyncio
    async def test_get_secret_returns_secret_str(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test get_secret returns SecretStr type."""
        monkeypatch.setenv("TEST_SECRET_VALUE", "my-secret-value")

        provider = EnvSecretsProvider()
        secret = await provider.get_secret("TEST_SECRET_VALUE")

        assert isinstance(secret, SecretStr)
        assert secret.get_secret_value() == "my-secret-value"

    @pytest.mark.asyncio
    async def test_get_secret_with_prefix(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test get_secret with prefix prepends prefix to key."""
        monkeypatch.setenv("OMNIMEMORY_DB_PASSWORD", "db-secret")

        provider = EnvSecretsProvider(prefix="OMNIMEMORY_")
        secret = await provider.get_secret("DB_PASSWORD")

        assert isinstance(secret, SecretStr)
        assert secret.get_secret_value() == "db-secret"

    @pytest.mark.asyncio
    async def test_get_secret_raises_key_error_for_missing(self) -> None:
        """Test get_secret raises KeyError for missing secret."""
        provider = EnvSecretsProvider()

        # Use a unique key that definitely doesn't exist
        with pytest.raises(KeyError) as exc_info:
            await provider.get_secret("NONEXISTENT_SECRET_12345_XYZ")

        # Error message should include the key name
        assert "NONEXISTENT_SECRET_12345_XYZ" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_secret_key_error_includes_prefixed_name(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test KeyError message includes full prefixed key name."""
        provider = EnvSecretsProvider(prefix="MY_PREFIX_")

        with pytest.raises(KeyError) as exc_info:
            await provider.get_secret("MISSING_KEY")

        # Should show the full expected env var name
        assert "MY_PREFIX_MISSING_KEY" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_secret_empty_value_is_valid(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test get_secret accepts empty string as valid value."""
        monkeypatch.setenv("EMPTY_SECRET", "")

        provider = EnvSecretsProvider()
        secret = await provider.get_secret("EMPTY_SECRET")

        assert isinstance(secret, SecretStr)
        assert secret.get_secret_value() == ""


class TestGetSecretOrDefault:
    """Tests for get_secret_or_default() method."""

    @pytest.mark.asyncio
    async def test_returns_env_value_when_exists(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test returns environment value when it exists."""
        monkeypatch.setenv("EXISTING_SECRET", "env-value")

        provider = EnvSecretsProvider()
        secret = await provider.get_secret_or_default("EXISTING_SECRET", "default")

        assert isinstance(secret, SecretStr)
        assert secret.get_secret_value() == "env-value"

    @pytest.mark.asyncio
    async def test_returns_default_when_missing(self) -> None:
        """Test returns default when environment variable is missing."""
        provider = EnvSecretsProvider()
        secret = await provider.get_secret_or_default(
            "MISSING_SECRET_ABC123", "default-value"
        )

        assert isinstance(secret, SecretStr)
        assert secret.get_secret_value() == "default-value"

    @pytest.mark.asyncio
    async def test_default_wrapped_in_secret_str(self) -> None:
        """Test default value is wrapped in SecretStr."""
        provider = EnvSecretsProvider()
        secret = await provider.get_secret_or_default(
            "MISSING_SECRET_DEF456", "my-default"
        )

        # Even the default should be a SecretStr
        assert isinstance(secret, SecretStr)
        # And the default should not appear in repr
        assert "my-default" not in repr(secret)

    @pytest.mark.asyncio
    async def test_with_prefix(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test get_secret_or_default with prefix."""
        monkeypatch.setenv("APP_API_KEY", "prefixed-key")

        provider = EnvSecretsProvider(prefix="APP_")
        secret = await provider.get_secret_or_default("API_KEY", "default")

        assert secret.get_secret_value() == "prefixed-key"

    @pytest.mark.asyncio
    async def test_with_prefix_falls_back_to_default(self) -> None:
        """Test get_secret_or_default with prefix falls back to default."""
        provider = EnvSecretsProvider(prefix="NONEXISTENT_PREFIX_")
        secret = await provider.get_secret_or_default("SOME_KEY", "fallback")

        assert secret.get_secret_value() == "fallback"


class TestHasSecret:
    """Tests for has_secret() method."""

    @pytest.mark.asyncio
    async def test_returns_true_when_exists(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test has_secret returns True when env var exists."""
        monkeypatch.setenv("EXISTS_SECRET_CHECK", "value")

        provider = EnvSecretsProvider()
        result = await provider.has_secret("EXISTS_SECRET_CHECK")

        assert result is True

    @pytest.mark.asyncio
    async def test_returns_false_when_missing(self) -> None:
        """Test has_secret returns False when env var is missing."""
        provider = EnvSecretsProvider()
        result = await provider.has_secret("DOES_NOT_EXIST_SECRET_99999")

        assert result is False

    @pytest.mark.asyncio
    async def test_with_prefix(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test has_secret respects prefix."""
        monkeypatch.setenv("PREFIX_MY_SECRET", "value")

        provider = EnvSecretsProvider(prefix="PREFIX_")

        # Key with prefix exists
        assert await provider.has_secret("MY_SECRET") is True
        # Key without prefix doesn't exist
        assert await provider.has_secret("OTHER_SECRET") is False

    @pytest.mark.asyncio
    async def test_empty_value_counts_as_exists(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test has_secret returns True even for empty string value."""
        monkeypatch.setenv("EMPTY_VALUE_SECRET", "")

        provider = EnvSecretsProvider()
        result = await provider.has_secret("EMPTY_VALUE_SECRET")

        assert result is True


class TestPrefix:
    """Tests for prefix functionality."""

    @pytest.mark.asyncio
    async def test_no_prefix_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test no prefix is applied by default."""
        monkeypatch.setenv("DIRECT_KEY", "direct-value")

        provider = EnvSecretsProvider()  # No prefix
        secret = await provider.get_secret("DIRECT_KEY")

        assert secret.get_secret_value() == "direct-value"

    @pytest.mark.asyncio
    async def test_empty_prefix_same_as_no_prefix(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test empty string prefix behaves same as no prefix."""
        monkeypatch.setenv("SOME_KEY", "some-value")

        provider = EnvSecretsProvider(prefix="")
        secret = await provider.get_secret("SOME_KEY")

        assert secret.get_secret_value() == "some-value"

    @pytest.mark.asyncio
    async def test_prefix_with_trailing_underscore(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test prefix with trailing underscore."""
        monkeypatch.setenv("OMNIMEMORY_SECRET_KEY", "secret-value")

        provider = EnvSecretsProvider(prefix="OMNIMEMORY_")
        secret = await provider.get_secret("SECRET_KEY")

        assert secret.get_secret_value() == "secret-value"

    @pytest.mark.asyncio
    async def test_prefix_without_trailing_underscore(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test prefix without trailing underscore."""
        monkeypatch.setenv("APPKEY", "value")

        provider = EnvSecretsProvider(prefix="APP")
        secret = await provider.get_secret("KEY")

        assert secret.get_secret_value() == "value"

    @pytest.mark.asyncio
    async def test_different_providers_different_prefixes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test multiple providers with different prefixes work independently."""
        monkeypatch.setenv("SERVICE_A_PASSWORD", "password-a")
        monkeypatch.setenv("SERVICE_B_PASSWORD", "password-b")

        provider_a = EnvSecretsProvider(prefix="SERVICE_A_")
        provider_b = EnvSecretsProvider(prefix="SERVICE_B_")

        secret_a = await provider_a.get_secret("PASSWORD")
        secret_b = await provider_b.get_secret("PASSWORD")

        assert secret_a.get_secret_value() == "password-a"
        assert secret_b.get_secret_value() == "password-b"


class TestSecretStrSecurity:
    """Tests for SecretStr security properties."""

    @pytest.mark.asyncio
    async def test_secret_not_in_str_repr(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test secret value does not appear in str() or repr()."""
        monkeypatch.setenv("SENSITIVE_SECRET", "super-sensitive-password-123")

        provider = EnvSecretsProvider()
        secret = await provider.get_secret("SENSITIVE_SECRET")

        assert "super-sensitive-password-123" not in str(secret)
        assert "super-sensitive-password-123" not in repr(secret)

    @pytest.mark.asyncio
    async def test_get_secret_value_required_to_access(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test get_secret_value() is required to access actual value."""
        monkeypatch.setenv("PROTECTED_SECRET", "protected-value")

        provider = EnvSecretsProvider()
        secret = await provider.get_secret("PROTECTED_SECRET")

        # Cannot access without get_secret_value()
        with pytest.raises(AttributeError):
            _ = secret.value  # SecretStr doesn't have .value attribute

        # Must use get_secret_value()
        assert secret.get_secret_value() == "protected-value"

    @pytest.mark.asyncio
    async def test_secret_masked_by_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test SecretStr shows masked value by default."""
        monkeypatch.setenv("MASKED_SECRET", "my-actual-secret")

        provider = EnvSecretsProvider()
        secret = await provider.get_secret("MASKED_SECRET")

        # str() should show masked placeholder
        str_repr = str(secret)
        assert "**" in str_repr or "***" in str_repr or "secret" in str_repr.lower()
        assert "my-actual-secret" not in str_repr
