# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Environment-based secrets provider implementation.

This module provides the EnvSecretsProvider class that implements
ProtocolSecretsProvider using environment variables as the backend.

All secrets are returned as SecretStr to prevent accidental exposure in
logs, traces, or error messages.

Example:
    from omnimemory.secrets import EnvSecretsProvider

    provider = EnvSecretsProvider(prefix="MYAPP_")
    password = await provider.get_secret("DB_PASSWORD")
    # Looks up MYAPP_DB_PASSWORD in environment
"""

from __future__ import annotations

import os

from pydantic import SecretStr


class EnvSecretsProvider:
    """Secrets provider that reads from environment variables.

    This implementation provides a simple way to manage secrets during
    development and in containerized deployments where secrets are
    typically injected as environment variables.

    Attributes:
        prefix: Optional prefix added to all secret keys when looking up
                environment variables. Useful for namespace isolation.

    Example:
        # Without prefix - looks up MY_SECRET directly
        provider = EnvSecretsProvider()
        secret = await provider.get_secret("MY_SECRET")

        # With prefix - looks up APP_MY_SECRET
        provider = EnvSecretsProvider(prefix="APP_")
        secret = await provider.get_secret("MY_SECRET")
    """

    def __init__(self, prefix: str = "") -> None:
        """Initialize the environment secrets provider.

        Args:
            prefix: Optional prefix to prepend to all secret key lookups.
                   Defaults to empty string (no prefix).
        """
        self._prefix = prefix

    def _resolve_key(self, key: str) -> str:
        """Resolve the full environment variable name.

        Args:
            key: The secret key name.

        Returns:
            The full environment variable name with prefix applied.
        """
        return f"{self._prefix}{key}"

    async def get_secret(self, key: str) -> SecretStr:
        """Get a secret value by key from environment variables.

        Args:
            key: The secret key/name to retrieve. The prefix (if configured)
                 will be prepended to form the full environment variable name.

        Returns:
            SecretStr containing the secret value.

        Raises:
            KeyError: If the environment variable is not set.
        """
        env_key = self._resolve_key(key)
        value = os.environ.get(env_key)
        if value is None:
            raise KeyError(f"Secret not found: {key} (env var: {env_key})")
        return SecretStr(value)

    async def get_secret_or_default(self, key: str, default: str) -> SecretStr:
        """Get a secret value with fallback default.

        Args:
            key: The secret key/name to retrieve.
            default: Default value if the environment variable is not set.

        Returns:
            SecretStr containing the secret or default value.
        """
        env_key = self._resolve_key(key)
        value = os.environ.get(env_key, default)
        return SecretStr(value)

    async def has_secret(self, key: str) -> bool:
        """Check if a secret exists in environment variables.

        Args:
            key: The secret key/name to check.

        Returns:
            True if the environment variable is set, False otherwise.
        """
        env_key = self._resolve_key(key)
        return env_key in os.environ


__all__ = ["EnvSecretsProvider"]
