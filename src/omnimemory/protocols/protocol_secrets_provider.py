"""
Protocol for secrets resolution - allows env vars now, Vault later.

This module defines the ProtocolSecretsProvider interface that abstracts
secret retrieval from various backends. Phase 1 uses environment variables;
future phases will add HashiCorp Vault integration.

All secrets are returned as SecretStr to prevent accidental exposure in
logs, traces, or error messages.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from pydantic import SecretStr


@runtime_checkable
class ProtocolSecretsProvider(Protocol):
    """Interface for resolving secrets from various backends.

    This protocol defines the contract for secret retrieval that can be
    implemented by different backends:

    - Phase 1: EnvSecretsProvider (environment variables)
    - Future: VaultSecretsProvider (HashiCorp Vault)

    All methods are async to support backends that require network calls
    (like Vault) without breaking the interface.

    Security guarantees:
    - All returned values use SecretStr to prevent accidental logging
    - Implementations must never cache secrets in plain text
    - Error messages must not include secret values

    Example:
        async def get_db_connection(provider: ProtocolSecretsProvider) -> str:
            password = await provider.get_secret("DB_PASSWORD")
            # password.get_secret_value() to access actual value
            return f"postgresql://user:{password.get_secret_value()}@host/db"
    """

    async def get_secret(self, key: str) -> SecretStr:
        """Get a secret value by key.

        Args:
            key: The secret key/name to retrieve. The actual lookup
                 may include provider-specific transformations
                 (e.g., prefix addition).

        Returns:
            SecretStr containing the secret value. Use .get_secret_value()
            to access the underlying string.

        Raises:
            KeyError: If secret not found and no default available.
                      Error message should indicate the key name but
                      never include the secret value.
        """
        ...

    async def get_secret_or_default(self, key: str, default: str) -> SecretStr:
        """Get a secret value with fallback default.

        Use this method when a secret is optional or has a sensible default.
        Note that the default value will also be wrapped in SecretStr for
        consistent handling.

        Args:
            key: The secret key/name to retrieve.
            default: Default value if secret not found. This value will be
                     wrapped in SecretStr before being returned.

        Returns:
            SecretStr containing the secret or default value.
        """
        ...

    async def has_secret(self, key: str) -> bool:
        """Check if a secret exists without retrieving it.

        This method allows checking for secret existence without triggering
        potential side effects (like Vault lease creation) that get_secret
        might cause.

        Args:
            key: The secret key/name to check.

        Returns:
            True if secret exists and is accessible, False otherwise.
        """
        ...
