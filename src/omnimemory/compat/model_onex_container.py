"""
ModelOnexContainer - compatibility stub.

This is a local implementation of ModelOnexContainer until
omnibase_core.core.model_onex_container is available.

Technical Debt Notes:
- This stub exists because omnibase_core.core.model_onex_container is not yet available
- Once omnibase_core provides this component, this stub should be removed
- The auto-injection of container parameter follows ONEX DI patterns
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any, TypeVar, cast

if TYPE_CHECKING:
    from collections.abc import Callable

T = TypeVar("T")


class ModelOnexContainer:
    """
    Simple dependency injection container for ONEX nodes.

    Provides singleton and transient registration patterns
    for service resolution with automatic container injection.

    The container automatically detects when a class constructor
    requires a 'container' parameter and injects itself.
    """

    def __init__(self) -> None:
        """Initialize the container with empty registries."""
        self._singletons: dict[type[Any], Any] = {}
        self._singleton_factories: dict[type[Any], Callable[..., Any]] = {}
        self._transient_factories: dict[type[Any], Callable[..., Any]] = {}

    def register_singleton(
        self,
        interface: type[T],
        implementation: type[T] | Callable[..., T],
    ) -> None:
        """
        Register a singleton service.

        The same instance will be returned for all resolve calls.

        Args:
            interface: The interface/protocol type to register
            implementation: The implementation class or factory function
        """
        # Both classes and factory functions are stored directly as factories
        self._singleton_factories[interface] = implementation

    def register_transient(
        self,
        interface: type[T],
        implementation: type[T] | Callable[..., T],
    ) -> None:
        """
        Register a transient service.

        A new instance will be created for each resolve call.

        Args:
            interface: The interface/protocol type to register
            implementation: The implementation class or factory function
        """
        # Both classes and factory functions are stored directly as factories
        self._transient_factories[interface] = implementation

    def _create_instance(self, factory: Callable[..., T]) -> T:
        """
        Create an instance from a factory, auto-injecting container if needed.

        This method inspects the factory's signature and automatically
        injects the container if the factory accepts a 'container' parameter.

        Args:
            factory: The factory function or class to instantiate

        Returns:
            The created instance
        """
        # Check if factory accepts a 'container' parameter
        try:
            sig = inspect.signature(factory)
            params = sig.parameters

            # Check for 'container' parameter
            if "container" in params:
                param = params["container"]
                # Only inject if positional/keyword param (not *args/**kwargs)
                pos_or_kw = inspect.Parameter.POSITIONAL_OR_KEYWORD
                kw_only = inspect.Parameter.KEYWORD_ONLY
                if param.kind in (pos_or_kw, kw_only):
                    return factory(container=self)
        except (ValueError, TypeError):
            # inspect.signature can fail for some built-in types
            pass

        # Default: call without arguments
        return factory()

    def resolve(self, interface: type[T]) -> T:
        """
        Resolve a registered service.

        Automatically injects the container if the service's constructor
        accepts a 'container' parameter.

        Args:
            interface: The interface/protocol type to resolve

        Returns:
            The resolved service instance

        Raises:
            KeyError: If the interface is not registered
        """
        # Check if we have a cached singleton
        if interface in self._singletons:
            return cast(T, self._singletons[interface])

        # Check if we have a singleton factory
        if interface in self._singleton_factories:
            factory = self._singleton_factories[interface]
            instance = self._create_instance(factory)
            self._singletons[interface] = instance
            return cast(T, instance)

        # Check if we have a transient factory
        if interface in self._transient_factories:
            factory = self._transient_factories[interface]
            return cast(T, self._create_instance(factory))

        raise KeyError(f"No registration found for {interface}")

    def is_registered(self, interface: type[Any]) -> bool:
        """
        Check if an interface is registered.

        Args:
            interface: The interface/protocol type to check

        Returns:
            True if the interface is registered
        """
        return (
            interface in self._singletons
            or interface in self._singleton_factories
            or interface in self._transient_factories
        )

    def clear(self) -> None:
        """Clear all registrations and cached instances."""
        self._singletons.clear()
        self._singleton_factories.clear()
        self._transient_factories.clear()


# Alias for upstream compatibility with omnibase_core
# omnibase_core uses ModelONEXContainer (uppercase ONEX)
ModelONEXContainer = ModelOnexContainer
