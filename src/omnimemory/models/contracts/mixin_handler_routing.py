# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Mixin for handler routing support in extended contracts.

Provides the handler_routing field extension for ONEX contracts.
Temporary workaround until OMN-1588 adds native support to omnibase_core.

Note: This mixin provides only the field definition. Each extended class
must define its own model_config with extra="ignore" to properly override
the base class configuration, as Pydantic's config merging only works
with BaseModel subclasses.

.. versionadded:: 0.1.0
    Temporary workaround for OMN-1588.
"""

from __future__ import annotations

from pydantic import Field

from omnimemory.models.contracts.model_handler_routing import (
    ModelHandlerRouting,  # noqa: TC001
)

__all__ = ["MixinHandlerRouting"]


class MixinHandlerRouting:
    """Mixin adding handler_routing field to ONEX contract models.

    Pydantic Mixin Pattern Explanation
    ----------------------------------
    This class uses an unconventional but valid pattern: defining a Pydantic
    ``Field()`` annotation on a plain Python class (not a BaseModel subclass).

    **Why this works:**

    When this mixin is combined with a BaseModel subclass via multiple inheritance,
    Pydantic's metaclass (``ModelMetaclass``) inspects all classes in the MRO
    (Method Resolution Order) during model construction. It collects field
    definitions from:

    1. Class annotations (``__annotations__``)
    2. Default values that are ``FieldInfo`` instances (created by ``Field()``)

    Because Python's MRO includes this mixin class, Pydantic discovers the
    ``handler_routing`` annotation and its ``Field()`` default, treating it as
    a proper model field in the final combined class.

    **Why use this pattern:**

    - **DRY (Don't Repeat Yourself)**: Defines the field once, reuses across
      multiple extended contract models (Effect, Compute, Reducer, Orchestrator)
    - **Consistent field definition**: All extended contracts get identical
      field configuration (type, default, description)
    - **Clean inheritance**: Avoids diamond inheritance issues that would arise
      from creating an intermediate BaseModel subclass

    **Important caveats:**

    - This mixin provides only the field definition, not ``model_config``
    - Each consuming class must define its own ``model_config`` with
      ``extra="ignore"`` to properly handle unknown fields from YAML
    - The mixin must appear before the BaseModel subclass in the inheritance
      list to ensure correct MRO

    Usage::

        class ModelContractEffectExtended(MixinHandlerRouting, ModelContractEffect):
            model_config = ConfigDict(extra="ignore", ...)

    See Also
    --------
    - Pydantic documentation on model inheritance
    - Python MRO and ``__mro__`` attribute
    - OMN-1588 for native omnibase_core support (renders this mixin obsolete)
    """

    handler_routing: ModelHandlerRouting | None = Field(
        default=None,
        description="Handler routing configuration for declarative dispatch",
    )
