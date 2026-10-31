# SPDX-FileCopyrightText: 2026 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The retrieval handler modules cleared of bare module functions stay clear.

The handler-shape validator of omnibase_core reports ``module_function_handler``
for a handler module that holds module functions and no handler class. Each
module listed here has been turned into a handler class with a typed ``handle``
and must stay one. The validator is run on each module as it stands, so the rule
and its patterns stay the validator's own.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from omnibase_core.validation.validator_handler_shape import ValidatorHandlerShape

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[3]
_HANDLERS = "src/omnimemory/nodes/node_memory_retrieval_effect/handlers"

_CLEARED_MODULES: tuple[str, ...] = (
    f"{_HANDLERS}/handler_decay.py",
    f"{_HANDLERS}/handler_fusion.py",
)


@pytest.mark.parametrize("module", _CLEARED_MODULES)
def test_the_handler_module_holds_a_handler_class_not_bare_functions(
    module: str,
) -> None:
    path = _ROOT / module
    assert path.is_file()
    result = ValidatorHandlerShape().validate_file(path)
    assert [i.message for i in result.issues] == []
