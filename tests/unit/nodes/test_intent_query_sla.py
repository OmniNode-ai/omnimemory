# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The intent query SLA is the node contract's value, held by the config model."""

from __future__ import annotations

import importlib.resources

import pytest
import yaml
from omnibase_core.container import ModelONEXContainer

from omnimemory.nodes.node_intent_query_effect.handlers.handler_intent_query import (
    HandlerIntentQuery,
)
from omnimemory.nodes.node_intent_query_effect.models import (
    ModelHandlerIntentQueryConfig,
)


def _contract_sla_ms() -> float:
    contract = yaml.safe_load(
        importlib.resources.files("omnimemory.nodes.node_intent_query_effect")
        .joinpath("contract.yaml")
        .read_text(encoding="utf-8")
    )
    return float(contract["performance"]["max_response_time_ms"])


@pytest.mark.unit
class TestIntentQuerySla:
    def test_config_default_equals_the_contract_value(self) -> None:
        assert (
            ModelHandlerIntentQueryConfig().max_response_time_ms == _contract_sla_ms()
        )

    def test_config_default_keeps_its_previous_value(self) -> None:
        assert ModelHandlerIntentQueryConfig().max_response_time_ms == 100.0

    def test_handler_starts_from_the_config_default(self) -> None:
        handler = HandlerIntentQuery(ModelONEXContainer())
        assert handler._max_response_time_ms == _contract_sla_ms()

    def test_non_positive_sla_is_refused(self) -> None:
        with pytest.raises(ValueError):
            ModelHandlerIntentQueryConfig(max_response_time_ms=0.0)
