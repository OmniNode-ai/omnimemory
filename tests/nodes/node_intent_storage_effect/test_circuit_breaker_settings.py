# SPDX-FileCopyrightText: 2026 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Intent-storage circuit breaker settings cross the contract/handler boundary."""

from unittest.mock import patch

import pytest
from pydantic import ValidationError

from omnimemory.models.config.model_intent_storage_settings import (
    ModelIntentStorageSettings,
)
from omnimemory.models.utils.model_circuit_breaker_config import (
    ModelCircuitBreakerConfig,
)
from omnimemory.nodes.node_intent_storage_effect.adapters import (
    HandlerIntentStorageAdapter,
)

pytestmark = pytest.mark.unit


def test_contract_defaults_reach_handler() -> None:
    handler = HandlerIntentStorageAdapter()
    assert handler._circuit_breaker.failure_threshold == 5
    assert handler._circuit_breaker.recovery_timeout == 60
    assert handler._circuit_breaker.success_threshold == 3


def test_overlay_reaches_handler(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OMNIMEMORY_INTENT_STORAGE_RECOVERY_SECONDS", "17.5")
    monkeypatch.setenv("OMNIMEMORY_INTENT_STORAGE_FAILURE_THRESHOLD", "2")
    monkeypatch.setenv("OMNIMEMORY_INTENT_STORAGE_SUCCESS_THRESHOLD", "4")
    handler = HandlerIntentStorageAdapter()
    assert handler._circuit_breaker.recovery_timeout == 17.5
    assert handler._circuit_breaker.failure_threshold == 2
    assert handler._circuit_breaker.success_threshold == 4


@pytest.mark.parametrize("value", ["-1", "nan", "inf", "broken"])
def test_invalid_overlay_fails_at_binding(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("OMNIMEMORY_INTENT_STORAGE_RECOVERY_SECONDS", value)
    with pytest.raises(ValidationError):
        HandlerIntentStorageAdapter()


def test_injected_config_bypasses_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OMNIMEMORY_INTENT_STORAGE_RECOVERY_SECONDS", "invalid")
    config = ModelCircuitBreakerConfig(
        failure_threshold=2, recovery_timeout=7, success_threshold=4
    )
    handler = HandlerIntentStorageAdapter(circuit_breaker_config=config)
    assert handler._circuit_breaker.recovery_timeout == 7
    assert handler._circuit_breaker.failure_threshold == 2
    assert handler._circuit_breaker.success_threshold == 4


def test_settings_have_no_python_defaults() -> None:
    with pytest.raises(ValidationError):
        ModelIntentStorageSettings()


def test_missing_contract_field_fails() -> None:
    with (
        patch(
            "omnimemory.models.config.model_intent_storage_settings.yaml.safe_load",
            return_value={"descriptor": {}},
        ),
        pytest.raises(ValidationError),
    ):
        ModelIntentStorageSettings.from_node_contract()
