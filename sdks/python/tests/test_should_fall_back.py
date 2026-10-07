"""Tests for terminal policy errors that must not fail-open."""

from tensorcost._errors import (
    TensorCostGuardrailHardStopError,
    TensorCostModelGovernanceDeniedError,
    TensorCostPeriodBudgetExceededError,
    TensorCostProxyError,
    TensorCostRunBudgetExceededError,
)
from tensorcost._proxy_client import should_fall_back_to_provider


def test_run_budget_refusal_does_not_fall_back():
    assert should_fall_back_to_provider(TensorCostRunBudgetExceededError("cap")) is False


def test_period_budget_refusal_does_not_fall_back():
    assert (
        should_fall_back_to_provider(TensorCostPeriodBudgetExceededError("cap"))
        is False
    )


def test_model_governance_denial_does_not_fall_back():
    assert (
        should_fall_back_to_provider(TensorCostModelGovernanceDeniedError("denied"))
        is False
    )


def test_guardrail_hard_stop_does_not_fall_back():
    assert (
        should_fall_back_to_provider(TensorCostGuardrailHardStopError("stop"))
        is False
    )


def test_proxy_error_falls_back():
    assert should_fall_back_to_provider(TensorCostProxyError("503")) is True
