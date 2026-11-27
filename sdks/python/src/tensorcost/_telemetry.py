"""Telemetry hook types for the TensorCost Python SDK.

Customers pass ``on_lifecycle_event`` in ``wrap()`` to receive typed events at
each stage of a request lifecycle.  The default is no-op — not providing the
callback adds zero overhead to the hot path.

What is included in every event:

* ``provider``       — ``"openai"`` or ``"anthropic"``
* ``model``          — as passed to the create() call
* ``operation``      — e.g. ``"chat.completions"`` or ``"messages"``
* ``attempt_number`` — 1-based; increments on each retry
* ``elapsed_ms``     — wall-clock ms since the first attempt for this call

What is never included:

* Prompt / completion content
* Request or response bodies
* Credentials of any kind

The event kinds::

    before_request   fires before each attempt (including retried ones)
    after_response   fires when a request completes successfully
    on_retry         fires when a request is going to be retried
    on_error         fires when the SDK gives up (retries exhausted or
                     a non-retriable error occurred)
    on_fallback      fires whenever a steer/route client sends a call straight
                     to the provider because TensorCost is unavailable.
                     ``reason`` says which way:
                       proxy_unavailable  the proxy call failed (network,
                                          timeout, 5xx, 429/402)
                       circuit_open       the circuit breaker is open, the
                                          proxy was not tried
                       proxy_throttled    the proxy still answered a bare 429
                                          after its retries (1.9.0)
                       proxy_plan_limit   the proxy answered a bare 402
                       layer_decayed      the published layer couldn't be
                                          fetched, so the client decayed to
                                          govern and skipped the proxy
                     and, since 1.9.0, when a govern call goes ahead without
                     an /admit answer: admit_unavailable (outage),
                     admit_throttled (still a bare 429 when the admit budget
                     ran out), admit_plan_limit (a bare 402). The same value
                     is the call's observation ``decision_reason``.
                     Never for a refusal: those raise and never fall back.
    on_budget_would_refuse
                     fires when /admit let a call through but named budgets
                     that would have refused it, had the SDK enforced them
                     (the licence lacks feature.enforce_sdk). The call is
                     not affected.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal, Optional, Union


@dataclass(frozen=True)
class BeforeRequestEvent:
    kind: Literal["before_request"]
    provider: str
    model: str
    operation: str
    attempt_number: int
    elapsed_ms: float


@dataclass(frozen=True)
class AfterResponseEvent:
    kind: Literal["after_response"]
    provider: str
    model: str
    operation: str
    attempt_number: int
    elapsed_ms: float
    status: int
    request_id: Optional[str]


@dataclass(frozen=True)
class OnRetryEvent:
    kind: Literal["on_retry"]
    provider: str
    model: str
    operation: str
    attempt_number: int
    elapsed_ms: float
    reason: Literal["5xx", "network_error", "quota_429"]
    status: Optional[int]
    delay_ms: float


@dataclass(frozen=True)
class OnErrorEvent:
    kind: Literal["on_error"]
    provider: str
    model: str
    operation: str
    attempt_number: int
    elapsed_ms: float
    error_class: str
    status: Optional[int]
    #: Set when the status alone doesn't say why. ``provider_unreachable``:
    #: the proxy answered 502 because it could not connect to the provider,
    #: so nothing was sent and the wrapper may call the provider directly.
    reason: Optional[Literal["provider_unreachable"]] = None


@dataclass(frozen=True)
class OnFallbackEvent:
    kind: Literal["on_fallback"]
    provider: str
    model: str
    operation: str
    attempt_number: int
    elapsed_ms: float
    consecutive_failures: int
    reason: Literal[
        "proxy_unavailable",
        "proxy_throttled",
        "proxy_plan_limit",
        "circuit_open",
        "layer_decayed",
        "admit_unavailable",
        "admit_throttled",
        "admit_plan_limit",
    ] = "proxy_unavailable"


@dataclass(frozen=True)
class OnBudgetWouldRefuseEvent:
    kind: Literal["on_budget_would_refuse"]
    provider: str
    model: str
    operation: str
    attempt_number: int
    elapsed_ms: float
    budget_ids: tuple[str, ...]


LifecycleEvent = Union[
    BeforeRequestEvent,
    AfterResponseEvent,
    OnRetryEvent,
    OnErrorEvent,
    OnFallbackEvent,
    OnBudgetWouldRefuseEvent,
]

LifecycleEventCallback = Callable[[LifecycleEvent], None]


def emit_event(
    callback: Optional[LifecycleEventCallback],
    event: LifecycleEvent,
) -> None:
    """Invoke *callback* with *event*, swallowing any exception it raises.

    Customer callbacks must never blow up the SDK's control flow.
    """
    if callback is None:
        return
    try:
        callback(event)
    except Exception:  # noqa: BLE001
        pass
