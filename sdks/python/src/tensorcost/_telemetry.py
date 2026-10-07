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

The five event kinds::

    before_request   fires before each attempt (including retried ones)
    after_response   fires when a request completes successfully
    on_retry         fires when a request is going to be retried
    on_error         fires when the SDK gives up (retries exhausted or
                     a non-retriable error occurred)
    on_fallback      fires when the circuit breaker opens and the SDK routes
                     directly to the provider instead of the proxy
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


@dataclass(frozen=True)
class OnFallbackEvent:
    kind: Literal["on_fallback"]
    provider: str
    model: str
    operation: str
    attempt_number: int
    elapsed_ms: float
    consecutive_failures: int


LifecycleEvent = Union[
    BeforeRequestEvent,
    AfterResponseEvent,
    OnRetryEvent,
    OnErrorEvent,
    OnFallbackEvent,
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
