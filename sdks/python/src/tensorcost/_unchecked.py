"""Calls that went to the provider without TensorCost's check.

Fail-open lets a call through when TensorCost cannot decide on it: /admit or
the proxy is down, the circuit is open, or TensorCost is pacing this process
(a bare 429) or the plan's ceiling is reached (a bare 402). Each such call
carries ``decision_reason`` on its observation, so the server can count them.
This module is the in-process side: a count per reason since the process
started, readable with ``unchecked_calls()``, and a warning per reason at most
once a minute, so it is never silent and never floods. Mirrors the Node SDK's
unchecked.ts.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Dict, Literal, Optional

from ._telemetry import LifecycleEventCallback, OnFallbackEvent, emit_event

_log = logging.getLogger("tensorcost")

UncheckedReason = Literal[
    "admit_unavailable",
    "admit_throttled",
    "admit_plan_limit",
    "proxy_unavailable",
    "proxy_throttled",
    "proxy_plan_limit",
    "circuit_open",
]

REASONS: tuple = (
    "admit_unavailable",
    "admit_throttled",
    "admit_plan_limit",
    "proxy_unavailable",
    "proxy_throttled",
    "proxy_plan_limit",
    "circuit_open",
)

WARN_INTERVAL_S = 60.0

_WHY = {
    "admit_unavailable": "TensorCost's admission check could not be reached",
    "admit_throttled": "TensorCost's admission check is rate limiting this process (429)",
    "admit_plan_limit": "TensorCost's admission check says the plan's request ceiling is reached (402)",
    "proxy_unavailable": "the TensorCost proxy could not be reached",
    "proxy_throttled": "the TensorCost proxy is rate limiting this process (429)",
    "proxy_plan_limit": "the TensorCost proxy says the plan's request ceiling is reached (402)",
    "circuit_open": "the TensorCost proxy circuit is open",
}

_lock = threading.Lock()
_counts: Dict[str, int] = {}
_since_warn: Dict[str, int] = {}
_last_warn: Dict[str, float] = {}


def record_unchecked(reason: str) -> None:
    """Count one call that went ahead unchecked; warn at most once a minute per reason."""
    with _lock:
        _counts[reason] = _counts.get(reason, 0) + 1
        _since_warn[reason] = _since_warn.get(reason, 0) + 1
        now = time.monotonic()
        last = _last_warn.get(reason)
        if last is not None and now - last < WARN_INTERVAL_S:
            return
        n = _since_warn[reason]
        _last_warn[reason] = now
        _since_warn[reason] = 0
    window = "" if last is None else " since the last warning"
    _log.warning(
        "tensorcost: %s; %d call(s)%s went to the provider without a budget or "
        "governance check (fail-open). Set fail_open_enabled=False to refuse them "
        "instead. unchecked_calls() has the totals.",
        _WHY.get(reason, reason),
        n,
        window,
    )


def unchecked_calls() -> Dict[str, int]:
    """Calls this process sent to the provider without TensorCost's check, by reason."""
    with _lock:
        return {r: _counts.get(r, 0) for r in REASONS}


def unchecked_metadata(reason: str) -> dict:
    """Observation metadata for a call that went ahead unchecked."""
    return {"decision_action": "allow", "decision_reason": reason}


def going_unchecked(
    reason: str,
    *,
    on_lifecycle_event: Optional[LifecycleEventCallback],
    provider: str,
    model: str,
    operation: str,
    consecutive_failures: int = 0,
) -> dict:
    """Count, warn, raise ``on_fallback`` with the reason, and return the
    metadata the call's observation carries."""
    record_unchecked(reason)
    emit_event(
        on_lifecycle_event,
        OnFallbackEvent(
            kind="on_fallback",
            provider=provider,  # type: ignore[arg-type]
            model=model,
            operation=operation,
            attempt_number=1,
            elapsed_ms=0.0,
            consecutive_failures=consecutive_failures,
            reason=reason,  # type: ignore[arg-type]
        ),
    )
    return unchecked_metadata(reason)


def proxy_fallback_reason(exc: Optional[BaseException]) -> str:
    """Why the proxy path fell back. ``None``: the circuit was open."""
    from ._errors import TensorCostQuotaError

    if exc is None:
        return "circuit_open"
    if isinstance(exc, TensorCostQuotaError):
        return "proxy_plan_limit" if exc.status == 402 else "proxy_throttled"
    return "proxy_unavailable"


def admit_error_reason(exc: Optional[BaseException]) -> str:
    """Why the token exchange in front of /admit left the call undecided."""
    from ._errors import TensorCostQuotaError

    if isinstance(exc, TensorCostQuotaError):
        return "admit_plan_limit" if exc.status == 402 else "admit_throttled"
    return "admit_unavailable"


def _reset() -> None:
    with _lock:
        _counts.clear()
        _since_warn.clear()
        _last_warn.clear()
