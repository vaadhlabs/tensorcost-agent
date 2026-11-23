"""Budgets that would have refused a call the SDK let through.

/v1/admit answers 204 with ``X-TensorCost-Budget-Would-Refuse: <ids>`` when
a team or agent is over a hard cap that isn't enforced at the SDK (the
licence lacks feature.enforce_sdk). Warn-only budgets are part of the base
product, so the app hears about it twice: the ``on_budget_would_refuse``
lifecycle event on every such call, and one log warning per budget id at
most once a minute, so a hot loop can't flood the log.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

from ._telemetry import LifecycleEventCallback, OnBudgetWouldRefuseEvent, emit_event

_log = logging.getLogger("tensorcost")

WARN_INTERVAL_S = 60.0
# Ids remembered for rate limiting; past this the oldest are forgotten.
_MAX_TRACKED = 1024

_last_warned: dict[str, float] = {}
_lock = threading.Lock()


def _reset() -> None:
    with _lock:
        _last_warned.clear()


def report_would_refuse(
    callback: Optional[LifecycleEventCallback],
    budget_ids: tuple[str, ...],
    *,
    provider: str,
    model: str,
    operation: str,
) -> None:
    if not budget_ids:
        return
    emit_event(
        callback,
        OnBudgetWouldRefuseEvent(
            kind="on_budget_would_refuse",
            provider=provider,
            model=model,
            operation=operation,
            attempt_number=1,
            elapsed_ms=0.0,
            budget_ids=budget_ids,
        ),
    )
    now = time.monotonic()
    due: list[str] = []
    with _lock:
        for budget_id in budget_ids:
            last = _last_warned.get(budget_id)
            if last is not None and now - last < WARN_INTERVAL_S:
                continue
            _last_warned[budget_id] = now
            due.append(budget_id)
        while len(_last_warned) > _MAX_TRACKED:
            _last_warned.pop(next(iter(_last_warned)))
    for budget_id in due:
        _log.warning(
            "tensorcost: budget %s is over its cap and would refuse this %s call, but it is "
            "not enforced at the SDK (warn-only), so the call went ahead. Repeats for this "
            "budget are logged at most once a minute.",
            budget_id,
            model,
        )
