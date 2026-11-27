"""Govern-layer admit — metadata-only POST /api/inference-proxy/v1/admit.

Fail-open only when TensorCost gave no decision: a network error, a timeout,
a 5xx, or a 429 / 402 with no refusal body. The customer's call then proceeds.

A bare 429 is TensorCost pacing this process, not an outage (1.9.0). It is
retried inside the admit budget: after Retry-After when that fits (and every
call to the same /admit holds off until then), else after a short jittered
backoff. Still throttled when the budget runs out, it takes the same fail mode
as an outage, with the outcome reason ``admit_throttled``. Through 1.8.2 a 429
was not retried and was recorded as ``admit_unavailable``. A bare 402 (the
plan's ceiling) is not retried: ``admit_plan_limit``. ``admit_call``
says so in its outcome (``unavailable``), and the provider wrappers mark the
call's own observation ``admit_unavailable``, so the row keeps its agent,
team and token usage. ``admit_request`` (the older tuple API) still posts a
separate ``admit_unavailable`` observation when given a transport.

A 204 can carry ``X-TensorCost-Budget-Would-Refuse: <budget ids>``: hard caps
that are over but not enforced at the SDK (the licence lacks
feature.enforce_sdk). The outcome carries the ids; the wrappers raise the
``on_budget_would_refuse`` event and log them.

Every other non-2xx is an answer and raises a ``TensorCostRefusedError``
subclass, fail-open or not: RUN_STOPPED with its who / when / why,
BUDGET_HARD_CAP as the budget error, a code this SDK doesn't know as a plain
``TensorCostRefusedError``. /admit only answers 403 when a gate refused, so a
403 with no readable code is still a refusal. A 401/400/404 means admission
could not be checked for this call; that is not an outage either.
"""

from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

import httpx

from ._errors import (
    format_seconds,
    TensorCostError,
    TensorCostGatewayRejectedError,
    TensorCostKeyRejectedError,
    TensorCostNetworkError,
    TensorCostProxyError,
    TensorCostQuotaError,
    TensorCostRefusedError,
    TensorCostTimeoutError,
)
from ._observation_scope import observation_scope
from ._layer import ADMIT_PATH, ADMIT_TIMEOUT_S, ControlLayer
from ._refusal import decision_says_refuse, parse_body, parse_refusal_body, refusal_error
from ._retry import parse_retry_after_ms
from ._transport import ObservationTransport
from ._version import SDK_VERSION, sdk_capability_header


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


WOULD_REFUSE_HEADER = "x-tensorcost-budget-would-refuse"


@dataclass(frozen=True)
class AdmitOutcome:
    """What /admit said about a call that may go ahead."""

    allowed: bool
    decision_header: Optional[str]
    # True when /admit gave no answer (fail-open let the call through).
    unavailable: bool = False
    # With ``unavailable``: why. ``admit_throttled`` is a bare 429 still there
    # when the admit budget ran out, ``admit_plan_limit`` a bare 402,
    # ``admit_unavailable`` an outage.
    reason: Optional[str] = None
    # Budgets that would have refused the call had the SDK enforced them.
    would_refuse: tuple[str, ...] = ()


def _would_refuse_ids(resp: httpx.Response) -> tuple[str, ...]:
    raw = resp.headers.get(WOULD_REFUSE_HEADER)
    if not raw:
        return ()
    return tuple(i.strip() for i in raw.split(",") if i.strip())


def admit_request(
    *,
    base_url: str,
    bearer_token: str,
    max_layer: ControlLayer,
    metadata: dict[str, Any],
    transport: Optional[ObservationTransport] = None,
    fail_open: bool = True,
    timeout_s: float = ADMIT_TIMEOUT_S,
    http_client: Optional[httpx.Client] = None,
) -> tuple[bool, Optional[str]]:
    """Returns ``(True, decision_header)`` when the call may proceed.

    Raises a ``TensorCostRefusedError`` subclass for any refusal. Raises the
    typed outage error (network / timeout / proxy / quota) when
    ``fail_open`` is False; otherwise an outage returns ``(True, None)`` and,
    with ``transport``, posts an ``admit_unavailable`` observation.
    """
    outcome = admit_call(
        base_url=base_url,
        bearer_token=bearer_token,
        max_layer=max_layer,
        metadata=metadata,
        fail_open=fail_open,
        timeout_s=timeout_s,
        http_client=http_client,
    )
    if outcome.unavailable and transport is not None:
        _post_admit_unavailable(transport, metadata)
    return outcome.allowed, outcome.decision_header


def admit_call(
    *,
    base_url: str,
    bearer_token: str,
    max_layer: ControlLayer,
    metadata: dict[str, Any],
    fail_open: bool = True,
    timeout_s: float = ADMIT_TIMEOUT_S,
    http_client: Optional[httpx.Client] = None,
) -> AdmitOutcome:
    """The /admit check for one call. Posts nothing.

    Raises a ``TensorCostRefusedError`` subclass for any refusal, and the
    typed outage error when ``fail_open`` is False. With fail-open, an
    outage returns ``AdmitOutcome(True, None, unavailable=True)``.
    """
    url = base_url.rstrip("/") + ADMIT_PATH
    headers = {
        "Authorization": f"Bearer {bearer_token}",
        "Content-Type": "application/json",
        "x-tc-sdk-version": SDK_VERSION,
        "x-tc-sdk-capabilities": sdk_capability_header(),
        "x-tc-max-layer": max_layer,
    }

    own_client = http_client is None
    client = http_client or httpx.Client(timeout=timeout_s)
    deadline = time.monotonic() + timeout_s
    # Raised after the try below: inside it, fail-open would swallow it.
    refusal: Optional[TensorCostError] = None
    try:
        attempt = 0
        while True:
            # A Retry-After another call got from this /admit holds every
            # call: wait it out when it ends inside this call's budget.
            with _throttle_lock:
                held_until = _throttled_until.get(url, 0.0)
            now = time.monotonic()
            if held_until > now:
                if held_until + _MIN_ATTEMPT_S > deadline:
                    raise _throttled(429, (held_until - now) * 1000.0, None)
                time.sleep(held_until - now)
            remaining = max(0.001, deadline - time.monotonic())
            try:
                resp = client.post(url, headers=headers, json=metadata, timeout=remaining)
            except httpx.TimeoutException as exc:
                raise TensorCostTimeoutError(
                    f"tensorcost: admit timed out after {format_seconds(timeout_s)}", "total", root_cause=exc
                ) from exc
            except (httpx.TransportError, OSError) as exc:
                raise TensorCostNetworkError(
                    f"tensorcost: network error reaching admit: {exc}", root_cause=exc
                ) from exc
            decision_header = resp.headers.get("x-tc-decision")
            request_id = resp.headers.get("x-tc-request-id")
            if 400 <= resp.status_code < 500:
                refusal = _refusal_from_4xx(resp, decision_header)
                if refusal is not None:
                    break
                # A bare 429 (TensorCost pacing this process) or 402 (the
                # plan's ceiling). Neither is an outage, and neither decided.
                ra_ms = parse_retry_after_ms(resp.headers.get("retry-after"))
                if resp.status_code == 402:
                    raise _throttled(402, ra_ms, request_id)
                if ra_ms is not None:
                    with _throttle_lock:
                        _throttled_until[url] = time.monotonic() + ra_ms / 1000.0
                # Retry-After is the least we wait; the backoff is the floor,
                # so "Retry-After: 0" can't turn into a busy loop.
                wait = max(ra_ms / 1000.0 if ra_ms is not None else 0.0, _backoff_s(attempt))
                if time.monotonic() + wait + _MIN_ATTEMPT_S > deadline:
                    raise _throttled(429, ra_ms, request_id)
                time.sleep(wait)
                attempt += 1
                continue
            if resp.status_code >= 400:
                raise TensorCostProxyError(
                    f"admit POST failed status={resp.status_code}",
                    status=resp.status_code,
                    request_id=request_id,
                )
            return AdmitOutcome(True, decision_header, would_refuse=_would_refuse_ids(resp))
    except Exception as exc:
        if fail_open:
            return AdmitOutcome(True, None, unavailable=True, reason=_reason_of(exc))
        raise
    finally:
        if own_client:
            client.close()
    assert refusal is not None
    raise refusal


# The shortest attempt worth sending: below this the budget is spent.
_MIN_ATTEMPT_S = 0.05
# Backoff for a 429 with no Retry-After: 25 ms, doubling, with jitter.
_BACKOFF_BASE_S = 0.025

# Per /admit URL: no call is sent before this monotonic time (a Retry-After).
_throttled_until: dict[str, float] = {}
_throttle_lock = threading.Lock()


def _backoff_s(attempt: int) -> float:
    return _BACKOFF_BASE_S * (2 ** attempt) * (0.5 + random.random() * 0.5)


def _throttled(status: int, ra_ms: Optional[float], request_id: Optional[str]) -> TensorCostQuotaError:
    return TensorCostQuotaError(
        "tensorcost: admit is rate limiting this process (429) and the admit budget ran out"
        if status == 429
        else "tensorcost: admit says the plan's request ceiling is reached (402)",
        retry_after_ms=int(ra_ms) if ra_ms is not None else None,
        status=status,
        request_id=request_id,
    )


def _reason_of(exc: BaseException) -> str:
    if isinstance(exc, TensorCostQuotaError):
        return "admit_plan_limit" if exc.status == 402 else "admit_throttled"
    return "admit_unavailable"


def _reset() -> None:
    """Test hook: forget any Retry-After hold."""
    with _throttle_lock:
        _throttled_until.clear()


ADMIT_UNAVAILABLE_METADATA = {
    "decision_action": "allow",
    "decision_reason": "admit_unavailable",
}


def _post_admit_unavailable(transport: ObservationTransport, metadata: dict[str, Any]) -> None:
    tags = metadata.get("tags")
    tag_list = [t for t in tags.split(",") if t] if isinstance(tags, str) else tags
    try:
        transport.post(
            {
                "sdk_version": SDK_VERSION,
                "provider": metadata.get("provider", "openai"),
                "model": metadata.get("model", "unknown"),
                "operation": metadata.get("operation", "chat.completions"),
                "request_at": _now_iso(),
                "response_at": _now_iso(),
                "input_tokens": None,
                "output_tokens": None,
                "cost_usd_cents": None,
                "status": "success",
                "error_message": None,
                "correlation_id": metadata.get("correlation_id", ""),
                **({"agent_id": metadata["agent_id"]} if metadata.get("agent_id") else {}),
                **({"workflow_id": metadata["workflow_id"]} if metadata.get("workflow_id") else {}),
                **observation_scope(metadata.get("application"), tag_list),
                "metadata": dict(ADMIT_UNAVAILABLE_METADATA),
            }
        )
    except Exception:
        pass


def _refusal_from_4xx(
    resp: httpx.Response, decision_header: Optional[str]
) -> Optional[TensorCostError]:
    """The refusal in a 4xx from /admit, or None for a bare 429 / 402."""
    body = parse_body(resp.content)
    parsed = parse_refusal_body(body)
    if (
        parsed is None
        and resp.status_code in (402, 429)
        and not decision_says_refuse(decision_header)
    ):
        return None
    request_id = resp.headers.get("x-tc-request-id")
    retry_after = resp.headers.get("retry-after")
    if parsed is not None:
        return refusal_error(
            parsed, status=resp.status_code, request_id=request_id, retry_after_header=retry_after
        )
    if not isinstance(body, dict):
        if decision_says_refuse(decision_header):
            return refusal_error(None, status=resp.status_code, request_id=request_id)
        # HTML or nothing: something in front of TensorCost said no.
        return TensorCostGatewayRejectedError(
            resp.status_code,
            body[:500] if isinstance(body, str) else None,
            request_id=request_id,
        )
    if resp.status_code == 403:
        return refusal_error(None, status=403, request_id=request_id)
    if resp.status_code == 401:
        # /admit rejected the TensorCost token.
        return TensorCostKeyRejectedError(401, request_id=request_id)
    message = body.get("message")
    return TensorCostRefusedError(
        f"tensorcost: admission check returned {resp.status_code}"
        + (f" ({message})" if isinstance(message, str) and message else "")
        + "; the call was not sent",
        status=resp.status_code,
        request_id=request_id,
    )
