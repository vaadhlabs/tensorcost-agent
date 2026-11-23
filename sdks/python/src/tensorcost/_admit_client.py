"""Govern-layer admit — metadata-only POST /api/inference-proxy/v1/admit.

Fail-open only on an outage: network error, timeout, 5xx, or a 429 / 402
with no refusal body. The customer's call then proceeds. ``admit_call``
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

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

import httpx

from ._errors import (
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
    # True when /admit couldn't be reached (fail-open let the call through).
    unavailable: bool = False
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
    # Raised after the try below: inside it, fail-open would swallow it.
    refusal: Optional[TensorCostError] = None
    try:
        try:
            resp = client.post(url, headers=headers, json=metadata)
        except httpx.TimeoutException as exc:
            raise TensorCostTimeoutError(
                f"tensorcost: admit timed out after {timeout_s}s", "total", root_cause=exc
            ) from exc
        except (httpx.TransportError, OSError) as exc:
            raise TensorCostNetworkError(
                f"tensorcost: network error reaching admit: {exc}", root_cause=exc
            ) from exc
        decision_header = resp.headers.get("x-tc-decision")
        request_id = resp.headers.get("x-tc-request-id")
        if 400 <= resp.status_code < 500:
            refusal = _refusal_from_4xx(resp, decision_header)
            if refusal is None:  # a bare 429 (throttler) or 402 (plan ceiling)
                ra_ms = parse_retry_after_ms(resp.headers.get("retry-after"))
                raise TensorCostQuotaError(
                    f"admit POST failed status={resp.status_code}",
                    retry_after_ms=int(ra_ms) if ra_ms is not None else None,
                    status=resp.status_code,
                    request_id=request_id,
                )
        elif resp.status_code >= 400:
            raise TensorCostProxyError(
                f"admit POST failed status={resp.status_code}",
                status=resp.status_code,
                request_id=request_id,
            )
        else:
            return AdmitOutcome(True, decision_header, would_refuse=_would_refuse_ids(resp))
    except Exception:
        if fail_open:
            return AdmitOutcome(True, None, unavailable=True)
        raise
    finally:
        if own_client:
            client.close()
    assert refusal is not None
    raise refusal


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
