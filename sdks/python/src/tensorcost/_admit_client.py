"""Govern-layer admit — metadata-only POST /api/inference-proxy/v1/admit.

Fail-open only on an outage: network error, timeout, 5xx, or a 429 / 402
with no refusal body. The customer's call then proceeds and an ``admit_unavailable``
observation is posted when transport is set.

Every other non-2xx is an answer and raises a ``TensorCostRefusedError``
subclass, fail-open or not: RUN_STOPPED with its who / when / why,
BUDGET_HARD_CAP as the budget error, a code this SDK doesn't know as a plain
``TensorCostRefusedError``. /admit only answers 403 when a gate refused, so a
403 with no readable code is still a refusal. A 401/400/404 means admission
could not be checked for this call; that is not an outage either.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

import httpx

from ._errors import (
    TensorCostNetworkError,
    TensorCostProxyError,
    TensorCostQuotaError,
    TensorCostRefusedError,
    TensorCostTimeoutError,
)
from ._layer import ADMIT_PATH, ADMIT_TIMEOUT_S, ControlLayer
from ._refusal import decision_says_refuse, parse_body, parse_refusal_body, refusal_error
from ._retry import parse_retry_after_ms
from ._transport import ObservationTransport
from ._version import SDK_VERSION, sdk_capability_header


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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
    ``fail_open`` is False; otherwise an outage returns ``(True, None)``.
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
    refusal: Optional[TensorCostRefusedError] = None
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
            return True, decision_header
    except Exception:
        if fail_open:
            if transport is not None:
                tags = metadata.get("tags")
                tag_list = (
                    [t for t in tags.split(",") if t] if isinstance(tags, str) else tags
                )
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
                            **(
                                {"application": metadata["application"]}
                                if metadata.get("application")
                                else {}
                            ),
                            **({"tags": list(tag_list)} if tag_list else {}),
                            "metadata": {
                                "decision_action": "allow",
                                "decision_reason": "admit_unavailable",
                            },
                        }
                    )
                except Exception:
                    pass
            return True, None
        raise
    finally:
        if own_client:
            client.close()
    assert refusal is not None
    raise refusal


def _refusal_from_4xx(
    resp: httpx.Response, decision_header: Optional[str]
) -> Optional[TensorCostRefusedError]:
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
    if parsed is not None or resp.status_code == 403:
        return refusal_error(
            parsed,
            status=resp.status_code,
            request_id=request_id,
            retry_after_header=resp.headers.get("retry-after"),
        )
    message = body.get("message") if isinstance(body, dict) else None
    return TensorCostRefusedError(
        f"tensorcost: admission check returned {resp.status_code}"
        + (f" ({message})" if isinstance(message, str) and message else "")
        + "; the call was not sent",
        status=resp.status_code,
        request_id=request_id,
    )
