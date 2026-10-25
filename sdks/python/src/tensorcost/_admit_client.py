"""Govern-layer admit — metadata-only POST /api/inference-proxy/v1/admit.

Fail-open on transport errors. A 403 is a refusal, never a transport error;
when its body says RUN_STOPPED this raises TensorCostRunStoppedError so the
caller gets who stopped the run and why instead of a generic govern refusal.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable, Optional

import httpx

from ._errors import TensorCostRunStoppedError
from ._layer import ADMIT_PATH, ADMIT_TIMEOUT_S, ControlLayer
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
    """Returns (allowed, decision_header). Fail-open on transport errors.

    Raises TensorCostRunStoppedError for a RUN_STOPPED 403.
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
    stopped: Optional[TensorCostRunStoppedError] = None
    try:
        resp = client.post(url, headers=headers, json=metadata)
        decision_header = resp.headers.get("x-tc-decision")
        if resp.status_code == 403:
            stopped = _run_stopped_from(resp)
            if stopped is None:
                return False, decision_header
        elif resp.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"admit POST failed status={resp.status_code}",
                request=resp.request,
                response=resp,
            )
        else:
            return True, decision_header
    except Exception as exc:
        if fail_open:
            if transport is not None:
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
    assert stopped is not None
    raise stopped


def _run_stopped_from(resp: httpx.Response) -> Optional[TensorCostRunStoppedError]:
    try:
        body = resp.json()
    except Exception:
        return None
    err = body.get("error") if isinstance(body, dict) else None
    if isinstance(err, dict) and err.get("code") == "RUN_STOPPED":
        return TensorCostRunStoppedError.from_proxy_detail(
            err, request_id=resp.headers.get("x-tc-request-id")
        )
    return None
