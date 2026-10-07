"""Govern-layer admit — metadata-only POST /api/inference-proxy/v1/admit."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable, Optional

import httpx

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
    """Returns (allowed, decision_header). Fail-open on transport errors."""
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
    try:
        resp = client.post(url, headers=headers, json=metadata)
        decision_header = resp.headers.get("x-tc-decision")
        if resp.status_code == 403:
            return False, decision_header
        if resp.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"admit POST failed status={resp.status_code}",
                request=resp.request,
                response=resp,
            )
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
