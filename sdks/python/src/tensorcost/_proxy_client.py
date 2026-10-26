"""Applied-mode proxy HTTP client for the Python SDK.

Contract:

* Returns the parsed response dict on success.
* On all non-success outcomes, raises a subclass of ``TensorCostError``.
* Never swallows errors — the provider wrapper's fail-open logic lives there,
  not here.

Timeout handling:

* ``headers_timeout_s`` (default 2s): abort if response headers never arrive.
* ``timeout_s`` (default 60s): wall-clock budget for the full response body.

How a non-2xx response is classified (unchanged split from 1.4.0, plus the
refusal check in front of it):

* 4xx whose body carries a TensorCost refusal code, or whose x-tc-decision
  says action=refuse → a ``TensorCostRefusedError`` subclass. Terminal,
  never retried, not a circuit failure.
* 429 otherwise → ``TensorCostQuotaError`` (TensorCost's rate limit).
* 402 otherwise → ``TensorCostQuotaError``, status 402 (the plan's monthly
  proxy-request ceiling). Not retried.
* 5xx → ``TensorCostProxyError`` (proxy or upstream unavailable).
* any other 4xx → ``TensorCostProviderError``: the provider's own error,
  relayed verbatim by the proxy.
"""

from __future__ import annotations

import json
import sys
import time
from typing import Any, Optional

import httpx

from ._circuit import CircuitBreaker
from ._errors import (
    TensorCostNetworkError,
    TensorCostProxyError,
    TensorCostProviderError,
    TensorCostQuotaError,
    TensorCostRefusedError,
    TensorCostTimeoutError,
)
from ._refusal import (
    decision_says_refuse,
    parse_body,
    parse_refusal_body,
    provider_message_from,
    refusal_error,
)
from ._retry import (
    DEFAULT_RETRY_CONFIG,
    RetryConfig,
    compute_delay,
    parse_retry_after_ms,
    sleep_sync,
)
from ._layer import PROXY_HEADERS_TIMEOUT_S, ControlLayer, proxy_path_for_operation
from ._transport import TenantIdMissingError
from ._telemetry import (
    AfterResponseEvent,
    BeforeRequestEvent,
    LifecycleEventCallback,
    OnErrorEvent,
    OnRetryEvent,
    emit_event,
)
from ._version import SDK_VERSION, sdk_capability_header

def _post_with_headers_timeout(
    client: httpx.Client,
    *,
    url: str,
    json_body: dict[str, Any],
    headers: dict[str, str],
    headers_wait_budget_s: float,
    short_headers_timeout_s: float,
    body_timeout_s: float,
    attempt: int,
) -> httpx.Response:
    """Wait for headers within headers budget, then read body before body deadline."""
    effective_headers = min(headers_wait_budget_s, body_timeout_s)
    call_start = time.monotonic()
    body_deadline = call_start + body_timeout_s
    stream_timeout = httpx.Timeout(
        connect=effective_headers,
        write=effective_headers,
        pool=effective_headers,
        read=body_timeout_s,
    )

    try:
        with client.stream(
            "POST",
            url,
            json=json_body,
            headers=headers,
            timeout=stream_timeout,
        ) as resp:
            chunks: list[bytes] = []
            for chunk in resp.iter_bytes():
                if time.monotonic() > body_deadline:
                    raise TensorCostTimeoutError(
                        "tensorcost: proxy response body timed out",
                        "total",
                        attempt=attempt,
                    )
                chunks.append(chunk)
            content = b"".join(chunks)
            return httpx.Response(
                status_code=resp.status_code,
                headers=resp.headers,
                content=content,
                request=resp.request,
            )
    except httpx.TimeoutException as exc:
        kind = (
            "headers"
            if headers_wait_budget_s == short_headers_timeout_s
            else "total"
        )
        message = (
            f"tensorcost: proxy did not respond with headers within {short_headers_timeout_s:.0f}s"
            if kind == "headers"
            else f"tensorcost: proxy request timed out after {body_timeout_s:.0f}s"
        )
        raise TensorCostTimeoutError(
            message,
            kind,
            root_cause=exc,
            attempt=attempt,
        ) from exc


def proxy_request(
    *,
    proxy_url: str,
    bearer_token: str,
    provider_url: str,
    provider_auth: str,
    body: dict[str, Any],
    model: str,
    provider: str,
    operation: str,
    correlation_id: str,
    environment: Optional[str],
    application: Optional[str] = None,
    tags: Optional[list[str]] = None,
    customer: Optional[str] = None,
    feature: Optional[str] = None,
    agent_id: Optional[str] = None,
    workflow_id: Optional[str] = None,
    retry: RetryConfig = DEFAULT_RETRY_CONFIG,
    timeout_s: float = 60.0,
    headers_timeout_s: float = PROXY_HEADERS_TIMEOUT_S,
    on_lifecycle_event: Optional[LifecycleEventCallback] = None,
    circuit: Optional[CircuitBreaker] = None,
    max_layer: ControlLayer = "observe",
    fail_open_fast: bool = False,
    http_client: Optional[httpx.Client] = None,
) -> tuple[dict[str, Any], Optional[str]]:
    """Execute a proxied request with retries, timeouts, and telemetry."""
    call_start = time.monotonic()

    # Circuit bypass is decided in the provider wrapper — do not re-check here.

    path = proxy_path_for_operation(operation) or (
        "/api/inference-proxy/v1/chat/completions"
    )
    url = proxy_url.rstrip("/") + path
    headers: dict[str, str] = {
        "Authorization": f"Bearer {bearer_token}",
        "Content-Type": "application/json",
        "x-tc-sdk-version": SDK_VERSION,
        "x-tc-sdk-capabilities": sdk_capability_header(),
        "x-tc-max-layer": max_layer,
        "x-tc-provider-url": provider_url,
        "x-tc-provider-auth": provider_auth,
        "x-tc-correlation-id": correlation_id,
    }
    if environment:
        headers["x-tc-environment"] = environment
    if application:
        headers["x-tc-application"] = application
    if tags:
        headers["x-tc-tags"] = ",".join(tags)
    if customer:
        headers["x-tc-customer"] = customer
    if feature:
        headers["x-tc-feature"] = feature
    if agent_id:
        headers["x-tc-agent-id"] = agent_id
    if workflow_id:
        headers["x-tc-workflow-id"] = workflow_id

    # Buffered JSON: proxy may not send headers until upstream completes.
    is_buffered_json = body.get("stream") is not True
    effective_headers_timeout_s = (
        timeout_s if is_buffered_json else min(headers_timeout_s, timeout_s)
    )

    last_error: Optional[Exception] = None

    for attempt in range(retry.max_attempts):
        elapsed_ms = (time.monotonic() - call_start) * 1000.0

        emit_event(
            on_lifecycle_event,
            BeforeRequestEvent(
                kind="before_request",
                provider=provider,
                model=model,
                operation=operation,
                attempt_number=attempt + 1,
                elapsed_ms=elapsed_ms,
            ),
        )

        client = http_client or httpx.Client(timeout=timeout_s)
        close_client = http_client is None

        try:
            resp = _post_with_headers_timeout(
                client,
                url=url,
                json_body=body,
                headers=headers,
                headers_wait_budget_s=effective_headers_timeout_s,
                short_headers_timeout_s=headers_timeout_s,
                body_timeout_s=timeout_s,
                attempt=attempt + 1,
            )
        except TensorCostTimeoutError as exc:
            if close_client:
                client.close()
            if circuit is not None:
                circuit.record_failure()
            if exc.timeout_kind == "headers":
                emit_event(
                    on_lifecycle_event,
                    OnErrorEvent(
                        kind="on_error",
                        provider=provider,
                        model=model,
                        operation=operation,
                        attempt_number=attempt + 1,
                        elapsed_ms=(time.monotonic() - call_start) * 1000.0,
                        error_class=exc.__class__.__name__,
                        status=None,
                    ),
                )
                raise exc
            last_error = exc
            if not fail_open_fast and attempt + 1 < retry.max_attempts:
                delay = compute_delay(attempt, retry)
                emit_event(
                    on_lifecycle_event,
                    OnRetryEvent(
                        kind="on_retry",
                        provider=provider,
                        model=model,
                        operation=operation,
                        attempt_number=attempt + 1,
                        elapsed_ms=(time.monotonic() - call_start) * 1000.0,
                        reason="network_error",
                        status=None,
                        delay_ms=delay,
                    ),
                )
                sleep_sync(delay)
                continue
            emit_event(
                on_lifecycle_event,
                OnErrorEvent(
                    kind="on_error",
                    provider=provider,
                    model=model,
                    operation=operation,
                    attempt_number=attempt + 1,
                    elapsed_ms=(time.monotonic() - call_start) * 1000.0,
                    error_class=exc.__class__.__name__,
                    status=None,
                ),
            )
            raise exc
        except httpx.TimeoutException as exc:
            if close_client:
                client.close()
            tc_err = TensorCostTimeoutError(
                f"tensorcost: proxy request timed out after {timeout_s:.0f}s",
                "total",
                root_cause=exc,
                attempt=attempt + 1,
            )
            last_error = tc_err
            if circuit is not None:
                circuit.record_failure()
            if not fail_open_fast and attempt + 1 < retry.max_attempts:
                delay = compute_delay(attempt, retry)
                emit_event(
                    on_lifecycle_event,
                    OnRetryEvent(
                        kind="on_retry",
                        provider=provider,
                        model=model,
                        operation=operation,
                        attempt_number=attempt + 1,
                        elapsed_ms=(time.monotonic() - call_start) * 1000.0,
                        reason="network_error",
                        status=None,
                        delay_ms=delay,
                    ),
                )
                sleep_sync(delay)
                continue
            emit_event(
                on_lifecycle_event,
                OnErrorEvent(
                    kind="on_error",
                    provider=provider,
                    model=model,
                    operation=operation,
                    attempt_number=attempt + 1,
                    elapsed_ms=(time.monotonic() - call_start) * 1000.0,
                    error_class=tc_err.__class__.__name__,
                    status=None,
                ),
            )
            raise tc_err
        except (httpx.NetworkError, httpx.RemoteProtocolError, OSError) as exc:
            if close_client:
                client.close()
            tc_err = TensorCostNetworkError(
                f"tensorcost: network error reaching proxy: {exc}",
                root_cause=exc,
                attempt=attempt + 1,
            )
            last_error = tc_err
            if circuit is not None:
                circuit.record_failure()
            if not fail_open_fast and attempt + 1 < retry.max_attempts:
                delay = compute_delay(attempt, retry)
                emit_event(
                    on_lifecycle_event,
                    OnRetryEvent(
                        kind="on_retry",
                        provider=provider,
                        model=model,
                        operation=operation,
                        attempt_number=attempt + 1,
                        elapsed_ms=(time.monotonic() - call_start) * 1000.0,
                        reason="network_error",
                        status=None,
                        delay_ms=delay,
                    ),
                )
                sleep_sync(delay)
                continue
            emit_event(
                on_lifecycle_event,
                OnErrorEvent(
                    kind="on_error",
                    provider=provider,
                    model=model,
                    operation=operation,
                    attempt_number=attempt + 1,
                    elapsed_ms=(time.monotonic() - call_start) * 1000.0,
                    error_class=tc_err.__class__.__name__,
                    status=None,
                ),
            )
            raise tc_err
        except Exception as exc:  # noqa: BLE001
            if close_client:
                client.close()
            tc_err = TensorCostNetworkError(
                f"tensorcost: unexpected error during proxy request: {exc}",
                root_cause=exc,
                attempt=attempt + 1,
            )
            last_error = tc_err
            if circuit is not None:
                circuit.record_failure()
            if not fail_open_fast and attempt + 1 < retry.max_attempts:
                delay = compute_delay(attempt, retry)
                emit_event(
                    on_lifecycle_event,
                    OnRetryEvent(
                        kind="on_retry",
                        provider=provider,
                        model=model,
                        operation=operation,
                        attempt_number=attempt + 1,
                        elapsed_ms=(time.monotonic() - call_start) * 1000.0,
                        reason="network_error",
                        status=None,
                        delay_ms=delay,
                    ),
                )
                sleep_sync(delay)
                continue
            emit_event(
                on_lifecycle_event,
                OnErrorEvent(
                    kind="on_error",
                    provider=provider,
                    model=model,
                    operation=operation,
                    attempt_number=attempt + 1,
                    elapsed_ms=(time.monotonic() - call_start) * 1000.0,
                    error_class=tc_err.__class__.__name__,
                    status=None,
                ),
            )
            raise tc_err
        finally:
            if close_client:
                try:
                    client.close()
                except Exception:  # noqa: BLE001
                    pass

        request_id = resp.headers.get("x-tc-request-id")

        if resp.is_success:
            if circuit is not None:
                circuit.record_success()
            emit_event(
                on_lifecycle_event,
                AfterResponseEvent(
                    kind="after_response",
                    provider=provider,
                    model=model,
                    operation=operation,
                    attempt_number=attempt + 1,
                    elapsed_ms=(time.monotonic() - call_start) * 1000.0,
                    status=resp.status_code,
                    request_id=request_id,
                ),
            )
            return json.loads(resp.content), resp.headers.get("x-tc-decision")

        error_body: Any = None
        if 400 <= resp.status_code < 500:
            # A TensorCost refusal is terminal whatever its status (a 429
            # refusal included): checked before the quota arm, never
            # retried, never counted against the circuit.
            error_body = parse_body(resp.content)
            refusal = parse_refusal_body(error_body)
            if refusal is not None or decision_says_refuse(resp.headers.get("x-tc-decision")):
                tc_refusal = refusal_error(
                    refusal,
                    status=resp.status_code,
                    request_id=request_id,
                    attempt=attempt + 1,
                    retry_after_header=resp.headers.get("retry-after"),
                )
                emit_event(
                    on_lifecycle_event,
                    OnErrorEvent(
                        kind="on_error",
                        provider=provider,
                        model=model,
                        operation=operation,
                        attempt_number=attempt + 1,
                        elapsed_ms=(time.monotonic() - call_start) * 1000.0,
                        error_class=tc_refusal.__class__.__name__,
                        status=resp.status_code,
                    ),
                )
                raise tc_refusal

        if resp.status_code == 402:
            # No refusal body: the plan's monthly proxy-request ceiling (Scale
            # COGS gate). TensorCost declining to carry the call is not a
            # refusal of the call, and the server means for it to go direct,
            # so it counts as an outage. Not retried: it won't clear soon.
            ra_ms = parse_retry_after_ms(resp.headers.get("retry-after"))
            tc_err = TensorCostQuotaError(
                "tensorcost: proxy returned 402 (plan request ceiling reached)",
                retry_after_ms=int(ra_ms) if ra_ms is not None else None,
                status=402,
                request_id=request_id,
                attempt=attempt + 1,
            )
            emit_event(
                on_lifecycle_event,
                OnErrorEvent(
                    kind="on_error",
                    provider=provider,
                    model=model,
                    operation=operation,
                    attempt_number=attempt + 1,
                    elapsed_ms=(time.monotonic() - call_start) * 1000.0,
                    error_class=tc_err.__class__.__name__,
                    status=402,
                ),
            )
            raise tc_err

        if resp.status_code == 429:
            ra_header = resp.headers.get("retry-after")
            ra_ms = parse_retry_after_ms(ra_header)
            ra_ms_int = int(ra_ms) if ra_ms is not None else None
            tc_err = TensorCostQuotaError(
                "tensorcost: proxy returned 429 Too Many Requests",
                retry_after_ms=ra_ms_int,
                status=429,
                request_id=request_id,
                attempt=attempt + 1,
            )
            last_error = tc_err
            if attempt + 1 < retry.max_attempts:
                delay = compute_delay(attempt, retry, ra_ms)
                emit_event(
                    on_lifecycle_event,
                    OnRetryEvent(
                        kind="on_retry",
                        provider=provider,
                        model=model,
                        operation=operation,
                        attempt_number=attempt + 1,
                        elapsed_ms=(time.monotonic() - call_start) * 1000.0,
                        reason="quota_429",
                        status=429,
                        delay_ms=delay,
                    ),
                )
                sleep_sync(delay)
                continue
            emit_event(
                on_lifecycle_event,
                OnErrorEvent(
                    kind="on_error",
                    provider=provider,
                    model=model,
                    operation=operation,
                    attempt_number=attempt + 1,
                    elapsed_ms=(time.monotonic() - call_start) * 1000.0,
                    error_class=tc_err.__class__.__name__,
                    status=429,
                ),
            )
            raise tc_err

        if resp.status_code >= 500:
            tc_err = TensorCostProxyError(
                f"tensorcost: proxy returned {resp.status_code}",
                status=resp.status_code,
                request_id=request_id,
                attempt=attempt + 1,
            )
            last_error = tc_err
            if circuit is not None:
                circuit.record_failure()
            if attempt + 1 < retry.max_attempts:
                delay = compute_delay(attempt, retry)
                emit_event(
                    on_lifecycle_event,
                    OnRetryEvent(
                        kind="on_retry",
                        provider=provider,
                        model=model,
                        operation=operation,
                        attempt_number=attempt + 1,
                        elapsed_ms=(time.monotonic() - call_start) * 1000.0,
                        reason="5xx",
                        status=resp.status_code,
                        delay_ms=delay,
                    ),
                )
                sleep_sync(delay)
                continue
            emit_event(
                on_lifecycle_event,
                OnErrorEvent(
                    kind="on_error",
                    provider=provider,
                    model=model,
                    operation=operation,
                    attempt_number=attempt + 1,
                    elapsed_ms=(time.monotonic() - call_start) * 1000.0,
                    error_class=tc_err.__class__.__name__,
                    status=resp.status_code,
                ),
            )
            raise tc_err

        # Any other status: the provider's own error, relayed by the proxy.
        # Not retried and never a fallback; the provider would answer the
        # same way again.
        provider_msg = provider_message_from(error_body)
        tc_err = TensorCostProviderError(
            f"tensorcost: upstream provider returned {resp.status_code}"
            + (f": {provider_msg}" if provider_msg else ""),
            body=error_body,
            status=resp.status_code,
            request_id=request_id,
            attempt=attempt + 1,
        )
        emit_event(
            on_lifecycle_event,
            OnErrorEvent(
                kind="on_error",
                provider=provider,
                model=model,
                operation=operation,
                attempt_number=attempt + 1,
                elapsed_ms=(time.monotonic() - call_start) * 1000.0,
                error_class=tc_err.__class__.__name__,
                status=resp.status_code,
            ),
        )
        raise tc_err

    raise last_error or TensorCostProxyError("tensorcost: max attempts reached")


def _terminal_proxy_refusal_error(
    content: bytes,
    *,
    request_id: Optional[str],
    attempt: int,
) -> Optional[TensorCostRefusedError]:
    """The typed refusal in a proxy error body, or None if it isn't one.

    Kept for callers of the 1.4.0 helper; the request paths now use
    ``_refusal`` directly, for every 4xx rather than only 403.
    """
    refusal = parse_refusal_body(parse_body(content))
    if refusal is None:
        return None
    return refusal_error(refusal, status=403, request_id=request_id, attempt=attempt)


def proxy_request_stream(
    *,
    proxy_url: str,
    bearer_token: str,
    provider_url: str,
    provider_auth: str,
    body: dict[str, Any],
    model: str,
    provider: str,
    operation: str,
    correlation_id: str,
    environment: Optional[str] = None,
    application: Optional[str] = None,
    tags: Optional[list[str]] = None,
    customer: Optional[str] = None,
    feature: Optional[str] = None,
    agent_id: Optional[str] = None,
    workflow_id: Optional[str] = None,
    retry: Optional[RetryConfig] = None,
    timeout_s: float = 60.0,
    headers_timeout_s: float = PROXY_HEADERS_TIMEOUT_S,
    on_lifecycle_event: Optional[LifecycleEventCallback] = None,
    circuit: Optional[CircuitBreaker] = None,
    max_layer: ControlLayer = "observe",
    fail_open_fast: bool = False,
    http_client: Optional[httpx.Client] = None,
) -> tuple[Any, Optional[str]]:
    """Steer/route a streaming request; returns (sse_iterable, decision_header)."""
    retry_cfg = retry or DEFAULT_RETRY_CONFIG
    path = proxy_path_for_operation(operation) or "/api/inference-proxy/v1/chat/completions"
    url = f"{proxy_url.rstrip('/')}{path}"
    headers: dict[str, str] = {
        "Authorization": f"Bearer {bearer_token}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        "x-tc-sdk-version": SDK_VERSION,
        "x-tc-sdk-capabilities": sdk_capability_header(),
        "x-tc-max-layer": max_layer,
        "x-tc-provider-url": provider_url,
        "x-tc-provider-auth": provider_auth,
        "x-tc-correlation-id": correlation_id,
    }
    if environment:
        headers["x-tc-environment"] = environment
    if agent_id:
        headers["x-tc-agent-id"] = agent_id
    if workflow_id:
        headers["x-tc-workflow-id"] = workflow_id
    if customer:
        headers["x-tc-customer"] = customer
    if feature:
        headers["x-tc-feature"] = feature
    if application:
        headers["x-tc-application"] = application
    if tags:
        headers["x-tc-tags"] = ",".join(tags)

    client = http_client or httpx.Client(timeout=timeout_s)
    close_client = http_client is None
    stream_ctx = client.stream(
        "POST",
        url,
        json=body,
        headers=headers,
        timeout=httpx.Timeout(
            connect=min(headers_timeout_s, timeout_s),
            read=timeout_s,
            write=min(headers_timeout_s, timeout_s),
            pool=min(headers_timeout_s, timeout_s),
        ),
    )
    # Transport failures are typed here so the fail-open predicate (an
    # allow-list since 1.4.1) recognises them as outages.
    try:
        resp = stream_ctx.__enter__()
    except httpx.TimeoutException as exc:
        if close_client:
            client.close()
        if circuit is not None:
            circuit.record_failure()
        raise TensorCostTimeoutError(
            f"tensorcost: proxy did not respond with headers within {headers_timeout_s:.0f}s",
            "headers",
            root_cause=exc,
            attempt=1,
        ) from exc
    except (httpx.TransportError, OSError) as exc:
        if close_client:
            client.close()
        if circuit is not None:
            circuit.record_failure()
        raise TensorCostNetworkError(
            f"tensorcost: network error reaching proxy: {exc}",
            root_cause=exc,
            attempt=1,
        ) from exc
    request_id = resp.headers.get("x-tc-request-id")
    try:
        if resp.status_code >= 400:
            error_body = parse_body(resp.read())
            if resp.status_code < 500:
                refusal = parse_refusal_body(error_body)
                if refusal is not None or decision_says_refuse(
                    resp.headers.get("x-tc-decision")
                ):
                    tc_refusal = refusal_error(
                        refusal,
                        status=resp.status_code,
                        request_id=request_id,
                        attempt=1,
                        retry_after_header=resp.headers.get("retry-after"),
                    )
                    emit_event(
                        on_lifecycle_event,
                        OnErrorEvent(
                            kind="on_error",
                            provider=provider,
                            model=model,
                            operation=operation,
                            attempt_number=1,
                            elapsed_ms=0.0,
                            error_class=tc_refusal.__class__.__name__,
                            status=resp.status_code,
                        ),
                    )
                    raise tc_refusal
            if resp.status_code in (402, 429):
                ra_ms = parse_retry_after_ms(resp.headers.get("retry-after"))
                raise TensorCostQuotaError(
                    "tensorcost: proxy returned 429 Too Many Requests"
                    if resp.status_code == 429
                    else "tensorcost: proxy returned 402 (plan request ceiling reached)",
                    retry_after_ms=int(ra_ms) if ra_ms is not None else None,
                    status=resp.status_code,
                    request_id=request_id,
                    attempt=1,
                )
            if resp.status_code >= 500:
                if circuit is not None:
                    circuit.record_failure()
                raise TensorCostProxyError(
                    f"tensorcost: proxy returned {resp.status_code}",
                    status=resp.status_code,
                    request_id=request_id,
                    attempt=1,
                )
            provider_msg = provider_message_from(error_body)
            raise TensorCostProviderError(
                f"tensorcost: upstream provider returned {resp.status_code}"
                + (f": {provider_msg}" if provider_msg else ""),
                body=error_body,
                status=resp.status_code,
                request_id=request_id,
                attempt=1,
            )
        if circuit is not None:
            circuit.record_success()

        def _iter_sse() -> Any:
            try:
                for line in resp.iter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data in ("", "[DONE]"):
                        continue
                    yield json.loads(data)
            finally:
                stream_ctx.__exit__(None, None, None)
                if close_client:
                    client.close()

        return _iter_sse(), resp.headers.get("x-tc-decision")
    except BaseException:
        stream_ctx.__exit__(*sys.exc_info())
        if close_client:
            client.close()
        raise


def should_fall_back_to_provider(err: BaseException) -> bool:
    """Should a failed proxy attempt fall back to calling the provider directly?

    Only when TensorCost itself is unavailable. Fail-open exists so that
    TensorCost being down never takes the customer's inference down with it,
    so the answer is yes for exactly these and no for everything else:

    * ``TensorCostNetworkError``  couldn't connect, DNS, connection refused
    * ``TensorCostTimeoutError``  no headers, or no body inside the budget
    * ``TensorCostProxyError``    5xx from the proxy, or the circuit is open
    * ``TensorCostQuotaError``    429 (rate limit) or 402 (plan ceiling) from
                                  the proxy, with no refusal body

    plus one case where TensorCost was never asked at all: no tenant id, so no
    token (``TenantIdMissingError``, which warns once). That can't be a skipped
    refusal, and 1.4.0 behaved the same way.

    This is an allow-list on purpose. 1.4.0 had the inverse, a tuple of
    refusal classes that must NOT fall back, and every refusal code missing
    from it fell back by default. That is how a money budget's hard cap
    (BUDGET_HARD_CAP) got routed around. Now a refusal this SDK has never
    heard of is terminal, a relayed provider 4xx surfaces as itself, and an
    error the SDK didn't classify at all is surfaced rather than retried
    against the provider.
    """
    return isinstance(
        err,
        (
            TensorCostNetworkError,
            TensorCostTimeoutError,
            TensorCostProxyError,
            TensorCostQuotaError,
            TenantIdMissingError,
        ),
    )
