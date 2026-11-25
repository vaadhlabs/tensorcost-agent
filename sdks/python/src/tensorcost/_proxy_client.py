"""Applied-mode proxy HTTP client for the Python SDK.

Contract:

* Returns the parsed response dict on success.
* On all non-success outcomes, raises a subclass of ``TensorCostError``.
* Never swallows errors — the provider wrapper's fail-open logic lives there,
  not here.

Timeout handling:

* ``headers_timeout_s`` (default 2s): abort if response headers never arrive.
* ``timeout_s`` (default 60s): wall-clock budget for the full response body.

Retries and fallback (1.8.0, ``_delivery``): only an attempt that provably
never reached the provider is retried, or may fall back: a connect failure, a
429 or 402 from TensorCost, a 503 from the gateway or the proxy, a gateway 502
that says it couldn't connect to the proxy. A timeout once the request was
sent, a reset, a 5xx the proxy didn't prove was its own pre-forward answer,
and anything with ``x-should-retry: false`` raise with
``may_have_reached_provider`` set. Through 1.7.0 every 5xx, timeout and
network error was retried five times and then fell back, so one call the
provider had already billed could run six times.

Since 1.8.2 the proxy's 502 ``x-tc-upstream-error: provider_unreachable`` (it
couldn't connect to the provider, nothing was sent) is neither retried nor a
circuit failure, and raises with ``may_have_reached_provider`` false so
fail-open goes direct.

How a non-2xx response is classified (unchanged split from 1.4.0, plus the
refusal check in front of it):

* 4xx whose body carries a TensorCost refusal code, or whose x-tc-decision
  says action=refuse → a ``TensorCostRefusedError`` subclass. Terminal,
  never retried, not a circuit failure.
* 429 otherwise → ``TensorCostQuotaError`` (TensorCost's rate limit).
* 402 otherwise → ``TensorCostQuotaError``, status 402 (the plan's monthly
  proxy-request ceiling). Not retried.
* 5xx → ``TensorCostProxyError`` (proxy or upstream unavailable).
* 429, 402 or 5xx marked ``x-tc-error-origin: provider`` →
  ``TensorCostProviderError``: the provider's answer, never resent.
* any other 4xx with a JSON body → ``TensorCostProviderError``: the
  provider's own error, relayed verbatim by the proxy. The wrappers re-raise
  it as the provider SDK's own error type when they can.
* any other 4xx without one (HTML, text, empty) →
  ``TensorCostGatewayRejectedError``: turned away before TensorCost.
"""

from __future__ import annotations

import json
import sys
import time
from typing import Any, Optional

import httpx

from ._circuit import CircuitBreaker
from ._delivery import classify_error_answer, is_provider_unreachable, send_error_sent_nothing
from ._errors import (
    format_seconds,
    TensorCostError,
    TensorCostGatewayRejectedError,
    TensorCostKeyRejectedError,
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
from ._telemetry import (
    AfterResponseEvent,
    BeforeRequestEvent,
    LifecycleEventCallback,
    OnErrorEvent,
    OnRetryEvent,
    emit_event,
)
from ._version import SDK_VERSION, sdk_capability_header


def _mark(err: TensorCostError, may_have_reached_provider: bool) -> TensorCostError:
    """Set whether ``err``'s call may already have reached the provider."""
    err.may_have_reached_provider = may_have_reached_provider
    return err

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
    # Reaching the proxy at all is bounded by the short headers budget even
    # when the headers themselves may take the whole body budget (buffered
    # JSON): through 1.7.0 a blackholed proxy got timeout_s (60 s) to connect.
    connect_budget = min(short_headers_timeout_s, body_timeout_s)
    call_start = time.monotonic()
    body_deadline = call_start + body_timeout_s
    stream_timeout = httpx.Timeout(
        connect=connect_budget,
        write=effective_headers,
        pool=connect_budget,
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
                    # The proxy has answered: never a reason to resend.
                    raise _mark(
                        TensorCostTimeoutError(
                            "tensorcost: proxy response body timed out",
                            "total",
                            attempt=attempt,
                        ),
                        True,
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
        if isinstance(exc, (httpx.ConnectTimeout, httpx.PoolTimeout)):
            message = f"tensorcost: could not connect to the proxy within {format_seconds(connect_budget)}"
        elif kind == "headers":
            message = f"tensorcost: proxy did not respond with headers within {format_seconds(short_headers_timeout_s)}"
        else:
            message = f"tensorcost: proxy request timed out after {format_seconds(body_timeout_s)}"
        # Only a connect or pool timeout comes before the request is written.
        # A read or write timeout may follow a forward the provider is
        # running right now, so it must never be sent again.
        raise _mark(
            TensorCostTimeoutError(
                message,
                kind,
                root_cause=exc,
                attempt=attempt,
            ),
            not send_error_sent_nothing(exc),
        ) from exc


def _may_retry(
    attempt: int,
    retry: RetryConfig,
    circuit: Optional[CircuitBreaker],
    probe: bool,
) -> bool:
    """Whether a failed attempt that never reached the provider may be
    followed by another.

    Never for the half-open probe: it is one attempt, and when it fails the
    call falls back to the provider at once and the circuit stays open for
    the next probe window. Never once failures have opened the circuit
    mid-call either, since every attempt after that is a wait on a proxy the
    breaker has already given up on. Through 1.8.0 a probe retried up to
    max_attempts times through a dead proxy (5-7.6 s per call) before
    falling back.
    """
    if probe or (circuit is not None and circuit.state != "CLOSED"):
        return False
    return attempt + 1 < retry.max_attempts


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
    extra_headers: Optional[dict[str, str]] = None,
    extra_query: Optional[dict[str, Any]] = None,
    response_sink: Optional[list] = None,
) -> tuple[dict[str, Any], Optional[str]]:
    """Execute a proxied request with retries, timeouts, and telemetry.

    ``extra_headers`` / ``extra_query`` are the caller's request options
    (the provider SDK's own), sent as headers and query parameters; they
    never replace a header this function sets. On success the httpx
    response is appended to ``response_sink`` when one is given.
    """
    call_start = time.monotonic()
    # The circuit was not CLOSED as this call began: it is the half-open probe.
    probe = circuit is not None and circuit.state != "CLOSED"

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
    _add_caller_headers(headers, extra_headers)
    url = _with_query(url, extra_query)

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
            # A timeout after the request was written is never retried and
            # never falls back (the provider may be running it); the circuit
            # counts it, so the calls after it go direct once a wedged proxy
            # opens the circuit. Only a connect timeout may be retried.
            if exc.may_have_reached_provider or exc.timeout_kind == "headers":
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
            if not fail_open_fast and _may_retry(attempt, retry, circuit, probe):
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
            sent_nothing = send_error_sent_nothing(exc)
            tc_err = _mark(
                TensorCostTimeoutError(
                    f"tensorcost: proxy request timed out after {format_seconds(timeout_s)}",
                    "total",
                    root_cause=exc,
                    attempt=attempt + 1,
                ),
                not sent_nothing,
            )
            last_error = tc_err
            if circuit is not None:
                circuit.record_failure()
            if sent_nothing and not fail_open_fast and _may_retry(attempt, retry, circuit, probe):
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
            # Only a connect failure (refused, DNS, TLS handshake) proves
            # nothing was written: TensorCost unreachable, where fail-open is
            # the product's promise. A reset or a protocol error may come
            # after the proxy forwarded the call, so it raises instead.
            sent_nothing = send_error_sent_nothing(exc)
            tc_err = _mark(
                TensorCostNetworkError(
                    f"tensorcost: network error reaching proxy: {exc}",
                    root_cause=exc,
                    attempt=attempt + 1,
                ),
                not sent_nothing,
            )
            last_error = tc_err
            if circuit is not None:
                circuit.record_failure()
            if sent_nothing and not fail_open_fast and _may_retry(attempt, retry, circuit, probe):
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
            # An error the SDK can't place: it may have come after the send.
            tc_err = _mark(
                TensorCostNetworkError(
                    f"tensorcost: unexpected error during proxy request: {exc}",
                    root_cause=exc,
                    attempt=attempt + 1,
                ),
                True,
            )
            last_error = tc_err
            if circuit is not None:
                circuit.record_failure()
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
            if response_sink is not None:
                response_sink.append(resp)
            return json.loads(resp.content), resp.headers.get("x-tc-decision")

        error_body: Any = parse_body(resp.content) if resp.status_code >= 400 else None
        if 400 <= resp.status_code < 500:
            # A TensorCost refusal is terminal whatever its status (a 429
            # refusal included): checked before the quota arm, never
            # retried, never counted against the circuit.
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

        # Where did the call get to? (_delivery) Only "unsent" may be retried
        # or fall back; "provider" is the provider's own answer.
        answer = (
            classify_error_answer(resp.status_code, resp.headers, error_body)
            if resp.status_code in (402, 429) or resp.status_code >= 500
            else None
        )
        unsent = answer == "unsent"
        if answer == "provider":
            # The proxy is healthy and relayed the provider: not a circuit failure.
            tc_err = _provider_answer_error(resp, error_body, request_id, attempt + 1)
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

        if unsent and is_provider_unreachable(resp.status_code, resp.headers):
            # The proxy couldn't connect to the provider and nothing was sent
            # (_delivery.is_provider_unreachable). Not retried through the
            # proxy, which already retried the provider once, and not a
            # circuit failure: the proxy is fine. Raised as unsent so the
            # wrapper can go direct.
            tc_err = _provider_unreachable_error(resp, request_id, attempt + 1)
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
                    reason="provider_unreachable",
                ),
            )
            raise tc_err

        if resp.status_code == 402:
            # No refusal body: the plan's monthly proxy-request ceiling (Scale
            # COGS gate). TensorCost declining to carry the call is not a
            # refusal of the call, and the server means for it to go direct,
            # so it counts as an outage. Not retried: it won't clear soon.
            ra_ms = parse_retry_after_ms(resp.headers.get("retry-after"))
            tc_err = _mark(
                TensorCostQuotaError(
                    "tensorcost: proxy returned 402 (plan request ceiling reached)",
                    retry_after_ms=int(ra_ms) if ra_ms is not None else None,
                    status=402,
                    request_id=request_id,
                    attempt=attempt + 1,
                ),
                not unsent,
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
            tc_err = _mark(
                TensorCostQuotaError(
                    "tensorcost: proxy returned 429 Too Many Requests",
                    retry_after_ms=ra_ms_int,
                    status=429,
                    request_id=request_id,
                    attempt=attempt + 1,
                ),
                not unsent,
            )
            last_error = tc_err
            if unsent and _may_retry(attempt, retry, circuit, probe):
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
            # TensorCost's own failure (the provider's was handled above).
            # Retried, and allowed to fall back, only when TensorCost proves
            # it answered before forwarding. Any other 5xx (a 500 or 502 the
            # proxy sent after the forward, a gateway 502/504 for a proxy that
            # reset or timed out mid-call, a 5xx with no origin header) may
            # follow a call the provider already ran and billed, so it is
            # raised to the caller instead, as the provider's SDK would raise
            # a 5xx it was told not to retry.
            tc_err = _mark(
                TensorCostProxyError(
                    f"tensorcost: proxy returned {resp.status_code}",
                    status=resp.status_code,
                    request_id=request_id,
                    attempt=attempt + 1,
                ),
                not unsent,
            )
            last_error = tc_err
            if circuit is not None:
                circuit.record_failure()
            if unsent and _may_retry(attempt, retry, circuit, probe):
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

        # Any other status: the provider's own error, relayed by the proxy
        # (always JSON), or, when the body isn't JSON, something in front of
        # TensorCost turning the request away. Not retried and never a
        # fallback; the provider would answer the same way again.
        tc_err = _non_refusal_4xx(resp, error_body, request_id, attempt + 1)
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


def _provider_unreachable_error(
    resp: httpx.Response, request_id: Optional[str], attempt: int
) -> TensorCostError:
    """The proxy's 502 for a provider it could not connect to: unsent, so fail-open may go direct."""
    return _mark(
        TensorCostProxyError(
            f"tensorcost: proxy returned {resp.status_code} "
            "(it could not reach the provider; nothing was sent)",
            status=resp.status_code,
            request_id=request_id,
            attempt=attempt,
        ),
        False,
    )


def _non_refusal_4xx(
    resp: httpx.Response, error_body: Any, request_id: Optional[str], attempt: int
) -> TensorCostError:
    """A 4xx that isn't a refusal: the provider's relayed error when the body is
    JSON, a gateway rejection when it isn't (HTML, text, or empty), and the
    proxy's own NestJS error (``{statusCode, message}``, which providers never
    send) as a TensorCost error."""
    if error_body is None or isinstance(error_body, str):
        return TensorCostGatewayRejectedError(
            resp.status_code,
            error_body[:500] if isinstance(error_body, str) else None,
            request_id=request_id,
            attempt=attempt,
        )
    if (
        isinstance(error_body, dict)
        and isinstance(error_body.get("statusCode"), int)
        and not isinstance(error_body.get("error"), dict)
    ):
        if resp.status_code == 401:
            return TensorCostKeyRejectedError(401, request_id=request_id, attempt=attempt)
        msg = error_body.get("message")
        return TensorCostRefusedError(
            f"tensorcost: the proxy rejected the request ({resp.status_code}"
            + (f": {msg}" if isinstance(msg, str) and msg else "")
            + ")",
            detail=error_body,
            status=resp.status_code,
            request_id=request_id,
            attempt=attempt,
        )
    return _provider_answer_error(resp, error_body, request_id, attempt)


def _provider_answer_error(
    resp: httpx.Response, error_body: Any, request_id: Optional[str], attempt: int
) -> TensorCostProviderError:
    """The provider's own answer, relayed by the proxy: a 4xx with a JSON body,
    or any status marked ``x-tc-error-origin: provider``. The call reached the
    provider, so it is never resent; the wrappers re-raise it as the provider
    SDK's own error type when they can."""
    provider_msg = provider_message_from(error_body)
    err = TensorCostProviderError(
        f"tensorcost: upstream provider returned {resp.status_code}"
        + (f": {provider_msg}" if provider_msg else ""),
        body=error_body,
        headers=dict(resp.headers),
        status=resp.status_code,
        request_id=request_id,
        attempt=attempt,
    )
    err.may_have_reached_provider = True
    return err


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
    extra_headers: Optional[dict[str, str]] = None,
    extra_query: Optional[dict[str, Any]] = None,
    raw: bool = False,
) -> tuple[Any, Optional[str]]:
    """Steer/route a streaming request; returns (sse_iterable, decision_header).

    With ``raw`` the first item is a ``RawProxyStream`` instead: the
    response's status, headers and undecoded-SSE bytes, for the provider
    SDK to parse into its own Stream (see _typed_response).
    """
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
    _add_caller_headers(headers, extra_headers)
    url = _with_query(url, extra_query)

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
    # allow-list since 1.5.0) recognises them as outages.
    try:
        resp = stream_ctx.__enter__()
    # Not retried at all; whether it may fall back depends on whether the
    # request can have been written (_delivery): a connect failure or connect
    # timeout can, a reset or a wait for headers after the send cannot.
    except httpx.TimeoutException as exc:
        if close_client:
            client.close()
        if circuit is not None:
            circuit.record_failure()
        raise _mark(
            TensorCostTimeoutError(
                f"tensorcost: proxy did not respond with headers within {format_seconds(headers_timeout_s)}",
                "headers",
                root_cause=exc,
                attempt=1,
            ),
            not send_error_sent_nothing(exc),
        ) from exc
    except (httpx.TransportError, OSError) as exc:
        if close_client:
            client.close()
        if circuit is not None:
            circuit.record_failure()
        raise _mark(
            TensorCostNetworkError(
                f"tensorcost: network error reaching proxy: {exc}",
                root_cause=exc,
                attempt=1,
            ),
            not send_error_sent_nothing(exc),
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
            answer = (
                classify_error_answer(resp.status_code, resp.headers, error_body)
                if resp.status_code in (402, 429) or resp.status_code >= 500
                else None
            )
            if answer == "provider":
                raise _provider_answer_error(resp, error_body, request_id, 1)
            if answer == "unsent" and is_provider_unreachable(resp.status_code, resp.headers):
                # See proxy_request: unreachable provider, nothing sent, proxy healthy.
                tc_err = _provider_unreachable_error(resp, request_id, 1)
                emit_event(
                    on_lifecycle_event,
                    OnErrorEvent(
                        kind="on_error",
                        provider=provider,
                        model=model,
                        operation=operation,
                        attempt_number=1,
                        elapsed_ms=0.0,
                        error_class=tc_err.__class__.__name__,
                        status=resp.status_code,
                        reason="provider_unreachable",
                    ),
                )
                raise tc_err
            if resp.status_code in (402, 429):
                ra_ms = parse_retry_after_ms(resp.headers.get("retry-after"))
                raise _mark(
                    TensorCostQuotaError(
                        "tensorcost: proxy returned 429 Too Many Requests"
                        if resp.status_code == 429
                        else "tensorcost: proxy returned 402 (plan request ceiling reached)",
                        retry_after_ms=int(ra_ms) if ra_ms is not None else None,
                        status=resp.status_code,
                        request_id=request_id,
                        attempt=1,
                    ),
                    answer != "unsent",
                )
            if resp.status_code >= 500:
                # See proxy_request: a 5xx not proven pre-forward is raised, never resent.
                if circuit is not None:
                    circuit.record_failure()
                raise _mark(
                    TensorCostProxyError(
                        f"tensorcost: proxy returned {resp.status_code}",
                        status=resp.status_code,
                        request_id=request_id,
                        attempt=1,
                    ),
                    answer != "unsent",
                )
            raise _non_refusal_4xx(resp, error_body, request_id, 1)
        if circuit is not None:
            circuit.record_success()

        if raw:
            def _close() -> None:
                try:
                    stream_ctx.__exit__(None, None, None)
                finally:
                    if close_client:
                        client.close()

            return RawProxyStream(resp, _close), resp.headers.get("x-tc-decision")

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


class RawProxyStream:
    """A successful streamed proxy response, not yet read."""

    def __init__(self, resp: httpx.Response, close: Any) -> None:
        self.status_code = resp.status_code
        self.headers = resp.headers
        self._resp = resp
        self._close = close
        self._closed = False

    def iter_bytes(self) -> Any:
        return self._resp.iter_bytes()

    def iter_json(self) -> Any:
        """The SSE data lines as dicts (the pre-1.7.1 return shape)."""
        try:
            for line in self._resp.iter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data in ("", "[DONE]"):
                    continue
                yield json.loads(data)
        finally:
            self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._close()


def _add_caller_headers(headers: dict[str, str], extra: Optional[dict[str, str]]) -> None:
    if not extra:
        return
    taken = {k.lower() for k in headers}
    for name, value in extra.items():
        if name.lower() not in taken:
            headers[name] = value


def _with_query(url: str, query: Optional[dict[str, Any]]) -> str:
    if not query:
        return url
    return str(httpx.URL(url).copy_merge_params(query))


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

    This is an allow-list on purpose. 1.4.0 had the inverse, a tuple of
    refusal classes that must NOT fall back, and every refusal code missing
    from it fell back by default. That is how a money budget's hard cap
    (BUDGET_HARD_CAP) got routed around. Now a refusal this SDK has never
    heard of is terminal, a relayed provider 4xx surfaces as itself, and an
    error the SDK didn't classify at all is surfaced rather than retried
    against the provider.

    Since 1.8.0 an outage falls back only when the call provably never
    reached the provider (``may_have_reached_provider`` false, ``_delivery``).
    A timeout, a reset, or a 5xx the proxy may have sent after forwarding is
    raised instead: calling the provider again would bill the call twice.
    """
    outage = isinstance(
        err,
        (
            TensorCostNetworkError,
            TensorCostTimeoutError,
            TensorCostProxyError,
            TensorCostQuotaError,
        ),
    )
    return outage and not getattr(err, "may_have_reached_provider", False)
