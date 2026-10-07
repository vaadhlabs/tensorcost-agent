"""Applied-mode proxy HTTP client for the Python SDK.

Contract:

* Returns the parsed response dict on success.
* On all non-success outcomes, raises a subclass of ``TensorCostError``.
* Never swallows errors — the provider wrapper's fail-open logic lives there,
  not here.

Timeout handling:

* ``headers_timeout_s`` (default 2s): abort if response headers never arrive.
* ``timeout_s`` (default 60s): wall-clock budget for the full response body.
"""

from __future__ import annotations

import json
import sys
import time
from typing import Any, Optional

import httpx

from ._circuit import CircuitBreaker
from ._errors import (
    TensorCostComplianceDeniedError,
    TensorCostComplianceTeamMismatchError,
    TensorCostGuardrailHardStopError,
    TensorCostModelGovernanceDeniedError,
    TensorCostNetworkError,
    TensorCostProxyError,
    TensorCostProviderError,
    TensorCostQuotaError,
    TensorCostPeriodBudgetExceededError,
    TensorCostRunBudgetExceededError,
    TensorCostTimeoutError,
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

        if resp.status_code == 403:
            tc_err = _terminal_proxy_refusal_error(
                resp.content,
                request_id=request_id,
                attempt=attempt + 1,
            )
            if tc_err is not None:
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
                        status=403,
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

        tc_err = TensorCostProviderError(
            f"tensorcost: upstream provider returned {resp.status_code}",
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


def _extract_proxy_refusal_code(content: bytes) -> tuple[Optional[str], dict[str, Any]]:
    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict):
            if isinstance(parsed.get("code"), str):
                return parsed["code"], parsed
            err_obj = parsed.get("error")
            if isinstance(err_obj, dict) and isinstance(err_obj.get("code"), str):
                return err_obj["code"], err_obj
    except (json.JSONDecodeError, TypeError):
        pass
    return None, {}


def _terminal_proxy_refusal_error(
    content: bytes,
    *,
    request_id: Optional[str],
    attempt: int,
) -> Optional[BaseException]:
    """Map a 403 proxy body to a terminal refusal error, or None if not ours."""
    code, detail = _extract_proxy_refusal_code(content)
    if code in ("RUN_BUDGET_EXCEEDED", "PERIOD_BUDGET_EXCEEDED"):
        if code == "PERIOD_BUDGET_EXCEEDED":
            return TensorCostPeriodBudgetExceededError(
                str(detail.get("message") or "tensorcost: agent period spend cap reached"),
                scope=detail.get("scope")
                if isinstance(detail.get("scope"), str)
                else None,
                period=detail.get("period")
                if isinstance(detail.get("period"), str)
                else None,
                cap_cents=detail.get("cap_cents")
                if isinstance(detail.get("cap_cents"), (int, float))
                else None,
                spent_cents=detail.get("spent_cents")
                if isinstance(detail.get("spent_cents"), (int, float))
                else None,
                status=403,
                request_id=request_id,
                attempt=attempt,
            )
        return TensorCostRunBudgetExceededError(
            str(detail.get("message") or "tensorcost: agent run spend cap reached"),
            workflow_id=detail.get("workflow_id")
            if isinstance(detail.get("workflow_id"), str)
            else None,
            cap_cents=detail.get("cap_cents")
            if isinstance(detail.get("cap_cents"), (int, float))
            else None,
            spent_cents=detail.get("spent_cents")
            if isinstance(detail.get("spent_cents"), (int, float))
            else None,
            status=403,
            request_id=request_id,
            attempt=attempt,
        )
    if code == "MODEL_GOVERNANCE_DENIED":
        return TensorCostModelGovernanceDeniedError(
            str(detail.get("message") or "tensorcost: model denied by tenant governance"),
            provider=detail.get("provider")
            if isinstance(detail.get("provider"), str)
            else None,
            model=detail.get("model") if isinstance(detail.get("model"), str) else None,
            governance_status=detail.get("governance_status")
            if isinstance(detail.get("governance_status"), str)
            else None,
            status=403,
            request_id=request_id,
            attempt=attempt,
        )
    if code == "GUARDRAIL_HARD_STOP":
        return TensorCostGuardrailHardStopError(
            str(detail.get("message") or "tensorcost: guardrail hard_stop"),
            policy_id=detail.get("policy_id")
            if isinstance(detail.get("policy_id"), str)
            else None,
            status=403,
            request_id=request_id,
            attempt=attempt,
        )
    if code == "COMPLIANCE_DENIED":
        return TensorCostComplianceDeniedError(
            str(detail.get("message") or "tensorcost: request refused by compliance policy"),
            status=403,
            request_id=request_id,
            attempt=attempt,
        )
    if code == "COMPLIANCE_TEAM_MISMATCH":
        return TensorCostComplianceTeamMismatchError(
            str(
                detail.get("message")
                or "tensorcost: SDK team scope does not match token binding"
            ),
            status=403,
            request_id=request_id,
            attempt=attempt,
        )
    return None


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
    resp = stream_ctx.__enter__()
    request_id = resp.headers.get("x-tc-request-id")
    try:
        if resp.status_code >= 400:
            body = resp.read()
            if resp.status_code == 403:
                tc_err = _terminal_proxy_refusal_error(
                    body,
                    request_id=request_id,
                    attempt=1,
                )
                if tc_err is not None:
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
                            status=403,
                        ),
                    )
                    raise tc_err
            detail = body.decode("utf-8", errors="replace")
            raise TensorCostProviderError(
                f"tensorcost: upstream provider returned {resp.status_code}: {detail}",
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
    """Return False for terminal policy refusals that must not fail-open."""
    return not isinstance(
        err,
        (
            TensorCostRunBudgetExceededError,
            TensorCostPeriodBudgetExceededError,
            TensorCostModelGovernanceDeniedError,
            TensorCostGuardrailHardStopError,
            TensorCostComplianceDeniedError,
            TensorCostComplianceTeamMismatchError,
        ),
    )
