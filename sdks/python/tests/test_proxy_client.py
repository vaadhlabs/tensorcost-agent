"""Tests for the proxy_request() applied-mode HTTP engine."""

from __future__ import annotations

import json
from typing import Optional
from unittest.mock import MagicMock

import httpx
import pytest

from tensorcost._circuit import CircuitBreaker, CircuitBreakerConfig
from tensorcost._errors import (
    TensorCostNetworkError,
    TensorCostProxyError,
    TensorCostProviderError,
    TensorCostQuotaError,
    TensorCostTimeoutError,
)
from tensorcost._proxy_client import proxy_request
from tensorcost._retry import RetryConfig
from tensorcost._telemetry import (
    AfterResponseEvent,
    BeforeRequestEvent,
    OnErrorEvent,
    OnRetryEvent,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client(responses: list) -> httpx.Client:
    """Build an httpx.Client backed by a FIFO list of responses.

    Each item in *responses* is either an httpx.Response or an Exception to
    be raised.
    """
    call_count = [0]

    def handler(request: httpx.Request) -> httpx.Response:
        idx = call_count[0]
        call_count[0] += 1
        item = responses[idx] if idx < len(responses) else responses[-1]
        if isinstance(item, Exception):
            raise item
        return item

    transport = httpx.MockTransport(handler)
    return httpx.Client(transport=transport)


def _json_resp(body: dict, status: int = 200) -> httpx.Response:
    return httpx.Response(
        status_code=status,
        headers={"Content-Type": "application/json"},
        content=json.dumps(body).encode(),
    )


def _base_kwargs(
    *,
    retry: Optional[RetryConfig] = None,
    circuit: Optional[CircuitBreaker] = None,
    on_lifecycle_event=None,
    http_client: Optional[httpx.Client] = None,
    fail_open_fast: bool = False,
) -> dict:
    return dict(
        proxy_url="https://proxy.example.com",
        bearer_token="jwt-tok",
        provider_url="https://api.openai.com",
        provider_auth="Bearer sk-test",
        body={"model": "gpt-4o-mini", "messages": []},
        model="gpt-4o-mini",
        provider="openai",
        operation="chat.completions",
        correlation_id="corr-1",
        environment=None,
        retry=retry or RetryConfig(max_attempts=1),
        on_lifecycle_event=on_lifecycle_event,
        circuit=circuit,
        fail_open_fast=fail_open_fast,
        http_client=http_client,
    )


# ---------------------------------------------------------------------------
# Success path
# ---------------------------------------------------------------------------


class TestProxyRequestSuccess:
    def test_returns_parsed_body(self):
        body = {"id": "cmp-1", "choices": []}
        client = _make_client([_json_resp(body)])
        body_out, decision_header = proxy_request(**_base_kwargs(http_client=client))
        assert body_out == body
        assert decision_header is None

    def test_records_circuit_success(self):
        cb = CircuitBreaker()
        client = _make_client([_json_resp({})])
        proxy_request(**_base_kwargs(circuit=cb, http_client=client))
        assert cb.state == "CLOSED"

    def test_emits_before_request_and_after_response_events(self):
        events = []
        client = _make_client([_json_resp({"id": "x"})])
        proxy_request(**_base_kwargs(on_lifecycle_event=events.append, http_client=client))
        kinds = [e.kind for e in events]
        assert "before_request" in kinds
        assert "after_response" in kinds

    def test_environment_header_included(self):
        captured = []

        def handler(req: httpx.Request) -> httpx.Response:
            captured.append(dict(req.headers))
            return _json_resp({})

        client = httpx.Client(transport=httpx.MockTransport(handler))
        kwargs = _base_kwargs(http_client=client)
        kwargs["environment"] = "staging"
        proxy_request(**kwargs)
        assert captured[0].get("x-tc-environment") == "staging"


# ---------------------------------------------------------------------------
# 4xx provider error (non-retriable)
# ---------------------------------------------------------------------------


class TestProviderError:
    def test_4xx_raises_provider_error(self):
        client = _make_client([_json_resp({}, status=401)])
        with pytest.raises(TensorCostProviderError) as exc_info:
            proxy_request(**_base_kwargs(http_client=client))
        assert exc_info.value.status == 401

    def test_4xx_not_retried(self):
        call_count = [0]

        def handler(req: httpx.Request) -> httpx.Response:
            call_count[0] += 1
            return _json_resp({}, status=400)

        client = httpx.Client(transport=httpx.MockTransport(handler))
        with pytest.raises(TensorCostProviderError):
            proxy_request(**_base_kwargs(
                retry=RetryConfig(max_attempts=3),
                http_client=client,
            ))
        assert call_count[0] == 1

    def test_4xx_emits_on_error_event(self):
        events = []
        client = _make_client([_json_resp({}, status=403)])
        with pytest.raises(TensorCostProviderError):
            proxy_request(**_base_kwargs(on_lifecycle_event=events.append, http_client=client))
        assert any(e.kind == "on_error" for e in events)


# ---------------------------------------------------------------------------
# 5xx proxy error with retries
# ---------------------------------------------------------------------------


class TestProxyError:
    def test_5xx_raises_proxy_error_after_exhausting_retries(self):
        client = _make_client([_json_resp({}, status=502)])
        with pytest.raises(TensorCostProxyError) as exc_info:
            proxy_request(**_base_kwargs(
                retry=RetryConfig(max_attempts=1),
                http_client=client,
            ))
        assert exc_info.value.status == 502

    def test_5xx_retried_until_success(self):
        client = _make_client([
            _json_resp({}, status=503),
            _json_resp({"id": "ok"}, status=200),
        ])
        body_out, _ = proxy_request(**_base_kwargs(
            retry=RetryConfig(max_attempts=2, base_delay_ms=0.0, max_delay_ms=0.0),
            http_client=client,
        ))
        assert body_out == {"id": "ok"}

    def test_5xx_increments_circuit_failure(self):
        cb = CircuitBreaker(CircuitBreakerConfig(open_threshold=3))
        client = _make_client([_json_resp({}, status=500)])
        with pytest.raises(TensorCostProxyError):
            proxy_request(**_base_kwargs(circuit=cb, http_client=client))
        assert cb.failures == 1

    def test_5xx_emits_on_retry_then_on_error(self):
        events = []
        client = _make_client([
            _json_resp({}, status=500),
            _json_resp({}, status=500),
        ])
        with pytest.raises(TensorCostProxyError):
            proxy_request(**_base_kwargs(
                retry=RetryConfig(max_attempts=2, base_delay_ms=0.0, max_delay_ms=0.0),
                on_lifecycle_event=events.append,
                http_client=client,
            ))
        kinds = [e.kind for e in events]
        assert "on_retry" in kinds
        assert "on_error" in kinds


# ---------------------------------------------------------------------------
# 429 quota error with retries
# ---------------------------------------------------------------------------


class TestQuotaError:
    def test_429_raises_quota_error_after_exhausting_retries(self):
        resp = httpx.Response(
            status_code=429,
            headers={"Content-Type": "application/json", "Retry-After": "1"},
            content=b"{}",
        )
        client = _make_client([resp])
        with pytest.raises(TensorCostQuotaError) as exc_info:
            proxy_request(**_base_kwargs(retry=RetryConfig(max_attempts=1), http_client=client))
        assert exc_info.value.status == 429
        assert exc_info.value.retry_after_ms == 1000

    def test_429_retried_until_success(self):
        resp_429 = httpx.Response(
            status_code=429,
            headers={"Content-Type": "application/json"},
            content=b"{}",
        )
        client = _make_client([resp_429, _json_resp({"ok": True})])
        body_out, _ = proxy_request(**_base_kwargs(
            retry=RetryConfig(max_attempts=2, base_delay_ms=0.0, max_delay_ms=0.0),
            http_client=client,
        ))
        assert body_out == {"ok": True}

    def test_429_does_not_count_as_circuit_failure(self):
        resp_429 = httpx.Response(
            status_code=429,
            headers={"Content-Type": "application/json"},
            content=b"{}",
        )
        cb = CircuitBreaker(CircuitBreakerConfig(open_threshold=1))
        client = _make_client([resp_429])
        with pytest.raises(TensorCostQuotaError):
            proxy_request(**_base_kwargs(circuit=cb, http_client=client))
        # Circuit should stay closed — 429 is a quota, not a proxy failure.
        assert cb.state == "CLOSED"


# ---------------------------------------------------------------------------
# Network errors
# ---------------------------------------------------------------------------


class TestNetworkErrors:
    def test_connect_error_raises_network_error(self):
        client = _make_client([httpx.ConnectError("connection refused")])
        with pytest.raises(TensorCostNetworkError):
            proxy_request(**_base_kwargs(http_client=client))

    def test_network_error_retried(self):
        client = _make_client([
            httpx.ConnectError("failed"),
            _json_resp({"ok": True}),
        ])
        body_out, _ = proxy_request(**_base_kwargs(
            retry=RetryConfig(max_attempts=2, base_delay_ms=0.0, max_delay_ms=0.0),
            http_client=client,
        ))
        assert body_out == {"ok": True}

    def test_network_error_not_retried_when_fail_open_fast(self):
        call_count = [0]

        def handler(_req: httpx.Request) -> httpx.Response:
            call_count[0] += 1
            raise httpx.ConnectError("failed")

        client = httpx.Client(transport=httpx.MockTransport(handler))
        with pytest.raises(TensorCostNetworkError):
            proxy_request(**_base_kwargs(
                retry=RetryConfig(max_attempts=3, base_delay_ms=0.0, max_delay_ms=0.0),
                fail_open_fast=True,
                http_client=client,
            ))
        assert call_count[0] == 1

    def test_network_error_increments_circuit_failures(self):
        cb = CircuitBreaker(CircuitBreakerConfig(open_threshold=3))
        client = _make_client([httpx.ConnectError("no route")])
        with pytest.raises(TensorCostNetworkError):
            proxy_request(**_base_kwargs(circuit=cb, http_client=client))
        assert cb.failures == 1


# ---------------------------------------------------------------------------
# Timeout errors
# ---------------------------------------------------------------------------


class TestTimeoutErrors:
    def test_read_timeout_before_headers_is_headers_kind(self):
        client = _make_client([httpx.ReadTimeout("timed out")])
        kwargs = _base_kwargs(http_client=client)
        kwargs["body"] = {"model": "gpt-4o-mini", "messages": [], "stream": True}
        with pytest.raises(TensorCostTimeoutError) as exc_info:
            proxy_request(**kwargs)
        assert exc_info.value.timeout_kind == "headers"

    def test_headers_timeout_raises_headers_kind(self):
        from tensorcost._proxy_client import _post_with_headers_timeout

        class _StuckClient:
            def stream(self, *_args, **_kwargs):
                class _CM:
                    def __enter__(self):
                        raise httpx.ReadTimeout("no headers")

                    def __exit__(self, *_a):
                        return False

                return _CM()

            def close(self) -> None:
                pass

        with pytest.raises(TensorCostTimeoutError) as exc_info:
            _post_with_headers_timeout(
                _StuckClient(),  # type: ignore[arg-type]
                url="https://proxy.example.com/x",
                json_body={},
                headers={},
                headers_wait_budget_s=2.0,
                short_headers_timeout_s=2.0,
                body_timeout_s=60.0,
                attempt=1,
            )
        assert exc_info.value.timeout_kind == "headers"

    def test_body_read_may_exceed_headers_timeout(self):
        import time

        from tensorcost._proxy_client import _post_with_headers_timeout

        class _SlowBodyStream:
            def __init__(self) -> None:
                self._entered = False

            def stream(self, *_args, **_kwargs):
                outer = self

                class _CM:
                    def __enter__(self):
                        outer._entered = True
                        return _SlowBodyResponse()

                    def __exit__(self, *_a):
                        return False

                return _CM()

            def close(self) -> None:
                pass

        class _SlowBodyResponse:
            status_code = 200
            headers = httpx.Headers({"content-type": "application/json"})
            request = httpx.Request("POST", "https://proxy.example.com/x")

            def iter_bytes(self, chunk_size: int = 1024):
                time.sleep(0.05)
                yield b'{"ok": true}'

        client = _SlowBodyStream()
        resp = _post_with_headers_timeout(
            client,  # type: ignore[arg-type]
            url="https://proxy.example.com/x",
            json_body={},
            headers={},
            headers_wait_budget_s=5.0,
            short_headers_timeout_s=2.0,
            body_timeout_s=5.0,
            attempt=1,
        )
        assert resp.status_code == 200
        assert json.loads(resp.content) == {"ok": True}


# ---------------------------------------------------------------------------
# Circuit breaker open — immediate raise
# ---------------------------------------------------------------------------


class TestCircuitBreakerOpen:
    def test_half_open_probe_still_reaches_http(self):
        cb = CircuitBreaker(CircuitBreakerConfig(open_threshold=1))
        cb.record_failure()
        assert cb.should_probe()
        assert cb.state == "HALF_OPEN"

        client = _make_client([_json_resp({"id": "ok", "usage": {}})])
        body, _ = proxy_request(
            **_base_kwargs(circuit=cb, http_client=client, retry=RetryConfig(max_attempts=1)),
        )
        assert body["id"] == "ok"

    def test_attempt_count_in_network_error(self):
        cb = CircuitBreaker(CircuitBreakerConfig(open_threshold=1))
        cb.record_failure()
        cb.should_probe()
        client = _make_client([httpx.ConnectError("refused")])
        with pytest.raises(TensorCostNetworkError) as exc_info:
            proxy_request(
                **_base_kwargs(
                    circuit=cb,
                    http_client=client,
                    retry=RetryConfig(max_attempts=1),
                    fail_open_fast=True,
                ),
            )
        assert exc_info.value.attempt == 1
