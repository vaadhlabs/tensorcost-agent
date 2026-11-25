"""1.8.2: the proxy's answer for a provider it could not connect to.

inference-proxy-service answers a provider send that got no answer in one of
three ways (unanswered-send.ts), each with ``x-tc-upstream-error``:

provider_unreachable      502, no x-should-retry. Connect, DNS or TLS failed
                          before anything was written, after the proxy's own
                          one retry. Nothing was billed.
provider_connection_lost  502, x-should-retry: false. May have run.
provider_timeout          504, x-should-retry: false. May have run.

All three carry ``x-tc-error-origin: provider``. Through 1.8.1 the SDK raised
all three as the provider's error and never fell open. The first is proven
unsent, so going to the provider directly is safe: 1.8.2 does, when fail-open
is on, without retrying through the proxy or counting a circuit failure (the
proxy is healthy).
"""

from __future__ import annotations

import types
from typing import Any

import httpx
import pytest

from tensorcost import _proxy_client
from tensorcost._circuit import CircuitBreaker, CircuitBreakerConfig
from tensorcost._delivery import classify_error_answer
from tensorcost._errors import TensorCostError, TensorCostProxyError
from tensorcost._providers import _openai
from tensorcost._proxy_client import proxy_request, proxy_request_stream, should_fall_back_to_provider
from tensorcost._retry import DEFAULT_RETRY_CONFIG, RetryConfig
from tensorcost._sdk_layer import SdkLayerClient

from .conftest import FakeOpenAIClient

TENANT = "11111111-1111-1111-1111-111111111111"
MSGS = [{"role": "user", "content": "x"}]
FAST_RETRY = RetryConfig(max_attempts=3, base_delay_ms=1, max_delay_ms=2)


def _unanswered(kind: str, **extra: str) -> httpx.Response:
    """The proxy's answer for ``kind``, headers as unanswered-send.ts and error-origin.ts write them."""
    status = 504 if kind == "provider_timeout" else 502
    headers = {"x-tc-error-origin": "provider", "x-tc-upstream-error": kind}
    if kind != "provider_unreachable":
        headers["x-should-retry"] = "false"
    headers.update({k.replace("_", "-"): v for k, v in extra.items()})
    err_type = "upstream_timeout" if kind == "provider_timeout" else "upstream_unavailable"
    return httpx.Response(
        status,
        headers=headers,
        json={"error": {"type": err_type, "code": kind, "message": "TensorCost couldn't reach OpenAI"}},
    )


# --- the classifier ------------------------------------------------------------


@pytest.mark.parametrize("origin", ["provider", "proxy", None])
def test_502_provider_unreachable_without_should_retry_is_unsent_whatever_the_origin(origin):
    headers = httpx.Headers({"x-tc-upstream-error": "provider_unreachable"})
    if origin:
        headers["x-tc-error-origin"] = origin
    assert classify_error_answer(502, headers, None) == "unsent"


def test_should_retry_false_still_wins_over_provider_unreachable():
    headers = httpx.Headers({
        "x-tc-error-origin": "provider",
        "x-tc-upstream-error": "provider_unreachable",
        "x-should-retry": "false",
    })
    assert classify_error_answer(502, headers, None) == "provider"
    del headers["x-tc-error-origin"]
    assert classify_error_answer(502, headers, None) == "maybe_sent"


def test_provider_unreachable_on_anything_but_a_502_is_classified_as_before():
    headers = httpx.Headers({"x-tc-error-origin": "provider", "x-tc-upstream-error": "provider_unreachable"})
    assert classify_error_answer(504, headers, None) == "provider"
    assert classify_error_answer(500, headers, None) == "provider"


@pytest.mark.parametrize(
    "status, kind",
    [(502, "provider_connection_lost"), (504, "provider_timeout"), (502, "something_new")],
)
def test_other_upstream_error_kinds_are_classified_as_before(status, kind):
    for origin, expected in (("provider", "provider"), ("proxy", "maybe_sent")):
        headers = httpx.Headers({"x-tc-error-origin": origin, "x-tc-upstream-error": kind})
        assert classify_error_answer(status, headers, None) == expected


# --- the proxy client, buffered and streamed ---------------------------------


@pytest.fixture
def no_sleep(monkeypatch):
    slept: list[float] = []
    monkeypatch.setattr(_proxy_client, "sleep_sync", slept.append)
    return slept


def _send(send, respond, circuit: CircuitBreaker, events: list) -> tuple[BaseException, int]:
    hits = {"n": 0}

    def handler(_req: httpx.Request) -> httpx.Response:
        hits["n"] += 1
        return respond()

    with pytest.raises(Exception) as ei:
        send(
            proxy_url="https://proxy.test", bearer_token="jwt", provider_url="https://api.openai.com/v1",
            provider_auth="Bearer sk", body={"model": "m", "messages": []}, model="m", provider="openai",
            operation="chat.completions", correlation_id="c", environment=None,
            retry=FAST_RETRY, circuit=circuit, on_lifecycle_event=events.append,
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        )
    return ei.value, hits["n"]


SENDS = [pytest.param(proxy_request, id="proxy_request"), pytest.param(proxy_request_stream, id="proxy_request_stream")]


@pytest.mark.parametrize("send", SENDS)
def test_provider_unreachable_is_one_attempt_no_circuit_failure_and_may_fall_back(send, no_sleep):
    circuit = CircuitBreaker(CircuitBreakerConfig(open_threshold=1))
    events: list = []
    err, hits = _send(send, lambda: _unanswered("provider_unreachable"), circuit, events)

    assert isinstance(err, TensorCostProxyError)
    assert err.status == 502
    assert err.may_have_reached_provider is False
    assert should_fall_back_to_provider(err)
    assert hits == 1 and no_sleep == []
    assert circuit.failures == 0 and circuit.state == "CLOSED"
    assert [e for e in events if e.kind == "on_retry"] == []
    errors = [e for e in events if e.kind == "on_error"]
    assert len(errors) == 1
    assert errors[0].error_class == "TensorCostProxyError"
    assert errors[0].status == 502
    assert errors[0].reason == "provider_unreachable"


@pytest.mark.parametrize("send", SENDS)
@pytest.mark.parametrize("kind", ["provider_connection_lost", "provider_timeout"])
def test_connection_lost_and_timeout_are_as_in_1_8_1_never_retried_never_a_fallback(send, kind, no_sleep):
    circuit = CircuitBreaker(CircuitBreakerConfig(open_threshold=1))
    err, hits = _send(send, lambda: _unanswered(kind), circuit, [])
    assert isinstance(err, TensorCostError)
    assert not isinstance(err, TensorCostProxyError)
    assert err.may_have_reached_provider is True
    assert not should_fall_back_to_provider(err)
    assert hits == 1 and no_sleep == []
    assert circuit.failures == 0


@pytest.mark.parametrize("send", SENDS)
def test_provider_unreachable_with_should_retry_false_may_have_been_sent(send, no_sleep):
    err, hits = _send(
        send, lambda: _unanswered("provider_unreachable", x_should_retry="false"), CircuitBreaker(), []
    )
    assert err.may_have_reached_provider is True
    assert not should_fall_back_to_provider(err)
    assert hits == 1


# --- through the OpenAI wrapper: circuit, fallback, fail-open switch ---------


def _install(monkeypatch, recording_transport, kind: str, **kw: Any):
    hits = {"n": 0}

    def proxy_handler(_req: httpx.Request) -> httpx.Response:
        hits["n"] += 1
        return _unanswered(kind)

    real_proxy_request = _openai.proxy_request

    def via_proxy(**k: Any):
        k["http_client"] = httpx.Client(transport=httpx.MockTransport(proxy_handler))
        return real_proxy_request(**k)

    monkeypatch.setattr(_openai, "proxy_request", via_proxy)
    layer = SdkLayerClient(
        base_url="https://tc.test", get_token=lambda: "jwt",
        http_client=httpx.Client(transport=httpx.MockTransport(
            lambda _r: httpx.Response(200, json={"published_layer": "route"}))),
    )
    client = FakeOpenAIClient()
    direct = client.chat.completions.create
    circuit = CircuitBreaker(CircuitBreakerConfig())
    events: list = []
    _openai.install(client, recording_transport, tenant_id=TENANT, environment=None, connection_id=None,
                    max_layer="route", proxy_url="https://proxy.test", sdk_layer=layer,
                    base_url="https://tc.test", on_lifecycle_event=events.append,
                    retry=DEFAULT_RETRY_CONFIG, circuit=circuit, **kw)
    return types.SimpleNamespace(client=client, direct=direct, circuit=circuit, events=events, hits=hits)


def test_fail_open_on_one_proxy_request_then_the_provider_and_the_circuit_counts_nothing(
    recording_transport, monkeypatch, no_sleep
):
    w = _install(monkeypatch, recording_transport, "provider_unreachable")
    # Four calls, more than the three failures that open the circuit: each
    # still goes through the proxy first, once, then direct.
    for i in range(1, 5):
        w.client.chat.completions.create(model="gpt-4o-mini", messages=MSGS)
        assert w.hits["n"] == i
        assert w.direct.call_count == i
    assert w.circuit.failures == 0 and w.circuit.state == "CLOSED"
    assert no_sleep == []
    fallbacks = [e for e in w.events if e.kind == "on_fallback"]
    assert len(fallbacks) == 4
    assert all(e.consecutive_failures == 0 for e in fallbacks)


def test_fail_open_off_raises_the_proxy_error_unsent_and_never_calls_the_provider(
    recording_transport, monkeypatch, no_sleep
):
    w = _install(monkeypatch, recording_transport, "provider_unreachable", fail_open_enabled=False)
    with pytest.raises(TensorCostProxyError) as ei:
        w.client.chat.completions.create(model="gpt-4o-mini", messages=MSGS)
    assert ei.value.status == 502
    assert ei.value.may_have_reached_provider is False
    assert w.hits["n"] == 1
    assert w.direct.call_count == 0


@pytest.mark.parametrize("kind", ["provider_connection_lost", "provider_timeout"])
def test_connection_lost_and_timeout_with_fail_open_on_are_raised_after_one_request(
    recording_transport, monkeypatch, no_sleep, kind
):
    w = _install(monkeypatch, recording_transport, kind)
    with pytest.raises(Exception) as ei:
        w.client.chat.completions.create(model="gpt-4o-mini", messages=MSGS)
    assert not isinstance(ei.value, TensorCostProxyError)
    assert w.hits["n"] == 1
    assert w.direct.call_count == 0
