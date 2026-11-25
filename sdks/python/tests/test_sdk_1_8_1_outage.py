"""Findings from a self-hosted run of 1.8.0 with inference-proxy scaled to 0,
so the gateway answered 502 and then 503 itself (x-tc-error-origin: gateway).

F1. Calls the provider answered while TensorCost was down left no spend
    row: a failed token exchange or POST dropped the observation, and
    flush_observations() after recovery had nothing to send.
F2. With the circuit open, the call that became the half-open probe retried
    up to five times through the dead proxy before falling back: 5.0-7.6 s.
Nit. A 0.4 s timeout was reported as "timed out after 0s".
"""

from __future__ import annotations

import json
import logging
import threading
import time
import types
from typing import Any

import httpx
import pytest

from tensorcost import _circuit, _proxy_client
from tensorcost._circuit import CircuitBreaker, CircuitBreakerConfig
from tensorcost._errors import TensorCostTimeoutError, format_seconds
from tensorcost._providers import _openai
from tensorcost._proxy_client import proxy_request, should_fall_back_to_provider
from tensorcost._retry import DEFAULT_RETRY_CONFIG
from tensorcost._sdk_layer import SdkLayerClient
from tensorcost._transport import ObservationTransport

from .conftest import FakeOpenAIClient

TENANT = "11111111-1111-1111-1111-111111111111"
MSGS = [{"role": "user", "content": "x"}]


def _gateway_down(status: int) -> httpx.Response:
    return httpx.Response(status, text="upstream unavailable", headers={"x-tc-error-origin": "gateway"})


def _wait_for(pred, timeout_s: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.01)
    return pred()


class _TensorCost:
    """TensorCost behind a gateway: down (502, then 503) until ``up()``."""

    def __init__(self) -> None:
        self.down = True
        self.exchanges = 0
        self.accepted: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def up(self) -> None:
        self.down = False

    def handler(self, req: httpx.Request) -> httpx.Response:
        with self._lock:
            if req.url.path.endswith("/sdk-token/exchange"):
                self.exchanges += 1
                if self.down:
                    return _gateway_down(502 if self.exchanges == 1 else 503)
                return httpx.Response(200, json={"token": "jwt", "expires_in": 900})
            if self.down:
                return _gateway_down(503)
            self.accepted.append(json.loads(req.content))
            return httpx.Response(202)


def _transport(tc: _TensorCost, **kw: Any) -> ObservationTransport:
    return ObservationTransport(
        "https://tc.test", "k", tenant_id=TENANT,
        http_client=httpx.Client(transport=httpx.MockTransport(tc.handler)),
        backoff_base_s=0.01, backoff_max_s=0.05, flush_on_exit=False, **kw,
    )


# --- F1 ----------------------------------------------------------------------


def test_twelve_calls_during_an_outage_are_delivered_once_each_after_it_with_no_flush():
    tc = _TensorCost()
    t = _transport(tc)
    ids = [f"call-{i}" for i in range(12)]
    for i in ids:
        t.post({"correlation_id": i, "input_tokens": 10, "output_tokens": 2})
    time.sleep(0.3)
    assert tc.accepted == []
    assert t.stats().buffered + t.stats().in_flight == 12 and t.stats().dropped == 0
    # One probe per backoff step while down, not one exchange per observation.
    assert tc.exchanges < 40

    tc.up()
    assert _wait_for(lambda: len(tc.accepted) == 12)
    got = [o["correlation_id"] for o in tc.accepted]
    assert sorted(got) == sorted(ids) and len(set(got)) == 12
    s = t.stats()
    assert (s.buffered, s.in_flight, s.dropped, s.delivered_after_retry) == (0, 0, 0, 12)
    assert t.flush(1.0) == 0
    time.sleep(0.2)
    assert len(tc.accepted) == 12


def test_with_a_token_held_a_503_post_is_retried_and_lands_once():
    tc = _TensorCost()
    tc.up()
    t = _transport(tc, max_workers=1)
    t.post({"correlation_id": "warm"})
    assert t.flush(1.0) == 0
    tc.down = True
    t.post({"correlation_id": "x"})
    time.sleep(0.2)
    tc.up()
    assert t.flush(2.0) == 0
    assert [o["correlation_id"] for o in tc.accepted] == ["warm", "x"]
    assert tc.exchanges == 1


def test_flush_returns_how_many_are_still_waiting_while_tensorcost_is_down():
    tc = _TensorCost()
    t = ObservationTransport(
        "https://tc.test", "k", tenant_id=TENANT,
        http_client=httpx.Client(transport=httpx.MockTransport(tc.handler)),
        backoff_base_s=10, flush_on_exit=False,
    )
    for i in range(3):
        t.post({"correlation_id": f"c{i}"})
    started = time.monotonic()
    assert t.flush(2.0) == 3
    # It gives up once the next retry would land past the deadline.
    assert time.monotonic() - started < 1.0
    tc.up()
    assert t.flush(2.0) == 0
    assert len(tc.accepted) == 3


def test_the_resend_carries_the_same_bytes_so_the_same_correlation_id():
    bodies: list[bytes] = []
    n = {"posts": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path.endswith("/sdk-token/exchange"):
            return httpx.Response(200, json={"token": "jwt", "expires_in": 900})
        n["posts"] += 1
        bodies.append(req.content)
        if n["posts"] == 1:
            # The server took it but the answer was lost.
            raise httpx.ReadError("connection reset", request=req)
        return httpx.Response(202)

    t = ObservationTransport(
        "https://tc.test", "k", tenant_id=TENANT,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        backoff_base_s=0.01, flush_on_exit=False,
    )
    t.post({"correlation_id": "corr-1", "input_tokens": 3})
    assert t.flush(2.0) == 0
    assert len(bodies) == 2 and bodies[0] == bodies[1]
    assert json.loads(bodies[1])["correlation_id"] == "corr-1"


def test_a_rejected_observation_warns_once_then_at_most_once_a_minute(caplog):
    posts = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path.endswith("/sdk-token/exchange"):
            return httpx.Response(200, json={"token": "jwt", "expires_in": 900})
        posts["n"] += 1
        return httpx.Response(422)

    t = ObservationTransport(
        "https://tc.test", "k", tenant_id=TENANT,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)), flush_on_exit=False,
    )
    with caplog.at_level(logging.WARNING, logger="tensorcost"):
        for i in range(20):
            t.post({"correlation_id": f"r{i}"})
        assert t.flush(2.0) == 0
    assert posts["n"] == 20
    rejected = [r for r in caplog.records if "not retried" in r.getMessage()]
    assert len(rejected) == 1
    assert "status=422" in rejected[0].getMessage()
    assert t.stats().rejected == 20 and t.stats().buffered == 0


# --- F2 ----------------------------------------------------------------------


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def monotonic(self) -> float:
        return self.now


@pytest.fixture
def fake_time(monkeypatch):
    """The circuit's clock, and the proxy client's backoff sleeps, faked."""
    clock = _Clock()
    slept: list[float] = []
    monkeypatch.setattr(_circuit, "time", types.SimpleNamespace(monotonic=clock.monotonic))

    def fake_sleep(s: float) -> None:
        slept.append(s)
        clock.now += s

    monkeypatch.setattr(_proxy_client, "sleep_sync", fake_sleep)
    return clock, slept


def test_the_half_open_probe_is_one_attempt_then_the_call_goes_direct_at_once(recording_transport, monkeypatch, fake_time):
    clock, slept = fake_time
    proxy_hits = {"n": 0}

    def proxy_handler(_req: httpx.Request) -> httpx.Response:
        proxy_hits["n"] += 1
        return _gateway_down(503)

    real_proxy_request = _openai.proxy_request

    def via_dead_proxy(**kw: Any):
        kw["http_client"] = httpx.Client(transport=httpx.MockTransport(proxy_handler))
        return real_proxy_request(**kw)

    monkeypatch.setattr(_openai, "proxy_request", via_dead_proxy)
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
                    retry=DEFAULT_RETRY_CONFIG, circuit=circuit)

    # Call 1, CLOSED: retried until three failures open the circuit, not after.
    client.chat.completions.create(model="gpt-4o-mini", messages=MSGS)
    assert proxy_hits["n"] == 3 and len(slept) == 2
    assert circuit.state == "OPEN" and direct.call_count == 1

    # Call 2, inside the window: straight to the provider.
    client.chat.completions.create(model="gpt-4o-mini", messages=MSGS)
    assert proxy_hits["n"] == 3 and direct.call_count == 2

    # Call 3, past the window: the probe. One request, no backoff, direct.
    # In 1.8.0: five requests and 3.75-7.5 s of backoff first.
    clock.now += 5.01
    slept.clear()
    wall = time.monotonic()
    client.chat.completions.create(model="gpt-4o-mini", messages=MSGS)
    assert proxy_hits["n"] == 4
    assert slept == []
    assert time.monotonic() - wall < 0.5
    assert direct.call_count == 3
    assert circuit.state == "OPEN"

    # Open for a fresh window: no probe until it has passed, then one.
    clock.now += 4.0
    client.chat.completions.create(model="gpt-4o-mini", messages=MSGS)
    assert proxy_hits["n"] == 4
    clock.now += 1.1
    client.chat.completions.create(model="gpt-4o-mini", messages=MSGS)
    assert proxy_hits["n"] == 5 and slept == []
    assert direct.call_count == 5


def test_a_probe_answered_with_tensorcosts_own_429_is_not_retried_either(fake_time):
    _, slept = fake_time
    hits = {"n": 0}

    def handler(_req: httpx.Request) -> httpx.Response:
        hits["n"] += 1
        return httpx.Response(429, headers={"x-tc-error-origin": "proxy", "retry-after": "1"})

    cb = CircuitBreaker(CircuitBreakerConfig(open_threshold=1, probe_interval_s=0))
    cb.record_failure()
    assert cb.should_probe() and cb.state == "HALF_OPEN"
    with pytest.raises(Exception) as ei:
        proxy_request(
            proxy_url="https://proxy.test", bearer_token="jwt", provider_url="https://api.openai.com/v1",
            provider_auth="Bearer sk", body={"model": "m", "messages": []}, model="m", provider="openai",
            operation="chat.completions", correlation_id="c", environment=None,
            retry=DEFAULT_RETRY_CONFIG, circuit=cb,
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        )
    assert hits["n"] == 1 and slept == []
    assert should_fall_back_to_provider(ei.value)


# --- Nit ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "seconds, text",
    [(0.4, "0.4s"), (2.0, "2s"), (1.25, "1.25s"), (60.0, "60s"), (0.05, "50ms"), (600.0, "600s")],
)
def test_timeouts_are_written_as_people_read_them(seconds, text):
    assert format_seconds(seconds) == text


def test_a_sub_second_proxy_timeout_reads_as_such():
    def handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=req)

    with pytest.raises(TensorCostTimeoutError) as ei:
        proxy_request(
            proxy_url="https://proxy.test", bearer_token="jwt", provider_url="https://api.openai.com/v1",
            provider_auth="Bearer sk", body={"model": "m", "messages": []}, model="m", provider="openai",
            operation="chat.completions", correlation_id="c", environment=None,
            timeout_s=0.4, headers_timeout_s=0.4,
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        )
    assert "0.4s" in str(ei.value) and "after 0s" not in str(ei.value)
