"""A bare 429 from TensorCost is pacing, not an outage.

Found by a self-hosted load test: the proxy's admit limit was 600/min for the
install's one tenant, and every call over it went to the provider with no
budget or governance check, marked ``admit_unavailable`` like an outage and
logged nowhere. Now /admit retries a 429 inside its own budget, honouring
Retry-After (also for the calls after it), and when it is still throttled the
configured fail mode applies exactly as for an outage, with its own reason on
the observation and the fall-back event, a warning at most once a minute, and
a per-process count in ``unchecked_calls()``. Mirrors the Node SDK's
admit-throttled.spec.ts.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import httpx
import pytest

import tensorcost
from tensorcost import _admit_client, _unchecked
from tensorcost._admit_client import admit_call
from tensorcost._errors import TensorCostProxyError, TensorCostQuotaError, TensorCostRefusedError
from tensorcost._providers import _anthropic, _openai
from tensorcost._sdk_layer import SdkLayerClient

from .conftest import FakeAnthropicClient, FakeOpenAIClient

TENANT = "11111111-1111-1111-1111-111111111111"
MSGS = [{"role": "user", "content": "x"}]
META = {"provider": "openai", "model": "gpt-4o-mini", "operation": "chat.completions", "correlation_id": "c-1"}


@pytest.fixture(autouse=True)
def _fresh():
    _admit_client._reset()
    _unchecked._reset()
    yield
    _admit_client._reset()
    _unchecked._reset()


class Script:
    """/v1/admit answers each call from the script in turn; the last repeats."""

    def __init__(self, script: list[httpx.Response]) -> None:
        self.script = script
        self.at: list[float] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.at.append(time.monotonic())
        r = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        return httpx.Response(r.status_code, headers=r.headers, content=r.content)

    @property
    def count(self) -> int:
        return len(self.at)


def bare(status: int, **headers: str) -> httpx.Response:
    return httpx.Response(status, headers={k.replace("_", "-"): v for k, v in headers.items()})


def admit(script: Script, **kw: Any):
    return admit_call(
        base_url="https://tc.test",
        bearer_token="tok",
        max_layer="govern",
        metadata=META,
        http_client=httpx.Client(transport=httpx.MockTransport(script)),
        **kw,
    )


# ----------------------------------------------------------------------
# admit_call on a 429


def test_429_without_retry_after_is_retried_and_admitted():
    s = Script([bare(429), bare(204)])
    out = admit(s)
    assert out.allowed and not out.unavailable
    assert s.count == 2


def test_retry_after_inside_the_budget_is_waited_out():
    s = Script([bare(429, retry_after="0.15"), bare(204)])
    out = admit(s)
    assert out.allowed and not out.unavailable
    assert s.count == 2
    assert s.at[1] - s.at[0] >= 0.145


def test_retry_after_past_the_budget_gives_up_at_once_as_throttled():
    s = Script([bare(429, retry_after="2")])
    t0 = time.monotonic()
    out = admit(s)
    assert time.monotonic() - t0 < 0.1
    assert out.allowed and out.unavailable and out.reason == "admit_throttled"
    assert s.count == 1


def test_that_retry_after_holds_the_next_calls_without_sending():
    s = Script([bare(429, retry_after="2")])
    admit(s)
    assert admit(s).reason == "admit_throttled"
    assert s.count == 1


def test_every_answer_a_429_keeps_retrying_inside_the_budget():
    s = Script([bare(429)])
    t0 = time.monotonic()
    out = admit(s, timeout_s=0.3)
    assert time.monotonic() - t0 <= 0.35
    assert s.count > 2
    assert out.reason == "admit_throttled"


def test_retry_after_zero_does_not_spin():
    s = Script([bare(429, retry_after="0")])
    admit(s, timeout_s=0.3)
    assert s.count < 10


def test_fail_open_off_raises_the_429():
    s = Script([bare(429, retry_after="5")])
    with pytest.raises(TensorCostQuotaError) as ei:
        admit(s, fail_open=False)
    assert ei.value.status == 429
    assert ei.value.retry_after_ms == 5000


def test_bare_402_is_not_retried_and_is_named():
    s = Script([bare(402)])
    assert admit(s).reason == "admit_plan_limit"
    assert s.count == 1


def test_429_with_a_refusal_code_is_still_a_refusal():
    body = json.dumps({"error": {"code": "RATE_LIMITED_BY_POLICY", "message": "no"}}).encode()
    s = Script([httpx.Response(429, headers={"content-type": "application/json"}, content=body)])
    with pytest.raises(TensorCostRefusedError):
        admit(s)
    assert s.count == 1


def test_an_outage_still_says_admit_unavailable():
    def boom(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=req)

    out = admit_call(
        base_url="https://tc.test", bearer_token="tok", max_layer="govern", metadata=META,
        http_client=httpx.Client(transport=httpx.MockTransport(boom)),
    )
    assert out.unavailable and out.reason == "admit_unavailable"


# ----------------------------------------------------------------------
# govern: still throttled when the budget runs out


def _admit_routed(monkeypatch, respond):
    real = httpx.Client

    class Routed(real):  # type: ignore[misc, valid-type]
        def __init__(self, *a: Any, **kw: Any) -> None:
            kw["transport"] = httpx.MockTransport(respond)
            super().__init__(*a, **kw)

    monkeypatch.setattr("tensorcost._admit_client.httpx.Client", Routed)


def _govern(recording_transport, events=None, client=None, module=_openai, **opts):
    client = client or FakeOpenAIClient()
    module.install(client, recording_transport, tenant_id=TENANT, environment=None, connection_id=None,
                   max_layer="govern", base_url="https://tc.test",
                   on_lifecycle_event=(events.append if events is not None else None), **opts)
    return client


def test_govern_fail_open_marks_counts_and_warns_once(recording_transport, monkeypatch, caplog):
    _admit_routed(monkeypatch, Script([bare(429, retry_after="30")]))
    events: list = []
    client = _govern(recording_transport, events)
    with caplog.at_level(logging.WARNING, logger="tensorcost"):
        for _ in range(5):
            client.chat.completions.create(model="gpt-4o-mini", messages=MSGS)
    assert client.chat.completions.create.__wrapped__.call_count == 5  # type: ignore[attr-defined]
    assert len(recording_transport.observations) == 5
    for o in recording_transport.observations:
        assert o["metadata"]["decision_action"] == "allow"
        assert o["metadata"]["decision_reason"] == "admit_throttled"
    assert tensorcost.unchecked_calls()["admit_throttled"] == 5
    fb = [e for e in events if e.kind == "on_fallback"]
    assert len(fb) == 5 and all(e.reason == "admit_throttled" for e in fb)
    warned = [r for r in caplog.records if "rate limiting" in r.getMessage()]
    assert len(warned) == 1
    assert "without a budget or governance check" in warned[0].getMessage()


def test_govern_fail_open_off_refuses_and_never_calls_the_provider(recording_transport, monkeypatch):
    _admit_routed(monkeypatch, Script([bare(429)]))
    client = _govern(recording_transport, fail_open_enabled=False)
    with pytest.raises(TensorCostQuotaError):
        client.chat.completions.create(model="gpt-4o-mini", messages=MSGS)
    assert client.chat.completions.create.__wrapped__.call_count == 0  # type: ignore[attr-defined]
    assert tensorcost.unchecked_calls()["admit_throttled"] == 0


def test_anthropic_behaves_the_same(recording_transport, monkeypatch):
    _admit_routed(monkeypatch, Script([bare(429, retry_after="30")]))
    client = _govern(recording_transport, client=FakeAnthropicClient(), module=_anthropic)
    client.messages.create(model="claude-3-5-haiku-latest", max_tokens=5, messages=MSGS)
    assert client.messages.create.__wrapped__.call_count == 1  # type: ignore[attr-defined]
    assert recording_transport.observations[0]["metadata"]["decision_reason"] == "admit_throttled"
    assert tensorcost.unchecked_calls()["admit_throttled"] == 1


def test_admit_outage_is_counted_as_admit_unavailable(recording_transport, monkeypatch):
    _admit_routed(monkeypatch, Script([bare(503)]))
    client = _govern(recording_transport)
    client.chat.completions.create(model="gpt-4o-mini", messages=MSGS)
    assert recording_transport.observations[0]["metadata"]["decision_reason"] == "admit_unavailable"
    counts = tensorcost.unchecked_calls()
    assert (counts["admit_unavailable"], counts["admit_throttled"]) == (1, 0)


# ----------------------------------------------------------------------
# route: the proxy still throttled after its retries


def _route(recording_transport, events=None, **opts):
    layer = SdkLayerClient(
        base_url="https://tc.test", get_token=lambda: "jwt",
        http_client=httpx.Client(transport=httpx.MockTransport(
            lambda r: httpx.Response(200, json={"published_layer": "route"}))),
    )
    client = FakeOpenAIClient()
    _openai.install(client, recording_transport, tenant_id=TENANT, environment=None, connection_id=None,
                    max_layer="route", proxy_url="https://proxy.test", sdk_layer=layer,
                    base_url="https://tc.test",
                    on_lifecycle_event=(events.append if events is not None else None), **opts)
    return client


def _proxy_raises(monkeypatch, exc):
    def raise_(**_kw):
        raise exc

    monkeypatch.setattr(_openai, "proxy_request", raise_)


def test_route_throttled_goes_direct_marked_proxy_throttled(recording_transport, monkeypatch, caplog):
    _proxy_raises(monkeypatch, TensorCostQuotaError("tensorcost: proxy returned 429", status=429))
    events: list = []
    client = _route(recording_transport, events)
    with caplog.at_level(logging.WARNING, logger="tensorcost"):
        client.chat.completions.create(model="gpt-4o-mini", messages=MSGS)
    assert client.chat.completions.create.__wrapped__.call_count == 1  # type: ignore[attr-defined]
    meta = recording_transport.observations[0]["metadata"]
    assert (meta["decision_action"], meta["decision_reason"]) == ("allow", "proxy_throttled")
    fb = [e for e in events if e.kind == "on_fallback"]
    assert len(fb) == 1 and fb[0].reason == "proxy_throttled"
    assert tensorcost.unchecked_calls()["proxy_throttled"] == 1
    assert any("proxy is rate limiting" in r.getMessage() for r in caplog.records)


def test_route_other_fallbacks_are_named_too(recording_transport, monkeypatch):
    _proxy_raises(monkeypatch, TensorCostProxyError("tensorcost: proxy returned 503", status=503))
    client = _route(recording_transport)
    client.chat.completions.create(model="gpt-4o-mini", messages=MSGS)
    assert recording_transport.observations[0]["metadata"]["decision_reason"] == "proxy_unavailable"
    assert tensorcost.unchecked_calls()["proxy_unavailable"] == 1


def test_route_throttled_with_fail_open_off_raises(recording_transport, monkeypatch):
    _proxy_raises(monkeypatch, TensorCostQuotaError("tensorcost: proxy returned 429", status=429))
    client = _route(recording_transport, fail_open_enabled=False)
    with pytest.raises(TensorCostProxyError, match="429"):
        client.chat.completions.create(model="gpt-4o-mini", messages=MSGS)
    assert client.chat.completions.create.__wrapped__.call_count == 0  # type: ignore[attr-defined]
    assert tensorcost.unchecked_calls()["proxy_throttled"] == 0


def test_real_proxy_429_is_retried_then_falls_back_as_throttled(recording_transport, monkeypatch):
    """Through the real proxy client: two 429s, then the fall-back."""
    from tensorcost._retry import RetryConfig

    hits = {"n": 0}

    def proxy(req: httpx.Request) -> httpx.Response:
        hits["n"] += 1
        hits.setdefault("urls", set()).add(str(req.url))
        return httpx.Response(429, headers={"retry-after": "0"})

    # Built first, so its sdk-layer client keeps its own transport.
    client = _route(recording_transport, retry=RetryConfig(max_attempts=2, base_delay_ms=1.0, max_delay_ms=5.0))
    real = httpx.Client

    class Routed(real):  # type: ignore[misc, valid-type]
        def __init__(self, *a: Any, **kw: Any) -> None:
            kw["transport"] = httpx.MockTransport(proxy)
            super().__init__(*a, **kw)

    monkeypatch.setattr("tensorcost._proxy_client.httpx.Client", Routed)
    client.chat.completions.create(model="gpt-4o-mini", messages=MSGS)
    assert hits["n"] == 2, hits.get("urls")
    assert recording_transport.observations[0]["metadata"]["decision_reason"] == "proxy_throttled"
