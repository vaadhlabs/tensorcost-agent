"""TensorCost down or slow: what a call costs, what it reports, what is kept.

From a field test against a self-hosted install (tensorcost 1.7.0):

* blackholed TensorCost made every call 6.5-8.5 s: the token exchange ran on
  the caller's thread with no cap, before the 0.25 s sdk-layer GET;
* a route client that decayed to govern went to the provider directly and
  fired no on_fallback;
* observations carried no team, and every undelivered one was dropped;
* an admit timeout left an "admit_unavailable" row with no agent and 0 tokens;
* /v1/admit's X-TensorCost-Budget-Would-Refuse header was never read.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

import httpx
import pytest

from tensorcost import _transport as transport_mod
from tensorcost._config import resolve_config
from tensorcost._errors import TensorCostModelGovernanceDeniedError, TensorCostTimeoutError
from tensorcost._providers import _openai
from tensorcost._runtime import get_shared_runtime as real_get_shared_runtime
from tensorcost._sdk_layer import SdkLayerClient
from tensorcost._transport import ObservationTransport

from .conftest import FakeOpenAIClient

TENANT = "11111111-1111-1111-1111-111111111111"
MSGS = [{"role": "user", "content": "x"}]


def _blackholed(_req: httpx.Request) -> httpx.Response:
    time.sleep(3.0)
    return httpx.Response(503)


def _http(handler: Any) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


# --- 4: nothing on the call path costs more than its budget ----------------


def test_bounded_get_token_gives_up_inside_its_budget():
    t = ObservationTransport("https://tc.test", "k", tenant_id=TENANT, http_client=_http(_blackholed))
    t0 = time.monotonic()
    with pytest.raises(TensorCostTimeoutError):
        t.get_token(timeout_s=0.2)
    assert time.monotonic() - t0 < 0.5


def test_sdk_layer_refresh_including_the_token_is_bounded_by_the_cap():
    def slow_token() -> str:
        time.sleep(3.0)
        return "jwt"

    layer = SdkLayerClient(base_url="https://tc.test", get_token=slow_token,
                           http_client=_http(lambda r: httpx.Response(200, json={"published_layer": "route"})))
    t0 = time.monotonic()
    assert layer.effective_layer("route") == "govern"
    assert time.monotonic() - t0 < 0.45
    # A second call while that refresh is still stuck does not wait again.
    t1 = time.monotonic()
    layer.effective_layer("route")
    assert time.monotonic() - t1 < 0.1


def test_observe_call_with_tensorcost_blackholed_costs_at_most_the_layer_cap(recording_transport):
    t = ObservationTransport("https://tc.test", "k", tenant_id=TENANT, http_client=_http(_blackholed))
    layer = SdkLayerClient(base_url="https://tc.test", get_token=t.get_token, http_client=_http(_blackholed))
    client = FakeOpenAIClient()
    _openai.install(client, t, tenant_id=TENANT, environment=None, connection_id=None,
                    max_layer="observe", sdk_layer=layer, base_url="https://tc.test")
    t0 = time.monotonic()
    client.chat.completions.create(model="gpt-4o-mini", messages=MSGS)
    assert time.monotonic() - t0 < 0.45
    t1 = time.monotonic()
    client.chat.completions.create(model="gpt-4o-mini", messages=MSGS)
    assert time.monotonic() - t1 < 0.1


def test_govern_call_with_tensorcost_blackholed_stays_inside_layer_plus_admit_budget():
    t = ObservationTransport("https://tc.test", "k", tenant_id=TENANT, http_client=_http(_blackholed))
    layer = SdkLayerClient(base_url="https://tc.test", get_token=t.get_token, http_client=_http(_blackholed))
    client = FakeOpenAIClient()
    _openai.install(client, t, tenant_id=TENANT, environment=None, connection_id=None,
                    max_layer="govern", sdk_layer=layer, base_url="https://tc.test")
    t0 = time.monotonic()
    client.chat.completions.create(model="gpt-4o-mini", messages=MSGS)
    # 0.25 s sdk-layer cap + 0.5 s admit budget (token + POST).
    assert time.monotonic() - t0 < 0.95
    assert client.chat.completions.create.__wrapped__.call_count == 1  # type: ignore[attr-defined]


# --- 5: on_fallback for every way a route client goes direct ---------------


def _route_client(recording_transport, events, *, layer_status: int = 503, **opts):
    def layer_handler(_r: httpx.Request) -> httpx.Response:
        if layer_status == 200:
            return httpx.Response(200, json={"published_layer": "route"})
        return httpx.Response(layer_status)

    layer = SdkLayerClient(base_url="https://tc.test", get_token=lambda: "jwt", http_client=_http(layer_handler))
    client = FakeOpenAIClient()
    _openai.install(client, recording_transport, tenant_id=TENANT, environment=None, connection_id=None,
                    max_layer="route", proxy_url="https://proxy.test", sdk_layer=layer,
                    base_url="https://tc.test", on_lifecycle_event=events.append, **opts)
    return client


def _fallbacks(events):
    return [e for e in events if e.kind == "on_fallback"]


def test_route_client_that_decayed_to_govern_fires_on_fallback(recording_transport, monkeypatch):
    from tensorcost._admit_client import AdmitOutcome

    monkeypatch.setattr(_openai, "admit_call", lambda **kw: AdmitOutcome(True, None))
    events: list = []
    client = _route_client(recording_transport, events)
    client.chat.completions.create(model="gpt-4o-mini", messages=MSGS)
    fb = _fallbacks(events)
    assert len(fb) == 1 and fb[0].reason == "layer_decayed"


def test_proxy_failure_fallback_says_so(recording_transport, monkeypatch):
    from tensorcost._errors import TensorCostNetworkError

    def boom(**_kw):
        raise TensorCostNetworkError("tensorcost: network error reaching proxy: refused")

    monkeypatch.setattr(_openai, "proxy_request", boom)
    events: list = []
    client = _route_client(recording_transport, events, layer_status=200)
    client.chat.completions.create(model="gpt-4o-mini", messages=MSGS)
    fb = _fallbacks(events)
    assert len(fb) == 1 and fb[0].reason == "proxy_unavailable"


def test_governance_refusal_in_a_decayed_route_client_never_falls_back(recording_transport, monkeypatch):
    def refuse(**_kw):
        raise TensorCostModelGovernanceDeniedError("tensorcost: model not approved", status=403)

    monkeypatch.setattr(_openai, "admit_call", refuse)
    events: list = []
    client = _route_client(recording_transport, events)
    with pytest.raises(TensorCostModelGovernanceDeniedError):
        client.chat.completions.create(model="gpt-4.1", messages=MSGS)
    assert _fallbacks(events) == []
    assert client.chat.completions.create.__wrapped__.call_count == 0  # type: ignore[attr-defined]


# --- 6: the observation names its team -------------------------------------


def test_observation_post_carries_the_team_header():
    seen: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path.endswith("/sdk-token/exchange"):
            return httpx.Response(200, json={"token": "jwt", "expires_in": 900})
        seen.append(req)
        return httpx.Response(202)

    config = resolve_config(api_key="k", tenant_id=TENANT, base_url="https://tc.test", team_id="zelvora-matrix")
    t, _ = real_get_shared_runtime(config)
    t._client = _http(handler)
    t.post({"model": "m"})
    t.flush(2.0)
    assert seen and seen[0].headers["x-tc-team-id"] == "zelvora-matrix"


# --- 7: undelivered observations are kept and replayed ---------------------


def _flaky_server(fail_first: int, status_when_failing: int | None = None):
    state = {"n": 0, "delivered": []}
    lock = threading.Lock()

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path.endswith("/sdk-token/exchange"):
            return httpx.Response(200, json={"token": "jwt", "expires_in": 900})
        with lock:
            state["n"] += 1
            n = state["n"]
        if n <= fail_first:
            if status_when_failing is None:
                raise httpx.ConnectError("refused", request=req)
            return httpx.Response(status_when_failing)
        import json

        state["delivered"].append(json.loads(req.content)["correlation_id"])
        return httpx.Response(202)

    return state, handler


def _post_and_wait(t: ObservationTransport, ids: list[str]) -> None:
    for i in ids:
        t.post({"correlation_id": i})
        t.flush(2.0)


def test_observations_that_fail_during_an_outage_are_replayed_after_it():
    state, handler = _flaky_server(fail_first=3)
    t = ObservationTransport("https://tc.test", "k", tenant_id=TENANT, http_client=_http(handler), max_workers=1)
    _post_and_wait(t, ["a", "b", "c"])
    assert t.queued_observations == 3
    _post_and_wait(t, ["d"])
    assert sorted(state["delivered"]) == ["a", "b", "c", "d"]
    assert t.queued_observations == 0 and t.dropped_observations == 0


def test_503_is_kept_but_a_400_is_not():
    state, handler = _flaky_server(fail_first=1, status_when_failing=503)
    t = ObservationTransport("https://tc.test", "k", tenant_id=TENANT, http_client=_http(handler), max_workers=1)
    _post_and_wait(t, ["a"])
    assert t.queued_observations == 1
    state2, handler2 = _flaky_server(fail_first=1, status_when_failing=400)
    t2 = ObservationTransport("https://tc.test", "k", tenant_id=TENANT, http_client=_http(handler2), max_workers=1)
    _post_and_wait(t2, ["a"])
    assert t2.queued_observations == 0


def test_retry_queue_is_bounded_and_drops_the_oldest(monkeypatch):
    monkeypatch.setattr(transport_mod, "_RETRY_QUEUE_MAX", 2)
    state, handler = _flaky_server(fail_first=3)
    t = ObservationTransport("https://tc.test", "k", tenant_id=TENANT, http_client=_http(handler), max_workers=1)
    _post_and_wait(t, ["a", "b", "c"])
    assert t.queued_observations == 2 and t.dropped_observations == 1
    _post_and_wait(t, ["d"])
    assert sorted(state["delivered"]) == ["b", "c", "d"]


# --- 8: admit_unavailable keeps the agent and the tokens -------------------


def _admit_routed(monkeypatch, respond):
    real = httpx.Client

    class Routed(real):  # type: ignore[misc, valid-type]
        def __init__(self, *a: Any, **kw: Any) -> None:
            kw["transport"] = httpx.MockTransport(respond)
            super().__init__(*a, **kw)

    monkeypatch.setattr("tensorcost._admit_client.httpx.Client", Routed)


def _govern_client(recording_transport, events=None, **opts):
    client = FakeOpenAIClient()
    _openai.install(client, recording_transport, tenant_id=TENANT, environment=None, connection_id=None,
                    max_layer="govern", base_url="https://tc.test", agent_id="matrix-agent",
                    team_id="zelvora-matrix",
                    on_lifecycle_event=(events.append if events is not None else None), **opts)
    return client


def test_admit_timeout_observation_keeps_agent_and_real_usage(recording_transport, monkeypatch):
    def timeout(req: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=req)

    _admit_routed(monkeypatch, timeout)
    _govern_client(recording_transport).chat.completions.create(model="gpt-4o-mini", messages=MSGS)
    assert len(recording_transport.observations) == 1
    obs = recording_transport.observations[0]
    assert obs["agent_id"] == "matrix-agent"
    assert (obs["input_tokens"], obs["output_tokens"]) == (12, 34)
    assert obs["metadata"]["decision_reason"] == "admit_unavailable"


def test_budget_would_refuse_header_is_surfaced(recording_transport, monkeypatch, caplog):
    ids = "edd83e8a-1638-4631-ba88-3286cff53475,4bacb294-a989-4c2f-9768-5322646db5f1"
    _admit_routed(monkeypatch, lambda req: httpx.Response(204, headers={"X-TensorCost-Budget-Would-Refuse": ids}))
    from tensorcost import _budget_signals

    _budget_signals._reset()
    events: list = []
    client = _govern_client(recording_transport, events)
    with caplog.at_level(logging.WARNING, logger="tensorcost"):
        for _ in range(5):
            client.chat.completions.create(model="gpt-4o-mini", messages=MSGS)
    would = [e for e in events if e.kind == "on_budget_would_refuse"]
    assert len(would) == 5
    assert would[0].budget_ids == tuple(ids.split(","))
    warned = [r for r in caplog.records if "would refuse" in r.getMessage()]
    assert len(warned) == 2
    assert {i for i in ids.split(",") if any(i in r.getMessage() for r in warned)} == set(ids.split(","))
    # The call itself went ahead.
    assert client.chat.completions.create.__wrapped__.call_count == 5  # type: ignore[attr-defined]
