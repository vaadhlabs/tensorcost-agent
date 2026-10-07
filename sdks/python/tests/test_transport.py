import json
import time

import httpx
import pytest

from tensorcost._transport import ObservationTransport


def _make_handler(state: dict):
    def handler(request: httpx.Request) -> httpx.Response:
        state["calls"].append(request)
        if request.url.path.endswith("/sdk-token/exchange"):
            state["token_calls"] += 1
            return httpx.Response(
                200,
                json={"token": "fake-jwt", "expires_in": 900},
            )
        if request.url.path.endswith("/observation"):
            state["observation_calls"] += 1
            return httpx.Response(202, json={"ok": True})
        return httpx.Response(404)

    return handler


def _build_transport(handler, **kwargs) -> ObservationTransport:
    mock = httpx.MockTransport(handler)
    client = httpx.Client(transport=mock, timeout=2.0)
    return ObservationTransport(
        base_url="https://api.test",
        api_key="long-lived-key",
        tenant_id="e82770de-06d6-415c-911f-c0d65cf45190",
        http_client=client,
        **kwargs,
    )


def _drain(transport: ObservationTransport, timeout_s: float = 2.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        with transport._pending_lock:
            pending = [f for f in transport._pending if not f.done()]
        if not pending:
            return
        time.sleep(0.01)


def test_post_hits_observation_url_with_bearer_token():
    state = {"calls": [], "token_calls": 0, "observation_calls": 0}
    t = _build_transport(_make_handler(state))
    try:
        t.post({"hello": "world"})
        _drain(t)
    finally:
        t.close()

    exchange_requests = [
        r for r in state["calls"] if r.url.path.endswith("/sdk-token/exchange")
    ]
    assert len(exchange_requests) == 1
    assert json.loads(exchange_requests[0].content) == {
        "tenant_id": "e82770de-06d6-415c-911f-c0d65cf45190",
        "sdk_long_lived_token": "long-lived-key",
    }

    obs_requests = [
        r for r in state["calls"] if r.url.path.endswith("/observation")
    ]
    assert len(obs_requests) == 1
    req = obs_requests[0]
    assert req.headers["authorization"] == "Bearer fake-jwt"
    assert json.loads(req.content) == {"hello": "world"}
    assert str(req.url) == "https://api.test/api/inference-proxy/observation"


def test_token_exchange_is_cached():
    state = {"calls": [], "token_calls": 0, "observation_calls": 0}
    t = _build_transport(_make_handler(state))
    try:
        for _ in range(5):
            t.post({"i": 1})
        _drain(t)
    finally:
        t.close()

    assert state["token_calls"] == 1
    assert state["observation_calls"] == 5


def test_fire_and_forget_does_not_block_caller():
    """The caller's `post` returns far faster than the transport's
    underlying response time."""

    def slow_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/sdk-token/exchange"):
            return httpx.Response(200, json={"token": "t", "expires_in": 900})
        time.sleep(0.5)
        return httpx.Response(202)

    t = _build_transport(slow_handler)
    try:
        start = time.perf_counter()
        t.post({"x": 1})
        elapsed = time.perf_counter() - start
        assert elapsed < 0.1, f"post() took {elapsed:.3f}s; should be non-blocking"
    finally:
        t.close()


def test_fail_open_swallows_errors_when_enabled(caplog):
    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("nope", request=request)

    t = _build_transport(boom, fail_open=True)
    try:
        # Should not raise.
        t.post({"x": 1})
        _drain(t)
    finally:
        t.close()


def test_token_exchange_failure_is_swallowed_when_fail_open():
    def reject(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "down"})

    t = _build_transport(reject, fail_open=True)
    try:
        t.post({"x": 1})  # must not raise
        _drain(t)
    finally:
        t.close()
