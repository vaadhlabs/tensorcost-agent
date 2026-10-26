"""wrap() against the real provider SDKs, not MagicMocks.

openai and anthropic aren't dependencies of this package, so this module
skips unless both import. To run it, put a copy on PYTHONPATH (the 1.5.0
run used openai 2.32.0 and anthropic 0.96.0 from an existing local venv).
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

openai = pytest.importorskip("openai")
anthropic = pytest.importorskip("anthropic")

from tensorcost import RetryConfig, TensorCostBudgetHardCapError, wrap  # noqa: E402

TENANT = "11111111-1111-1111-1111-111111111111"
CHAT_OK = {
    "id": "chatcmpl-direct",
    "object": "chat.completion",
    "created": 0,
    "model": "gpt-4o",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}
MSG_OK = {
    "id": "msg_direct",
    "type": "message",
    "role": "assistant",
    "content": [{"type": "text", "text": "hi"}],
    "model": "claude-sonnet-4-5",
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 1, "output_tokens": 1},
}


@pytest.fixture
def api(monkeypatch):
    """TensorCost endpoints (proxy, admit) and the provider, all recorded."""
    state: dict[str, Any] = {"calls": [], "proxy": None, "admit": None}
    real_client = httpx.Client

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        state["calls"].append((url, request.headers.get("authorization")))
        if "/v1/admit" in url:
            return state["admit"]() if state["admit"] else httpx.Response(204)
        if url.startswith("https://proxy.test"):
            return state["proxy"]()
        if url.startswith("https://api.openai.com"):
            return httpx.Response(200, json=CHAT_OK)
        if url.startswith("https://api.anthropic.com"):
            return httpx.Response(200, json=MSG_OK)
        raise httpx.ConnectError("unrouted " + url, request=request)

    # A subclass, not a function: patching the module attribute patches
    # httpx.Client for everyone, and the provider SDKs isinstance-check it.
    class Routed(real_client):  # type: ignore[misc, valid-type]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            kwargs["transport"] = httpx.MockTransport(handler)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr("tensorcost._proxy_client.httpx.Client", Routed)
    state["http_client"] = Routed()
    return state


def _openai(api, **opts):
    c = openai.OpenAI(api_key="sk-provider", http_client=api["http_client"], max_retries=0)
    wrap(c, api_key="tc", tenant_id=TENANT, base_url="https://tc.test", proxy_url="https://proxy.test",
         max_layer="route", retry=RetryConfig(max_attempts=1), **opts)
    c._tensorcost_transport._get_token = lambda: "jwt"
    return c


def _chat(c):
    return c.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])


def test_relayed_400_is_openai_bad_request_error(api):
    body = {"error": {"message": "bad", "type": "invalid_request_error", "param": "messages", "code": "invalid_value"}}
    api["proxy"] = lambda: httpx.Response(400, json=body, headers={"x-request-id": "req_1"})
    c = _openai(api)
    with pytest.raises(openai.BadRequestError) as info:
        _chat(c)
    err = info.value
    assert err.status_code == 400
    assert err.code == "invalid_value"
    assert err.type == "invalid_request_error"
    assert getattr(err, "tensorcost", None) is not None
    assert not any(u.startswith("https://api.openai.com") for u, _ in api["calls"])


def test_relayed_401_is_openai_authentication_error(api):
    api["proxy"] = lambda: httpx.Response(
        401, json={"error": {"message": "Incorrect API key", "type": "invalid_request_error", "code": "invalid_api_key"}}
    )
    with pytest.raises(openai.AuthenticationError):
        _chat(_openai(api))


def test_proxy_503_falls_back_to_api_openai_com_with_the_provider_key(api):
    api["proxy"] = lambda: httpx.Response(503, text="<html>503</html>")
    r = _chat(_openai(api))
    assert r.choices[0].message.content == "hi"
    direct = [(u, a) for u, a in api["calls"] if u.startswith("https://api.openai.com")]
    assert direct == [("https://api.openai.com/v1/chat/completions", "Bearer sk-provider")]


def test_money_budget_hard_cap_is_terminal(api):
    api["proxy"] = lambda: httpx.Response(
        403, json={"error": {"code": "PERIOD_BUDGET_EXCEEDED", "budget_code": "BUDGET_HARD_CAP", "budget_id": "b-1", "retryable": False}}
    )
    with pytest.raises(TensorCostBudgetHardCapError):
        _chat(_openai(api))
    assert [u for u, _ in api["calls"]] == ["https://proxy.test/api/inference-proxy/v1/chat/completions"]


def _anthropic(api):
    c = anthropic.Anthropic(api_key="sk-ant", http_client=api["http_client"], max_retries=0)
    wrap(c, api_key="tc", tenant_id=TENANT, base_url="https://tc.test", proxy_url="https://proxy.test",
         max_layer="route")
    c._tensorcost_transport._get_token = lambda: "jwt"
    return c


def _msg(c):
    return c.messages.create(model="claude-sonnet-4-5", max_tokens=8, messages=[{"role": "user", "content": "x"}])


def test_anthropic_is_pinned_to_govern_and_admitted_before_the_call(api):
    api["proxy"] = lambda: httpx.Response(500)
    r = _msg(_anthropic(api))
    assert r.id == "msg_direct"
    assert [u for u, _ in api["calls"]] == [
        "https://tc.test/api/inference-proxy/v1/admit",
        "https://api.anthropic.com/v1/messages",
    ]


def test_anthropic_budget_refusal_on_admit_stops_the_call(api):
    api["admit"] = lambda: httpx.Response(
        403,
        json={"error": {"code": "PERIOD_BUDGET_EXCEEDED", "budget_code": "BUDGET_HARD_CAP",
                        "reason": "spend_not_known_yet", "retryable": True, "budget_id": "b-2"}},
        headers={"retry-after": "5"},
    )
    with pytest.raises(TensorCostBudgetHardCapError) as info:
        _msg(_anthropic(api))
    assert info.value.retryable is True
    assert info.value.retry_after_seconds == 5
    assert [u for u, _ in api["calls"]] == ["https://tc.test/api/inference-proxy/v1/admit"]
