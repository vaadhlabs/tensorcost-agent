"""Route mode hands back what the unwrapped openai client would.

Field test against a self-hosted install (tensorcost 1.7.0, openai 3.57.0):
route-mode create() returned a plain dict, with_raw_response sent
extra_headers inside the JSON body (OpenAI answered 400 "Unrecognized
request argument supplied: extra_headers"), and a streamed call yielded
dicts and was observed as 0/0 tokens.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

openai = pytest.importorskip("openai")

from tensorcost import RetryConfig, wrap  # noqa: E402

TENANT = "11111111-1111-1111-1111-111111111111"
CHAT_OK = {
    "id": "chatcmpl-proxy",
    "object": "chat.completion",
    "created": 0,
    "model": "gpt-4o-mini",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
}


def _sse(chunks: list[dict]) -> bytes:
    return b"".join(f"data: {json.dumps(c)}\n\n".encode() for c in chunks) + b"data: [DONE]\n\n"


STREAM_CHUNKS = [
    {"id": "c1", "object": "chat.completion.chunk", "created": 0, "model": "gpt-4o-mini",
     "choices": [{"index": 0, "delta": {"role": "assistant", "content": "O"}, "finish_reason": None}]},
    {"id": "c1", "object": "chat.completion.chunk", "created": 0, "model": "gpt-4o-mini",
     "choices": [{"index": 0, "delta": {"content": "k."}, "finish_reason": "stop"}]},
    {"id": "c1", "object": "chat.completion.chunk", "created": 0, "model": "gpt-4o-mini",
     "choices": [], "usage": {"prompt_tokens": 9, "completion_tokens": 3, "total_tokens": 12}},
]


@pytest.fixture
def api(monkeypatch):
    state: dict[str, Any] = {"proxy_requests": [], "direct": 0}
    real_client = httpx.Client

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url.startswith("https://proxy.test"):
            state["proxy_requests"].append(request)
            body = json.loads(request.content)
            if body.get("stream"):
                return httpx.Response(
                    200, content=_sse(STREAM_CHUNKS),
                    headers={"content-type": "text/event-stream", "x-request-id": "req_s"},
                )
            return httpx.Response(200, json=CHAT_OK, headers={"x-request-id": "req_p", "x-tc-request-id": "tc_1"})
        if url.startswith("https://api.openai.com"):
            state["direct"] += 1
            return httpx.Response(200, json=CHAT_OK)
        raise httpx.ConnectError("unrouted " + url, request=request)

    class Routed(real_client):  # type: ignore[misc, valid-type]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            kwargs["transport"] = httpx.MockTransport(handler)
            super().__init__(*args, **kwargs)

    # Only the proxy client's httpx is swapped: openai isinstance-checks the
    # http_client it is given against the real httpx.Client.
    import types

    from tensorcost import _proxy_client

    shim = types.ModuleType("httpx_routed")
    shim.__dict__.update(httpx.__dict__)
    shim.Client = Routed  # type: ignore[attr-defined]
    monkeypatch.setattr(_proxy_client, "httpx", shim)
    state["http_client"] = Routed()
    return state


def _client(api, recording_transport=None, **opts):
    c = openai.OpenAI(api_key="sk-provider", http_client=api["http_client"], max_retries=0)
    wrap(c, api_key="tc", tenant_id=TENANT, base_url="https://tc.test", proxy_url="https://proxy.test",
         max_layer="route", retry=RetryConfig(max_attempts=1), **opts)
    c._tensorcost_transport._get_token = lambda: "jwt"
    return c


MSGS = [{"role": "user", "content": "x"}]


def test_route_create_returns_a_chat_completion(api):
    r = _client(api).chat.completions.create(model="gpt-4o-mini", messages=MSGS)
    assert isinstance(r, openai.types.chat.ChatCompletion)
    assert r.choices[0].message.content == "hi"
    assert r.usage.prompt_tokens == 10
    assert len(api["proxy_requests"]) == 1 and api["direct"] == 0


def test_route_with_raw_response_returns_the_raw_wrapper(api):
    raw = _client(api).chat.completions.with_raw_response.create(model="gpt-4o-mini", messages=MSGS)
    assert raw.headers.get("x-request-id") == "req_p"
    parsed = raw.parse()
    assert isinstance(parsed, openai.types.chat.ChatCompletion)
    assert parsed.choices[0].message.content == "hi"
    body = json.loads(api["proxy_requests"][0].content)
    assert "extra_headers" not in body


def test_route_request_options_are_not_sent_in_the_body(api):
    _client(api).chat.completions.create(
        model="gpt-4o-mini", messages=MSGS,
        extra_headers={"X-Custom": "1"},
        extra_body={"custom_flag": True},
        extra_query={"api-version": "2024-10-21"},
        timeout=12.5,
    )
    req = api["proxy_requests"][0]
    body = json.loads(req.content)
    for key in ("extra_headers", "extra_body", "extra_query", "timeout"):
        assert key not in body
    assert body["custom_flag"] is True
    assert body["model"] == "gpt-4o-mini" and body["messages"] == MSGS
    assert req.headers["x-custom"] == "1"
    assert req.url.params["api-version"] == "2024-10-21"
    # Our headers are not overridable by the caller's.
    assert req.headers["authorization"] == "Bearer jwt"
    assert req.extensions["timeout"]["read"] == 12.5


def test_route_extra_headers_cannot_replace_tensorcost_auth(api):
    _client(api).chat.completions.create(
        model="gpt-4o-mini", messages=MSGS, extra_headers={"Authorization": "Bearer nope"},
    )
    assert api["proxy_requests"][0].headers["authorization"] == "Bearer jwt"


def test_route_stream_returns_an_openai_stream_and_observes_usage(api, recording_transport):
    c = _client(api)
    s = c.chat.completions.create(
        model="gpt-4o-mini", messages=MSGS, stream=True, stream_options={"include_usage": True},
    )
    assert isinstance(s, openai.Stream)
    with s as stream:
        chunks = list(stream)
    assert all(isinstance(ch, openai.types.chat.ChatCompletionChunk) for ch in chunks)
    assert "".join(ch.choices[0].delta.content or "" for ch in chunks if ch.choices) == "Ok."
    obs = [o for o in recording_transport.observations if o.get("operation") == "chat.completions"]
    assert len(obs) == 1
    assert (obs[0]["input_tokens"], obs[0]["output_tokens"]) == (9, 3)


def test_route_connect_timeout_is_the_headers_budget_not_the_body_budget(api):
    # A buffered JSON call may wait the whole body budget for headers (the
    # proxy answers when upstream does), but a blackholed proxy must not get
    # 60 s just to accept the connection.
    _client(api).chat.completions.create(model="gpt-4o-mini", messages=MSGS)
    t = api["proxy_requests"][0].extensions["timeout"]
    assert t["connect"] == 2.0
    assert t["read"] == 60.0
