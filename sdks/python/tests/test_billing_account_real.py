"""billing_account against the real provider SDKs, not fakes.

Skips per SDK unless it imports. 1.7.0 was checked against anthropic 2.8.0
and openai 3.51.0 (both send through httpx2), google-genai 3.3.0 and
boto3/botocore 1.44.3.
"""

from __future__ import annotations

from typing import Any

import pytest

from tensorcost import _billing_account as ba
from tensorcost._providers import _anthropic, _bedrock, _openai, _vertex

MSG = {
    "id": "msg_direct",
    "type": "message",
    "role": "assistant",
    "content": [{"type": "text", "text": "hi"}],
    "model": "claude-sonnet-4-5",
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 1, "output_tokens": 1},
}
CHAT = {
    "id": "chatcmpl-direct",
    "object": "chat.completion",
    "created": 0,
    "model": "gpt-4o",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}


def _http_module(sdk: Any) -> Any:
    """httpx2 for anthropic 2.x / openai 3.x, httpx for older ones."""
    import inspect

    ann = str(inspect.signature(sdk.__init__).parameters["http_client"].annotation)
    return pytest.importorskip("httpx2" if "httpx2" in ann else "httpx")


def _routed(sdk: Any, respond: Any) -> tuple[Any, list]:
    hx = _http_module(sdk)
    calls: list = []

    def handler(request: Any) -> Any:
        calls.append(str(request.url))
        return respond(hx, request)

    return hx.Client(transport=hx.MockTransport(handler)), calls


def _install(mod: Any, c: Any, t: Any) -> None:
    mod.install(c, t, tenant_id=None, environment=None, connection_id=None)


def test_real_anthropic(recording_transport):
    anthropic = pytest.importorskip("anthropic")
    http, calls = _routed(
        anthropic.Anthropic,
        lambda hx, req: hx.Response(200, json=MSG, headers={"anthropic-organization-id": "org-anth-real"}),
    )
    c = anthropic.Anthropic(api_key="sk-ant", http_client=http, max_retries=0)
    _install(_anthropic, c, recording_transport)
    out = c.messages.create(model="claude-sonnet-4-5", max_tokens=8, messages=[{"role": "user", "content": "x"}])
    assert out.id == "msg_direct"
    assert calls == ["https://api.anthropic.com/v1/messages"]
    assert recording_transport.observations[-1]["billing_account"] == "org-anth-real"


def test_real_anthropic_error_and_no_header(recording_transport):
    anthropic = pytest.importorskip("anthropic")
    http, _ = _routed(
        anthropic.Anthropic,
        lambda hx, req: hx.Response(
            529, json={"type": "error", "error": {"type": "overloaded_error", "message": "x"}},
            headers={"anthropic-organization-id": "org-anth-real"},
        ),
    )
    c = anthropic.Anthropic(api_key="sk-ant", http_client=http, max_retries=0)
    _install(_anthropic, c, recording_transport)
    with pytest.raises(anthropic.APIStatusError):
        c.messages.create(model="claude-sonnet-4-5", max_tokens=8, messages=[{"role": "user", "content": "x"}])
    assert recording_transport.observations[-1]["billing_account"] == "org-anth-real"

    http, _ = _routed(anthropic.Anthropic, lambda hx, req: hx.Response(200, json=MSG))
    c = anthropic.Anthropic(api_key="sk-ant", http_client=http, max_retries=0)
    _install(_anthropic, c, recording_transport)
    c.messages.create(model="claude-sonnet-4-5", max_tokens=8, messages=[{"role": "user", "content": "x"}])
    assert "billing_account" not in recording_transport.observations[-1]


def test_real_openai_and_stream(recording_transport):
    openai = pytest.importorskip("openai")
    sse = (
        'data: {"id":"c1","object":"chat.completion.chunk","created":0,"model":"gpt-4o",'
        '"choices":[{"index":0,"delta":{"content":"hi"}}]}\n\ndata: [DONE]\n\n'
    )

    def respond(hx: Any, req: Any) -> Any:
        h = {"openai-organization": "org-oai-real", "openai-project": "proj_1"}
        if b'"stream":true' in req.content.replace(b" ", b""):
            return hx.Response(200, content=sse.encode(), headers={**h, "content-type": "text/event-stream"})
        return hx.Response(200, json=CHAT, headers=h)

    http, calls = _routed(openai.OpenAI, respond)
    c = openai.OpenAI(api_key="sk", http_client=http, max_retries=0)
    _install(_openai, c, recording_transport)
    assert c.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}]).id == "chatcmpl-direct"
    assert recording_transport.observations[-1]["billing_account"] == "org-oai-real"

    stream = c.chat.completions.create(model="gpt-4o", stream=True, messages=[{"role": "user", "content": "x"}])
    assert recording_transport.observations[-1]["billing_account"] == "org-oai-real"
    assert "".join(ch.choices[0].delta.content or "" for ch in stream) == "hi"
    assert len(calls) == 2


def test_real_azure_openai(recording_transport):
    openai = pytest.importorskip("openai")
    http, _ = _routed(openai.AzureOpenAI, lambda hx, req: hx.Response(200, json=CHAT))
    c = openai.AzureOpenAI(
        api_key="az", azure_endpoint="https://Contoso.openai.azure.com", api_version="2024-10-21",
        http_client=http, max_retries=0,
    )
    _install(_openai, c, recording_transport)
    c.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
    obs = recording_transport.observations[-1]
    assert obs["provider"] == "azure_openai"
    assert obs["billing_account"] == "contoso"


def test_real_genai_vertex_url_matches_the_clients_own(recording_transport):
    genai = pytest.importorskip("google.genai")
    c = genai.Client(vertexai=True, project="my-proj", location="us-central1")
    api = c._api_client
    # The client's own path rule, on its own base URL.
    own = f"{api._http_options.base_url.rstrip('/')}/{api._http_options.api_version}/projects/{api.project}/locations/{api.location}/publishers/google/models/m:generateContent"
    assert ba.billing_account_of_call(own, None) == "my-proj"
    assert ba.billing_account_of_call(ba.vertex_call_url(c), None) == "my-proj"
    assert ba.vertex_call_url(genai.Client(vertexai=True, api_key="k")) is None
    assert ba.vertex_call_url(genai.Client(api_key="k")) is None

    c.models.generate_content = lambda **kw: "resp"  # no network
    _install(_vertex, c, recording_transport)
    assert c.models.generate_content(model="gemini-2.0-flash", contents="x") == "resp"
    assert recording_transport.observations[-1]["billing_account"] == "my-proj"


@pytest.mark.parametrize("account,expected", [("123456789012", "123456789012"), (None, None)])
def test_real_boto3_bedrock(recording_transport, monkeypatch, account, expected):
    boto3 = pytest.importorskip("boto3")
    from botocore.stub import Stubber

    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIATEST")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
    if account:
        monkeypatch.setenv("AWS_ACCOUNT_ID", account)
    else:
        monkeypatch.delenv("AWS_ACCOUNT_ID", raising=False)
    c = boto3.Session().client("bedrock-runtime", region_name="us-east-1")
    resp = {
        "output": {"message": {"role": "assistant", "content": [{"text": "hi"}]}},
        "stopReason": "end_turn",
        "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2},
        "metrics": {"latencyMs": 1},
    }
    with Stubber(c) as stub:
        stub.add_response("converse", resp)
        _install(_bedrock, c, recording_transport)
        out = c.converse(modelId="anthropic.claude-3-haiku", messages=[{"role": "user", "content": [{"text": "x"}]}])
    assert out["usage"]["inputTokens"] == 1
    assert recording_transport.observations[-1].get("billing_account") == expected
