"""Every observation carries the model the provider's response named.

Field test (Zelvora N6): an Azure OpenAI deployment named for its job,
"support-bot", runs gpt-5.4-mini 2026-03-17. The SDK sent only the
request's model, the deployment name, which has no price, so the call was
stored at $0. The response names the model Azure ran; the server prices the
call from ``response_model`` and keeps the deployment beside it
(ObservationDto.response_model, at most 200 characters).

A stream sends the last chunk's model: Azure's first chunk (prompt filter
results) names none.
"""

from __future__ import annotations

import inspect
import json
from typing import Any

import pytest

from tensorcost._providers import _anthropic, _openai
from tensorcost._usage import response_model, response_model_field

SERVED = "gpt-5.4-mini-2026-03-17"
CHAT = {
    "id": "chatcmpl-azure",
    "object": "chat.completion",
    "created": 0,
    "model": SERVED,
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 9, "completion_tokens": 4, "total_tokens": 13},
}
CHUNKS = [
    # Azure's first chunk: prompt filter results, no model, no choices.
    {"id": "", "object": "", "created": 0, "model": "", "choices": [],
     "prompt_filter_results": [{"prompt_index": 0, "content_filter_results": {}}]},
    {"id": "chatcmpl-azure-stream", "object": "chat.completion.chunk", "created": 0, "model": SERVED,
     "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Ok."}, "finish_reason": "stop"}]},
    {"id": "chatcmpl-azure-stream", "object": "chat.completion.chunk", "created": 0, "model": SERVED,
     "choices": [], "usage": {"prompt_tokens": 9, "completion_tokens": 4, "total_tokens": 13}},
]
MESSAGE = {
    "id": "msg_direct", "type": "message", "role": "assistant", "content": [{"type": "text", "text": "hi"}],
    "model": "claude-haiku-4-5-20251001", "stop_reason": "end_turn", "usage": {"input_tokens": 10, "output_tokens": 4},
}
ANTHROPIC_EVENTS = [
    ("message_start", {"type": "message_start", "message": {
        "id": "msg_stream", "type": "message", "role": "assistant", "content": [],
        "model": "claude-haiku-4-5-20251001", "stop_reason": None, "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 1}}}),
    ("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
    ("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hi"}}),
    ("content_block_stop", {"type": "content_block_stop", "index": 0}),
    ("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                       "usage": {"output_tokens": 4}}),
    ("message_stop", {"type": "message_stop"}),
]
MSGS = [{"role": "user", "content": "x"}]


def _http_module(sdk: Any) -> Any:
    ann = str(inspect.signature(sdk.__init__).parameters["http_client"].annotation)
    return pytest.importorskip("httpx2" if "httpx2" in ann else "httpx")


def _routed(sdk: Any, respond: Any) -> Any:
    hx = _http_module(sdk)
    return hx.Client(transport=hx.MockTransport(lambda req: respond(hx, req)))


def _install(mod: Any, c: Any, t: Any) -> None:
    mod.install(c, t, tenant_id=None, environment=None, connection_id=None)


def _azure_respond(hx: Any, req: Any) -> Any:
    if json.loads(req.content).get("stream"):
        sse = b"".join(f"data: {json.dumps(c)}\n\n".encode() for c in CHUNKS) + b"data: [DONE]\n\n"
        return hx.Response(200, content=sse, headers={"content-type": "text/event-stream"})
    return hx.Response(200, json=CHAT)


@pytest.fixture
def azure_client(recording_transport):
    openai = pytest.importorskip("openai")
    c = openai.AzureOpenAI(
        api_key="k", api_version="2024-10-21", azure_endpoint="https://acme.openai.azure.com",
        http_client=_routed(openai.AzureOpenAI, _azure_respond), max_retries=0,
    )
    _install(_openai, c, recording_transport)
    return c


def _anthropic_respond(hx: Any, req: Any) -> Any:
    if json.loads(req.content).get("stream"):
        sse = b"".join(f"event: {n}\ndata: {json.dumps(d)}\n\n".encode() for n, d in ANTHROPIC_EVENTS)
        return hx.Response(200, content=sse, headers={"content-type": "text/event-stream"})
    return hx.Response(200, json=MESSAGE)


@pytest.fixture
def anthropic_client(recording_transport):
    anthropic = pytest.importorskip("anthropic")
    c = anthropic.Anthropic(
        api_key="sk-ant", http_client=_routed(anthropic.Anthropic, _anthropic_respond), max_retries=0,
    )
    _install(_anthropic, c, recording_transport)
    return c


def _only(recording_transport) -> dict:
    assert len(recording_transport.observations) == 1
    return recording_transport.observations[0]


def test_azure_create_sends_the_model_azure_served_beside_the_deployment(azure_client, recording_transport):
    azure_client.chat.completions.create(model="support-bot", messages=MSGS)
    obs = _only(recording_transport)
    assert obs["provider"] == "azure_openai"
    assert obs["model"] == "support-bot"
    assert obs["response_model"] == SERVED


def test_azure_with_raw_response_sends_the_served_model(azure_client, recording_transport):
    raw = azure_client.chat.completions.with_raw_response.create(model="support-bot", messages=MSGS)
    assert raw.parse().model == SERVED
    assert _only(recording_transport)["response_model"] == SERVED


def test_azure_stream_sends_the_last_chunk_that_names_a_model(azure_client, recording_transport):
    s = azure_client.chat.completions.create(
        model="support-bot", messages=MSGS, stream=True, stream_options={"include_usage": True},
    )
    with s:
        list(s)
    obs = _only(recording_transport)
    assert obs["model"] == "support-bot"
    assert obs["response_model"] == SERVED
    assert (obs["input_tokens"], obs["output_tokens"]) == (9, 4)


def test_anthropic_create_sends_the_served_model(anthropic_client, recording_transport):
    anthropic_client.messages.create(model="claude-haiku-4-5", max_tokens=8, messages=MSGS)
    assert _only(recording_transport)["response_model"] == "claude-haiku-4-5-20251001"


def test_anthropic_stream_sends_message_starts_model(anthropic_client, recording_transport):
    s = anthropic_client.messages.create(model="claude-haiku-4-5", max_tokens=8, messages=MSGS, stream=True)
    for _ in s:
        pass
    assert _only(recording_transport)["response_model"] == "claude-haiku-4-5-20251001"


def test_a_failed_call_sends_no_response_model(recording_transport):
    openai = pytest.importorskip("openai")
    c = openai.AzureOpenAI(
        api_key="k", api_version="2024-10-21", azure_endpoint="https://acme.openai.azure.com",
        http_client=_routed(openai.AzureOpenAI, lambda hx, req: hx.Response(500, json={"error": {"message": "x"}})),
        max_retries=0,
    )
    _install(_openai, c, recording_transport)
    with pytest.raises(Exception):
        c.chat.completions.create(model="support-bot", messages=MSGS)
    assert "response_model" not in _only(recording_transport)


def test_only_a_model_id_the_server_takes_is_sent():
    assert response_model(" gpt-4o ") == "gpt-4o"
    for bad in (None, "", "   ", 7, "x" * 201, "gpt\x004o"):
        assert response_model(bad) is None
    assert response_model_field(None) == {}
    assert response_model_field(SERVED) == {"response_model": SERVED}
