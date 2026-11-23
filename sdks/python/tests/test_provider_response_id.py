"""Every observation carries the provider's id for the call when it has one.

Field test against a self-hosted install: observe-mode SDK rows had no
provider_response_id, so the same call seen by the proxy, a gateway or a
cost import that knows the provider's id couldn't be matched to them
(same-call rule, packages/contracts/src/same-call.ts). The server reads
``provider_response_id`` off the observation (ObservationDto, at most 200
characters).
"""

from __future__ import annotations

import inspect
import io
import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from tensorcost._providers import _anthropic, _bedrock, _openai, _vertex

CHAT = {
    "id": "chatcmpl-direct",
    "object": "chat.completion",
    "created": 0,
    "model": "gpt-4o-mini",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
}
LEGACY = {
    "id": "cmpl-legacy",
    "object": "text_completion",
    "created": 0,
    "model": "gpt-3.5-turbo-instruct",
    "choices": [{"index": 0, "text": "hi", "finish_reason": "stop", "logprobs": None}],
    "usage": {"prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5},
}
CHUNKS = [
    {"id": "chatcmpl-stream", "object": "chat.completion.chunk", "created": 0, "model": "gpt-4o-mini",
     "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Ok."}, "finish_reason": "stop"}]},
    {"id": "chatcmpl-stream", "object": "chat.completion.chunk", "created": 0, "model": "gpt-4o-mini",
     "choices": [], "usage": {"prompt_tokens": 9, "completion_tokens": 3, "total_tokens": 12}},
]
MESSAGE = {
    "id": "msg_direct", "type": "message", "role": "assistant", "content": [{"type": "text", "text": "hi"}],
    "model": "claude-haiku-4-5", "stop_reason": "end_turn", "usage": {"input_tokens": 10, "output_tokens": 4},
}
ANTHROPIC_EVENTS = [
    ("message_start", {"type": "message_start", "message": {
        "id": "msg_stream", "type": "message", "role": "assistant", "content": [], "model": "claude-haiku-4-5",
        "stop_reason": None, "stop_sequence": None, "usage": {"input_tokens": 10, "output_tokens": 1}}}),
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


def _openai_respond(hx: Any, req: Any) -> Any:
    body = json.loads(req.content)
    if body.get("stream"):
        sse = b"".join(f"data: {json.dumps(c)}\n\n".encode() for c in CHUNKS) + b"data: [DONE]\n\n"
        return hx.Response(200, content=sse, headers={"content-type": "text/event-stream"})
    if req.url.path.endswith("/chat/completions"):
        return hx.Response(200, json=CHAT)
    return hx.Response(200, json=LEGACY)


@pytest.fixture
def openai_client(recording_transport):
    openai = pytest.importorskip("openai")
    c = openai.OpenAI(api_key="sk", http_client=_routed(openai.OpenAI, _openai_respond), max_retries=0)
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


# OpenAI


def test_openai_create_sends_the_completion_id(openai_client, recording_transport):
    openai_client.chat.completions.create(model="gpt-4o-mini", messages=MSGS)
    assert _only(recording_transport)["provider_response_id"] == "chatcmpl-direct"


def test_openai_with_raw_response_sends_the_completion_id(openai_client, recording_transport):
    raw = openai_client.chat.completions.with_raw_response.create(model="gpt-4o-mini", messages=MSGS)
    assert raw.parse().id == "chatcmpl-direct"
    assert _only(recording_transport)["provider_response_id"] == "chatcmpl-direct"


def test_openai_stream_sends_the_chunks_id(openai_client, recording_transport):
    s = openai_client.chat.completions.create(
        model="gpt-4o-mini", messages=MSGS, stream=True, stream_options={"include_usage": True},
    )
    with s:
        list(s)
    obs = _only(recording_transport)
    assert obs["provider_response_id"] == "chatcmpl-stream"
    assert (obs["input_tokens"], obs["output_tokens"]) == (9, 3)


def test_openai_stream_abandoned_after_one_chunk_still_sends_its_id(openai_client, recording_transport):
    import gc

    s = openai_client.chat.completions.create(model="gpt-4o-mini", messages=MSGS, stream=True)
    next(iter(s))
    s.close()
    del s
    gc.collect()
    assert _only(recording_transport)["provider_response_id"] == "chatcmpl-stream"


def test_openai_legacy_completion_sends_its_id(openai_client, recording_transport):
    openai_client.completions.create(model="gpt-3.5-turbo-instruct", prompt="x")
    assert _only(recording_transport)["provider_response_id"] == "cmpl-legacy"


def test_openai_error_has_no_response_id(recording_transport):
    openai = pytest.importorskip("openai")
    c = openai.OpenAI(
        api_key="sk", max_retries=0,
        http_client=_routed(openai.OpenAI, lambda hx, req: hx.Response(
            400, json={"error": {"message": "bad", "type": "invalid_request_error"}})),
    )
    _install(_openai, c, recording_transport)
    with pytest.raises(openai.BadRequestError):
        c.chat.completions.create(model="gpt-4o-mini", messages=MSGS)
    obs = _only(recording_transport)
    assert obs["status"] == "error"
    assert "provider_response_id" not in obs


def test_an_id_the_server_would_refuse_is_left_off(recording_transport):
    # ObservationDto refuses an id over 200 characters, and with it the
    # whole observation (forbidNonWhitelisted / MaxLength).
    openai = pytest.importorskip("openai")
    body = dict(CHAT, id="x" * 201)
    c = openai.OpenAI(
        api_key="sk", max_retries=0,
        http_client=_routed(openai.OpenAI, lambda hx, req: hx.Response(200, json=body)),
    )
    _install(_openai, c, recording_transport)
    c.chat.completions.create(model="gpt-4o-mini", messages=MSGS)
    obs = _only(recording_transport)
    assert "provider_response_id" not in obs
    assert obs["input_tokens"] == 10


# Anthropic


def test_anthropic_create_sends_the_message_id(anthropic_client, recording_transport):
    anthropic_client.messages.create(model="claude-haiku-4-5", max_tokens=8, messages=MSGS)
    assert _only(recording_transport)["provider_response_id"] == "msg_direct"


def test_anthropic_with_raw_response_sends_the_message_id(anthropic_client, recording_transport):
    raw = anthropic_client.messages.with_raw_response.create(
        model="claude-haiku-4-5", max_tokens=8, messages=MSGS,
    )
    assert raw.parse().id == "msg_direct"
    assert _only(recording_transport)["provider_response_id"] == "msg_direct"


def test_anthropic_stream_sends_message_start_id(anthropic_client, recording_transport):
    s = anthropic_client.messages.create(
        model="claude-haiku-4-5", max_tokens=8, messages=MSGS, stream=True,
    )
    list(s)
    obs = _only(recording_transport)
    assert obs["provider_response_id"] == "msg_stream"
    assert (obs["input_tokens"], obs["output_tokens"]) == (10, 4)


# Bedrock: the request id (x-amzn-requestid), the key the proxy, the
# invocation log and the CUR share. Anthropic on Bedrock's msg_bdrk_ body id
# is known to none of them.


def _bedrock_meta(request_id: str) -> dict:
    return {"RequestId": request_id, "HTTPStatusCode": 200,
            "HTTPHeaders": {"x-amzn-requestid": request_id}}


def test_bedrock_invoke_model_sends_the_request_id(bedrock_client, recording_transport):
    body = json.dumps({"id": "msg_bdrk_1", "usage": {"input_tokens": 15, "output_tokens": 30}}).encode()
    bedrock_client.invoke_model.return_value = {
        "body": io.BytesIO(body), "contentType": "application/json",
        "ResponseMetadata": _bedrock_meta("req-invoke"),
    }
    _install(_bedrock, bedrock_client, recording_transport)
    bedrock_client.invoke_model(modelId="anthropic.claude-3-haiku", body="{}")
    assert _only(recording_transport)["provider_response_id"] == "req-invoke"


def test_bedrock_converse_sends_the_request_id(bedrock_client, recording_transport):
    bedrock_client.converse.return_value = {
        "output": {"message": {"role": "assistant", "content": []}},
        "usage": {"inputTokens": 12, "outputTokens": 24},
        "ResponseMetadata": _bedrock_meta("req-converse"),
    }
    _install(_bedrock, bedrock_client, recording_transport)
    bedrock_client.converse(modelId="amazon.nova-lite-v1:0", messages=[])
    assert _only(recording_transport)["provider_response_id"] == "req-converse"


def test_bedrock_stream_sends_the_request_id(bedrock_client, recording_transport):
    bedrock_client.converse_stream.return_value = {
        "stream": iter([]), "ResponseMetadata": _bedrock_meta("req-stream"),
    }
    _install(_bedrock, bedrock_client, recording_transport)
    bedrock_client.converse_stream(modelId="amazon.nova-lite-v1:0", messages=[])
    assert _only(recording_transport)["provider_response_id"] == "req-stream"


def test_bedrock_without_response_metadata_sends_no_id(bedrock_client, recording_transport):
    _install(_bedrock, bedrock_client, recording_transport)
    bedrock_client.converse(modelId="amazon.nova-lite-v1:0", messages=[])
    assert "provider_response_id" not in _only(recording_transport)


# Vertex: google-genai's GenerateContentResponse.response_id.


def test_vertex_generate_content_sends_the_response_id(vertex_client, recording_transport):
    resp = MagicMock()
    resp.usage_metadata.prompt_token_count = 42
    resp.usage_metadata.candidates_token_count = 17
    resp.response_id = "vertex-resp-1"
    vertex_client.models.generate_content.return_value = resp
    _install(_vertex, vertex_client, recording_transport)
    vertex_client.models.generate_content(model="gemini-2.5-flash", contents="x")
    assert _only(recording_transport)["provider_response_id"] == "vertex-resp-1"


def test_vertex_without_a_response_id_sends_none(vertex_client, recording_transport):
    # The conftest fake is a MagicMock: an attribute it invents isn't an id.
    _install(_vertex, vertex_client, recording_transport)
    vertex_client.models.generate_content(model="gemini-2.5-flash", contents="x")
    assert "provider_response_id" not in _only(recording_transport)
