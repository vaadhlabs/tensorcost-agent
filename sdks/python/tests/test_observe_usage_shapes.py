"""Observe mode records real usage whatever shape the call returns.

Field test (tensorcost 1.7.0, openai 3.57.0): with_raw_response calls were
recorded as 0/0 tokens, and so were streamed calls with
stream_options={"include_usage": True}, where the usage is in the stream.
"""

from __future__ import annotations

import inspect
import json
from typing import Any

import pytest

from tensorcost._providers import _anthropic, _openai

CHAT = {
    "id": "chatcmpl-direct",
    "object": "chat.completion",
    "created": 0,
    "model": "gpt-4o-mini",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
}
CHUNKS = [
    {"id": "c1", "object": "chat.completion.chunk", "created": 0, "model": "gpt-4o-mini",
     "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Ok."}, "finish_reason": "stop"}]},
    {"id": "c1", "object": "chat.completion.chunk", "created": 0, "model": "gpt-4o-mini",
     "choices": [], "usage": {"prompt_tokens": 9, "completion_tokens": 3, "total_tokens": 12}},
]
ANTHROPIC_EVENTS = [
    ("message_start", {"type": "message_start", "message": {
        "id": "msg_1", "type": "message", "role": "assistant", "content": [], "model": "claude-haiku-4-5",
        "stop_reason": None, "stop_sequence": None, "usage": {"input_tokens": 10, "output_tokens": 1}}}),
    ("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
    ("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hi"}}),
    ("content_block_stop", {"type": "content_block_stop", "index": 0}),
    ("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                       "usage": {"output_tokens": 4}}),
    ("message_stop", {"type": "message_stop"}),
]


def _http_module(sdk: Any) -> Any:
    ann = str(inspect.signature(sdk.__init__).parameters["http_client"].annotation)
    return pytest.importorskip("httpx2" if "httpx2" in ann else "httpx")


def _routed(sdk: Any, respond: Any) -> Any:
    hx = _http_module(sdk)
    return hx.Client(transport=hx.MockTransport(lambda req: respond(hx, req)))


def _openai_client(respond: Any) -> Any:
    openai = pytest.importorskip("openai")
    return openai, openai.OpenAI(api_key="sk", http_client=_routed(openai.OpenAI, respond), max_retries=0)


def _install(mod: Any, c: Any, t: Any) -> None:
    mod.install(c, t, tenant_id=None, environment=None, connection_id=None)


def _openai_respond(hx: Any, req: Any) -> Any:
    if json.loads(req.content).get("stream"):
        body = b"".join(f"data: {json.dumps(c)}\n\n".encode() for c in CHUNKS) + b"data: [DONE]\n\n"
        return hx.Response(200, content=body, headers={"content-type": "text/event-stream"})
    return hx.Response(200, json=CHAT)


def test_with_raw_response_records_usage(recording_transport):
    openai, c = _openai_client(_openai_respond)
    _install(_openai, c, recording_transport)
    raw = c.chat.completions.with_raw_response.create(model="gpt-4o-mini", messages=[{"role": "user", "content": "x"}])
    assert raw.parse().choices[0].message.content == "hi"
    obs = recording_transport.observations[-1]
    assert (obs["input_tokens"], obs["output_tokens"]) == (10, 2)


def test_streamed_call_records_usage_from_the_final_chunk(recording_transport):
    openai, c = _openai_client(_openai_respond)
    _install(_openai, c, recording_transport)
    s = c.chat.completions.create(
        model="gpt-4o-mini", messages=[{"role": "user", "content": "x"}],
        stream=True, stream_options={"include_usage": True},
    )
    assert isinstance(s, openai.Stream)
    # Nothing is recorded before the stream has been read.
    assert recording_transport.observations == []
    with s:
        chunks = list(s)
    assert len(chunks) == 2
    assert len(recording_transport.observations) == 1
    obs = recording_transport.observations[0]
    assert (obs["input_tokens"], obs["output_tokens"]) == (9, 3)
    assert obs["status"] == "success"


def test_streamed_call_abandoned_early_is_still_recorded_once(recording_transport):
    openai, c = _openai_client(_openai_respond)
    _install(_openai, c, recording_transport)
    s = c.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": "x"}], stream=True)
    next(iter(s))
    s.close()
    del s
    import gc

    gc.collect()
    assert len(recording_transport.observations) == 1


def test_anthropic_streamed_call_records_usage_from_its_events(recording_transport):
    anthropic = pytest.importorskip("anthropic")

    def respond(hx: Any, req: Any) -> Any:
        body = b"".join(
            f"event: {name}\ndata: {json.dumps(data)}\n\n".encode() for name, data in ANTHROPIC_EVENTS
        )
        return hx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    c = anthropic.Anthropic(api_key="sk-ant", http_client=_routed(anthropic.Anthropic, respond), max_retries=0)
    _install(_anthropic, c, recording_transport)
    s = c.messages.create(
        model="claude-haiku-4-5", max_tokens=8, messages=[{"role": "user", "content": "x"}], stream=True,
    )
    assert isinstance(s, anthropic.Stream)
    events = list(s)
    assert events[0].type == "message_start"
    obs = recording_transport.observations[-1]
    assert (obs["input_tokens"], obs["output_tokens"]) == (10, 4)


def test_anthropic_with_raw_response_records_usage(recording_transport):
    anthropic = pytest.importorskip("anthropic")
    msg = {"id": "msg_1", "type": "message", "role": "assistant", "content": [{"type": "text", "text": "hi"}],
           "model": "claude-haiku-4-5", "stop_reason": "end_turn", "usage": {"input_tokens": 10, "output_tokens": 4}}
    c = anthropic.Anthropic(
        api_key="sk-ant", http_client=_routed(anthropic.Anthropic, lambda hx, req: hx.Response(200, json=msg)),
        max_retries=0,
    )
    _install(_anthropic, c, recording_transport)
    raw = c.messages.with_raw_response.create(
        model="claude-haiku-4-5", max_tokens=8, messages=[{"role": "user", "content": "x"}],
    )
    assert raw.parse().id == "msg_1"
    obs = recording_transport.observations[-1]
    assert (obs["input_tokens"], obs["output_tokens"]) == (10, 4)
