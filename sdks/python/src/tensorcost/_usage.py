"""Token usage from every shape a provider call can return.

* A parsed response: ``response.usage``.
* ``with_raw_response``: a LegacyAPIResponse, usage only after ``parse()``.
  Its parse is cached, so parsing here hands the customer the same object.
* ``stream=True``: usage arrives in the stream (openai's final chunk with
  ``stream_options={"include_usage": True}``; anthropic's message_start and
  message_delta events). The observation is posted when the stream ends.

Through 1.7.0 the last two were posted at once as 0/0 tokens.

The provider's id for the call rides along the same way (``id`` on a
response or an openai chunk, ``message.id`` on anthropic's message_start).
The server keys the same call seen by the SDK, the proxy, a gateway or a
cost import on it (provider_response_id).

So does the model the provider says it served (``model`` on a response or
an openai chunk, ``message.model`` on anthropic's message_start), sent as
``response_model``. An Azure OpenAI request names a deployment
("support-bot"), which has no price; the server prices the call from the
model behind it. A stream sends the last chunk that names one: Azure's
first chunk (prompt filter results) names none.
"""

from __future__ import annotations

import inspect
import threading
import weakref
from typing import Any, Callable, Iterator, Optional

Usage = tuple[Optional[int], Optional[int]]
FinishFn = Callable[[Optional[int], Optional[int], Optional[BaseException], Optional[str]], None]
# With ``chunk_model``: finish(input, output, error, response_id, response_model).
FinishWithModelFn = Callable[
    [Optional[int], Optional[int], Optional[BaseException], Optional[str], Optional[str]], None
]

# ObservationDto.provider_response_id is @MaxLength(200), and the DTO refuses
# the whole observation over a bad field, so a longer id is left off.
MAX_RESPONSE_ID_LENGTH = 200
# ObservationDto.response_model is @MaxLength(200) too.
MAX_RESPONSE_MODEL_LENGTH = 200


def _field(obj: Any, name: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _int(value: Any) -> Optional[int]:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def response_id(value: Any) -> Optional[str]:
    """``value`` as a provider response id the server will take, else None."""
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value if 0 < len(value) <= MAX_RESPONSE_ID_LENGTH else None


def response_id_of(response: Any) -> Optional[str]:
    """``response.id``: chatcmpl-/cmpl-/resp_ on openai, msg_ on anthropic."""
    return response_id(_field(response, "id"))


def response_id_field(rid: Optional[str]) -> dict[str, str]:
    """The observation's ``provider_response_id``, absent when there is none."""
    rid = response_id(rid)
    return {"provider_response_id": rid} if rid is not None else {}


def response_model(value: Any) -> Optional[str]:
    """``value`` as a model id the server will take, else None."""
    if not isinstance(value, str):
        return None
    value = value.strip()
    if not 0 < len(value) <= MAX_RESPONSE_MODEL_LENGTH or "\x00" in value:
        return None
    return value


def response_model_of(response: Any) -> Optional[str]:
    """``response.model``: the model the provider ran the call on."""
    return response_model(_field(response, "model"))


def response_model_field(model: Optional[str]) -> dict[str, str]:
    """The observation's ``response_model``, absent when there is none."""
    model = response_model(model)
    return {"response_model": model} if model is not None else {}


def parsed_for_usage(response: Any) -> Any:
    """The parsed object behind a ``with_raw_response`` result, else ``response``.

    Leaves alone ``with_streaming_response`` (its body is the customer's to
    read) and a raw response to a streamed request.
    """
    http = getattr(response, "http_response", None)
    parse = getattr(response, "parse", None)
    if http is None or not callable(parse):
        return response
    try:
        # "true" (openai, anthropic 0.x) or "raw" (anthropic 2.x) is
        # with_raw_response; "stream" is with_streaming_response.
        if http.request.headers.get("X-Stainless-Raw-Response") not in ("true", "raw"):
            return response
        if "text/event-stream" in (http.headers.get("content-type") or ""):
            return response
        return parse()
    except Exception:  # noqa: BLE001
        return response


def openai_chunk_usage(chunk: Any) -> Usage:
    usage = _field(chunk, "usage")
    if usage is None:
        return None, None
    return _int(_field(usage, "prompt_tokens")), _int(_field(usage, "completion_tokens"))


def openai_chunk_id(chunk: Any) -> Optional[str]:
    return response_id_of(chunk)


def openai_chunk_model(chunk: Any) -> Optional[str]:
    return response_model_of(chunk)


def anthropic_event_model(event: Any) -> Optional[str]:
    if _field(event, "type") != "message_start":
        return None
    return response_model_of(_field(event, "message"))


def anthropic_event_id(event: Any) -> Optional[str]:
    if _field(event, "type") != "message_start":
        return None
    return response_id_of(_field(event, "message"))


def anthropic_event_usage(event: Any) -> Usage:
    kind = _field(event, "type")
    if kind == "message_start":
        usage = _field(_field(event, "message"), "usage")
    elif kind == "message_delta":
        usage = _field(event, "usage")
    else:
        return None, None
    if usage is None:
        return None, None
    return _int(_field(usage, "input_tokens")), _int(_field(usage, "output_tokens"))


def can_observe_stream(stream: Any) -> bool:
    return inspect.isgenerator(getattr(stream, "_iterator", None)) or inspect.isgenerator(stream)


def observe_stream(
    stream: Any,
    chunk_usage: Callable[[Any], Usage],
    chunk_id: Callable[[Any], Optional[str]],
    finish: Any,
    chunk_model: Optional[Callable[[Any], Optional[str]]] = None,
) -> Any:
    """Calls ``finish(input, output, error, response_id)`` once, when ``stream`` ends.

    With ``chunk_model``, ``finish`` also gets the last model a chunk named
    (FinishWithModelFn).

    Check ``can_observe_stream`` first. A provider SDK Stream is returned as
    itself (same type, still a context manager): its ``_iterator`` generator
    is swapped for one that reads usage on the way past. Ending means
    exhausted, raised, closed, or garbage-collected unread. A bare generator
    (the proxy's dict chunks, for clients we can't type) comes back wrapped.
    """
    state: dict[str, Any] = {"in": None, "out": None, "id": None, "model": None, "done": False}
    lock = threading.Lock()

    def done(error: Optional[BaseException] = None) -> None:
        with lock:
            if state["done"]:
                return
            state["done"] = True
        try:
            if chunk_model is None:
                finish(state["in"], state["out"], error, state["id"])
            else:
                finish(state["in"], state["out"], error, state["id"], state["model"])
        except Exception:  # noqa: BLE001
            pass

    def watch(items: Iterator[Any]) -> Iterator[Any]:
        try:
            for item in items:
                try:
                    i, o = chunk_usage(item)
                    if i is not None:
                        state["in"] = i
                    if o is not None:
                        state["out"] = o
                    if state["id"] is None:
                        state["id"] = chunk_id(item)
                    if chunk_model is not None:
                        m = chunk_model(item)
                        if m is not None:
                            state["model"] = m
                except Exception:  # noqa: BLE001
                    pass
                yield item
        except Exception as exc:
            done(exc)
            raise
        finally:
            done()

    inner = getattr(stream, "_iterator", None)
    if inspect.isgenerator(inner):
        try:
            stream._iterator = watch(inner)
            weakref.finalize(stream, done)
            return stream
        except Exception:  # noqa: BLE001
            done()
            return stream
    if inspect.isgenerator(stream):
        return watch(stream)
    done()
    return stream
