"""Token usage from every shape a provider call can return.

* A parsed response: ``response.usage``.
* ``with_raw_response``: a LegacyAPIResponse, usage only after ``parse()``.
  Its parse is cached, so parsing here hands the customer the same object.
* ``stream=True``: usage arrives in the stream (openai's final chunk with
  ``stream_options={"include_usage": True}``; anthropic's message_start and
  message_delta events). The observation is posted when the stream ends.

Through 1.7.0 the last two were posted at once as 0/0 tokens.
"""

from __future__ import annotations

import inspect
import threading
import weakref
from typing import Any, Callable, Iterator, Optional

Usage = tuple[Optional[int], Optional[int]]
FinishFn = Callable[[Optional[int], Optional[int], Optional[BaseException]], None]


def _field(obj: Any, name: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _int(value: Any) -> Optional[int]:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


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


def observe_stream(stream: Any, chunk_usage: Callable[[Any], Usage], finish: FinishFn) -> Any:
    """Calls ``finish(input, output, error)`` once, when ``stream`` ends.

    Check ``can_observe_stream`` first. A provider SDK Stream is returned as
    itself (same type, still a context manager): its ``_iterator`` generator
    is swapped for one that reads usage on the way past. Ending means
    exhausted, raised, closed, or garbage-collected unread. A bare generator
    (the proxy's dict chunks, for clients we can't type) comes back wrapped.
    """
    state: dict[str, Any] = {"in": None, "out": None, "done": False}
    lock = threading.Lock()

    def done(error: Optional[BaseException] = None) -> None:
        with lock:
            if state["done"]:
                return
            state["done"] = True
        try:
            finish(state["in"], state["out"], error)
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
