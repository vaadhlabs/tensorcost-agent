"""Turn the proxy's answer into what the provider SDK itself would return.

Route mode sends the call to the TensorCost proxy over our own HTTP client,
so what comes back is bytes. Through 1.7.0 the wrapper returned the parsed
JSON dict (and a generator of dicts for a stream), so
``r.choices[0].message.content`` raised AttributeError where the unwrapped
client returns a ChatCompletion.

Rather than rebuild the SDK's response types by hand (they differ between
openai 2.x and 3.x, ``with_raw_response`` wants a LegacyAPIResponse, a stream
wants a Stream that is also a context manager), the proxy's response is
replayed through a copy of the customer's own client: ``with_options`` with
an HTTP client whose transport returns that response. The SDK parses it
exactly as it parses an answer from the provider, with the same kwargs the
customer passed, so the return type, ``with_raw_response`` and
``stream=True`` all come out as the unwrapped client would give them. Only
public client API is used (``with_options(http_client=, max_retries=)``).

Clients that are not real openai / anthropic clients (test fakes, anything
without a sync httpx / httpx2 ``_client``) get ``UNTYPED`` back and the
caller keeps the dict.
"""

from __future__ import annotations

import sys
import threading
from typing import Any, Callable, Iterator, Mapping, Optional

UNTYPED = object()

# Already applied by our HTTP client when it read the proxy's response.
_DROP_HEADERS = frozenset({"content-encoding", "content-length", "transfer-encoding"})


def _http_classes(client: Any) -> Optional[tuple[Any, type]]:
    """(httpx or httpx2 module, its Client class) behind ``client._client``."""
    http = getattr(client, "_client", None)
    if http is None:
        return None
    for cls in type(http).__mro__:
        mod = cls.__module__.split(".")[0]
        if mod in ("httpx", "httpx2") and cls.__name__ == "Client":
            module = sys.modules.get(mod)
            return (module, cls) if module is not None else None
    return None


def replay_through_client(
    client: Any,
    method_of: Callable[[Any], Callable[..., Any]],
    args: tuple,
    kwargs: Mapping[str, Any],
    *,
    status_code: int,
    headers: Any,
    content: Optional[bytes] = None,
    chunks: Optional[Iterator[bytes]] = None,
    on_close: Optional[Callable[[], None]] = None,
) -> Any:
    """``method_of(copy)(*args, **kwargs)`` on a copy of ``client`` whose HTTP
    answer is the given response, or ``UNTYPED``.

    ``chunks`` (with ``on_close``) replays a streamed body: the SDK's Stream
    reads it as it iterates and closing that Stream calls ``on_close``.
    """
    found = _http_classes(client)
    with_options = getattr(client, "with_options", None)
    if found is None or not callable(with_options):
        return UNTYPED
    hx, client_cls = found
    items = headers.multi_items() if hasattr(headers, "multi_items") else list(dict(headers).items())
    replay_headers = [(k, v) for k, v in items if k.lower() not in _DROP_HEADERS]

    try:
        stream_cls = _byte_stream_class(hx)

        class _Replay(hx.BaseTransport):  # type: ignore[misc, name-defined]
            def handle_request(self, request: Any) -> Any:
                if chunks is not None:
                    return hx.Response(
                        status_code, headers=replay_headers,
                        stream=stream_cls(chunks, on_close), request=request,
                    )
                return hx.Response(status_code, headers=replay_headers, content=content or b"", request=request)

        copy = with_options(http_client=client_cls(transport=_Replay()), max_retries=0)
        method = method_of(copy)
    except Exception:  # noqa: BLE001 - an SDK we can't drive this way
        return UNTYPED
    return method(*args, **kwargs)


_stream_classes: dict[str, type] = {}


def _byte_stream_class(hx: Any) -> type:
    cached = _stream_classes.get(hx.__name__)
    if cached is not None:
        return cached

    class _ChunkStream(hx.SyncByteStream):  # type: ignore[misc, name-defined]
        def __init__(self, chunks: Iterator[bytes], on_close: Optional[Callable[[], None]]) -> None:
            self._chunks = chunks
            self._on_close = on_close
            self._closed = False
            self._lock = threading.Lock()

        def __iter__(self) -> Iterator[bytes]:
            yield from self._chunks

        def close(self) -> None:
            with self._lock:
                if self._closed:
                    return
                self._closed = True
            if self._on_close is not None:
                try:
                    self._on_close()
                except Exception:  # noqa: BLE001
                    pass

    _stream_classes[hx.__name__] = _ChunkStream
    return _ChunkStream


def route_stream_result(
    client: Any,
    method_of: Callable[[Any], Callable[..., Any]],
    raw_stream: Any,
    chunk_usage: Callable[[Any], Any],
    chunk_id: Callable[[Any], Optional[str]],
    finish: Callable[[Any, Any, Optional[BaseException], Optional[str]], None],
    args: tuple,
    kwargs: Mapping[str, Any],
) -> Any:
    """A streamed proxy answer as the SDK's own Stream, observed when it ends.

    ``raw_stream`` is the ``RawProxyStream`` from proxy_request_stream(raw=True);
    anything else (a stand-in in tests) is observed as it is.
    """
    from ._proxy_client import RawProxyStream
    from ._usage import can_observe_stream, observe_stream

    if isinstance(raw_stream, RawProxyStream):
        try:
            typed = replay_through_client(
                client, method_of, args, kwargs,
                status_code=raw_stream.status_code,
                headers=raw_stream.headers,
                chunks=raw_stream.iter_bytes(),
                on_close=raw_stream.close,
            )
        except BaseException:
            raw_stream.close()
            raise
        stream = raw_stream.iter_json() if typed is UNTYPED else typed
    else:
        stream = raw_stream
    if can_observe_stream(stream):
        return observe_stream(stream, chunk_usage, chunk_id, finish)
    finish(None, None, None, None)
    return stream
