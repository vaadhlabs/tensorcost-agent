"""Re-raising a relayed provider error as the provider SDK's own type.

When the proxy relays an OpenAI 400, a caller written against the openai
package expects ``openai.BadRequestError``, not a TensorCost class. Both
openai-python and anthropic-python build their status errors with the
client's ``_make_status_error_from_response(response)``; there is no public
equivalent. So the SDK rebuilds the relayed response and asks the client,
exactly as the client would have for a direct call. A client without that
method (or one that fails to build it) gets the ``TensorCostProviderError``.
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from ._errors import TensorCostProviderError

_HOP_HEADERS = {"content-length", "content-encoding", "transfer-encoding", "connection"}


def provider_native_error(client: Any, err: BaseException, url: str) -> BaseException:
    if not isinstance(err, TensorCostProviderError) or err.status is None:
        return err
    make = getattr(type(client), "_make_status_error_from_response", None)
    if not callable(make):
        return err
    try:
        body = err.body
        content = (
            body.encode() if isinstance(body, str) else json.dumps(body).encode()
            if body is not None
            else b""
        )
        headers = {k: v for k, v in err.headers.items() if k.lower() not in _HOP_HEADERS}
        response = httpx.Response(
            err.status,
            headers=headers,
            content=content,
            request=httpx.Request("POST", url),
        )
        native = make(client, response)
    except Exception:  # noqa: BLE001
        return err
    if not isinstance(native, BaseException):
        return err
    try:
        native.tensorcost = err  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass
    return native
