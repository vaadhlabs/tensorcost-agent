"""Split a provider SDK call's request options from its body.

Every openai / anthropic ``create()`` takes four keyword arguments that are
not API parameters: ``extra_headers``, ``extra_query``, ``extra_body`` and
``timeout``. The SDKs turn them into the request's headers, query string,
extra JSON fields and timeout. When the proxy path built its body with
``dict(kwargs)`` they went into the JSON instead, and OpenAI answered 400
"Unrecognized request argument supplied: extra_headers"; with_raw_response
adds extra_headers to every call, so every raw call broke.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

REQUEST_OPTION_KEYS = frozenset({"extra_headers", "extra_query", "extra_body", "timeout"})


def _given(value: Any) -> bool:
    """False for the SDKs' NotGiven / Omit placeholders (matched by name, so
    neither SDK has to be importable)."""
    return type(value).__name__ not in ("NotGiven", "Omit")


def _internal_header(name: str) -> bool:
    # X-Stainless-* tell the provider SDK how to parse its own response
    # (with_raw_response sets X-Stainless-Raw-Response); the override-cast
    # "header" holds a type, not a string. Neither means anything to the proxy.
    lower = name.lower()
    return lower.startswith("x-stainless") or lower.startswith("____stainless")


@dataclass
class RequestOptions:
    body: dict[str, Any]
    headers: dict[str, str] = field(default_factory=dict)
    query: dict[str, Any] = field(default_factory=dict)
    timeout_s: Optional[float] = None


def split_request_options(kwargs: Mapping[str, Any]) -> RequestOptions:
    body = {k: v for k, v in kwargs.items() if k not in REQUEST_OPTION_KEYS and _given(v)}
    extra_body = kwargs.get("extra_body")
    if isinstance(extra_body, Mapping):
        # The SDKs merge extra_body over the JSON body, extra_body winning.
        body.update(extra_body)

    headers: dict[str, str] = {}
    extra_headers = kwargs.get("extra_headers")
    if isinstance(extra_headers, Mapping):
        for name, value in extra_headers.items():
            if isinstance(name, str) and isinstance(value, str) and not _internal_header(name):
                headers[name] = value

    query: dict[str, Any] = {}
    extra_query = kwargs.get("extra_query")
    if isinstance(extra_query, Mapping):
        query = {k: v for k, v in extra_query.items() if _given(v) and v is not None}

    return RequestOptions(body=body, headers=headers, query=query, timeout_s=_timeout_seconds(kwargs.get("timeout")))


def _timeout_seconds(value: Any) -> Optional[float]:
    """A float, or an httpx(2).Timeout's read timeout. None (the SDKs' "no
    timeout") and NotGiven leave the wrapper's own timeout in place."""
    if value is None or not _given(value) or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value) if value > 0 else None
    read = getattr(value, "read", None)
    if isinstance(read, (int, float)) and not isinstance(read, bool) and read > 0:
        return float(read)
    return None
