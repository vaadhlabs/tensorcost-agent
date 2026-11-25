"""Did a failed proxied call reach the provider?

A proxied call is sent once to TensorCost, which forwards it to the provider.
When that goes wrong the SDK has two ways to try again: retry through the
proxy, or fall open and call the provider directly. Either one sends the call
to the provider a second time, so either one is safe only when the first
attempt provably never got there. Through 1.7.0 the SDK retried every 5xx,
timeout and network error up to five times and then fell back, so a call the
provider had already answered and billed (the proxy then failed, timed out,
or lost the response) could run six times.

This module answers the question from what the SDK can see, and answers
"maybe" whenever it can't prove "no":

* A send that raised. Only a failure while connecting proves nothing was
  written: ``httpx.ConnectError`` (refused, DNS, TLS handshake),
  ``httpx.ConnectTimeout`` and ``httpx.PoolTimeout``. A read or write error,
  a protocol error or a read timeout may come after the body was written and
  the proxy forwarded it.

* A non-2xx answer. The proxy marks every error with ``x-tc-error-origin``:
  ``provider`` when the status is the provider's (relayed, or the proxy's
  answer about a provider that did not answer in time), ``proxy`` when it is
  the proxy's own. The gateway marks its own answers ``gateway``. The proxy
  also sends ``x-should-retry: false`` (the OpenAI and Anthropic SDK
  convention) on every error it chose not to resend because the provider may
  have run the call. ``proxy`` alone does not prove "not sent": the proxy
  also marks its 502 for a provider answer it could not translate, and any
  unhandled error after the forward.

Same rules as the Node SDK (sdks/node/src/delivery.ts).
"""

from __future__ import annotations

import re
import socket
from typing import Any, Literal, Mapping

import httpx

#: Set by inference-proxy-service and the gateway on every 4xx/5xx.
ERROR_ORIGIN_HEADER = "x-tc-error-origin"
#: OpenAI/Anthropic SDK convention; "false" means do not send this again.
SHOULD_RETRY_HEADER = "x-should-retry"
#: Set by inference-proxy-service on its answer for a provider send that got
#: no answer (unanswered-send.ts): ``provider_unreachable``,
#: ``provider_connection_lost`` or ``provider_timeout``.
UPSTREAM_ERROR_HEADER = "x-tc-upstream-error"

#: "unsent": TensorCost answered before forwarding, safe to retry and to fall
#: back. "provider": the provider's own answer, relayed. "maybe_sent": can't
#: tell, the provider may have run and billed it; never resend.
ErrorAnswer = Literal["unsent", "provider", "maybe_sent"]

# Connect-phase codes as they appear in a message (the gateway relaying its
# own connect failure to the proxy in a 502 body).
_CONNECT_MESSAGE_RE = re.compile(
    r"\b(ECONNREFUSED|ENOTFOUND|EAI_AGAIN|ENETUNREACH|EHOSTUNREACH|EHOSTDOWN|EADDRNOTAVAIL|UND_ERR_CONNECT_TIMEOUT)\b"
)


def send_error_sent_nothing(exc: BaseException) -> bool:
    """True when a send that raised provably wrote nothing.

    httpx raises ``ConnectError`` for a refused connection, a DNS failure and
    a TLS handshake failure, and ``ConnectTimeout`` / ``PoolTimeout`` before
    a connection exists. A bare ``ConnectionRefusedError`` or ``socket.gaierror``
    (from a custom transport) counts too. Everything else, a reset or a read
    timeout included, may come after the proxy forwarded the call.
    """
    if isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout)):
        return True
    if isinstance(exc, (ConnectionRefusedError, socket.gaierror)):
        return True
    return False


def classify_error_answer(status: int, headers: Mapping[str, str], body: Any) -> ErrorAnswer:
    """Classify a non-2xx, non-refusal answer from the proxy or the gateway.

    In order:

    0. The proxy's 502 ``x-tc-upstream-error: provider_unreachable``, marked
       origin ``provider``, without ``x-should-retry: false``
       (``is_provider_unreachable``): the proxy
       could not connect to the provider (connect, DNS or TLS, after its own
       one retry), so nothing of the call left it and nothing was billed.
       The origin is ``provider``, so it has to come before rule 1. The
       same header with another origin or none is not the proxy's answer as
       unanswered-send.ts writes it, and stays "maybe_sent". A routed call's answer carries ``x-should-retry: false`` (an
       earlier route may have run) and stays "maybe_sent".
    1. origin ``provider``: the provider's answer. Never resent: it would
       answer the same, or (a 5xx, the proxy's 504) it may already have run.
    2. ``x-should-retry: false``: whoever sent it says don't. Always honoured.
    3. 429 and 402 not from the provider: TensorCost's rate limit and plan
       ceiling, both checked before the call is forwarded. An unmarked 429 (a
       proxy older than the origin header) counts too: a 429 is a rejection,
       and nobody runs or bills one.
    4. 503 from the gateway (its breaker is open: no forward attempt) or the
       proxy (every 503 it sends is about something it needs before
       forwarding), or any 503 with ``x-should-retry: true``.
    5. 502 from the gateway carrying its own connect failure to the proxy.
    6. Anything else is "maybe_sent": a 500 or 502 marked ``proxy`` (an
       unhandled error, or a provider answer the proxy could not translate,
       both after the forward), a gateway 502/504 for a proxy that reset or
       timed out mid-call, any 5xx with no origin header. The provider may
       have run those, so the SDK surfaces the error rather than risk billing
       the call twice.
    """
    origin = (headers.get(ERROR_ORIGIN_HEADER) or "").strip().lower() or None
    should_retry = (headers.get(SHOULD_RETRY_HEADER) or "").strip().lower() or None

    if is_provider_unreachable(status, headers):
        return "unsent"
    if origin == "provider":
        return "provider"
    if should_retry == "false":
        return "maybe_sent"
    if status in (429, 402):
        return "unsent"
    if status == 503 and (origin in ("gateway", "proxy") or should_retry == "true"):
        return "unsent"
    if status == 502 and origin == "gateway" and _gateway_could_not_connect(body):
        return "unsent"
    return "maybe_sent"


def is_provider_unreachable(status: int, headers: Mapping[str, str]) -> bool:
    """The proxy's 502 for a provider it could not connect to, nothing sent.

    Origin ``provider``, ``x-tc-upstream-error: provider_unreachable``, no
    ``x-should-retry: false``. Nothing was sent, so it is "unsent" to
    ``classify_error_answer``. The proxy already retried the provider once,
    so the proxy client doesn't retry it, but it does count it against the
    circuit: when the proxy's own egress is broken every call gets this
    answer after the proxy's two connect attempts (about 20 s at a 10 s
    connect timeout), and an open circuit is what stops each call paying
    that before going direct. The wrapper calls the provider directly when
    fail-open is on. ``provider_connection_lost`` and ``provider_timeout`` may have run, carry
    ``x-should-retry: false``, and never match.
    """
    if status != 502:
        return False
    origin = (headers.get(ERROR_ORIGIN_HEADER) or "").strip().lower()
    kind = (headers.get(UPSTREAM_ERROR_HEADER) or "").strip().lower()
    should_retry = (headers.get(SHOULD_RETRY_HEADER) or "").strip().lower()
    return origin == "provider" and kind == "provider_unreachable" and should_retry != "false"


def _gateway_could_not_connect(body: Any) -> bool:
    if not isinstance(body, dict):
        return False
    error = body.get("error")
    if not isinstance(error, dict):
        return False
    message = error.get("message")
    return (
        error.get("code") == "UPSTREAM_PROXY_ERROR"
        and isinstance(message, str)
        and _CONNECT_MESSAGE_RE.search(message) is not None
    )
