"""The proxy's caps on observation prose, applied before the POST.

``error_message`` is built as ``f"{type(exc).__name__}: {exc}"``, and a
provider's message (an Azure content-filter refusal, an OpenAI rate-limit
notice) runs well past 200 characters. Proxies before the fix answered such an
observation 400, and a 4xx is final here, so the call's spend row was lost.
Newer proxies cut the message themselves; cutting it here as well means an
older proxy keeps the row too, and nothing is sent that the proxy would only
throw away.

Keys (model, correlation_id, agent_id, customer, environment ...) are left as
the caller set them: cut short, one would name something else.
"""

from __future__ import annotations

ERROR_MESSAGE_MAX = 200

_ELLIPSIS = "…"


def _utf16_units(ch: str) -> int:
    return 2 if ord(ch) > 0xFFFF else 1


def truncate_free_text(s: str, max_units: int) -> str:
    """``s`` cut to at most ``max_units`` UTF-16 code units, ending in an ellipsis when cut.

    UTF-16 code units are what the proxy measures (JavaScript ``length``), so
    a message full of emoji is cut where the proxy would cut it. A character
    outside the BMP is kept whole or dropped whole.
    """
    if len(s) <= max_units // 2:
        return s
    total = sum(_utf16_units(ch) for ch in s)
    if total <= max_units:
        return s
    budget = max_units - 1  # room for the ellipsis
    out: list[str] = []
    used = 0
    for ch in s:
        n = _utf16_units(ch)
        if used + n > budget:
            break
        out.append(ch)
        used += n
    return "".join(out) + _ELLIPSIS


def bound_observation(observation: dict) -> dict:
    """The observation with its error_message within the proxy's cap; the same dict when it already is."""
    msg = observation.get("error_message")
    if not isinstance(msg, str):
        return observation
    cut = truncate_free_text(msg, ERROR_MESSAGE_MAX)
    if cut is msg:
        return observation
    return {**observation, "error_message": cut}
