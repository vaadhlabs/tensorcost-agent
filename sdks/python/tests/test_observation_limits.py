"""error_message is cut to the proxy's 200 before the POST.

Proxies before the server fix answered a longer one 400, and a 4xx is final
in the transport, so the call's spend row was lost.
"""

import json
import time

import httpx

from tensorcost._observation_limits import (
    ERROR_MESSAGE_MAX,
    bound_observation,
    truncate_free_text,
)
from tensorcost._transport import ObservationTransport

AZURE_FILTER = (
    "BadRequestError: Error code: 400 - The response was filtered due to the prompt triggering "
    "Azure OpenAI's content management policy. Please modify your prompt and retry. To learn more "
    "about our content filtering policies please read our documentation: "
    "https://go.microsoft.com/fwlink/?linkid=2198766"
)


def _utf16_len(s: str) -> int:
    return len(s.encode("utf-16-le")) // 2


def test_leaves_a_string_at_the_cap_alone():
    s = "x" * 200
    assert truncate_free_text(s, 200) is s


def test_cuts_to_exactly_the_cap_with_an_ellipsis():
    assert len(AZURE_FILTER) > 200
    out = truncate_free_text(AZURE_FILTER, 200)
    assert out == AZURE_FILTER[:199] + "…"
    assert _utf16_len(out) == 200


def test_measures_utf16_units_like_the_proxy_and_keeps_emoji_whole():
    # 150 emoji are 150 Python characters but 300 UTF-16 units: the proxy
    # would cut this, so the SDK does.
    s = "\U0001F600" * 150
    out = truncate_free_text(s, 200)
    assert _utf16_len(out) <= 200
    assert out == "\U0001F600" * 99 + "…"
    # An emoji that would straddle the cut is dropped whole.
    out = truncate_free_text("x" * 198 + "\U0001F600" + "y" * 10, 200)
    assert out == "x" * 198 + "…"


def test_bound_observation_cuts_only_error_message():
    obs = {"error_message": AZURE_FILTER, "agent_id": "a" * 300, "model": "m" * 300}
    out = bound_observation(obs)
    assert len(out["error_message"]) == ERROR_MESSAGE_MAX
    assert out["agent_id"] == "a" * 300
    assert out["model"] == "m" * 300
    # The caller's dict is not changed.
    assert obs["error_message"] == AZURE_FILTER


def test_bound_observation_returns_the_same_dict_when_nothing_is_cut():
    for obs in ({"error_message": None}, {"error_message": "RateLimitError: slow down"}, {}):
        assert bound_observation(obs) is obs


def test_transport_posts_the_cut_error_message():
    bodies = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/sdk-token/exchange"):
            return httpx.Response(200, json={"token": "fake-jwt", "expires_in": 900})
        if request.url.path.endswith("/observation"):
            bodies.append(json.loads(request.content))
            return httpx.Response(202, json={"ok": True})
        return httpx.Response(404)

    client = httpx.Client(transport=httpx.MockTransport(handler), timeout=2.0)
    t = ObservationTransport(
        base_url="https://api.test",
        api_key="long-lived-key",
        tenant_id="e82770de-06d6-415c-911f-c0d65cf45190",
        http_client=client,
    )
    try:
        t.post({"status": "error", "input_tokens": 412, "error_message": AZURE_FILTER})
        deadline = time.time() + 2.0
        while not bodies and time.time() < deadline:
            time.sleep(0.01)
    finally:
        t.close()

    assert len(bodies) == 1
    assert bodies[0]["error_message"] == AZURE_FILTER[:199] + "…"
    assert bodies[0]["input_tokens"] == 412
