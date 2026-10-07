"""SdkLayerClient — timeout, decay, stale-while-revalidate."""

from __future__ import annotations

import httpx
import pytest

from tensorcost._layer import SDK_LAYER_DECAY_MS
from tensorcost._sdk_layer import SdkLayerClient


def test_cold_fetch_failure_decays_to_govern():
    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code=503)

    client = SdkLayerClient(
        base_url="https://api.tensorcost.com",
        get_token=lambda: "jwt",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    assert client.effective_layer("route") == "govern"


def test_stale_cache_returns_immediately_on_hung_refresh():
    calls = [0]

    def handler(_req: httpx.Request) -> httpx.Response:
        calls[0] += 1
        if calls[0] == 1:
            return httpx.Response(
                status_code=200,
                headers={"Content-Type": "application/json"},
                content=b'{"published_layer":"observe"}',
            )
        raise httpx.ConnectTimeout("hung")

    layer = SdkLayerClient(
        base_url="https://api.tensorcost.com",
        get_token=lambda: "jwt",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    assert layer.effective_layer("route") == "observe"
    layer._cache = ("observe", layer._cache[1] - (SDK_LAYER_DECAY_MS / 1000))
    assert layer.effective_layer("route") == "observe"
