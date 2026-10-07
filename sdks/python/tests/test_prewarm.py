"""prewarm() shares runtime cache with subsequent get_token()."""

from __future__ import annotations

from unittest.mock import patch

import httpx

from tensorcost._config import resolve_config
from tensorcost._prewarm import prewarm
from tensorcost._runtime import _runtimes, get_shared_runtime


def test_prewarm_warms_shared_runtime_token_cache():
    _runtimes.clear()
    calls = {"exchange": 0}
    real_post = httpx.Client.post

    def counting_post(self, url, **kwargs):
        req = httpx.Request("POST", str(url))
        if str(url).endswith("/sdk-token/exchange"):
            calls["exchange"] += 1
            return httpx.Response(
                200,
                json={"token": "jwt-1", "expires_in": 900},
                request=req,
            )
        return real_post(self, url, **kwargs)

    real_get = httpx.Client.get

    def counting_get(self, url, **kwargs):
        req = httpx.Request("GET", str(url))
        if str(url).endswith("/sdk-layer"):
            return httpx.Response(
                200,
                json={"published_layer": "observe"},
                request=req,
            )
        return real_get(self, url, **kwargs)

    with (
        patch.object(httpx.Client, "post", counting_post),
        patch.object(httpx.Client, "get", counting_get),
    ):
        prewarm(
            api_key="tc_test",
            tenant_id="tenant-1",
            base_url="https://api.tensorcost.com",
        )

        config = resolve_config(
            api_key="tc_test",
            tenant_id="tenant-1",
            base_url="https://api.tensorcost.com",
        )
        transport, _ = get_shared_runtime(config)
        before = calls["exchange"]
        transport.get_token()
        assert calls["exchange"] == before
