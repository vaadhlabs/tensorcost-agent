"""Fetches console-published control layer with 30s cache.

Expired cache is served immediately (stale-while-revalidate). Cold fetch is
capped at SDK_LAYER_FETCH_TIMEOUT_S so a hung TensorCost cannot stall the agent.
"""

from __future__ import annotations

import threading
import time
from typing import Callable, Optional

import httpx

from ._layer import (
    ControlLayer,
    SDK_LAYER_CACHE_TTL_MS,
    SDK_LAYER_DECAY_MS,
    SDK_LAYER_FETCH_TIMEOUT_S,
    SDK_LAYER_PATH,
    is_control_layer,
    min_layer,
)
from ._version import SDK_VERSION, sdk_capability_header


class PassthroughSdkLayer:
    """Test/default: assume published layer equals code ceiling (no network)."""

    def effective_layer(self, code_max_layer: ControlLayer) -> ControlLayer:
        return code_max_layer


class SdkLayerClient:
    def __init__(
        self,
        *,
        base_url: str,
        get_token: Callable[[], str],
        team_id: Optional[str] = None,
        http_client: Optional[httpx.Client] = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._get_token = get_token
        self._team_id = team_id.strip() if team_id and team_id.strip() else None
        self._http_client = http_client
        self._cache: Optional[tuple[ControlLayer, float, Optional[dict]]] = None
        # Pretend last success was one decay window ago — cold start must decay to govern.
        self._last_success_at = time.monotonic() - SDK_LAYER_DECAY_MS / 1000
        self._refreshing = False
        self._lock = threading.Lock()

    def effective_layer(self, code_max_layer: ControlLayer) -> ControlLayer:
        published = self._fetch_published_layer(code_max_layer)
        return min_layer(code_max_layer, published)

    def _fetch_published_layer(self, code_max_layer: ControlLayer) -> ControlLayer:
        now = time.monotonic()
        if (
            self._cache is not None
            and (now - self._cache[1]) * 1000 < SDK_LAYER_CACHE_TTL_MS
        ):
            return self._cache[0]

        if (
            self._cache is not None
            and (now - self._last_success_at) * 1000 < SDK_LAYER_DECAY_MS
        ):
            self._schedule_refresh()
            return self._cache[0]

        return self._refresh_now(code_max_layer)

    def get_compliance_snapshot(
        self, code_max_layer: ControlLayer = "observe"
    ) -> Optional[dict]:
        self._fetch_published_layer(code_max_layer)
        return self._cache[2] if self._cache else None

    def _schedule_refresh(self) -> None:
        with self._lock:
            if self._refreshing:
                return
            self._refreshing = True

        def run() -> None:
            try:
                self._refresh_now("observe")
            except Exception:
                pass
            finally:
                with self._lock:
                    self._refreshing = False

        threading.Thread(target=run, daemon=True).start()

    def _refresh_now(self, code_max_layer: ControlLayer) -> ControlLayer:
        now = time.monotonic()
        own_client = self._http_client is None
        client = self._http_client or httpx.Client(
            timeout=httpx.Timeout(SDK_LAYER_FETCH_TIMEOUT_S),
        )
        try:
            token = self._get_token()
            qs = (
                f"?teamId={self._team_id}"
                if self._team_id
                else ""
            )
            resp = client.get(
                f"{self._base_url}{SDK_LAYER_PATH}{qs}",
                headers={
                    "Authorization": f"Bearer {token}",
                    "x-tc-sdk-version": SDK_VERSION,
                    "x-tc-sdk-capabilities": sdk_capability_header(),
                },
            )
            resp.raise_for_status()
            body = resp.json()
            layer_raw = body.get("published_layer") or body.get("layer") or "observe"
            if not isinstance(layer_raw, str) or not is_control_layer(layer_raw):
                layer_raw = "observe"
            layer: ControlLayer = layer_raw  # type: ignore[assignment]
            if body.get("routing_paused"):
                layer = min_layer(layer, "govern")
            compliance = body.get("compliance")
            comp = compliance if isinstance(compliance, dict) else None
            fetched_at = time.monotonic()
            self._cache = (layer, fetched_at, comp)
            self._last_success_at = fetched_at
            return layer
        except Exception:
            if (now - self._last_success_at) * 1000 >= SDK_LAYER_DECAY_MS:
                return "govern"
            return self._cache[0] if self._cache else code_max_layer
        finally:
            if own_client:
                client.close()
