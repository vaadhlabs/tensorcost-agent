"""Fetches console-published control layer with 30s cache.

Expired cache is served immediately (stale-while-revalidate). A cold fetch,
token exchange included, costs a call at most SDK_LAYER_FETCH_TIMEOUT_S, so a
hung TensorCost cannot stall the agent.
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

    def resolve(self, code_max_layer: ControlLayer) -> tuple[ControlLayer, bool]:
        return code_max_layer, False


def resolve_layer(layer_client: object, code_max_layer: ControlLayer) -> tuple[ControlLayer, bool]:
    """``(effective layer, decayed)`` from any layer client, old fakes included."""
    resolve = getattr(layer_client, "resolve", None)
    if callable(resolve):
        out = resolve(code_max_layer)
        if isinstance(out, tuple) and len(out) == 2:
            return out  # type: ignore[return-value]
    return layer_client.effective_layer(code_max_layer), False  # type: ignore[attr-defined]


class SdkLayerClient:
    """The published layer, cached 30 s, never more than the cap on a call.

    A refresh (token exchange plus the GET) runs on its own thread, one at a
    time. A call that needs it waits at most SDK_LAYER_FETCH_TIMEOUT_S, and
    no longer than that cap after the refresh in flight started, so while
    TensorCost is unreachable a call pays the cap at most once per refresh
    attempt rather than once per call. A refresh that outlives its caller
    keeps going and fills the cache for the next call.
    """

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
        self._cache: Optional[tuple] = None
        # Pretend last success was one decay window ago — cold start must decay to govern.
        self._last_success_at = time.monotonic() - SDK_LAYER_DECAY_MS / 1000
        self._lock = threading.Lock()
        # (done event, monotonic start) of the refresh in flight, if any.
        self._inflight: Optional[tuple[threading.Event, float]] = None

    @property
    def _refreshing(self) -> bool:
        with self._lock:
            return self._inflight is not None

    def effective_layer(self, code_max_layer: ControlLayer) -> ControlLayer:
        return self.resolve(code_max_layer)[0]

    def resolve(self, code_max_layer: ControlLayer) -> tuple[ControlLayer, bool]:
        """``(effective layer, decayed)``.

        ``decayed`` is True when the published layer could not be had and the
        layer fell back to govern: a route client is then going around the
        proxy because TensorCost is unreachable, not because anyone chose it.
        """
        published, decayed = self._fetch_published_layer(code_max_layer)
        return min_layer(code_max_layer, published), decayed

    def _fetch_published_layer(self, code_max_layer: ControlLayer) -> tuple[ControlLayer, bool]:
        now = time.monotonic()
        cache = self._cache
        if cache is not None and (now - cache[1]) * 1000 < SDK_LAYER_CACHE_TTL_MS:
            return cache[0], False

        if cache is not None and (now - self._last_success_at) * 1000 < SDK_LAYER_DECAY_MS:
            self._schedule_refresh()
            return cache[0], False

        done, started = self._start_refresh()
        wait = min(SDK_LAYER_FETCH_TIMEOUT_S, started + SDK_LAYER_FETCH_TIMEOUT_S - now)
        if wait > 0:
            done.wait(wait)
        cache = self._cache
        if cache is not None and (time.monotonic() - self._last_success_at) * 1000 < SDK_LAYER_DECAY_MS:
            return cache[0], False
        return "govern", True

    def get_compliance_snapshot(
        self, code_max_layer: ControlLayer = "observe"
    ) -> Optional[dict]:
        self._fetch_published_layer(code_max_layer)
        cache = self._cache
        return cache[2] if cache is not None and len(cache) > 2 else None

    def _schedule_refresh(self) -> None:
        self._start_refresh()

    def _start_refresh(self) -> tuple[threading.Event, float]:
        with self._lock:
            if self._inflight is not None:
                return self._inflight
            done = threading.Event()
            self._inflight = (done, time.monotonic())
            inflight = self._inflight

        def run() -> None:
            try:
                self._refresh_now("observe")
            except Exception:
                pass
            finally:
                with self._lock:
                    self._inflight = None
                done.set()

        threading.Thread(target=run, name="tensorcost-sdk-layer", daemon=True).start()
        return inflight

    def _refresh_now(self, code_max_layer: ControlLayer) -> ControlLayer:
        """Fetch synchronously, unbounded by the cap (prewarm and the refresh thread)."""
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
