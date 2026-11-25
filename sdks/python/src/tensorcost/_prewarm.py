"""Prefetch JWT + published control layer before the first model call."""

from __future__ import annotations

from typing import Optional

from ._config import resolve_config
from ._layer import ControlLayer
from ._runtime import get_shared_runtime, shared_transports
from ._transport import TransportStats


def prewarm(
    *,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    tenant_id: Optional[str] = None,
    max_layer: Optional[ControlLayer] = None,
    applied_mode: bool = False,
) -> None:
    """Best-effort warmup — safe to call at Lambda/container init."""
    config = resolve_config(
        api_key=api_key,
        base_url=base_url,
        tenant_id=tenant_id,
        max_layer=max_layer,
        applied_mode=applied_mode,
    )
    if config.max_layer == "off":
        return

    transport, sdk_layer = get_shared_runtime(config)

    try:
        transport.get_token()
    except Exception:
        pass
    try:
        # Blocking on purpose: prewarm runs at init, off any call path, and
        # a call's own wait for the layer is capped (SdkLayerClient).
        refresh = getattr(sdk_layer, "_refresh_now", None)
        if callable(refresh):
            refresh(config.max_layer)
        else:
            sdk_layer.effective_layer(config.max_layer)
    except Exception:
        pass


def flush_observations(
    *,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    tenant_id: Optional[str] = None,
    max_layer: Optional[ControlLayer] = None,
    timeout_s: float = 5.0,
) -> int:
    """Best-effort drain before a Lambda freeze: sends what is queued and
    what is waiting to retry. If TensorCost is still unreachable it returns
    after one attempt rather than waiting out the timeout; the rest stays
    buffered and keeps retrying. Returns how many observations are still
    undelivered (0 when everything landed)."""
    config = resolve_config(
        api_key=api_key,
        base_url=base_url,
        tenant_id=tenant_id,
        max_layer=max_layer,
    )
    if config.max_layer == "off":
        return 0
    transport, _ = get_shared_runtime(config)
    return transport.flush(timeout_s=timeout_s)


def observation_stats() -> TransportStats:
    """Delivery counters summed over every transport wrap() has created in
    this process: in flight, waiting in memory, written to the disk spool,
    and lost for good (``dropped``)."""
    totals = [t.stats() for t in shared_transports()]
    return TransportStats(
        in_flight=sum(s.in_flight for s in totals),
        buffered=sum(s.buffered for s in totals),
        buffered_bytes=sum(s.buffered_bytes for s in totals),
        dropped=sum(s.dropped for s in totals),
        spooled=sum(s.spooled for s in totals),
        delivered_after_retry=sum(s.delivered_after_retry for s in totals),
        rejected=sum(s.rejected for s in totals),
    )
