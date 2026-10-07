"""Prefetch JWT + published control layer before the first model call."""

from __future__ import annotations

from typing import Optional

from ._config import resolve_config
from ._layer import ControlLayer
from ._runtime import get_shared_runtime


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
) -> None:
    """Best-effort drain of queued observations before Lambda freeze."""
    config = resolve_config(
        api_key=api_key,
        base_url=base_url,
        tenant_id=tenant_id,
        max_layer=max_layer,
    )
    if config.max_layer == "off":
        return
    transport, _ = get_shared_runtime(config)
    transport.flush(timeout_s=timeout_s)
