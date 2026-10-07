"""Process-wide shared transport + sdk-layer for prewarm() and wrap()."""

from __future__ import annotations

import threading

from ._config import ResolvedConfig
from ._sdk_layer import SdkLayerClient
from ._transport import ObservationTransport

_runtimes: dict[str, tuple[ObservationTransport, SdkLayerClient]] = {}


def _runtime_key(config: ResolvedConfig) -> str:
    return (
        f"{config.base_url}\0{config.tenant_id or ''}\0"
        f"{config.team_id or ''}\0{config.api_key}"
    )


def get_shared_runtime(
    config: ResolvedConfig,
) -> tuple[ObservationTransport, SdkLayerClient]:
    key = _runtime_key(config)
    existing = _runtimes.get(key)
    if existing is not None:
        return existing

    transport = ObservationTransport(
        base_url=config.base_url,
        api_key=config.api_key,
        tenant_id=config.tenant_id,
        fail_open=config.fail_open,
    )
    sdk_layer = SdkLayerClient(
        base_url=config.base_url,
        get_token=transport.get_token,
        team_id=config.team_id,
    )
    pair = (transport, sdk_layer)
    _runtimes[key] = pair
    return pair


def warmup_shared_runtime(config: ResolvedConfig) -> None:
    """Fire-and-forget JWT + sdk-layer warmup on the shared runtime."""
    if config.max_layer == "off":
        return
    transport, sdk_layer = get_shared_runtime(config)

    def _run() -> None:
        try:
            transport.get_token()
        except Exception:
            pass
        try:
            sdk_layer.effective_layer(config.max_layer)
        except Exception:
            pass

    threading.Thread(target=_run, daemon=True).start()
