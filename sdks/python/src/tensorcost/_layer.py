"""Control-layer ladder — mirrors packages/contracts control-layer."""

from __future__ import annotations

from typing import Literal, Optional

ControlLayer = Literal["off", "observe", "govern", "steer", "route"]

CONTROL_LAYERS: tuple[ControlLayer, ...] = (
    "off",
    "observe",
    "govern",
    "steer",
    "route",
)

_LAYER_RANK: dict[ControlLayer, int] = {
    "off": 0,
    "observe": 1,
    "govern": 2,
    "steer": 3,
    "route": 4,
}

PROXY_OPERATION_PATHS: dict[str, str] = {
    "chat.completions": "/api/inference-proxy/v1/chat/completions",
    "messages": "/api/inference-proxy/v1/messages",
    "completions": "/api/inference-proxy/v1/completions",
    "embeddings": "/api/inference-proxy/v1/embeddings",
    "responses": "/api/inference-proxy/v1/responses",
}

ADMIT_PATH = "/api/inference-proxy/v1/admit"
SDK_LAYER_PATH = "/api/inference-proxy/v1/sdk-layer"
SDK_LAYER_CACHE_TTL_MS = 30_000
SDK_LAYER_DECAY_MS = 5 * 60_000
SDK_LAYER_FETCH_TIMEOUT_S = 0.25
ADMIT_TIMEOUT_S = 0.5
PROXY_HEADERS_TIMEOUT_S = 2.0


def layer_rank(layer: ControlLayer) -> int:
    return _LAYER_RANK[layer]


def min_layer(a: ControlLayer, b: ControlLayer) -> ControlLayer:
    return a if layer_rank(a) <= layer_rank(b) else b


def is_control_layer(value: str) -> bool:
    return value in CONTROL_LAYERS


def needs_proxy_url(layer: ControlLayer) -> bool:
    return layer_rank(layer) >= layer_rank("steer")


def needs_admit(layer: ControlLayer) -> bool:
    return layer_rank(layer) >= layer_rank("govern")


def resolve_max_layer(
    *,
    max_layer: Optional[ControlLayer] = None,
    applied_mode: bool = False,
) -> ControlLayer:
    if max_layer is not None:
        return max_layer
    if applied_mode:
        import logging

        logging.getLogger("tensorcost").warning(
            "tensorcost: applied_mode is deprecated; use max_layer='route' instead",
        )
        return "route"
    return "observe"


def proxy_path_for_operation(operation: str) -> Optional[str]:
    return PROXY_OPERATION_PATHS.get(operation)


def code_layer_for_install(
    max_layer: ControlLayer,
    applied_mode: bool,
) -> ControlLayer:
    """Map legacy applied_mode=True to route when install() is called directly."""
    if applied_mode and max_layer == "observe":
        return "route"
    return max_layer
