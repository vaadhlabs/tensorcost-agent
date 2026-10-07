"""Central SDK identity — wire headers and observation sdk_version."""

from __future__ import annotations

SDK_VERSION = "tensorcost-python/1.3.0"

# Advertise only capabilities this build implements (non-streaming v1).
SDK_CAPABILITIES: tuple[str, ...] = ("refusal-v1",)


def sdk_capability_header() -> str:
    return ",".join(SDK_CAPABILITIES)
