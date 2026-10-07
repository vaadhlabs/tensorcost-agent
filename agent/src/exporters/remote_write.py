"""Prometheus remote_write push (optional)."""

from __future__ import annotations

import logging
import struct
import time
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

try:
    import snappy
except ImportError:
    snappy = None  # type: ignore


def _encode_varint(value: int) -> bytes:
    out = bytearray()
    while True:
        b = value & 0x7F
        value >>= 7
        if value:
            out.append(b | 0x80)
        else:
            out.append(b)
            break
    return bytes(out)


def _encode_label(name: str, value: str) -> bytes:
    name_b = name.encode("utf-8")
    val_b = value.encode("utf-8")
    inner = (
        b"\x0a"
        + _encode_varint(len(name_b))
        + name_b
        + b"\x12"
        + _encode_varint(len(val_b))
        + val_b
    )
    return b"\x0a" + _encode_varint(len(inner)) + inner


def _encode_sample(value: float, ts_ms: int) -> bytes:
    val_bytes = struct.pack("<d", value)
    inner = b"\x09" + val_bytes + b"\x10" + _encode_varint(ts_ms)
    return b"\x12" + _encode_varint(len(inner)) + inner


def _encode_timeseries(name: str, labels: Dict[str, str], value: float, ts_ms: int) -> bytes:
    parts = [_encode_label("__name__", name)]
    for k, v in sorted(labels.items()):
        if k == "__name__":
            continue
        parts.append(_encode_label(k, v))
    labels_blob = b"".join(parts)
    sample = _encode_sample(value, ts_ms)
    series_inner = b"".join(parts) + sample
    return b"\x0a" + _encode_varint(len(series_inner)) + series_inner


def _build_write_request(points: List[Dict[str, Any]]) -> bytes:
    series_chunks: List[bytes] = []
    now_ms = int(time.time() * 1000)
    for p in points:
        ts_ms = int(p.get("ts_unix_ms") or now_ms)
        labels = {
            k: str(v)
            for k, v in (
                ("cloud_provider", p.get("cloud_provider")),
                ("cloud_id", p.get("cloud_id")),
                ("instance_id", p.get("instance_id")),
                ("gpu_index", p.get("gpu_index")),
            )
            if v is not None and v != ""
        }
        for field, prom in (
            ("util_pct", "tensorcost_gpu_utilization_percent"),
            ("mem_pct", "tensorcost_gpu_memory_utilization_percent"),
            ("temp_c", "tensorcost_gpu_temperature_celsius"),
            ("power_w", "tensorcost_gpu_power_watts"),
        ):
            val = p.get(field)
            if val is None:
                continue
            series_chunks.append(_encode_timeseries(prom, labels, float(val), ts_ms))
    body = b"".join(series_chunks)
    return b"\x0a" + _encode_varint(len(body)) + body


class PrometheusRemoteWriteExporter:
    def __init__(self, url: str, timeout: float = 15.0) -> None:
        self._url = url
        self._timeout = timeout

    def push(self, points: List[Dict[str, Any]]) -> None:
        if not points:
            return
        if snappy is None:
            logger.warning("remote_write skipped: install python-snappy")
            return
        payload = snappy.compress(_build_write_request(points))
        headers = {
            "Content-Type": "application/x-protobuf",
            "Content-Encoding": "snappy",
            "X-Prometheus-Remote-Write-Version": "0.1.0",
        }
        resp = requests.post(self._url, data=payload, headers=headers, timeout=self._timeout)
        if resp.status_code >= 300:
            logger.warning("remote_write failed: %s %s", resp.status_code, resp.text[:200])
