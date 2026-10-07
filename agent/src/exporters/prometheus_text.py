"""Prometheus text exposition on a local HTTP port."""

from __future__ import annotations

import logging
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _metric_lines(points: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for p in points:
        labels = []
        for key in ("cloud_provider", "cloud_id", "instance_id", "gpu_index"):
            val = p.get(key)
            if val is None or val == "":
                continue
            labels.append(f'{key}="{val}"')
        label_str = "{" + ",".join(labels) + "}" if labels else ""
        ts_ms = p.get("ts_unix_ms") or p.get("timestamp")
        suffix = f" {int(ts_ms)}" if ts_ms else ""
        for field, prom in (
            ("util_pct", "gpu_utilization_percent"),
            ("mem_pct", "gpu_memory_utilization_percent"),
            ("temp_c", "gpu_temperature_celsius"),
            ("power_w", "gpu_power_watts"),
            ("memory_used_mb", "gpu_memory_used_megabytes"),
            ("memory_total_mb", "gpu_memory_total_megabytes"),
        ):
            val = p.get(field)
            if val is None:
                continue
            try:
                num = float(val)
            except (TypeError, ValueError):
                continue
            lines.append(f"tensorcost_{prom}{label_str} {num}{suffix}")
    return "\n".join(lines) + ("\n" if lines else "")


class _MetricsHandler(BaseHTTPRequestHandler):
    body: str = "# no samples yet\n"

    def do_GET(self) -> None:
        if self.path not in ("/metrics", "/"):
            self.send_response(404)
            self.end_headers()
            return
        payload = self.body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, fmt: str, *args: Any) -> None:
        logger.debug("prometheus_text %s", fmt % args)


class PrometheusTextExporter:
    def __init__(self, host: str, port: int) -> None:
        self._host = host
        self._port = port
        self._server: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._latest: List[Dict[str, Any]] = []

    def start(self) -> None:
        if self._server:
            return
        self._server = HTTPServer((self._host, self._port), _MetricsHandler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        logger.info("Prometheus text metrics on http://%s:%s/metrics", self._host, self._port)

    def stop(self) -> None:
        if self._server:
            self._server.shutdown()
            self._server = None

    def update(self, points: List[Dict[str, Any]]) -> None:
        with self._lock:
            self._latest = list(points)
            _MetricsHandler.body = _metric_lines(self._latest) or "# no samples yet\n"
