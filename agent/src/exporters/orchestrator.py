"""Wire standalone exporters from environment variables."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from exporters.focus_usage import FocusUsageExporter
from exporters.otlp_metrics import OtlpMetricsExporter, otlp_enabled
from exporters.prometheus_text import PrometheusTextExporter
from exporters.remote_write import PrometheusRemoteWriteExporter

logger = logging.getLogger(__name__)


class MetricExporterOrchestrator:
    def __init__(
        self,
        prometheus: Optional[PrometheusTextExporter],
        remote_write: Optional[PrometheusRemoteWriteExporter],
        otlp: Optional[OtlpMetricsExporter],
        focus: Optional[FocusUsageExporter],
        region: str,
        provider: str,
    ) -> None:
        self._prometheus = prometheus
        self._remote_write = remote_write
        self._otlp = otlp
        self._focus = focus
        self._region = region
        self._provider = provider

    def start(self) -> None:
        if self._prometheus:
            self._prometheus.start()

    def stop(self) -> None:
        if self._prometheus:
            self._prometheus.stop()

    def on_metrics(self, points: List[Dict[str, Any]]) -> None:
        if not points:
            return
        if self._prometheus:
            self._prometheus.update(points)
        if self._remote_write:
            try:
                self._remote_write.push(points)
            except Exception as exc:  # noqa: BLE001
                logger.warning("remote_write error: %s", exc)
        if self._otlp:
            try:
                self._otlp.export(points)
            except Exception as exc:  # noqa: BLE001
                logger.warning("OTLP metrics error: %s", exc)
        if self._focus:
            self._focus.record(points, self._region, self._provider)


def build_metric_exporter(
    region: str = "",
    provider: str = "",
) -> Optional[MetricExporterOrchestrator]:
    prom_enabled = os.getenv("PROMETHEUS_TEXT_ENABLED", "true").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    prom_host = os.getenv("PROMETHEUS_TEXT_HOST", "0.0.0.0")
    prom_port = int(os.getenv("PROMETHEUS_TEXT_PORT", "9090"))

    rw_url = os.getenv("PROMETHEUS_REMOTE_WRITE_URL", "").strip()
    otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()
    focus_path = os.getenv("FOCUS_USAGE_FILE", "").strip()

    if not any(
        [
            prom_enabled,
            rw_url,
            (otlp_enabled() and otlp_endpoint),
            focus_path,
        ]
    ):
        return None

    prometheus = (
        PrometheusTextExporter(prom_host, prom_port) if prom_enabled else None
    )
    remote = PrometheusRemoteWriteExporter(rw_url) if rw_url else None
    otlp = (
        OtlpMetricsExporter(otlp_endpoint)
        if otlp_enabled() and otlp_endpoint
        else None
    )
    focus = FocusUsageExporter(focus_path) if focus_path else None

    return MetricExporterOrchestrator(
        prometheus=prometheus,
        remote_write=remote,
        otlp=otlp,
        focus=focus,
        region=region,
        provider=provider,
    )
