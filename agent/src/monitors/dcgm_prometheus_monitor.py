"""DCGM / Prometheus GPU telemetry — agentless attach on the live sync path."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import requests

from host_identity import build_host_instance, metric_template_from_instance
from prometheus_parse import parse_prometheus_metrics

logger = logging.getLogger(__name__)

DCGM_UTIL = "DCGM_FI_DEV_GPU_UTIL"
DCGM_MEM_USED = "DCGM_FI_DEV_FB_USED"
DCGM_MEM_FREE = "DCGM_FI_DEV_FB_FREE"
DCGM_TEMP = "DCGM_FI_DEV_GPU_TEMP"
DCGM_POWER = "DCGM_FI_DEV_POWER_USAGE"


class DcgmPrometheusMonitor:
    """Scrape an existing DCGM Exporter or GPU Prometheus endpoint."""

    def __init__(
        self,
        *,
        hostname: str,
        cloud_provider: str,
        instance_id: str,
        metrics_url: Optional[str] = None,
    ) -> None:
        self.hostname = hostname
        self.cloud_provider = cloud_provider or "onprem"
        self.instance_id = instance_id or hostname
        self.metrics_url = (
            metrics_url
            or os.getenv("DCGM_EXPORTER_URL")
            or os.getenv("GPU_PROMETHEUS_URL")
        )

    def get_gpu_instances(self) -> List[Dict[str, Any]]:
        if not self.metrics_url:
            return []
        parsed = self._fetch_metrics()
        if not parsed:
            return []
        gpu_indices = self._gpu_indices(parsed)
        gpu_count = len(gpu_indices) if gpu_indices else 1
        return [
            build_host_instance(
                hostname=self.hostname,
                cloud_provider=self.cloud_provider,
                instance_id=self.instance_id,
                gpu_count=gpu_count,
            )
        ]

    def get_gpu_utilization(
        self, instances: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        if not self.metrics_url or not instances:
            return []
        parsed = self._fetch_metrics()
        if not parsed:
            return []

        template = metric_template_from_instance(instances[0])
        gpu_indices = self._gpu_indices(parsed) or [0]
        metrics: List[Dict[str, Any]] = []

        for gpu_idx in gpu_indices:
            util = self._metric_for_gpu(parsed, DCGM_UTIL, gpu_idx)
            mem_used = self._metric_for_gpu(parsed, DCGM_MEM_USED, gpu_idx)
            mem_free = self._metric_for_gpu(parsed, DCGM_MEM_FREE, gpu_idx)
            mem_total = (mem_used or 0) + (mem_free or 0)
            mem_util = (
                (mem_used / mem_total * 100.0) if mem_total > 0 and mem_used is not None else None
            )
            metrics.append(
                {
                    **template,
                    "gpu_index": gpu_idx,
                    "gpu_utilization": util if util is not None else 0.0,
                    "memory_utilization": mem_util if mem_util is not None else 0.0,
                    "temperature_c": self._metric_for_gpu(parsed, DCGM_TEMP, gpu_idx),
                    "power_usage_w": self._metric_for_gpu(parsed, DCGM_POWER, gpu_idx),
                    "memory_used_mb": (mem_used / (1024 * 1024)) if mem_used else None,
                    "memory_total_mb": (mem_total / (1024 * 1024)) if mem_total else None,
                }
            )
        return metrics

    def _fetch_metrics(self) -> Dict[str, float]:
        if not self.metrics_url:
            return {}
        try:
            resp = requests.get(self.metrics_url, timeout=5)
            resp.raise_for_status()
            return parse_prometheus_metrics(resp.text)
        except Exception as exc:
            logger.warning("DCGM/Prometheus scrape failed: %s", exc)
            return {}

    @staticmethod
    def _gpu_indices(parsed: Dict[str, float]) -> List[int]:
        indices: set[int] = set()
        for key in parsed:
            if "gpu=" in key and "{" in key:
                label_blob = key.split("{", 1)[1].rsplit("}", 1)[0]
                for part in label_blob.split(","):
                    if part.strip().startswith("gpu="):
                        raw = part.split("=", 1)[1].strip().strip('"')
                        try:
                            indices.add(int(raw))
                        except ValueError:
                            pass
            elif key == DCGM_UTIL or key.startswith(f"{DCGM_UTIL} "):
                indices.add(0)
        if not indices and any(k.startswith("DCGM_FI_") for k in parsed):
            indices.add(0)
        return sorted(indices)

    @staticmethod
    def _metric_for_gpu(
        parsed: Dict[str, float], prefix: str, gpu_index: int
    ) -> Optional[float]:
        candidates = [
            f'{prefix}{{gpu="{gpu_index}"}}',
            f"{prefix}{{gpu={gpu_index}}}",
        ]
        for key in candidates:
            if key in parsed:
                return parsed[key]
        for key, value in parsed.items():
            if not key.startswith(prefix + "{"):
                continue
            if f'gpu="{gpu_index}"' in key or f"gpu={gpu_index}" in key:
                return value
        if gpu_index == 0:
            return parsed.get(prefix)
        return None
