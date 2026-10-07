"""
Host GPU Monitor — standalone NVML telemetry for bare-metal / VM hosts.

When the agent runs directly on a GPU host (EC2 p4d, DGX, etc.) without
Kubernetes, NVML samples local devices and this monitor synthesizes the
instance + metric DTOs the sync pipeline expects.

Gated by NVML_ENABLED=true and K8S_ENABLED=false (or HOST_GPU_TELEMETRY_ENABLED=true).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from src import gpu_model_key

logger = logging.getLogger(__name__)


class HostGpuMonitor:
    """NVML-only monitor for non-K8s GPU hosts."""

    def __init__(
        self,
        *,
        hostname: str,
        cloud_provider: str,
        instance_id: str,
        nvml_sampler: Any,
    ) -> None:
        self.hostname = hostname
        self.cloud_provider = cloud_provider or "onprem"
        self.instance_id = instance_id or hostname
        self.nvml_sampler = nvml_sampler

    def get_gpu_instances(self) -> List[Dict[str, Any]]:
        if not self.nvml_sampler or not self.nvml_sampler.is_available():
            return []

        samples = self.nvml_sampler.get_latest_metrics()
        if not samples:
            return []

        gpu_count = len(samples)
        raw_gpu_name: str | None = None
        gpu_type: str | None = None
        pricing_key: str | None = None
        try:
            import pynvml

            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            raw_gpu_name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(raw_gpu_name, bytes):
                raw_gpu_name = raw_gpu_name.decode("utf-8", errors="replace")
            gpu_type = gpu_model_key.normalize_gpu_type(raw_gpu_name)
            pricing_key, _ = gpu_model_key.resolve_gpu_model_key(raw_gpu_name)
        except Exception:
            pass

        tags: Dict[str, str] = {"host_gpu": "true"}
        if raw_gpu_name and raw_gpu_name != gpu_type:
            tags["gpu_name_raw"] = raw_gpu_name

        row: Dict[str, Any] = {
            "instance_id": self.instance_id,
            "cloud_id": self.hostname,
            "instance_type": "host-gpu",
            "state": "running",
            "status": "running",
            "cloud_provider": self.cloud_provider,
            "tags": tags,
            "availability_zone": None,
            "launch_time": None,
            "gpu_count": gpu_count,
            "gpu_type": gpu_type,
            **self._mig_wire_fields(),
        }
        if pricing_key:
            row["pricing_key"] = pricing_key
        return [row]

    def _mig_wire_fields(self) -> Dict[str, Any]:
        """Read-only MIG inventory for sync InstanceUpdate.mig_partitions."""
        out: Dict[str, Any] = {"mig_enabled": False, "mig_partitions": []}
        if not self.nvml_sampler or not self.nvml_sampler.is_available():
            return out
        try:
            import pynvml
            from src import mig_inspector

            device_count = pynvml.nvmlDeviceGetCount()
            partitions: List[Dict[str, Any]] = []
            mig_enabled = False
            for gpu_index in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
                snap = mig_inspector.inspect_device(handle, gpu_index)
                if not snap.get("mig_supported", True):
                    continue
                if snap.get("mig_enabled"):
                    mig_enabled = True
                    gpu_family = snap.get("gpu_family")
                    for entry in snap.get("partitions") or []:
                        partitions.append(
                            mig_inspector.partition_to_wire(
                                entry, gpu_index, gpu_family=gpu_family
                            )
                        )
            out["mig_enabled"] = mig_enabled
            out["mig_partitions"] = partitions
        except Exception as exc:
            logger.debug("MIG wire snapshot failed: %s", exc)
        return out

    def get_gpu_utilization(
        self, instances: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        if not self.nvml_sampler or not self.nvml_sampler.is_available():
            return []

        if not instances:
            return []

        inst = instances[0]
        idle_devices = set(self.nvml_sampler.get_idle_devices())
        metrics: List[Dict[str, Any]] = []

        for nvml_m in self.nvml_sampler.get_latest_metrics():
            gpu_idx = nvml_m["gpu_index"]
            metrics.append(
                {
                    "instance_id": inst["instance_id"],
                    "cloud_id": inst["cloud_id"],
                    "cloud_provider": inst["cloud_provider"],
                    "gpu_index": gpu_idx,
                    "gpu_utilization": nvml_m["gpu_utilization"],
                    "memory_utilization": nvml_m["memory_utilization"],
                    "temperature_c": nvml_m.get("temperature_c"),
                    "power_usage_w": nvml_m.get("power_usage_w"),
                    "mem_bw_pct": nvml_m.get("mem_bw_pct"),
                    "ecc_errors_total": nvml_m.get("ecc_errors_uncorrected"),
                    "memory_used_mb": nvml_m.get("memory_used_mb"),
                    "memory_total_mb": nvml_m.get("memory_total_mb"),
                    "clock_mhz": nvml_m.get("clock_mhz"),
                    "memory_clock_mhz": nvml_m.get("memory_clock_mhz"),
                    "is_idle": gpu_idx in idle_devices,
                    "gpu_utilization_local": nvml_m["gpu_utilization"],
                    "memory_utilization_local": nvml_m["memory_utilization"],
                }
            )

        return metrics
