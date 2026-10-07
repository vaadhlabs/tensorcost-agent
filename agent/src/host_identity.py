"""Synthesize on-prem / neocloud host identity for the sync pipeline."""

from __future__ import annotations

from typing import Any, Dict, List


def build_host_instance(
    *,
    hostname: str,
    cloud_provider: str,
    instance_id: str,
    gpu_count: int = 0,
    gpu_type: str | None = None,
) -> Dict[str, Any]:
    """Build a single host instance row the gpu-service sync path expects."""
    return {
        "instance_id": instance_id or hostname,
        "cloud_id": hostname,
        "instance_type": "host-gpu" if gpu_count else "host",
        "state": "running",
        "status": "running",
        "cloud_provider": cloud_provider or "onprem",
        "tags": {"host_identity": "true"},
        "availability_zone": None,
        "launch_time": None,
        "gpu_count": gpu_count,
        "gpu_type": gpu_type,
        "mig_enabled": False,
        "mig_partitions": [],
    }


def metric_template_from_instance(inst: Dict[str, Any]) -> Dict[str, Any]:
    """Identity fields copied onto per-GPU metric rows."""
    return {
        "instance_id": inst["instance_id"],
        "cloud_id": inst.get("cloud_id", inst["instance_id"]),
        "cloud_provider": inst.get("cloud_provider", "onprem"),
    }


def merge_nvml_metrics(
    template: Dict[str, Any],
    nvml_metrics: List[Dict[str, Any]],
    idle_devices: set[int],
) -> List[Dict[str, Any]]:
    """Fan out NVML samples using an explicit identity template."""
    per_device: List[Dict[str, Any]] = []
    for nvml_m in nvml_metrics:
        gpu_idx = nvml_m["gpu_index"]
        dev_metric: Dict[str, Any] = dict(template)
        dev_metric.update(
            {
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
        per_device.append(dev_metric)
    return per_device
