"""Tests for HostGpuMonitor — standalone NVML path (gpu-agents #6)."""

import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from monitors.host_gpu_monitor import HostGpuMonitor


def _sampler(samples=None, idle=None):
    sampler = MagicMock()
    sampler.is_available.return_value = True
    sampler.get_latest_metrics.return_value = samples or [
        {
            "gpu_index": 0,
            "gpu_utilization": 42.0,
            "memory_utilization": 55.0,
            "temperature_c": 61.0,
            "power_usage_w": 180.0,
            "mem_bw_pct": 12.0,
            "ecc_errors_uncorrected": 0,
            "memory_used_mb": 8192.0,
            "memory_total_mb": 16384.0,
        }
    ]
    sampler.get_idle_devices.return_value = idle or []
    return sampler


def test_get_gpu_instances_returns_host_row():
    monitor = HostGpuMonitor(
        hostname="gpu-host-1",
        cloud_provider="aws",
        instance_id="inst-abc",
        nvml_sampler=_sampler(),
    )
    instances = monitor.get_gpu_instances()
    assert len(instances) == 1
    assert instances[0]["instance_id"] == "inst-abc"
    assert instances[0]["cloud_id"] == "gpu-host-1"
    assert instances[0]["gpu_count"] == 1
    assert instances[0]["cloud_provider"] == "aws"


def test_get_gpu_utilization_fans_out_per_device():
    monitor = HostGpuMonitor(
        hostname="gpu-host-1",
        cloud_provider="onprem",
        instance_id="inst-abc",
        nvml_sampler=_sampler(idle=[1]),
    )
    instances = monitor.get_gpu_instances()
    metrics = monitor.get_gpu_utilization(instances)
    assert len(metrics) == 1
    assert metrics[0]["gpu_index"] == 0
    assert metrics[0]["gpu_utilization"] == 42.0
    assert metrics[0]["is_idle"] is False


def test_get_gpu_instances_normalizes_h200_and_pricing_key():
    sampler = _sampler()
    monitor = HostGpuMonitor(
        hostname="gpu-host-h200",
        cloud_provider="aws",
        instance_id="inst-h200",
        nvml_sampler=sampler,
    )
    fake_pynvml = MagicMock()
    fake_pynvml.nvmlDeviceGetHandleByIndex.return_value = MagicMock()
    fake_pynvml.nvmlDeviceGetName.return_value = b"NVIDIA H200 141GB HBM3e"
    fake_pynvml.nvmlDeviceGetCount.return_value = 1
    with patch.dict(sys.modules, {"pynvml": fake_pynvml}), \
         patch("src.mig_inspector.inspect_device", return_value={
             "mig_supported": True,
             "mig_enabled": False,
             "gpu_family": "H200",
             "partitions": [],
         }):
        instances = monitor.get_gpu_instances()
    assert instances[0]["gpu_type"] == "H200"
    assert instances[0]["pricing_key"] == "h200-sxm"
    assert instances[0]["tags"]["gpu_name_raw"] == "NVIDIA H200 141GB HBM3e"


def test_returns_empty_when_nvml_unavailable():
    sampler = MagicMock()
    sampler.is_available.return_value = False
    monitor = HostGpuMonitor(
        hostname="h",
        cloud_provider="aws",
        instance_id="i",
        nvml_sampler=sampler,
    )
    assert monitor.get_gpu_instances() == []
    assert monitor.get_gpu_utilization([]) == []
