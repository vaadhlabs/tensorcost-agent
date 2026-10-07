"""Tests for host identity + DCGM prometheus monitor."""

import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from host_identity import build_host_instance, merge_nvml_metrics, metric_template_from_instance
from monitors.dcgm_prometheus_monitor import DcgmPrometheusMonitor
from monitors.host_identity_monitor import HostIdentityMonitor


def test_build_host_instance_onprem():
    inst = build_host_instance(
        hostname="neocloud-box",
        cloud_provider="onprem",
        instance_id="neocloud-box",
    )
    assert inst["cloud_provider"] == "onprem"
    assert inst["instance_id"] == "neocloud-box"


def test_merge_nvml_uses_template_without_cloud_metrics():
    template = metric_template_from_instance(
        build_host_instance(
            hostname="h1",
            cloud_provider="onprem",
            instance_id="h1",
        )
    )
    nvml = [
        {
            "gpu_index": 0,
            "gpu_utilization": 10.0,
            "memory_utilization": 20.0,
            "temperature_c": 50.0,
            "power_usage_w": None,
        }
    ]
    out = merge_nvml_metrics(template, nvml, set())
    assert out[0]["cloud_provider"] == "onprem"
    assert out[0]["gpu_utilization"] == 10.0


def test_host_identity_monitor_returns_instance_without_metrics():
    mon = HostIdentityMonitor(
        hostname="bare",
        cloud_provider="onprem",
        instance_id="bare",
    )
    inst = mon.get_gpu_instances()
    assert len(inst) == 1
    assert mon.get_gpu_utilization(inst) == []


@patch("monitors.dcgm_prometheus_monitor.requests.get")
def test_dcgm_monitor_parses_labeled_metrics(mock_get):
    mock_get.return_value = MagicMock(
        status_code=200,
        text=(
            'DCGM_FI_DEV_GPU_UTIL{gpu="0"} 55.0\n'
            'DCGM_FI_DEV_GPU_UTIL{gpu="1"} 33.0\n'
            'DCGM_FI_DEV_GPU_TEMP{gpu="0"} 62.0\n'
        ),
    )
    mock_get.return_value.raise_for_status = MagicMock()
    mon = DcgmPrometheusMonitor(
        hostname="cluster-node",
        cloud_provider="onprem",
        instance_id="cluster-node",
        metrics_url="http://localhost:9400/metrics",
    )
    instances = mon.get_gpu_instances()
    assert len(instances) == 1
    metrics = mon.get_gpu_utilization(instances)
    assert len(metrics) == 2
    assert metrics[0]["gpu_utilization"] == 55.0
    assert metrics[1]["gpu_utilization"] == 33.0


@patch("monitors.dcgm_prometheus_monitor.requests.get")
def test_dcgm_monitor_scrapes_unlabeled_prometheus(mock_get):
    mock_get.return_value = MagicMock(
        status_code=200,
        text="DCGM_FI_DEV_GPU_UTIL 55.0\nDCGM_FI_DEV_GPU_TEMP 62.0\n",
    )
    mock_get.return_value.raise_for_status = MagicMock()
    mon = DcgmPrometheusMonitor(
        hostname="cluster-node",
        cloud_provider="onprem",
        instance_id="cluster-node",
        metrics_url="http://localhost:9400/metrics",
    )
    instances = mon.get_gpu_instances()
    assert len(instances) == 1
    metrics = mon.get_gpu_utilization(instances)
    assert metrics[0]["gpu_utilization"] == 55.0
    assert metrics[0]["cloud_provider"] == "onprem"
