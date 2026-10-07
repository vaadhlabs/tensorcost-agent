"""Tests for standalone metric exporters."""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from exporters.focus_usage import FocusUsageExporter
from exporters.orchestrator import build_metric_exporter
from exporters.prometheus_text import PrometheusTextExporter, _metric_lines
from exporters.remote_write import PrometheusRemoteWriteExporter


def test_metric_lines_format():
    body = _metric_lines(
        [
            {
                "util_pct": 42.5,
                "gpu_index": 0,
                "cloud_provider": "aws",
                "instance_id": "i-abc",
                "ts_unix_ms": 1_700_000_000_000,
            }
        ]
    )
    assert "tensorcost_gpu_utilization_percent" in body
    assert 'cloud_provider="aws"' in body
    assert "42.5" in body


def test_prometheus_exporter_update():
    exp = PrometheusTextExporter("127.0.0.1", 19092)
    exp.update([{"util_pct": 1.0, "gpu_index": 0}])
    assert exp._latest[0]["util_pct"] == 1.0


def test_focus_usage_file(tmp_path):
    path = tmp_path / "usage.csv"
    exp = FocusUsageExporter(str(path))
    exp.record(
        [{"util_pct": 50.0, "instance_id": "i-1", "gpu_index": 0}],
        region="us-east-1",
        provider="aws",
    )
    text = path.read_text()
    assert "GPU-Hours" in text
    assert "aws" in text


def test_build_metric_exporter_defaults():
    with patch.dict(
        os.environ,
        {
            "PROMETHEUS_TEXT_ENABLED": "true",
            "PROMETHEUS_REMOTE_WRITE_URL": "",
            "FOCUS_USAGE_FILE": "",
            "OTEL_METRICS_ENABLED": "",
            "OTEL_EXPORTER_OTLP_ENDPOINT": "",
        },
        clear=False,
    ):
        orch = build_metric_exporter(region="r", provider="aws")
    assert orch is not None
    orch.start()
    orch.on_metrics([{"util_pct": 3.0, "gpu_index": 0, "cloud_provider": "aws"}])
    orch.stop()


def test_build_metric_exporter_disabled():
    with patch.dict(
        os.environ,
        {
            "PROMETHEUS_TEXT_ENABLED": "false",
            "PROMETHEUS_REMOTE_WRITE_URL": "",
            "FOCUS_USAGE_FILE": "",
            "OTEL_METRICS_ENABLED": "",
        },
        clear=False,
    ):
        assert build_metric_exporter() is None


@patch("exporters.remote_write.requests.post")
def test_remote_write_push(mock_post):
    mock_post.return_value = MagicMock(status_code=204, text="")
    pytest.importorskip("snappy")
    rw = PrometheusRemoteWriteExporter("http://localhost:9090/api/v1/write")
    rw.push([{"util_pct": 10.0, "gpu_index": 0, "ts_unix_ms": 1_700_000_000_000}])
    mock_post.assert_called_once()
