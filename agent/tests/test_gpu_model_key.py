"""Tests for gpu_type detection and catalog model_key resolution."""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gpu_model_key import (  # noqa: E402
    HARDWARE_CATALOG_KEYS,
    detect_gpu_family,
    is_mig_capable,
    normalize_gpu_type,
    resolve_gpu_model_key,
)

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures", "gpu_detection_strings.json")


def _load_fixtures():
    with open(FIXTURES, encoding="utf-8") as fh:
        return json.load(fh)


@pytest.mark.unit
class TestGpuDetectionFixtures:
    @pytest.fixture(scope="class")
    def fixtures(self):
        return _load_fixtures()

    def test_nvml_detection_strings(self, fixtures):
        for row in fixtures["nvml_names"]:
            assert detect_gpu_family(row["input"]) == row["family"]
            assert normalize_gpu_type(row["input"]) == row["normalized"]
            key, miss = resolve_gpu_model_key(row["input"])
            assert key == row.get("model_key")
            if row.get("miss"):
                assert miss == row["miss"]
            else:
                assert miss is None
                assert key in HARDWARE_CATALOG_KEYS
            assert is_mig_capable(row["input"]) == row["mig_capable"]

    def test_h200_resolves_to_h200_sxm(self, fixtures):
        h200_rows = [r for r in fixtures["nvml_names"] if r["family"] == "H200"]
        assert h200_rows, "fixture must include H200 NVML strings"
        for row in h200_rows:
            key, miss = resolve_gpu_model_key(row["input"])
            assert key == "h200-sxm"
            assert miss is None

    def test_aws_instance_type_fixtures(self, fixtures):
        from monitors.aws_monitor import AWSMonitor

        monitor = AWSMonitor.__new__(AWSMonitor)
        for row in fixtures["aws_instance_types"]:
            assert monitor._get_gpu_type(row["input"]) == row["gpu_type"]
            assert monitor._get_gpu_count(row["input"]) == row["gpu_count"]


@pytest.mark.unit
class TestResolveGpuModelKey:
    def test_bare_h100_is_ambiguous(self):
        key, miss = resolve_gpu_model_key("NVIDIA H100")
        assert key is None
        assert miss == "ambiguous_sku"

    def test_blackwell_resolves_to_catalog_keys(self):
        key, miss = resolve_gpu_model_key("NVIDIA B200")
        assert key == "b200-sxm"
        assert miss is None
        key, miss = resolve_gpu_model_key("NVIDIA GB200")
        assert key == "gb200"
        assert miss is None
        assert is_mig_capable("NVIDIA B200") is True
        assert is_mig_capable("NVIDIA GB200") is True

    def test_rtx_pro_blackwell_is_not_b200(self):
        """Bare 'Blackwell' must not alias workstation SKUs onto b200-sxm."""
        assert detect_gpu_family("NVIDIA RTX PRO 6000 Blackwell") is None
        key, miss = resolve_gpu_model_key("NVIDIA RTX PRO 6000 Blackwell")
        assert key is None
        assert miss == "unknown_gpu_type"
        assert is_mig_capable("NVIDIA RTX PRO 6000 Blackwell") is False
