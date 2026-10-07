"""Tests for MIG inspector (spec 4C)."""

import json
import os
from unittest.mock import MagicMock, patch

import pytest

import src.mig_inspector as mig_inspector

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures", "gpu_detection_strings.json")


@pytest.mark.unit
class TestGetMigMode:
    def test_returns_false_when_nvml_unavailable(self):
        with patch.object(mig_inspector, '_NVML_AVAILABLE', False):
            result = mig_inspector.get_mig_mode(MagicMock())
        assert result == {'current_enabled': False, 'pending_enabled': False}

    def test_returns_enabled_when_nvml_reports_1(self):
        fake_pynvml = MagicMock()
        fake_pynvml.nvmlDeviceGetMigMode.return_value = (1, 1)
        fake_pynvml.nvmlDeviceGetName.return_value = b"NVIDIA A100-SXM4-80GB"
        with patch.object(mig_inspector, '_NVML_AVAILABLE', True), \
             patch.object(mig_inspector, 'pynvml', fake_pynvml, create=True):
            result = mig_inspector.get_mig_mode(MagicMock())
        assert result == {'current_enabled': True, 'pending_enabled': True}

    def test_returns_disabled_when_nvml_reports_0(self):
        fake_pynvml = MagicMock()
        fake_pynvml.nvmlDeviceGetMigMode.return_value = (0, 0)
        with patch.object(mig_inspector, '_NVML_AVAILABLE', True), \
             patch.object(mig_inspector, 'pynvml', fake_pynvml, create=True):
            result = mig_inspector.get_mig_mode(MagicMock())
        assert result == {'current_enabled': False, 'pending_enabled': False}

    def test_tolerates_nvml_error(self):
        fake_pynvml = MagicMock()
        fake_pynvml.nvmlDeviceGetMigMode.side_effect = Exception("boom")
        with patch.object(mig_inspector, '_NVML_AVAILABLE', True), \
             patch.object(mig_inspector, 'pynvml', fake_pynvml, create=True):
            result = mig_inspector.get_mig_mode(MagicMock())
        assert result == {'current_enabled': False, 'pending_enabled': False}


@pytest.mark.unit
class TestGetMigPartitions:
    def test_empty_when_nvml_unavailable(self):
        with patch.object(mig_inspector, '_NVML_AVAILABLE', False):
            assert mig_inspector.get_mig_partitions(MagicMock()) == []

    def test_enumerates_partitions(self):
        fake_pynvml = MagicMock()
        fake_pynvml.nvmlDeviceGetName.return_value = b"NVIDIA H100 80GB HBM3"
        fake_pynvml.nvmlDeviceGetMaxMigDeviceCount.return_value = 3

        def handle_by_index(_parent, index):
            # Slot 1 has no instance — simulate that by raising.
            if index == 1:
                raise Exception("no instance")
            return MagicMock(name=f"mig-{index}")

        fake_pynvml.nvmlDeviceGetMigDeviceHandleByIndex.side_effect = handle_by_index
        fake_pynvml.nvmlDeviceGetUUID.return_value = b'MIG-uuid-xyz'

        mem_info = MagicMock()
        mem_info.total = 10 * 1024 * 1024 * 1024
        mem_info.used = 2 * 1024 * 1024 * 1024
        fake_pynvml.nvmlDeviceGetMemoryInfo.return_value = mem_info

        with patch.object(mig_inspector, '_NVML_AVAILABLE', True), \
             patch.object(mig_inspector, 'pynvml', fake_pynvml, create=True):
            parts = mig_inspector.get_mig_partitions(MagicMock())

        assert len(parts) == 2  # slot 1 was skipped
        assert parts[0]['memory_total_mb'] == 10240
        assert parts[0]['memory_used_mb'] == 2048
        assert parts[0]['uuid'] == 'MIG-uuid-xyz'


@pytest.mark.unit
class TestInspectDevice:
    def test_returns_benign_shape_when_mig_disabled(self):
        fake_pynvml = MagicMock()
        fake_pynvml.nvmlDeviceGetMigMode.return_value = (0, 0)
        fake_pynvml.nvmlDeviceGetUUID.return_value = b'GPU-abc'
        # Running processes enumeration returns an empty list.
        fake_pynvml.nvmlDeviceGetComputeRunningProcesses.return_value = []
        fake_pynvml.nvmlDeviceGetGraphicsRunningProcesses.return_value = []

        with patch.object(mig_inspector, '_NVML_AVAILABLE', True), \
             patch.object(mig_inspector, 'pynvml', fake_pynvml, create=True):
            snap = mig_inspector.inspect_device(MagicMock(), index=0)

        assert snap['gpu_index'] == 0
        assert snap['gpu_uuid'] == 'GPU-abc'
        assert snap['mig_enabled'] is False
        assert snap['partitions'] == []


@pytest.mark.unit
class TestMigProfiles:
    def test_h100_profile_from_fixture(self):
        with open(FIXTURES, encoding="utf-8") as fh:
            rows = json.load(fh)["mig_memory_profiles"]
        h100 = next(r for r in rows if r["gpu_family"] == "H100" and r["profile"] == "1g.10gb")
        profile, memory_gb, compute_slice = mig_inspector.infer_profile_from_memory_mb(
            h100["memory_mb"], gpu_family="H100"
        )
        assert profile == "1g.10gb"
        assert memory_gb == 10
        assert compute_slice == h100["compute_slice"]

    def test_h200_profile_from_fixture(self):
        with open(FIXTURES, encoding="utf-8") as fh:
            rows = json.load(fh)["mig_memory_profiles"]
        h200 = next(r for r in rows if r["gpu_family"] == "H200" and r["profile"] == "1g.18gb")
        profile, _, compute_slice = mig_inspector.infer_profile_from_memory_mb(
            h200["memory_mb"], gpu_family="H200"
        )
        assert profile == "1g.18gb"
        assert compute_slice == h200["compute_slice"]

    def test_b200_profile_from_fixture(self):
        with open(FIXTURES, encoding="utf-8") as fh:
            rows = json.load(fh)["mig_memory_profiles"]
        b200 = next(r for r in rows if r["gpu_family"] == "B200" and r["profile"] == "1g.23gb")
        profile, memory_gb, compute_slice = mig_inspector.infer_profile_from_memory_mb(
            b200["memory_mb"], gpu_family="B200"
        )
        assert profile == "1g.23gb"
        assert memory_gb == 23
        assert compute_slice == b200["compute_slice"]

    def test_gb200_uses_distinct_profile_table(self):
        """GB200 must not alias to B200's 2g.45gb / 3g.90gb / 7g.180gb names."""
        profile, memory_gb, compute_slice = mig_inspector.infer_profile_from_memory_mb(
            48128, gpu_family="GB200", gi_slice=2
        )
        assert profile == "2g.47gb"
        assert memory_gb == 47
        assert compute_slice == 28
        # Same memory on B200 resolves to the B200 name, not GB200's.
        b200_profile, _, _ = mig_inspector.infer_profile_from_memory_mb(
            46080, gpu_family="B200", gi_slice=2
        )
        assert b200_profile == "2g.45gb"
        # Trimmed table must not invent H100-96 / GH200-shared names.
        assert "1g.24gb" not in {r[0] for r in mig_inspector._MIG_PROFILES_GB200}
        assert "7g.189gb" not in {r[0] for r in mig_inspector._MIG_PROFILES_GB200}

    def test_same_memory_profiles_use_gi_slice(self):
        """1g.45gb/2g.45gb and 3g.90gb/4g.90gb share memory; gi_slice breaks ties."""
        p1, _, c1 = mig_inspector.infer_profile_from_memory_mb(
            46080, gpu_family="B200", gi_slice=1
        )
        p2, _, c2 = mig_inspector.infer_profile_from_memory_mb(
            46080, gpu_family="B200", gi_slice=2
        )
        assert (p1, c1) == ("1g.45gb", 14)
        assert (p2, c2) == ("2g.45gb", 28)

        p3, _, c3 = mig_inspector.infer_profile_from_memory_mb(
            92160, gpu_family="B200", gi_slice=3
        )
        p4, _, c4 = mig_inspector.infer_profile_from_memory_mb(
            92160, gpu_family="B200", gi_slice=4
        )
        assert (p3, c3) == ("3g.90gb", 42)
        assert (p4, c4) == ("4g.90gb", 57)

    def test_partition_to_wire_passes_gi_slice(self):
        wire = mig_inspector.partition_to_wire(
            {"index": 0, "uuid": "MIG-x", "memory_total_mb": 92160, "gi_slice": 4},
            gpu_index=0,
            gpu_family="B200",
        )
        assert wire["profile"] == "4g.90gb"
        assert wire["compute_slice"] == 57
        assert wire["memory_gb"] == 90

    def test_gi_slice_from_mig_device_name(self):
        assert mig_inspector._profile_gi_slice_from_device_name(
            "NVIDIA B200 MIG 4g.90gb"
        ) == 4
        assert mig_inspector._profile_gi_slice_from_device_name(
            "NVIDIA GB200 MIG 2g.47gb"
        ) == 2

    def test_mig_gi_slice_uses_by_id_not_profile_index(self):
        """profileId must go to ProfileInfoById(V), never ProfileInfo(index)."""
        fake_pynvml = MagicMock()
        fake_pynvml.nvmlDeviceGetName.return_value = b"NVIDIA B200"
        fake_pynvml.nvmlDeviceGetDeviceHandleFromMigDeviceHandle.return_value = MagicMock()
        fake_pynvml.nvmlDeviceGetGpuInstanceId.return_value = 7
        fake_pynvml.nvmlDeviceGetGpuInstanceById.return_value = MagicMock()
        info = MagicMock()
        info.profileId = 42  # not a valid profile *index*
        fake_pynvml.nvmlGpuInstanceGetInfo.return_value = info
        prof = MagicMock()
        prof.name = b"MIG 4g.90gb"
        prof.sliceCount = 4
        fake_pynvml.nvmlDeviceGetGpuInstanceProfileInfoByIdV.return_value = prof
        fake_pynvml.nvmlDeviceGetGpuInstanceProfileInfoById.return_value = prof
        # If the wrong API is used with profileId-as-index, blow up.
        fake_pynvml.nvmlDeviceGetGpuInstanceProfileInfo.side_effect = AssertionError(
            "must not call ProfileInfo with profileId"
        )
        fake_pynvml.nvmlDeviceGetGpuInstanceProfileInfoV.side_effect = AssertionError(
            "must not call ProfileInfoV with profileId"
        )
        with patch.object(mig_inspector, "_NVML_AVAILABLE", True), \
             patch.object(mig_inspector, "pynvml", fake_pynvml, create=True):
            assert mig_inspector._mig_gi_slice(MagicMock()) == 4
        fake_pynvml.nvmlDeviceGetGpuInstanceProfileInfoByIdV.assert_called()
        fake_pynvml.nvmlDeviceGetGpuInstanceProfileInfo.assert_not_called()

@pytest.mark.unit
class TestBlackwellMigSupport:
    def test_b200_device_supports_mig(self):
        fake_pynvml = MagicMock()
        fake_pynvml.nvmlDeviceGetName.return_value = b"NVIDIA B200"
        fake_pynvml.nvmlDeviceGetMigMode.return_value = (0, 0)
        with patch.object(mig_inspector, "_NVML_AVAILABLE", True), \
             patch.object(mig_inspector, "pynvml", fake_pynvml, create=True):
            assert mig_inspector.device_supports_mig(MagicMock()) is True
            assert mig_inspector.get_mig_mode(MagicMock()) == {
                "current_enabled": False,
                "pending_enabled": False,
            }

    def test_gb200_inspect_reports_supported(self):
        fake_pynvml = MagicMock()
        fake_pynvml.nvmlDeviceGetName.return_value = b"NVIDIA GB200 NVL72"
        fake_pynvml.nvmlDeviceGetUUID.return_value = b"GPU-gb200"
        fake_pynvml.nvmlDeviceGetMigMode.return_value = (0, 0)
        fake_pynvml.nvmlDeviceGetComputeRunningProcesses.return_value = []
        with patch.object(mig_inspector, "_NVML_AVAILABLE", True), \
             patch.object(mig_inspector, "pynvml", fake_pynvml, create=True):
            snap = mig_inspector.inspect_device(MagicMock(), index=0)
        assert snap["mig_supported"] is True
        assert snap["gpu_family"] == "GB200"
        assert snap["partitions"] == []
