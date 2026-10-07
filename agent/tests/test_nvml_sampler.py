"""
Comprehensive tests for NvmlSampler — covers initialization, device reading,
idle detection (ring buffer logic), sampling aggregation, and graceful
handling when NVML is unavailable.
"""

import pytest
import time
import os
import sys
import collections
from unittest.mock import Mock, patch, MagicMock, PropertyMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


# ---------------------------------------------------------------------------
# Helper: Create sampler with mocked pynvml
# ---------------------------------------------------------------------------

def _make_sampler(device_count=1, sample_interval=10, idle_gpu_threshold=10.0,
                  idle_memory_threshold=10.0, idle_duration_threshold=120,
                  available=True):
    """Create an NvmlSampler with mocked pynvml -- bypasses __init__."""
    from nvml_sampler import NvmlSampler, GpuSample
    sampler = object.__new__(NvmlSampler)
    sampler.sample_interval = sample_interval
    sampler.idle_gpu_threshold = idle_gpu_threshold
    sampler.idle_memory_threshold = idle_memory_threshold
    sampler.idle_duration_threshold = idle_duration_threshold
    sampler._buffer_size = max(1, idle_duration_threshold // sample_interval)
    sampler._available = available
    sampler._device_count = device_count
    sampler._buffers = {i: collections.deque(maxlen=sampler._buffer_size) for i in range(device_count)}
    sampler._lock = __import__('threading').Lock()
    sampler._stop_event = __import__('threading').Event()
    sampler._thread = None
    return sampler


def _make_sample(gpu_index=0, gpu_util=50.0, mem_util=30.0,
                 mem_used_mb=4096.0, mem_total_mb=16384.0,
                 temp=65.0, power=120.0, mem_bw_pct=None, timestamp=None,
                 ecc_errors_total=None, memory_errors_total=None,
                 clock_mhz=1500, memory_clock_mhz=5000):
    """Create a GpuSample namedtuple."""
    from nvml_sampler import GpuSample
    return GpuSample(
        gpu_index=gpu_index,
        gpu_utilization=gpu_util,
        memory_utilization=mem_util,
        memory_used_mb=mem_used_mb,
        memory_total_mb=mem_total_mb,
        temperature_c=temp,
        power_usage_w=power,
        mem_bw_pct=mem_bw_pct,
        ecc_errors_total=ecc_errors_total,
        memory_errors_total=memory_errors_total,
        clock_mhz=clock_mhz,
        memory_clock_mhz=memory_clock_mhz,
        timestamp=timestamp or time.time(),
    )


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestNvmlSamplerInit:
    def test_init_with_pynvml_available(self):
        """Should initialize successfully when pynvml is available."""
        mock_pynvml = MagicMock()
        mock_pynvml.nvmlInit.return_value = None
        mock_pynvml.nvmlDeviceGetCount.return_value = 2

        with patch.dict(sys.modules, {'pynvml': mock_pynvml}), \
             patch('nvml_sampler.PYNVML_AVAILABLE', True), \
             patch('nvml_sampler.pynvml', mock_pynvml):
            from nvml_sampler import NvmlSampler
            sampler = NvmlSampler(sample_interval=5, idle_duration_threshold=60)

        assert sampler.is_available() is True
        assert sampler._device_count == 2
        assert len(sampler._buffers) == 2
        assert sampler._buffer_size == 12  # 60 // 5

    def test_init_with_pynvml_unavailable(self):
        """Should gracefully handle missing pynvml."""
        with patch('nvml_sampler.PYNVML_AVAILABLE', False):
            from nvml_sampler import NvmlSampler
            sampler = NvmlSampler()

        assert sampler.is_available() is False
        assert sampler._device_count == 0

    def test_init_with_nvml_init_failure(self):
        """Should mark unavailable when nvmlInit raises."""
        mock_pynvml = MagicMock()
        mock_pynvml.nvmlInit.side_effect = Exception('Driver not loaded')

        import nvml_sampler
        orig_available = nvml_sampler.PYNVML_AVAILABLE
        orig_pynvml = getattr(nvml_sampler, 'pynvml', None)
        try:
            nvml_sampler.PYNVML_AVAILABLE = True
            nvml_sampler.pynvml = mock_pynvml
            sampler = nvml_sampler.NvmlSampler()
        finally:
            nvml_sampler.PYNVML_AVAILABLE = orig_available
            if orig_pynvml is not None:
                nvml_sampler.pynvml = orig_pynvml

        assert sampler.is_available() is False

    def test_init_with_zero_devices(self):
        """Should mark unavailable when no GPUs detected."""
        mock_pynvml = MagicMock()
        mock_pynvml.nvmlInit.return_value = None
        mock_pynvml.nvmlDeviceGetCount.return_value = 0

        import nvml_sampler
        orig_available = nvml_sampler.PYNVML_AVAILABLE
        orig_pynvml = getattr(nvml_sampler, 'pynvml', None)
        try:
            nvml_sampler.PYNVML_AVAILABLE = True
            nvml_sampler.pynvml = mock_pynvml
            sampler = nvml_sampler.NvmlSampler()
        finally:
            nvml_sampler.PYNVML_AVAILABLE = orig_available
            if orig_pynvml is not None:
                nvml_sampler.pynvml = orig_pynvml

        assert sampler.is_available() is False

    def test_buffer_size_calculation(self):
        """Should calculate buffer size from duration / interval."""
        sampler = _make_sampler(idle_duration_threshold=300, sample_interval=10)
        assert sampler._buffer_size == 30

    def test_buffer_size_minimum_one(self):
        """Should have minimum buffer size of 1."""
        sampler = _make_sampler(idle_duration_threshold=5, sample_interval=10)
        assert sampler._buffer_size == 1


# ---------------------------------------------------------------------------
# _read_device
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestReadDevice:
    def test_returns_correct_structure(self):
        """Should return GpuSample with all fields populated."""
        mock_pynvml = MagicMock()
        mock_pynvml.nvmlDeviceGetUtilizationRates.return_value = Mock(gpu=75.5, memory=42.3)
        mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = Mock(
            used=4 * 1024 * 1024 * 1024,
            total=16 * 1024 * 1024 * 1024,
        )
        mock_pynvml.nvmlDeviceGetTemperature.return_value = 68
        mock_pynvml.nvmlDeviceGetPowerUsage.return_value = 150000  # milliwatts
        mock_pynvml.NVML_TEMPERATURE_GPU = 0

        with patch.dict(sys.modules, {'pynvml': mock_pynvml}):
            import importlib
            import nvml_sampler
            nvml_sampler.pynvml = mock_pynvml
            sampler = _make_sampler()
            handle = Mock()
            ts = time.time()
            sample = sampler._read_device(handle, 0, ts)

        assert sample.gpu_index == 0
        assert sample.gpu_utilization == 75.5
        assert sample.memory_utilization == 42.3
        assert sample.memory_used_mb == pytest.approx(4096.0, rel=1e-1)
        assert sample.memory_total_mb == pytest.approx(16384.0, rel=1e-1)
        assert sample.temperature_c == 68
        assert sample.power_usage_w == 150.0
        assert sample.timestamp == ts

    def test_handles_temperature_failure(self):
        """Should set temperature to None on failure."""
        mock_pynvml = MagicMock()
        mock_pynvml.nvmlDeviceGetUtilizationRates.return_value = Mock(gpu=50, memory=30)
        mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = Mock(used=100, total=1000)
        mock_pynvml.nvmlDeviceGetTemperature.side_effect = Exception('No sensor')
        mock_pynvml.nvmlDeviceGetPowerUsage.return_value = 100000
        mock_pynvml.NVML_TEMPERATURE_GPU = 0

        import nvml_sampler
        nvml_sampler.pynvml = mock_pynvml
        sampler = _make_sampler()
        sample = sampler._read_device(Mock(), 0, time.time())

        assert sample.temperature_c is None
        assert sample.power_usage_w is not None

    def test_handles_power_failure(self):
        """Should set power to None on failure."""
        mock_pynvml = MagicMock()
        mock_pynvml.nvmlDeviceGetUtilizationRates.return_value = Mock(gpu=50, memory=30)
        mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = Mock(used=100, total=1000)
        mock_pynvml.nvmlDeviceGetTemperature.return_value = 60
        mock_pynvml.nvmlDeviceGetPowerUsage.side_effect = Exception('No power reading')
        mock_pynvml.NVML_TEMPERATURE_GPU = 0

        import nvml_sampler
        nvml_sampler.pynvml = mock_pynvml
        sampler = _make_sampler()
        sample = sampler._read_device(Mock(), 0, time.time())

        assert sample.temperature_c == 60
        assert sample.power_usage_w is None

    def test_rounds_values(self):
        """Should round utilization and memory values to 2 decimal places."""
        mock_pynvml = MagicMock()
        mock_pynvml.nvmlDeviceGetUtilizationRates.return_value = Mock(gpu=75.555, memory=42.333)
        mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = Mock(
            used=1234567890, total=9876543210
        )
        mock_pynvml.nvmlDeviceGetTemperature.return_value = 65
        mock_pynvml.nvmlDeviceGetPowerUsage.return_value = 123456
        mock_pynvml.NVML_TEMPERATURE_GPU = 0

        import nvml_sampler
        nvml_sampler.pynvml = mock_pynvml
        sampler = _make_sampler()
        sample = sampler._read_device(Mock(), 0, time.time())

        assert sample.gpu_utilization == 75.56  # rounded
        assert sample.memory_utilization == 42.33
        assert sample.power_usage_w == 123.46  # 123456 / 1000, rounded

    def test_correct_device_index(self):
        """Should preserve the device index in the returned sample."""
        mock_pynvml = MagicMock()
        mock_pynvml.nvmlDeviceGetUtilizationRates.return_value = Mock(gpu=50, memory=30)
        mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = Mock(used=100, total=1000)
        mock_pynvml.nvmlDeviceGetTemperature.return_value = 60
        mock_pynvml.nvmlDeviceGetPowerUsage.return_value = 100000
        mock_pynvml.NVML_TEMPERATURE_GPU = 0

        import nvml_sampler
        nvml_sampler.pynvml = mock_pynvml
        sampler = _make_sampler(device_count=4)
        sample = sampler._read_device(Mock(), 3, time.time())

        assert sample.gpu_index == 3


# ---------------------------------------------------------------------------
# Idle detection (ring buffer logic)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestIdleDetection:
    def test_not_idle_when_buffer_not_full(self):
        """Should return False when ring buffer is not yet full."""
        sampler = _make_sampler(idle_duration_threshold=30, sample_interval=10)
        # buffer_size = 3, add only 2 samples
        sampler._buffers[0].append(_make_sample(gpu_util=1, mem_util=1))
        sampler._buffers[0].append(_make_sample(gpu_util=1, mem_util=1))

        assert sampler._is_device_idle(sampler._buffers[0]) is False

    def test_idle_when_buffer_full_all_below_threshold(self):
        """Should return True when all samples in full buffer are below threshold."""
        sampler = _make_sampler(idle_duration_threshold=30, sample_interval=10)
        # buffer_size = 3
        for _ in range(3):
            sampler._buffers[0].append(_make_sample(gpu_util=5, mem_util=5))

        assert sampler._is_device_idle(sampler._buffers[0]) is True

    def test_not_idle_when_any_sample_above_gpu_threshold(self):
        """Should return False if any sample has GPU util above threshold."""
        sampler = _make_sampler(idle_duration_threshold=30, sample_interval=10)
        sampler._buffers[0].append(_make_sample(gpu_util=5, mem_util=5))
        sampler._buffers[0].append(_make_sample(gpu_util=15, mem_util=5))  # above threshold
        sampler._buffers[0].append(_make_sample(gpu_util=5, mem_util=5))

        assert sampler._is_device_idle(sampler._buffers[0]) is False

    def test_not_idle_when_any_sample_above_memory_threshold(self):
        """Should return False if any sample has memory util above threshold."""
        sampler = _make_sampler(idle_duration_threshold=30, sample_interval=10)
        sampler._buffers[0].append(_make_sample(gpu_util=5, mem_util=5))
        sampler._buffers[0].append(_make_sample(gpu_util=5, mem_util=15))  # above threshold
        sampler._buffers[0].append(_make_sample(gpu_util=5, mem_util=5))

        assert sampler._is_device_idle(sampler._buffers[0]) is False

    def test_idle_exactly_at_threshold_is_not_idle(self):
        """Should not be idle when utilization equals threshold (< not <=)."""
        sampler = _make_sampler(idle_gpu_threshold=10.0, idle_memory_threshold=10.0,
                                idle_duration_threshold=10, sample_interval=10)
        sampler._buffers[0].append(_make_sample(gpu_util=10.0, mem_util=10.0))

        assert sampler._is_device_idle(sampler._buffers[0]) is False

    def test_idle_just_below_threshold(self):
        """Should be idle when utilization is just below threshold."""
        sampler = _make_sampler(idle_gpu_threshold=10.0, idle_memory_threshold=10.0,
                                idle_duration_threshold=10, sample_interval=10)
        sampler._buffers[0].append(_make_sample(gpu_util=9.99, mem_util=9.99))

        assert sampler._is_device_idle(sampler._buffers[0]) is True

    def test_ring_buffer_evicts_old_samples(self):
        """Should evict oldest sample when buffer exceeds max length."""
        sampler = _make_sampler(idle_duration_threshold=20, sample_interval=10)
        # buffer_size = 2

        sampler._buffers[0].append(_make_sample(gpu_util=50, mem_util=50))  # will be evicted
        sampler._buffers[0].append(_make_sample(gpu_util=5, mem_util=5))
        sampler._buffers[0].append(_make_sample(gpu_util=5, mem_util=5))

        # Old high-util sample should be gone
        assert sampler._is_device_idle(sampler._buffers[0]) is True

    def test_get_idle_devices_returns_idle_indices(self):
        """Should return list of GPU indices that are idle."""
        sampler = _make_sampler(device_count=3, idle_duration_threshold=10, sample_interval=10)

        # GPU 0: idle
        sampler._buffers[0].append(_make_sample(gpu_index=0, gpu_util=1, mem_util=1))
        # GPU 1: not idle (high util)
        sampler._buffers[1].append(_make_sample(gpu_index=1, gpu_util=80, mem_util=50))
        # GPU 2: idle
        sampler._buffers[2].append(_make_sample(gpu_index=2, gpu_util=2, mem_util=2))

        idle = sampler.get_idle_devices()
        assert 0 in idle
        assert 1 not in idle
        assert 2 in idle

    def test_get_idle_devices_empty_buffer(self):
        """Should not return idle for empty buffers."""
        sampler = _make_sampler(device_count=1, idle_duration_threshold=10, sample_interval=10)
        # Buffer is empty
        assert sampler.get_idle_devices() == []


# ---------------------------------------------------------------------------
# get_latest_metrics
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestGetLatestMetrics:
    def test_returns_latest_sample_per_device(self):
        """Should return the most recent sample for each device."""
        sampler = _make_sampler(device_count=2, idle_duration_threshold=10, sample_interval=10)

        ts1 = time.time() - 20
        ts2 = time.time()

        sampler._buffers[0].append(_make_sample(gpu_index=0, gpu_util=30, timestamp=ts1))
        sampler._buffers[0].append(_make_sample(gpu_index=0, gpu_util=80, timestamp=ts2))
        sampler._buffers[1].append(_make_sample(gpu_index=1, gpu_util=45, timestamp=ts2))

        metrics = sampler.get_latest_metrics()

        assert len(metrics) == 2
        gpu0 = next(m for m in metrics if m['gpu_index'] == 0)
        assert gpu0['gpu_utilization'] == 80
        assert gpu0['source'] == 'nvml_local'

    def test_empty_buffer_skipped(self):
        """Should skip devices with no samples."""
        sampler = _make_sampler(device_count=2)
        # Only populate device 1
        sampler._buffers[1].append(_make_sample(gpu_index=1, gpu_util=50))

        metrics = sampler.get_latest_metrics()
        assert len(metrics) == 1
        assert metrics[0]['gpu_index'] == 1

    def test_no_devices_returns_empty(self):
        """Should return empty list when no devices available."""
        sampler = _make_sampler(device_count=0)
        sampler._buffers = {}
        assert sampler.get_latest_metrics() == []

    def test_includes_idle_status(self):
        """Should include is_idle based on full ring buffer analysis."""
        sampler = _make_sampler(device_count=1, idle_duration_threshold=10, sample_interval=10)
        sampler._buffers[0].append(_make_sample(gpu_index=0, gpu_util=2, mem_util=3))

        metrics = sampler.get_latest_metrics()
        assert len(metrics) == 1
        assert metrics[0]['is_idle'] is True

    def test_not_idle_in_latest_metrics(self):
        """Should report is_idle=False when utilization is high."""
        sampler = _make_sampler(device_count=1, idle_duration_threshold=10, sample_interval=10)
        sampler._buffers[0].append(_make_sample(gpu_index=0, gpu_util=80, mem_util=50))

        metrics = sampler.get_latest_metrics()
        assert metrics[0]['is_idle'] is False

    def test_includes_all_metric_fields(self):
        """Should include all expected fields in the output dict."""
        sampler = _make_sampler(device_count=1, idle_duration_threshold=10, sample_interval=10)
        sampler._buffers[0].append(_make_sample(
            gpu_index=0, gpu_util=70, mem_util=40,
            mem_used_mb=8192, mem_total_mb=16384,
            temp=72, power=200,
        ))

        metrics = sampler.get_latest_metrics()
        m = metrics[0]
        assert m['gpu_index'] == 0
        assert m['gpu_utilization'] == 70
        assert m['memory_utilization'] == 40
        assert m['memory_used_mb'] == 8192
        assert m['memory_total_mb'] == 16384
        assert m['temperature_c'] == 72
        assert m['power_usage_w'] == 200
        assert 'timestamp' in m
        assert m['source'] == 'nvml_local'

    def test_thread_safety(self):
        """Should be safe to call from multiple threads concurrently."""
        sampler = _make_sampler(device_count=1, idle_duration_threshold=10, sample_interval=10)
        sampler._buffers[0].append(_make_sample(gpu_index=0, gpu_util=50))

        results = []
        errors = []

        def reader():
            try:
                for _ in range(100):
                    metrics = sampler.get_latest_metrics()
                    results.append(len(metrics))
            except Exception as e:
                errors.append(e)

        threads = [__import__('threading').Thread(target=reader) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert all(r == 1 for r in results)


# ---------------------------------------------------------------------------
# _sample_all_devices
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestSampleAllDevices:
    def test_samples_all_devices(self):
        """Should read metrics from all GPU devices."""
        mock_pynvml = MagicMock()
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = Mock()
        mock_pynvml.nvmlDeviceGetUtilizationRates.return_value = Mock(gpu=50, memory=30)
        mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = Mock(used=4*1024**3, total=16*1024**3)
        mock_pynvml.nvmlDeviceGetTemperature.return_value = 60
        mock_pynvml.nvmlDeviceGetPowerUsage.return_value = 100000
        mock_pynvml.NVML_TEMPERATURE_GPU = 0

        import nvml_sampler
        nvml_sampler.pynvml = mock_pynvml
        sampler = _make_sampler(device_count=3)
        sampler._sample_all_devices()

        assert len(sampler._buffers[0]) == 1
        assert len(sampler._buffers[1]) == 1
        assert len(sampler._buffers[2]) == 1

    def test_isolates_device_failures(self):
        """Should continue sampling other devices when one fails."""
        mock_pynvml = MagicMock()
        good_handle = Mock()
        mock_pynvml.nvmlDeviceGetHandleByIndex.side_effect = [
            Exception('GPU 0 error'),
            good_handle,
        ]
        mock_pynvml.nvmlDeviceGetUtilizationRates.return_value = Mock(gpu=50, memory=30)
        mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = Mock(used=100, total=1000)
        mock_pynvml.nvmlDeviceGetTemperature.return_value = 60
        mock_pynvml.nvmlDeviceGetPowerUsage.return_value = 100000
        mock_pynvml.NVML_TEMPERATURE_GPU = 0

        import nvml_sampler
        nvml_sampler.pynvml = mock_pynvml
        sampler = _make_sampler(device_count=2)
        sampler._sample_all_devices()

        assert len(sampler._buffers[0]) == 0  # failed
        assert len(sampler._buffers[1]) == 1  # succeeded


# ---------------------------------------------------------------------------
# Lifecycle: start / stop
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestLifecycle:
    def test_start_skips_when_unavailable(self):
        """Should not start thread when NVML is unavailable."""
        sampler = _make_sampler(available=False)
        sampler.start()
        assert sampler._thread is None

    def test_start_creates_daemon_thread(self):
        """Should create a daemon thread for sampling."""
        sampler = _make_sampler()
        with patch.object(sampler, '_sample_loop'):
            sampler.start()
            assert sampler._thread is not None
            assert sampler._thread.daemon is True
            assert sampler._thread.name == 'nvml-sampler'
            sampler.stop()

    def test_start_idempotent(self):
        """Should not create second thread if already running."""
        sampler = _make_sampler()
        # Keep the thread alive by blocking on the stop event
        def blocking_loop():
            sampler._stop_event.wait()
        with patch.object(sampler, '_sample_loop', side_effect=blocking_loop):
            sampler.start()
            first_thread = sampler._thread
            sampler.start()  # Should be no-op since thread is alive
            assert sampler._thread is first_thread
            sampler.stop()

    def test_stop_sets_stop_event(self):
        """Should set the stop event to signal shutdown."""
        sampler = _make_sampler()
        sampler.stop()
        assert sampler._stop_event.is_set()

    def test_is_available_reflects_state(self):
        """Should return the availability state."""
        sampler = _make_sampler(available=True)
        assert sampler.is_available() is True

        sampler = _make_sampler(available=False)
        assert sampler.is_available() is False


# ---------------------------------------------------------------------------
# ECC Errors
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestEccErrors:
    """Tests for ECC error monitoring.

    ecc_errors_total is Optional[int] — uncorrected error count from nvmlDeviceGetTotalEccErrors.
    memory_errors_total is Optional[int] — corrected error count.
    get_devices_with_ecc_errors returns List[Dict] with gpu_index, ecc_errors_uncorrected, ecc_errors_corrected.
    get_latest_metrics returns ecc_errors_uncorrected / ecc_errors_corrected keys.
    """

    def test_get_devices_with_ecc_errors_above_threshold(self):
        """Should return devices with uncorrected ECC errors above threshold."""
        sampler = _make_sampler(device_count=3)

        sampler._buffers[0].append(_make_sample(gpu_index=0, ecc_errors_total=5, memory_errors_total=10))
        sampler._buffers[1].append(_make_sample(gpu_index=1, ecc_errors_total=0, memory_errors_total=20))
        sampler._buffers[2].append(_make_sample(gpu_index=2, ecc_errors_total=12, memory_errors_total=5))

        devices = sampler.get_devices_with_ecc_errors(threshold=10)

        assert len(devices) == 1
        assert devices[0]['gpu_index'] == 2
        assert devices[0]['ecc_errors_uncorrected'] == 12

    def test_get_devices_with_ecc_errors_threshold_zero(self):
        """Should return all devices with uncorrected ECC errors > 0 when threshold=0."""
        sampler = _make_sampler(device_count=3)

        sampler._buffers[0].append(_make_sample(gpu_index=0, ecc_errors_total=5, memory_errors_total=10))
        sampler._buffers[1].append(_make_sample(gpu_index=1, ecc_errors_total=0, memory_errors_total=20))
        sampler._buffers[2].append(_make_sample(gpu_index=2, ecc_errors_total=12, memory_errors_total=5))

        devices = sampler.get_devices_with_ecc_errors(threshold=0)

        gpu_indices = [d['gpu_index'] for d in devices]
        assert len(devices) == 2
        assert 0 in gpu_indices
        assert 2 in gpu_indices
        assert 1 not in gpu_indices  # GPU 1 has 0 uncorrected errors

    def test_get_devices_with_ecc_errors_no_errors(self):
        """Should return empty list when no devices have ECC errors."""
        sampler = _make_sampler(device_count=2)

        sampler._buffers[0].append(_make_sample(gpu_index=0, ecc_errors_total=0, memory_errors_total=5))
        sampler._buffers[1].append(_make_sample(gpu_index=1, ecc_errors_total=0, memory_errors_total=10))

        devices = sampler.get_devices_with_ecc_errors(threshold=0)

        assert devices == []

    def test_get_devices_with_ecc_errors_none_values(self):
        """Should return empty list when ECC values are None."""
        sampler = _make_sampler(device_count=2)

        sampler._buffers[0].append(_make_sample(gpu_index=0, ecc_errors_total=None))
        sampler._buffers[1].append(_make_sample(gpu_index=1, ecc_errors_total=None))

        devices = sampler.get_devices_with_ecc_errors(threshold=0)

        assert devices == []

    def test_get_devices_with_ecc_errors_mixed_none_and_values(self):
        """Should skip devices with None ECC values."""
        sampler = _make_sampler(device_count=3)

        sampler._buffers[0].append(_make_sample(gpu_index=0, ecc_errors_total=8, memory_errors_total=5))
        sampler._buffers[1].append(_make_sample(gpu_index=1, ecc_errors_total=None))
        sampler._buffers[2].append(_make_sample(gpu_index=2, ecc_errors_total=3, memory_errors_total=10))

        devices = sampler.get_devices_with_ecc_errors(threshold=5)

        assert len(devices) == 1
        assert devices[0]['gpu_index'] == 0
        assert devices[0]['ecc_errors_uncorrected'] == 8

    def test_get_latest_metrics_includes_ecc_fields(self):
        """Should include ECC error fields in latest metrics."""
        sampler = _make_sampler(device_count=1)

        sampler._buffers[0].append(_make_sample(
            gpu_index=0, ecc_errors_total=5, memory_errors_total=20
        ))

        metrics = sampler.get_latest_metrics()

        assert len(metrics) == 1
        assert metrics[0]['ecc_errors_uncorrected'] == 5
        assert metrics[0]['ecc_errors_corrected'] == 20

    def test_get_latest_metrics_ecc_fields_when_none(self):
        """Should include None ECC fields when not available."""
        sampler = _make_sampler(device_count=1)

        sampler._buffers[0].append(_make_sample(
            gpu_index=0, ecc_errors_total=None, memory_errors_total=None
        ))

        metrics = sampler.get_latest_metrics()

        assert len(metrics) == 1
        assert metrics[0]['ecc_errors_uncorrected'] is None
        assert metrics[0]['ecc_errors_corrected'] is None

    def test_get_devices_with_ecc_errors_empty_buffer(self):
        """Should return empty list when buffers are empty."""
        sampler = _make_sampler(device_count=2)

        devices = sampler.get_devices_with_ecc_errors(threshold=0)

        assert devices == []

    def test_get_devices_with_ecc_errors_dict_structure(self):
        """Return dicts should contain gpu_index, ecc_errors_uncorrected, ecc_errors_corrected."""
        sampler = _make_sampler(device_count=1)
        sampler._buffers[0].append(_make_sample(gpu_index=0, ecc_errors_total=3, memory_errors_total=15))

        devices = sampler.get_devices_with_ecc_errors(threshold=0)

        assert len(devices) == 1
        d = devices[0]
        assert 'gpu_index' in d
        assert 'ecc_errors_uncorrected' in d
        assert 'ecc_errors_corrected' in d
        assert d['gpu_index'] == 0
        assert d['ecc_errors_uncorrected'] == 3
        assert d['ecc_errors_corrected'] == 15

    def test_handles_ecc_errors_failure(self):
        """Should set ecc_errors to None on failure."""
        mock_pynvml = MagicMock()
        mock_pynvml.nvmlDeviceGetUtilizationRates.return_value = Mock(gpu=50, memory=30)
        mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = Mock(used=100, total=1000)
        mock_pynvml.nvmlDeviceGetTemperature.return_value = 60
        mock_pynvml.nvmlDeviceGetPowerUsage.return_value = 100000
        mock_pynvml.nvmlDeviceGetTotalEccErrors.side_effect = Exception('ECC not supported')
        mock_pynvml.NVML_TEMPERATURE_GPU = 0
        mock_pynvml.NVML_MEMORY_ERROR_TYPE_UNCORRECTED = 1
        mock_pynvml.NVML_VOLATILE_ECC_ERRORS = 0

        import nvml_sampler
        nvml_sampler.pynvml = mock_pynvml
        sampler = _make_sampler()
        sample = sampler._read_device(Mock(), 0, time.time())

        assert sample.ecc_errors_total is None
        assert sample.memory_errors_total is None

    def test_handles_memory_errors_failure(self):
        """Should set memory_errors to None on failure while ecc succeeds."""
        mock_pynvml = MagicMock()
        mock_pynvml.nvmlDeviceGetUtilizationRates.return_value = Mock(gpu=50, memory=30)
        mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = Mock(used=100, total=1000)
        mock_pynvml.nvmlDeviceGetTemperature.return_value = 60
        mock_pynvml.nvmlDeviceGetPowerUsage.return_value = 100000
        mock_pynvml.NVML_TEMPERATURE_GPU = 0
        mock_pynvml.NVML_MEMORY_ERROR_TYPE_UNCORRECTED = 1
        mock_pynvml.NVML_MEMORY_ERROR_TYPE_CORRECTED = 2
        mock_pynvml.NVML_VOLATILE_ECC_ERRORS = 0

        def ecc_side_effect(*args, **kwargs):
            if args[1] == 2:  # CORRECTED type
                raise Exception('Memory errors not supported')
            return 5  # Return uncorrected errors

        mock_pynvml.nvmlDeviceGetTotalEccErrors.side_effect = ecc_side_effect

        import nvml_sampler
        nvml_sampler.pynvml = mock_pynvml
        sampler = _make_sampler()
        sample = sampler._read_device(Mock(), 0, time.time())

        assert sample.ecc_errors_total == 5
        assert sample.memory_errors_total is None

    def test_sample_loop_handles_os_nice_failure(self):
        """Should handle OSError from os.nice gracefully."""
        mock_pynvml = MagicMock()
        mock_pynvml.nvmlInit.return_value = None
        mock_pynvml.nvmlDeviceGetCount.return_value = 1

        import nvml_sampler
        orig_pynvml = getattr(nvml_sampler, 'pynvml', None)
        try:
            nvml_sampler.pynvml = mock_pynvml
            with patch('nvml_sampler.os.nice', side_effect=OSError('Nice not available')):
                sampler = _make_sampler(available=True, device_count=1)
                sampler._stop_event.set()  # Stop immediately
                # Should not raise
                sampler._sample_loop()
        finally:
            if orig_pynvml is not None:
                nvml_sampler.pynvml = orig_pynvml

    def test_sample_loop_handles_attribute_error_on_nice(self):
        """Should handle AttributeError from os.nice on non-POSIX systems."""
        import nvml_sampler
        orig_pynvml = getattr(nvml_sampler, 'pynvml', None)
        try:
            with patch('nvml_sampler.os.nice', side_effect=AttributeError('nice not available')):
                sampler = _make_sampler(available=True, device_count=1)
                sampler._stop_event.set()  # Stop immediately
                # Should not raise
                sampler._sample_loop()
        finally:
            if orig_pynvml is not None:
                nvml_sampler.pynvml = orig_pynvml

    def test_sample_loop_handles_sampling_exception(self):
        """Should handle exceptions during _sample_all_devices gracefully."""
        mock_pynvml = MagicMock()

        import nvml_sampler
        orig_pynvml = getattr(nvml_sampler, 'pynvml', None)
        try:
            nvml_sampler.pynvml = mock_pynvml
            sampler = _make_sampler(available=True, device_count=1)

            call_count = [0]
            def side_effect():
                call_count[0] += 1
                if call_count[0] == 1:
                    raise Exception('Device reading failed')

            with patch.object(sampler, '_sample_all_devices', side_effect=side_effect):
                sampler._stop_event.set()
                # Should log error but not raise
                sampler._sample_loop()
        finally:
            if orig_pynvml is not None:
                nvml_sampler.pynvml = orig_pynvml
