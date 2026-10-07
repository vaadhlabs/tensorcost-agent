"""
Local NVML GPU Sampler

Samples GPU metrics locally via pynvml in a low-priority background thread.
Provides sub-minute idle detection by maintaining a ring buffer of samples
per device and requiring sustained low utilization before declaring idle.

Resource safety:
- Runs at OS nice(19) — lowest scheduling priority
- Read-only driver queries (~0.1ms each, no GPU compute)
- Bounded memory via deque(maxlen=N) ring buffers
- All failures isolated per-device
"""

import os
import time
import logging
import threading
import collections
from datetime import datetime, timezone
from typing import List, Dict, Optional, NamedTuple

logger = logging.getLogger(__name__)

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False


class GpuSample(NamedTuple):
    """Single point-in-time GPU measurement."""
    gpu_index: int
    gpu_utilization: float
    memory_utilization: float
    memory_used_mb: float
    memory_total_mb: float
    temperature_c: Optional[float]
    power_usage_w: Optional[float]
    mem_bw_pct: Optional[float]
    ecc_errors_total: Optional[int]
    memory_errors_total: Optional[int]
    clock_mhz: Optional[int]
    memory_clock_mhz: Optional[int]
    timestamp: float  # time.time()


class NvmlSampler:
    """Background GPU sampler using NVIDIA Management Library.

    Collects GPU metrics at a configurable interval and maintains a ring buffer
    per device. Idle detection requires ALL samples in the configured window to
    be below threshold — prevents false positives from momentary dips.

    Args:
        sample_interval: Seconds between samples (default 10).
        idle_gpu_threshold: GPU util % below which is considered idle (default 10).
        idle_memory_threshold: Memory util % below which is considered idle (default 10).
        idle_duration_threshold: Seconds of sustained idle before declaring device idle (default 120).
    """

    def __init__(
        self,
        sample_interval: int = 10,
        idle_gpu_threshold: float = 10.0,
        idle_memory_threshold: float = 10.0,
        idle_duration_threshold: int = 120,
        phase_detector=None,
    ):
        self.sample_interval = sample_interval
        self.idle_gpu_threshold = idle_gpu_threshold
        self.idle_memory_threshold = idle_memory_threshold
        self.idle_duration_threshold = idle_duration_threshold

        # Ring buffer size = enough samples to cover the idle duration window
        self._buffer_size = max(1, idle_duration_threshold // sample_interval)

        # Workload-aware idle (spec 3B): classify each buffer window and let
        # legitimate low-util phases (checkpoint, data loading, eval) suppress
        # a naive idle verdict.
        if phase_detector is None:
            try:
                from src.training_phase_detector import TrainingPhaseDetector
                phase_detector = TrainingPhaseDetector()
            except Exception:
                phase_detector = None
        self._phase_detector = phase_detector

        self._available = False
        self._device_count = 0
        self._buffers: Dict[int, collections.deque] = {}  # gpu_index -> deque of GpuSample
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._init_nvml()

    def _init_nvml(self):
        """Try to initialize pynvml. Silently marks unavailable on failure."""
        if not PYNVML_AVAILABLE:
            logger.info("pynvml not installed — NVML sampler unavailable")
            return

        try:
            pynvml.nvmlInit()
            self._device_count = pynvml.nvmlDeviceGetCount()
            if self._device_count == 0:
                logger.info("No NVIDIA GPUs detected — NVML sampler unavailable")
                return

            for i in range(self._device_count):
                self._buffers[i] = collections.deque(maxlen=self._buffer_size)

            self._available = True
            logger.info(
                f"NVML sampler initialized: {self._device_count} GPU(s), "
                f"sampling every {self.sample_interval}s, "
                f"idle window {self.idle_duration_threshold}s"
            )
        except Exception as e:
            logger.warning(f"Failed to initialize NVML: {e} — sampler unavailable")

    def is_available(self) -> bool:
        """Whether NVML initialized successfully and GPUs are present."""
        return self._available

    def start(self):
        """Start the background sampling thread."""
        if not self._available:
            logger.info("NVML sampler not available — skipping start")
            return

        if self._thread and self._thread.is_alive():
            logger.warning("NVML sampler already running")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True, name="nvml-sampler")
        self._thread.start()
        logger.info("NVML sampler thread started")

    def stop(self):
        """Stop the sampling thread gracefully."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=self.sample_interval + 2)
        logger.info("NVML sampler stopped")

    def _sample_loop(self):
        """Main sampling loop — runs at lowest OS priority."""
        try:
            os.nice(19)
        except (OSError, AttributeError):
            pass  # nice() not available on all platforms

        logger.debug("NVML sampler loop running at low priority")

        while not self._stop_event.is_set():
            try:
                self._sample_all_devices()
            except Exception as e:
                logger.error(f"NVML sampling cycle error: {e}")

            self._stop_event.wait(timeout=self.sample_interval)

    def _sample_all_devices(self):
        """Read metrics from all GPU devices."""
        samples = []
        now = time.time()

        for i in range(self._device_count):
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                sample = self._read_device(handle, i, now)
                samples.append(sample)
            except Exception as e:
                logger.debug(f"Failed to read GPU {i}: {e}")

        with self._lock:
            for sample in samples:
                self._buffers[sample.gpu_index].append(sample)

    def _read_device(self, handle, index: int, timestamp: float) -> GpuSample:
        """Read metrics from a single GPU device."""
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

        temperature = None
        try:
            temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        except Exception:
            pass

        power = None
        try:
            power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # milliwatts -> watts
        except Exception:
            pass

        mem_bw_pct = None
        try:
            mem_bw_pct = float(pynvml.nvmlDeviceGetUtilizationRates(handle).memory)
        except Exception:
            pass

        # ECC error counts (early hardware failure indicator)
        ecc_errors = None
        try:
            ecc_errors = pynvml.nvmlDeviceGetTotalEccErrors(
                handle,
                pynvml.NVML_MEMORY_ERROR_TYPE_UNCORRECTED,
                pynvml.NVML_VOLATILE_ECC_ERRORS,
            )
        except Exception:
            pass

        # Memory errors
        memory_errors = None
        try:
            memory_errors = pynvml.nvmlDeviceGetTotalEccErrors(
                handle,
                pynvml.NVML_MEMORY_ERROR_TYPE_CORRECTED,
                pynvml.NVML_VOLATILE_ECC_ERRORS,
            )
        except Exception:
            pass

        clock_mhz = None
        memory_clock_mhz = None
        try:
            clock_mhz = int(pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS))
            memory_clock_mhz = int(pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM))
        except Exception:
            pass

        return GpuSample(
            gpu_index=index,
            gpu_utilization=round(utilization.gpu, 2),
            memory_utilization=round(utilization.memory, 2),
            memory_used_mb=round(memory_info.used / 1024 / 1024, 2),
            memory_total_mb=round(memory_info.total / 1024 / 1024, 2),
            temperature_c=temperature,
            power_usage_w=round(power, 2) if power is not None else None,
            mem_bw_pct=round(mem_bw_pct, 2) if mem_bw_pct is not None else None,
            ecc_errors_total=ecc_errors,
            memory_errors_total=memory_errors,
            clock_mhz=clock_mhz,
            memory_clock_mhz=memory_clock_mhz,
            timestamp=timestamp,
        )

    def get_latest_metrics(self) -> List[Dict]:
        """Return the most recent sample for each device.

        Thread-safe. Returns dicts matching the existing metric schema so they
        can be merged directly into the monitoring pipeline.
        """
        results = []
        with self._lock:
            for gpu_index, buffer in self._buffers.items():
                if not buffer:
                    continue
                sample = buffer[-1]
                phase = self._classify_phase(buffer)
                is_idle = self._is_device_idle(buffer, phase=phase)
                results.append({
                    'gpu_index': sample.gpu_index,
                    'gpu_utilization': sample.gpu_utilization,
                    'memory_utilization': sample.memory_utilization,
                    'memory_used_mb': sample.memory_used_mb,
                    'memory_total_mb': sample.memory_total_mb,
                    'temperature_c': sample.temperature_c,
                    'power_usage_w': sample.power_usage_w,
                    'mem_bw_pct': sample.mem_bw_pct,
                    'ecc_errors_uncorrected': sample.ecc_errors_total,
                    'ecc_errors_corrected': sample.memory_errors_total,
                    'clock_mhz': sample.clock_mhz,
                    'memory_clock_mhz': sample.memory_clock_mhz,
                    'is_idle': is_idle,
                    'gpu_phase': phase,
                    'timestamp': datetime.fromtimestamp(sample.timestamp, tz=timezone.utc).isoformat(),
                    'source': 'nvml_local',
                })
        return results

    def get_idle_devices(self) -> List[int]:
        """Return GPU indices that have been continuously idle for >= idle_duration_threshold.

        A device is idle only if the ring buffer is full AND every sample in it
        shows utilization below the configured thresholds AND the inferred
        training phase is not a legitimate low-util phase (checkpoint, data
        loading, eval).
        """
        idle = []
        with self._lock:
            for gpu_index, buffer in self._buffers.items():
                phase = self._classify_phase(buffer)
                if self._is_device_idle(buffer, phase=phase):
                    idle.append(gpu_index)
        return idle

    def _classify_phase(self, buffer: collections.deque) -> Optional[str]:
        detector = getattr(self, '_phase_detector', None)
        if not detector or not buffer:
            return None
        try:
            return detector.classify(list(buffer))
        except Exception:
            return None

    def get_devices_with_ecc_errors(self, threshold: int = 0) -> List[Dict]:
        """Return devices with uncorrected ECC errors above threshold."""
        results = []
        with self._lock:
            for gpu_index, ring_buffer in self._buffers.items():
                if not ring_buffer:
                    continue
                latest = ring_buffer[-1]
                if latest.ecc_errors_total is not None and latest.ecc_errors_total > threshold:
                    results.append({
                        'gpu_index': gpu_index,
                        'ecc_errors_uncorrected': latest.ecc_errors_total,
                        'ecc_errors_corrected': latest.memory_errors_total,
                    })
        return results

    def _is_device_idle(self, buffer: collections.deque, phase: Optional[str] = None) -> bool:
        """Check if all samples in the buffer indicate idle state.

        Returns False if buffer is not full (not enough history yet) OR if the
        inferred training phase is a legitimate low-util phase (checkpoint,
        data loading, eval).
        """
        if len(buffer) < self._buffer_size:
            return False

        # Suppress naive idle when the workload is mid-phase.
        try:
            from src.training_phase_detector import is_low_util_phase_non_idle
        except Exception:
            is_low_util_phase_non_idle = lambda _: False  # noqa: E731
        if is_low_util_phase_non_idle(phase):
            return False

        return all(
            s.gpu_utilization < self.idle_gpu_threshold
            and s.memory_utilization < self.idle_memory_threshold
            for s in buffer
        )
