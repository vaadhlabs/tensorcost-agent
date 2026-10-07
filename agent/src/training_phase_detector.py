"""
TrainingPhaseDetector (spec 3B)

Rule-based classifier that labels each window of recent GpuSample readings
with one of these phases:

  - IDLE             — util < idle_util_pct and power near idle floor
  - DATA_LOADING     — util moderate (5-40%), memory I/O active, power mid
  - FORWARD          — util high (>70%), power high, steady
  - BACKWARD         — util very high (>85%), power at peak, memory rising
  - CHECKPOINT       — util low (<15%), power mid, memory stable (saving weights)
  - EVAL             — util moderate (30-70%), power mid, no gradient buffers
  - UNKNOWN          — insufficient signal

This is v1 — rule-based thresholds on a sliding window. A learned classifier
lands in a later sprint once the FeedbackOutcomeJob has labeled data.

Usage:
    detector = TrainingPhaseDetector()
    phase = detector.classify(samples)  # samples: Iterable[GpuSample]
"""

import collections
from typing import Iterable, Optional


PHASE_IDLE = "idle"
PHASE_DATA_LOADING = "data_loading"
PHASE_FORWARD = "forward"
PHASE_BACKWARD = "backward"
PHASE_CHECKPOINT = "checkpoint"
PHASE_EVAL = "eval"
PHASE_UNKNOWN = "unknown"

# Phases that are legitimately low-util but should NOT be treated as idle by
# policy engines — killing a training job during a checkpoint loses work.
NON_IDLE_LOW_UTIL_PHASES = frozenset({PHASE_CHECKPOINT, PHASE_DATA_LOADING, PHASE_EVAL})


class TrainingPhaseDetector:
    def __init__(
        self,
        idle_util_pct: float = 5.0,
        low_util_pct: float = 15.0,
        high_util_pct: float = 70.0,
        very_high_util_pct: float = 85.0,
        idle_power_floor_w: float = 50.0,
        min_samples: int = 3,
    ):
        self.idle_util_pct = idle_util_pct
        self.low_util_pct = low_util_pct
        self.high_util_pct = high_util_pct
        self.very_high_util_pct = very_high_util_pct
        self.idle_power_floor_w = idle_power_floor_w
        self.min_samples = min_samples

    def classify(self, samples: Iterable) -> str:
        """
        Classify the phase represented by the given sample window.

        `samples` should be an iterable of GpuSample (or any object exposing
        the same gpu_utilization / memory_utilization / power_usage_w
        attributes). The window should already be time-sorted; we look at
        the average plus the trend between the first and last sample.
        """
        sample_list = list(samples)
        if len(sample_list) < self.min_samples:
            return PHASE_UNKNOWN

        utils = [s.gpu_utilization for s in sample_list]
        mem_utils = [s.memory_utilization for s in sample_list]
        powers = [s.power_usage_w for s in sample_list if s.power_usage_w is not None]

        avg_util = sum(utils) / len(utils)
        avg_mem_util = sum(mem_utils) / len(mem_utils)
        avg_power = sum(powers) / len(powers) if powers else None

        # True idle: util near zero, power at baseline.
        if avg_util < self.idle_util_pct and (avg_power is None or avg_power <= self.idle_power_floor_w):
            return PHASE_IDLE

        # Very-high util with rising memory = backward pass (gradient buffers
        # inflating). Ordered before forward so the more specific rule wins.
        if avg_util >= self.very_high_util_pct and self._memory_rising(mem_utils):
            return PHASE_BACKWARD

        # High util, steady memory = forward pass.
        if avg_util >= self.high_util_pct:
            return PHASE_FORWARD

        # Low util BUT power above idle floor → we're doing something that
        # isn't streaming compute. Differentiate:
        #   - Stable memory → checkpoint (writing weights to disk).
        #   - Rising memory → data-loading (moving batches into memory).
        if avg_util < self.low_util_pct and avg_power is not None and avg_power > self.idle_power_floor_w:
            if self._memory_rising(mem_utils):
                return PHASE_DATA_LOADING
            return PHASE_CHECKPOINT

        # Moderate util (30-70%) with steady memory → eval loop.
        if self.low_util_pct <= avg_util < self.high_util_pct:
            if self._memory_stable(mem_utils):
                return PHASE_EVAL
            return PHASE_DATA_LOADING

        return PHASE_UNKNOWN

    @staticmethod
    def _memory_rising(mem_utils) -> bool:
        if len(mem_utils) < 2:
            return False
        first_half = mem_utils[: len(mem_utils) // 2]
        second_half = mem_utils[len(mem_utils) // 2 :]
        if not first_half or not second_half:
            return False
        return (sum(second_half) / len(second_half)) - (sum(first_half) / len(first_half)) > 3.0

    @staticmethod
    def _memory_stable(mem_utils) -> bool:
        if len(mem_utils) < 2:
            return True
        first_half = mem_utils[: len(mem_utils) // 2]
        second_half = mem_utils[len(mem_utils) // 2 :]
        if not first_half or not second_half:
            return True
        delta = abs((sum(second_half) / len(second_half)) - (sum(first_half) / len(first_half)))
        return delta < 3.0


def is_low_util_phase_non_idle(phase: Optional[str]) -> bool:
    """Helper used by the NvmlSampler + backend PolicyEngine to distinguish
    legitimate low-util phases (checkpoint, data loading, eval) from actual
    abandoned compute."""
    return phase in NON_IDLE_LOW_UTIL_PHASES
