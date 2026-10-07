"""Tests for TrainingPhaseDetector (spec 3B)."""

from collections import namedtuple

import pytest

from src.training_phase_detector import (
    TrainingPhaseDetector,
    is_low_util_phase_non_idle,
    PHASE_IDLE,
    PHASE_DATA_LOADING,
    PHASE_FORWARD,
    PHASE_BACKWARD,
    PHASE_CHECKPOINT,
    PHASE_EVAL,
    PHASE_UNKNOWN,
)

Sample = namedtuple('Sample', 'gpu_utilization memory_utilization power_usage_w')


def samples(util_seq, mem_seq=None, power_seq=None):
    if mem_seq is None:
        mem_seq = [u * 0.6 for u in util_seq]
    if power_seq is None:
        power_seq = [80.0 for _ in util_seq]
    return [Sample(u, m, p) for u, m, p in zip(util_seq, mem_seq, power_seq)]


@pytest.mark.unit
class TestTrainingPhaseDetector:
    def test_returns_unknown_for_sparse_input(self):
        d = TrainingPhaseDetector()
        assert d.classify(samples([5])) == PHASE_UNKNOWN

    def test_detects_idle_when_util_and_power_low(self):
        d = TrainingPhaseDetector()
        phase = d.classify(samples([1, 2, 1, 2], mem_seq=[5, 5, 5, 5], power_seq=[30, 30, 30, 30]))
        assert phase == PHASE_IDLE

    def test_detects_forward_pass_on_sustained_high_util(self):
        d = TrainingPhaseDetector()
        phase = d.classify(samples([75, 76, 74, 77], mem_seq=[40, 40, 40, 40], power_seq=[250, 250, 250, 250]))
        assert phase == PHASE_FORWARD

    def test_detects_backward_pass_on_very_high_util_rising_memory(self):
        d = TrainingPhaseDetector()
        phase = d.classify(samples(
            [90, 92, 93, 95],
            mem_seq=[50, 53, 58, 63],  # rising
            power_seq=[300, 310, 320, 320],
        ))
        assert phase == PHASE_BACKWARD

    def test_detects_checkpoint_on_low_util_but_sustained_power(self):
        d = TrainingPhaseDetector()
        phase = d.classify(samples(
            [3, 5, 6, 4],
            mem_seq=[70, 70, 70, 70],  # stable
            power_seq=[120, 120, 120, 120],  # above idle floor
        ))
        assert phase == PHASE_CHECKPOINT

    def test_detects_data_loading_on_low_util_rising_memory(self):
        d = TrainingPhaseDetector()
        phase = d.classify(samples(
            [3, 4, 5, 6],
            mem_seq=[20, 25, 30, 35],  # rising fast
            power_seq=[120, 120, 120, 120],
        ))
        assert phase == PHASE_DATA_LOADING

    def test_detects_eval_on_moderate_util_stable_memory(self):
        d = TrainingPhaseDetector()
        phase = d.classify(samples(
            [45, 45, 46, 45],
            mem_seq=[40, 40, 40, 40],  # stable
            power_seq=[180, 180, 180, 180],
        ))
        assert phase == PHASE_EVAL

    def test_low_util_phase_helper_flags_checkpoint_and_eval(self):
        assert is_low_util_phase_non_idle(PHASE_CHECKPOINT) is True
        assert is_low_util_phase_non_idle(PHASE_DATA_LOADING) is True
        assert is_low_util_phase_non_idle(PHASE_EVAL) is True
        assert is_low_util_phase_non_idle(PHASE_IDLE) is False
        assert is_low_util_phase_non_idle(PHASE_FORWARD) is False
        assert is_low_util_phase_non_idle(None) is False
