"""Tests for the circuit breaker."""

from __future__ import annotations

import pytest

from tensorcost._circuit import (
    CircuitBreaker,
    CircuitBreakerConfig,
    DEFAULT_CIRCUIT_CONFIG,
)


class TestCircuitBreakerConfig:
    def test_defaults(self):
        assert DEFAULT_CIRCUIT_CONFIG.open_threshold == 3
        assert DEFAULT_CIRCUIT_CONFIG.close_threshold == 5

    def test_custom(self):
        cfg = CircuitBreakerConfig(open_threshold=2, close_threshold=3)
        assert cfg.open_threshold == 2
        assert cfg.close_threshold == 3

    def test_is_frozen(self):
        cfg = CircuitBreakerConfig()
        with pytest.raises(Exception):
            cfg.open_threshold = 99  # type: ignore[misc]


class TestCircuitBreakerInitial:
    def test_starts_closed(self):
        cb = CircuitBreaker()
        assert cb.state == "CLOSED"

    def test_should_not_bypass_when_closed(self):
        cb = CircuitBreaker()
        assert cb.should_bypass() is False

    def test_failures_count_zero(self):
        cb = CircuitBreaker()
        assert cb.failures == 0


class TestCircuitBreakerClosedToOpen:
    def test_opens_after_threshold_failures(self):
        cb = CircuitBreaker(CircuitBreakerConfig(open_threshold=3))
        for _ in range(2):
            cb.record_failure()
            assert cb.state == "CLOSED"
        cb.record_failure()
        assert cb.state == "OPEN"

    def test_should_bypass_when_open(self):
        cb = CircuitBreaker(CircuitBreakerConfig(open_threshold=2))
        cb.record_failure()
        cb.record_failure()
        assert cb.should_bypass() is True

    def test_success_in_closed_resets_failure_count(self):
        cb = CircuitBreaker(CircuitBreakerConfig(open_threshold=3))
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        cb.record_failure()
        # Only 1 failure after the reset, still closed.
        assert cb.state == "CLOSED"


class TestCircuitBreakerOpenToHalfOpen:
    def test_should_probe_transitions_to_half_open(self):
        cb = CircuitBreaker(CircuitBreakerConfig(open_threshold=1))
        cb.record_failure()
        assert cb.state == "OPEN"
        result = cb.should_probe()
        assert result is True
        assert cb.state == "HALF_OPEN"

    def test_should_probe_returns_false_when_closed(self):
        cb = CircuitBreaker()
        assert cb.should_probe() is False
        assert cb.state == "CLOSED"

    def test_probe_failure_goes_back_to_open(self):
        cb = CircuitBreaker(CircuitBreakerConfig(open_threshold=1))
        cb.record_failure()
        cb.should_probe()  # transitions to HALF_OPEN
        cb.record_failure()
        assert cb.state == "OPEN"


class TestCircuitBreakerHalfOpenToClose:
    def test_closes_after_close_threshold_probe_successes(self):
        cfg = CircuitBreakerConfig(open_threshold=1, close_threshold=3)
        cb = CircuitBreaker(cfg)
        cb.record_failure()
        cb.should_probe()  # OPEN → HALF_OPEN
        for i in range(2):
            cb.record_success()
            assert cb.state == "HALF_OPEN", f"should still be HALF_OPEN after {i+1} success"
        cb.record_success()
        assert cb.state == "CLOSED"

    def test_on_transition_fires_once_per_change(self):
        seen: list[str] = []

        def listener(frm: str, to: str) -> None:
            seen.append(f"{frm}->{to}")

        cb = CircuitBreaker(
            CircuitBreakerConfig(open_threshold=1, close_threshold=1),
            on_transition=listener,
        )
        cb.record_failure()
        cb.should_probe()
        cb.record_success()
        assert seen == ["CLOSED->OPEN", "OPEN->HALF_OPEN", "HALF_OPEN->CLOSED"]

    def test_release_stuck_probe_returns_to_open(self):
        cb = CircuitBreaker(CircuitBreakerConfig(open_threshold=1, close_threshold=3))
        cb.record_failure()
        cb.should_probe()
        assert cb.state == "HALF_OPEN"
        cb.release_stuck_probe()
        assert cb.state == "OPEN"

    def test_after_close_failures_count_reset(self):
        cfg = CircuitBreakerConfig(open_threshold=1, close_threshold=2)
        cb = CircuitBreaker(cfg)
        cb.record_failure()
        cb.should_probe()
        cb.record_success()
        cb.record_success()
        assert cb.state == "CLOSED"
        assert cb.failures == 0
