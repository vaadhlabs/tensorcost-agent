"""Tests for retry utilities."""

from __future__ import annotations

import math

import pytest

from tensorcost._retry import (
    DEFAULT_RETRY_CONFIG,
    RetryConfig,
    compute_delay,
    is_retryable_status,
    parse_retry_after_ms,
    sleep_sync,
)


class TestRetryConfig:
    def test_default_values(self):
        cfg = DEFAULT_RETRY_CONFIG
        assert cfg.max_attempts == 5
        assert cfg.base_delay_ms == 500.0
        assert cfg.max_delay_ms == 30_000.0

    def test_custom_values(self):
        cfg = RetryConfig(max_attempts=3, base_delay_ms=1000.0, max_delay_ms=5000.0)
        assert cfg.max_attempts == 3
        assert cfg.base_delay_ms == 1000.0
        assert cfg.max_delay_ms == 5000.0

    def test_is_frozen(self):
        cfg = RetryConfig()
        with pytest.raises(Exception):
            cfg.max_attempts = 99  # type: ignore[misc]


class TestComputeDelay:
    def test_delay_within_range(self):
        cfg = RetryConfig(base_delay_ms=500.0, max_delay_ms=30_000.0)
        delay = compute_delay(0, cfg)
        # Jitter brings it to [0.5 * 500, 1.0 * 500] = [250, 500]
        assert 250.0 <= delay <= 500.0

    def test_delay_capped_at_max(self):
        cfg = RetryConfig(base_delay_ms=500.0, max_delay_ms=1000.0)
        # attempt=10: 500 * 2^10 = 512_000 >> 1000
        delay = compute_delay(10, cfg)
        assert delay <= 1000.0

    def test_exponential_growth(self):
        cfg = RetryConfig(base_delay_ms=100.0, max_delay_ms=100_000.0)
        # Max delay for attempt 0 is 100 ms; for attempt 3 is min(100*8, 100k) = 800 ms
        d0_max = 100.0
        d3_max = 800.0
        for _ in range(30):
            d0 = compute_delay(0, cfg)
            d3 = compute_delay(3, cfg)
            assert d0 <= d0_max
            assert d3 <= d3_max

    def test_retry_after_overrides_backoff(self):
        cfg = RetryConfig(base_delay_ms=500.0, max_delay_ms=30_000.0)
        delay = compute_delay(0, cfg, retry_after_ms=12_000.0)
        assert delay == 12_000.0

    def test_retry_after_still_capped(self):
        cfg = RetryConfig(base_delay_ms=500.0, max_delay_ms=5000.0)
        delay = compute_delay(0, cfg, retry_after_ms=99_000.0)
        assert delay == 5000.0

    def test_retry_after_zero_allowed(self):
        cfg = RetryConfig(base_delay_ms=500.0, max_delay_ms=30_000.0)
        delay = compute_delay(0, cfg, retry_after_ms=0.0)
        assert delay == 0.0


class TestIsRetryableStatus:
    def test_5xx_are_retryable(self):
        for status in (500, 502, 503, 504, 599):
            assert is_retryable_status(status) is True

    def test_429_is_retryable(self):
        assert is_retryable_status(429) is True

    def test_4xx_not_retryable(self):
        for status in (400, 401, 403, 404, 422):
            assert is_retryable_status(status) is False

    def test_2xx_not_retryable(self):
        for status in (200, 201, 204):
            assert is_retryable_status(status) is False


class TestParseRetryAfterMs:
    def test_none_returns_none(self):
        assert parse_retry_after_ms(None) is None

    def test_empty_returns_none(self):
        assert parse_retry_after_ms("") is None

    def test_integer_seconds(self):
        result = parse_retry_after_ms("5")
        assert result == 5000

    def test_float_seconds_ceiling(self):
        # math.ceil(1.5 * 1000) = 1500
        result = parse_retry_after_ms("1.5")
        assert result == 1500

    def test_zero_seconds(self):
        result = parse_retry_after_ms("0")
        assert result == 0

    def test_garbage_returns_none(self):
        result = parse_retry_after_ms("not-a-date-or-number")
        assert result is None

    def test_http_date_parses(self):
        # A valid RFC 2822 date 60s in the future should return ~60_000 ms.
        import datetime
        future = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(seconds=60)
        # Format as RFC 2822
        from email.utils import format_datetime
        header = format_datetime(future)
        result = parse_retry_after_ms(header)
        # Allow ±5 s slack for test execution time.
        assert result is not None
        assert 55_000 <= result <= 65_000


class TestSleepSync:
    def test_short_sleep_completes(self):
        # Just check it doesn't error; don't verify wall-clock timing.
        sleep_sync(1.0)  # 1 ms
