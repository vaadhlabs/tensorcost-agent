"""Retry utilities for the TensorCost applied-mode request path.

Retries use jittered exponential backoff::

    delay = min(max_delay_ms, base_delay_ms * 2^attempt) * (0.5 + random() * 0.5)

where ``attempt`` is 0-indexed.  With defaults (base=500 ms, max=30 000 ms),
the sequence of *maximum* delays is 500 → 1 000 → 2 000 → 4 000 → 8 000 ms.

Rules:

* Never retry 4xx responses (except 429).
* Always retry 5xx responses and network errors.
* On 429: honour the ``Retry-After`` response header when present.
* Chat/completions calls are idempotent at the proxy level (stateless proxy).
"""

from __future__ import annotations

import asyncio
import math
import random
import time
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class RetryConfig:
    """Retry policy for applied-mode proxy requests."""

    #: Maximum number of total attempts (1 = no retries).
    max_attempts: int = 5
    #: Base backoff interval in milliseconds.
    base_delay_ms: float = 500.0
    #: Cap on any single backoff interval.
    max_delay_ms: float = 30_000.0


DEFAULT_RETRY_CONFIG = RetryConfig()


def compute_delay(
    attempt: int,
    config: RetryConfig,
    retry_after_ms: Optional[float] = None,
) -> float:
    """Return the delay in milliseconds before the next attempt.

    *attempt* is 0-indexed (0 = before the second try).
    When *retry_after_ms* is provided (from a 429 Retry-After header) it
    overrides the exponential calculation but is still capped at
    ``config.max_delay_ms``.
    """
    if retry_after_ms is not None and retry_after_ms >= 0:
        return min(retry_after_ms, config.max_delay_ms)
    exponential = config.base_delay_ms * math.pow(2, attempt)
    capped = min(exponential, config.max_delay_ms)
    # Jitter: uniform in [0.5 * capped, 1.0 * capped]
    return capped * (0.5 + random.random() * 0.5)


def is_retryable_status(status: int) -> bool:
    """Return True for HTTP status codes the SDK should retry on."""
    return status == 429 or status >= 500


def parse_retry_after_ms(retry_after: Optional[str]) -> Optional[float]:
    """Parse a Retry-After header value to milliseconds.

    Accepts numeric seconds (including fractions) or an HTTP-date string.
    Returns None when the header is absent or unparseable.
    """
    if not retry_after:
        return None
    try:
        secs = float(retry_after)
        return math.ceil(secs * 1000)
    except ValueError:
        pass
    # Attempt HTTP-date parsing.
    from email.utils import parsedate_to_datetime
    try:
        import datetime
        dt = parsedate_to_datetime(retry_after)
        delta_ms = (dt - datetime.datetime.now(datetime.timezone.utc)).total_seconds() * 1000
        return max(0.0, delta_ms)
    except Exception:  # noqa: BLE001
        return None


def sleep_sync(ms: float) -> None:
    """Blocking sleep for *ms* milliseconds."""
    time.sleep(ms / 1000.0)


async def sleep_async(ms: float) -> None:
    """Async sleep for *ms* milliseconds."""
    await asyncio.sleep(ms / 1000.0)
