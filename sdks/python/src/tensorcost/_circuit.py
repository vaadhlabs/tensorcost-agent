"""Circuit breaker for the fail-open proxy fallback.

Tracks consecutive proxy failures.  When the failure count reaches
``open_threshold``, the circuit opens and subsequent requests go directly to
the provider instead of through the proxy.  The circuit closes again after
``close_threshold`` consecutive successful probe calls.

State machine::

    CLOSED ──(failures >= open_threshold)──> OPEN
    OPEN   ──(single-flight probe)─────────> HALF_OPEN
    HALF_OPEN ──(proxy 2xx)────────────────> probe_successes++
               ──(proxy 5xx/err)───────────> back to OPEN
    HALF_OPEN ──(probe_successes >= close_threshold)──> CLOSED

This is intentionally simple — the circuit lives on the SDK client instance.
A per-circuit lock makes HALF_OPEN single-flight safe under threaded servers
(gunicorn sync workers, ``ThreadPoolExecutor`` in the same process).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Callable, Literal, Optional


CircuitState = Literal["CLOSED", "OPEN", "HALF_OPEN"]


@dataclass(frozen=True)
class CircuitBreakerConfig:
    """Configuration for the proxy circuit breaker."""

    #: Consecutive proxy failures before the circuit opens.
    open_threshold: int = 3
    #: Consecutive probe successes before the circuit closes again.
    close_threshold: int = 5


DEFAULT_CIRCUIT_CONFIG = CircuitBreakerConfig()


CircuitTransitionListener = Callable[[CircuitState, CircuitState], None]


class CircuitBreaker:
    """Simple synchronous circuit breaker."""

    def __init__(
        self,
        config: CircuitBreakerConfig = DEFAULT_CIRCUIT_CONFIG,
        on_transition: Optional[CircuitTransitionListener] = None,
    ) -> None:
        self.config = config
        self._on_transition = on_transition
        self._state: CircuitState = "CLOSED"
        self._consecutive_failures: int = 0
        self._probe_successes: int = 0
        self._probe_in_flight: bool = False
        self._lock = threading.Lock()

    def _transition(self, to: CircuitState) -> None:
        if to == self._state:
            return
        prev = self._state
        self._state = to
        if self._on_transition is not None:
            self._on_transition(prev, to)

    @property
    def state(self) -> CircuitState:
        return self._state

    @property
    def failures(self) -> int:
        return self._consecutive_failures

    def should_bypass(self) -> bool:
        """Return True when calls should skip the proxy entirely."""
        with self._lock:
            if self._state == "OPEN":
                return True
            if self._state == "HALF_OPEN" and self._probe_in_flight:
                return True
            return False

    def should_probe(self) -> bool:
        """Return True when this caller should run the single-flight probe."""
        with self._lock:
            if self._state == "OPEN" and not self._probe_in_flight:
                self._probe_in_flight = True
                self._transition("HALF_OPEN")
                return True
            return False

    def release_stuck_probe(self) -> None:
        """Clear a HALF_OPEN probe that never reached record_success/failure."""
        with self._lock:
            if self._state == "HALF_OPEN" and self._probe_in_flight:
                self._probe_successes = 0
                self._probe_in_flight = False
                self._transition("OPEN")

    def record_success(self) -> None:
        """Call after a proxy request succeeds."""
        with self._lock:
            if self._state == "HALF_OPEN":
                self._probe_in_flight = False
                self._probe_successes += 1
                if self._probe_successes >= self.config.close_threshold:
                    self._reset_unlocked()
            elif self._state == "CLOSED":
                self._consecutive_failures = 0

    def record_failure(self) -> None:
        """Call after a proxy request fails (5xx or network error)."""
        with self._lock:
            if self._state == "HALF_OPEN":
                self._probe_successes = 0
                self._probe_in_flight = False
                self._transition("OPEN")
            elif self._state == "CLOSED":
                self._consecutive_failures += 1
                if self._consecutive_failures >= self.config.open_threshold:
                    self._transition("OPEN")
            # OPEN + another failure: stay open.

    def _reset_unlocked(self) -> None:
        self._consecutive_failures = 0
        self._probe_successes = 0
        self._probe_in_flight = False
        self._transition("CLOSED")
