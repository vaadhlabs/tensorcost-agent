"""Fire-and-forget HTTP transport for observation events.

Design (matches the Node SDK's transport.ts):

* ``post()`` serialises the observation once and puts it in a bounded
  in-memory buffer; a small pool of daemon worker threads sends it. The
  customer's call returns the moment the provider responds.
* A delivery that fails for a reason worth retrying (network error,
  timeout, 408, 429, 5xx, or a token exchange that hit one of those) goes
  back into the buffer and is resent later with the same bytes, so the same
  correlation_id; ingestion dedupes on it, so retrying a POST that actually
  landed is harmless.
* Backoff is transport-wide: the first failure blocks sending for 1 s, each
  failed probe doubles that up to 60 s (with jitter), and while blocked
  only one request is in flight, so a recovering endpoint sees one probe
  rather than the whole buffer. A 429's Retry-After sets the wait instead.
  The first success clears it.
* 400/401/403 (and any other 4xx but 408 and 429) are not retried: the
  same body would get the same answer. They warn once, then at most once a
  minute with a count, and are counted in ``stats().rejected``.
* ``stats()`` counts what is buffered, what was delivered only after a
  retry, what was dropped and what was rejected. When delivery recovers
  after an outage that is logged once, with how many are waiting.
* The buffer is capped by count and approximate bytes. When it is full the
  oldest observation goes (to the disk spool if one is configured,
  otherwise it is counted in ``dropped_count``), with at most one warning a
  minute. An observation still undelivered after ``max_retry_age_s`` (an
  hour) is given up on and counted too.
* A short-lived JWT is fetched via
  ``POST /api/inference-proxy/sdk-token/exchange`` using the long-lived API
  key, cached in memory and refreshed when within 60 s of expiry.
* Exit draining is opt-out and lives in ``_exit.py``.
* All errors are swallowed and logged via ``logging.warning``.
"""

from __future__ import annotations

import json
import logging
import os
import random as _random
import re
import threading
import time
import weakref
from collections import deque
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from concurrent.futures import Future
from concurrent.futures import TimeoutError as _FutureTimeout
from typing import Callable, Deque, List, Optional, Set, Tuple

import httpx

from ._config import MissingConfigError
from ._errors import (
    format_seconds,
    TensorCostKeyRejectedError,
    TensorCostNetworkError,
    TensorCostProxyError,
    TensorCostQuotaError,
    TensorCostTimeoutError,
)
from ._exit import register_exit_hooks
from ._observation_limits import bound_observation
from ._spool import Spool
from ._version import SDK_VERSION as _USER_AGENT

logger = logging.getLogger("tensorcost")

# Token is treated as expired this many seconds before its real expiry.
_TOKEN_REFRESH_LEEWAY_S = 60.0
# Default lifetime to assume if the server response doesn't carry one.
_DEFAULT_TOKEN_LIFETIME_S = 15 * 60.0

DEFAULT_MAX_BUFFERED_OBSERVATIONS = 10_000
DEFAULT_MAX_BUFFERED_BYTES = 32 * 1024 * 1024
DEFAULT_MAX_RETRY_AGE_S = 60 * 60.0
_WARN_INTERVAL_S = 60.0
# Lines waiting for a worker to write them to the spool before new spills
# are refused (and counted as dropped).
_MAX_PENDING_SPOOL = 10_000

_Outcome = Tuple  # ("ok",) | ("retry", detail, retry_after_s) | ("fatal", status, exc)


@dataclass(frozen=True)
class TransportStats:
    """Delivery counters. ``dropped`` is what was lost for good: buffer
    overflow without a spool, expiry, exit, spool overflow."""

    in_flight: int
    buffered: int
    buffered_bytes: int
    dropped: int
    spooled: int
    #: Held back by a failed delivery (their own or the probe's), then delivered.
    delivered_after_retry: int = 0
    #: Answered with a 4xx other than 408/429; never retried.
    rejected: int = 0


class _Entry:
    __slots__ = ("body", "first_at", "held")

    def __init__(self, body: bytes, first_at: float) -> None:
        self.body = body
        self.first_at = first_at
        # Held back by a failure: its own attempt failed, or it waited while
        # delivery was failing. Counted in delivered_after_retry when it lands.
        self.held = False


class ObservationTransport:
    """Thread-safe fire-and-forget poster for observation events."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        *,
        tenant_id: Optional[str] = None,
        team_id: Optional[str] = None,
        timeout_s: float = 2.0,
        fail_open: bool = True,
        max_workers: int = 4,
        http_client: Optional[httpx.Client] = None,
        max_buffered_observations: Optional[int] = None,
        max_buffered_bytes: Optional[int] = None,
        max_retry_age_s: Optional[float] = None,
        flush_on_exit: bool = False,
        spool_dir: Optional[str] = None,
        spool_max_bytes: Optional[int] = None,
        spool_segment_bytes: Optional[int] = None,
        backoff_base_s: float = 1.0,
        backoff_max_s: float = 60.0,
        random: Optional[Callable[[], float]] = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._tenant_id = tenant_id
        # Sent as x-tc-team-id on every observation POST: the ingest route
        # reads the team from the token binding, then that header, and has
        # no body field for it (observation-team.ts).
        self._team_id = team_id.strip() if team_id and team_id.strip() else None
        self._timeout_s = timeout_s
        self._fail_open = fail_open
        self._max_workers = max(1, max_workers)
        self._max_buffered = max(1, max_buffered_observations or DEFAULT_MAX_BUFFERED_OBSERVATIONS)
        self._max_bytes = max(1, max_buffered_bytes or DEFAULT_MAX_BUFFERED_BYTES)
        self._max_age_s = max_retry_age_s if max_retry_age_s is not None else DEFAULT_MAX_RETRY_AGE_S
        self._flush_on_exit = flush_on_exit
        self._backoff_base_s = backoff_base_s
        self._backoff_max_s = backoff_max_s
        self._random = random or _random.random

        self._client = http_client or httpx.Client(timeout=timeout_s)

        self._reset_state()
        self._dropped = 0
        self._spooled = 0
        self._drops_since_warn = 0
        self._last_drop_warn = float("-inf")
        self._last_fail_warn = float("-inf")
        self._delivered_after_retry = 0
        self._rejected = 0
        self._rejected_since_warn = 0
        self._last_reject_warn = float("-inf")
        self._closed = False

        # Short-lived JWT cache.
        self._token: Optional[str] = None
        self._token_expires_at: float = 0.0

        self._spool: Optional[Spool] = None
        if spool_dir:
            self._spool = Spool(
                spool_dir,
                max_bytes=spool_max_bytes,
                segment_bytes=spool_segment_bytes,
                on_lost=self._lose,
            )
        if flush_on_exit or self._spool is not None:
            register_exit_hooks(self)

        if hasattr(os, "register_at_fork"):
            ref = weakref.ref(self)

            def _child() -> None:
                t = ref()
                if t is not None:
                    t._reset_state()

            os.register_at_fork(after_in_child=_child)

        self._replay_done = threading.Event()
        if self._spool is not None and self._tenant_id:
            threading.Thread(
                target=self._replay_spool, name="tensorcost-spool-replay", daemon=True
            ).start()
        else:
            self._replay_done.set()

    def _reset_state(self) -> None:
        """Fresh locks, buffer and workers. Also run in a forked child: the
        parent keeps delivering its own copy of the buffer, and its threads
        and lock holders did not come along."""
        self._cond = threading.Condition()
        self._token_lock = threading.Lock()
        # The one token exchange a bounded get_token() waits on, and when it
        # started (see get_token).
        self._exchange: Optional[Tuple[Future, float]] = None
        self._exchange_lock = threading.Lock()
        self._queue: Deque[_Entry] = deque()
        self._queue_bytes = 0
        self._in_flight: Set[_Entry] = set()
        self._failures = 0
        self._blocked_until = 0.0
        self._dormant = False
        self._to_spool: List[bytes] = []
        self._spool_busy = False
        self._workers: List[threading.Thread] = []
        self._worker_pid: Optional[int] = None
        spool = getattr(self, "_spool", None)
        if spool is not None:
            spool._after_fork()

    # ------------------------------------------------------------------
    # Public API

    def post(self, observation: dict) -> None:
        """Enqueue an observation for fire-and-forget delivery.

        Returns immediately. Errors are logged, never raised (when
        ``fail_open`` is True, which is the default).
        """
        # No tenant id, no token, so nothing can be delivered. wrap() has said
        # so once already; failing every POST would only repeat it per call.
        if not self._tenant_id and self._fail_open:
            return
        try:
            # A long provider error must not cost the call its row
            # (_observation_limits.py).
            body = json.dumps(
                bound_observation(observation), ensure_ascii=False, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            logger.warning("tensorcost: observation could not be serialised: %s", exc)
            return
        with self._cond:
            if self._closed:
                return
            self._dormant = False
            try:
                self._ensure_workers_locked()
            except RuntimeError:
                return  # interpreter shutting down
            self._enqueue_locked(self._entry(body, time.monotonic()))
            self._cond.notify()

    @property
    def dropped_count(self) -> int:
        """Observations lost for good since this transport started."""
        return self._dropped

    @property
    def dropped_observations(self) -> int:
        """Same as ``dropped_count`` (the 1.8.0 name)."""
        return self._dropped

    @property
    def pending_count(self) -> int:
        """Observations not yet delivered: in flight plus buffered."""
        with self._cond:
            return len(self._in_flight) + len(self._queue)

    @property
    def queued_observations(self) -> int:
        """Same as ``pending_count`` (the 1.8.0 name)."""
        return self.pending_count

    def stats(self) -> TransportStats:
        with self._cond:
            return TransportStats(
                in_flight=len(self._in_flight),
                buffered=len(self._queue),
                buffered_bytes=self._queue_bytes,
                dropped=self._dropped,
                spooled=self._spooled,
                delivered_after_retry=self._delivered_after_retry,
                rejected=self._rejected,
            )

    def flush(self, timeout_s: float = 5.0, *, _for_exit: bool = False) -> int:
        """Try to deliver everything queued or waiting to retry (Lambda
        shutdown). A pending backoff is cut short so one probe goes now.
        Returns when everything has landed, when the endpoint is still
        failing and the next retry would fall past the deadline, or at the
        deadline. What is left stays buffered and keeps retrying. Returns how
        many observations are still undelivered (0 when all landed)."""
        deadline = time.monotonic() + timeout_s
        with self._cond:
            try:
                if self._idle_locked():
                    return 0
                self._dormant = False
                self._ensure_workers_locked()
                if not self._in_flight and self._blocked_until > time.monotonic():
                    self._blocked_until = 0.0  # probe now rather than wait out the backoff
                self._cond.notify_all()
                while not self._idle_locked():
                    now = time.monotonic()
                    if now >= deadline:
                        break
                    if not self._in_flight and self._blocked_until >= deadline:
                        break
                    self._cond.wait(min(0.05, deadline - now))
            except RuntimeError:
                pass  # no new threads at interpreter shutdown
            finally:
                if _for_exit:
                    self._dormant = True
            return len(self._in_flight) + len(self._queue)

    def close(self) -> None:
        """Best-effort flush, then stop the workers. Anything left goes to
        the spool if there is one and is counted as dropped otherwise."""
        self.flush(timeout_s=5.0)
        with self._cond:
            self._closed = True
            self._cond.notify_all()
            workers = list(self._workers)
        for th in workers:
            th.join(timeout=0.5)
        self._spill_remaining()
        try:
            self._client.close()
        except Exception:  # noqa: BLE001
            pass

    def spool_idle(self, timeout_s: float = 5.0) -> None:
        """Test hook: wait until queued spool writes are on disk."""
        deadline = time.monotonic() + timeout_s
        with self._cond:
            while (self._to_spool or self._spool_busy) and time.monotonic() < deadline:
                self._cond.wait(0.05)

    def replay_done(self, timeout_s: float = 5.0) -> None:
        """Test hook: wait until the start-up spool replay has been read in."""
        self._replay_done.wait(timeout_s)

    # ------------------------------------------------------------------
    # Exit hook (_exit.py)

    def _spill_remaining(self) -> None:
        with self._cond:
            left = list(self._in_flight) + list(self._queue)
            self._in_flight.clear()
            self._queue.clear()
            self._queue_bytes = 0
            self._dormant = True
            pending = self._to_spool
            self._to_spool = []
            self._cond.notify_all()
        if self._spool is not None:
            self._spool.append(pending + [e.body for e in left], seal=True)
            with self._cond:
                self._spooled += len(left)
            return
        if left:
            with self._cond:
                self._dropped += len(left)
            logger.warning(
                "tensorcost: %d observation(s) were not delivered before exit "
                "(set spool_dir to keep them on disk for the next start)",
                len(left),
            )

    # ------------------------------------------------------------------
    # Buffer (all *_locked methods run with self._cond held)

    @staticmethod
    def _entry(body: bytes, first_at: float) -> _Entry:
        return _Entry(body, first_at)

    def _idle_locked(self) -> bool:
        return not (self._queue or self._in_flight or self._to_spool or self._spool_busy)

    def _ensure_workers_locked(self) -> None:
        pid = os.getpid()
        if self._worker_pid == pid and self._workers:
            return
        self._worker_pid = pid
        self._workers = []
        for i in range(self._max_workers):
            th = threading.Thread(
                target=self._worker_loop, name=f"tensorcost-obs-{i}", daemon=True
            )
            th.start()
            self._workers.append(th)

    def _enqueue_locked(self, entry: _Entry) -> None:
        # Retries go back in first-posted order; new ones land at the end.
        q = self._queue
        i = len(q)
        while i > 0 and q[i - 1].first_at > entry.first_at:
            i -= 1
        q.insert(i, entry)
        self._queue_bytes += len(entry.body)
        if self._failures:
            entry.held = True
        while q and (len(q) > self._max_buffered or self._queue_bytes > self._max_bytes):
            oldest = q.popleft()
            self._queue_bytes -= len(oldest.body)
            self._overflow_locked(oldest)

    def _overflow_locked(self, entry: _Entry) -> None:
        if self._spool is not None and len(self._to_spool) < _MAX_PENDING_SPOOL:
            self._to_spool.append(entry.body)
            self._spooled += 1
            self._drops_since_warn += 1
            self._warn_drop_locked("spooled to disk")
            self._cond.notify()
            return
        self._dropped += 1
        self._drops_since_warn += 1
        self._warn_drop_locked("dropped")

    def _lose(self, count: int, reason: str) -> None:
        with self._cond:
            self._dropped += count
            self._drops_since_warn += count
            self._warn_drop_locked(f"dropped ({reason})")

    def _warn_drop_locked(self, what: str) -> None:
        now = time.monotonic()
        if now - self._last_drop_warn < _WARN_INTERVAL_S:
            return
        self._last_drop_warn = now
        if what == "dropped":
            what = "dropped (buffer full)"
        logger.warning(
            "tensorcost: observation buffer full (%d waiting); %d oldest %s since "
            "the last warning, %d dropped in total",
            len(self._queue),
            self._drops_since_warn,
            what,
            self._dropped,
        )
        self._drops_since_warn = 0

    def _expire_locked(self, now: float) -> None:
        while self._queue and now - self._queue[0].first_at > self._max_age_s:
            stale = self._queue.popleft()
            self._queue_bytes -= len(stale.body)
            self._dropped += 1
            self._drops_since_warn += 1
            self._warn_drop_locked("dropped (undelivered after the retry window)")

    def _backoff_s(self, failures: int) -> float:
        ceiling = min(self._backoff_max_s, self._backoff_base_s * (2 ** (failures - 1)))
        return ceiling * (0.5 + self._random() * 0.5)

    # ------------------------------------------------------------------
    # Workers

    def _worker_loop(self) -> None:
        while True:
            entry: Optional[_Entry] = None
            batch: Optional[List[bytes]] = None
            with self._cond:
                while True:
                    if self._closed:
                        return
                    if self._to_spool and not self._spool_busy:
                        batch, self._to_spool = self._to_spool, []
                        self._spool_busy = True
                        break
                    now = time.monotonic()
                    self._expire_locked(now)
                    wait: Optional[float] = None
                    if self._queue and not self._dormant:
                        if now < self._blocked_until:
                            wait = self._blocked_until - now
                        elif len(self._in_flight) < (1 if self._failures else self._max_workers):
                            entry = self._queue.popleft()
                            self._queue_bytes -= len(entry.body)
                            self._in_flight.add(entry)
                            break
                    self._cond.wait(wait)
            if batch is not None:
                assert self._spool is not None
                self._spool.append(batch)
                with self._cond:
                    self._spool_busy = False
                    self._cond.notify_all()
                continue
            assert entry is not None
            outcome = self._attempt(entry)
            with self._cond:
                if entry in self._in_flight:  # not spilled at exit meanwhile
                    self._in_flight.discard(entry)
                    self._settle_locked(entry, outcome)
                self._cond.notify_all()

    def _settle_locked(self, entry: _Entry, outcome: _Outcome) -> None:
        now = time.monotonic()
        kind = outcome[0]
        if kind == "ok":
            if entry.held:
                self._delivered_after_retry += 1
            if self._failures:
                logger.warning(
                    "tensorcost: observation delivery recovered; sending %d buffered "
                    "observation(s) (%d delivered after a retry so far)",
                    len(self._queue),
                    self._delivered_after_retry,
                )
            self._failures = 0
            self._blocked_until = 0.0
        elif kind == "fatal":
            _, status, exc = outcome
            if status is not None:
                self._warn_rejected_locked(status)
            else:
                logger.warning("tensorcost: observation delivery failed: %s", exc)
        else:
            _, detail, retry_after = outcome
            if now - entry.first_at > self._max_age_s:
                self._dropped += 1
                self._drops_since_warn += 1
                self._warn_drop_locked("dropped (undelivered after the retry window)")
                return
            if retry_after is not None:
                self._failures = max(1, self._failures)
                self._blocked_until = max(self._blocked_until, now + retry_after)
            elif now >= self._blocked_until:
                # Only a request sent while we were not already backing off
                # escalates; the rest of an outage's first wave just requeues.
                self._failures += 1
                self._blocked_until = now + self._backoff_s(self._failures)
            entry.held = True
            for waiting in self._queue:
                waiting.held = True
            self._enqueue_locked(entry)
            if now - self._last_fail_warn >= _WARN_INTERVAL_S:
                self._last_fail_warn = now
                logger.warning(
                    "tensorcost: observation delivery failed (%s); %d buffered, retrying",
                    detail,
                    len(self._queue),
                )

    def _warn_rejected_locked(self, status: int) -> None:
        """A 4xx TensorCost won't change its mind about: warn on the first,
        then once a minute."""
        self._rejected += 1
        self._rejected_since_warn += 1
        now = time.monotonic()
        if now - self._last_reject_warn < _WARN_INTERVAL_S:
            return
        self._last_reject_warn = now
        logger.warning(
            "tensorcost: observation POST failed status=%s; not retried "
            "(%d rejected since the last warning, %d in total)",
            status,
            self._rejected_since_warn,
            self._rejected,
        )
        self._rejected_since_warn = 0

    def _attempt(self, entry: _Entry) -> _Outcome:
        try:
            token = self._get_token()
        except (
            TensorCostNetworkError,
            TensorCostTimeoutError,
            TensorCostProxyError,
            TensorCostQuotaError,
        ) as exc:
            return ("retry", str(exc), None)
        except Exception as exc:  # noqa: BLE001
            return ("fatal", None, exc)
        url = f"{self._base_url}/api/inference-proxy/observation"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "User-Agent": _USER_AGENT,
        }
        if self._team_id:
            headers["x-tc-team-id"] = self._team_id
        try:
            resp = self._client.post(url, content=entry.body, headers=headers)
        except httpx.TimeoutException as exc:
            return ("retry", f"timed out: {exc}", None)
        except (httpx.TransportError, OSError) as exc:
            return ("retry", str(exc) or type(exc).__name__, None)
        except Exception as exc:  # noqa: BLE001
            return ("fatal", None, exc)
        status = resp.status_code
        if status < 400:
            return ("ok",)
        if status in (408, 429) or status >= 500:
            retry_after = (
                _parse_retry_after(resp.headers.get("retry-after"), self._max_age_s)
                if status in (429, 503)
                else None
            )
            return ("retry", f"status={status}", retry_after)
        return ("fatal", status, None)

    def _replay_spool(self) -> None:
        assert self._spool is not None
        try:
            # A segment at a time while there is room, so a large spool does
            # not push the buffer straight back out to disk.
            while True:
                with self._cond:
                    if self._closed or len(self._queue) >= self._max_buffered / 2:
                        return
                lines = self._spool.claim_next()
                if lines is None:
                    return
                now = time.monotonic()
                with self._cond:
                    self._ensure_workers_locked()
                    for body in lines:
                        self._enqueue_locked(_Entry(body, now))
                    self._cond.notify_all()
        except Exception:  # noqa: BLE001 — replay is best effort
            pass
        finally:
            self._replay_done.set()

    # ------------------------------------------------------------------
    # Token

    def get_token(self, timeout_s: Optional[float] = None) -> str:
        """Return a valid short-lived JWT for proxy/admit/sdk-layer calls.

        With ``timeout_s`` the caller waits at most that long, and for no
        longer than ``timeout_s`` after the exchange in flight started: the
        exchange runs on its own thread, one at a time, and a caller that
        gives up leaves it to finish and fill the cache for the next call.
        Giving up raises ``TensorCostTimeoutError``, which the fail-open
        rule treats as TensorCost being unavailable.
        """
        if timeout_s is None:
            return self._get_token()
        token = self._cached_token()
        if token is not None:
            return token
        now = time.monotonic()
        with self._exchange_lock:
            if self._exchange is None or self._exchange[0].done():
                fut: Future = Future()

                def run() -> None:
                    try:
                        fut.set_result(self._get_token())
                    except BaseException as exc:  # noqa: BLE001
                        fut.set_exception(exc)

                threading.Thread(target=run, name="tensorcost-token", daemon=True).start()
                self._exchange = (fut, now)
            fut, started = self._exchange
        wait = min(timeout_s, started + timeout_s - now)
        try:
            return fut.result(timeout=max(0.0, wait))
        except _FutureTimeout:
            raise TensorCostTimeoutError(
                f"tensorcost: token-exchange did not answer within {format_seconds(timeout_s)}",
                "total",
            ) from None

    def _cached_token(self) -> Optional[str]:
        token = self._token
        if token is not None and time.time() < (self._token_expires_at - _TOKEN_REFRESH_LEEWAY_S):
            return token
        return None

    def _get_token(self) -> str:
        """Return a valid short-lived JWT, refreshing if needed."""
        now = time.time()
        with self._token_lock:
            if (
                self._token is not None
                and now < (self._token_expires_at - _TOKEN_REFRESH_LEEWAY_S)
            ):
                return self._token

            url = f"{self._base_url}/api/inference-proxy/sdk-token/exchange"
            if not self._tenant_id:
                # wrap() refuses govern/steer/route without a tenant id, so
                # this is only reached by observation delivery (fail-open).
                raise MissingConfigError(
                    "tensorcost: tenant_id required for token exchange "
                    "(pass tenant_id=... to wrap() or set TENSORCOST_TENANT_ID)"
                )
            # Outages are typed so the wrappers' fail-open predicate can tell
            # "TensorCost is down" (network, timeout, 5xx, 429) from
            # "TensorCost said no" (any other status), which is not a reason
            # to go around it.
            try:
                resp = self._client.post(
                    url,
                    json={
                        "tenant_id": self._tenant_id,
                        "sdk_long_lived_token": self._api_key,
                    },
                    headers={
                        "Content-Type": "application/json",
                        "User-Agent": _USER_AGENT,
                    },
                )
            except httpx.TimeoutException as exc:
                raise TensorCostTimeoutError(
                    "tensorcost: token-exchange timed out", "total", root_cause=exc
                ) from exc
            except (httpx.TransportError, OSError) as exc:
                raise TensorCostNetworkError(
                    f"tensorcost: token-exchange network error: {exc}", root_cause=exc
                ) from exc
            if resp.status_code >= 500:
                raise TensorCostProxyError(
                    f"token-exchange failed status={resp.status_code}",
                    status=resp.status_code,
                )
            if resp.status_code == 429:
                raise TensorCostQuotaError(f"token-exchange failed status={resp.status_code}")
            if resp.status_code in (401, 403):
                raise TensorCostKeyRejectedError(resp.status_code)
            resp.raise_for_status()
            body = resp.json()
            token = body.get("token") or body.get("access_token")
            if not token:
                raise RuntimeError(
                    "tensorcost: token-exchange response missing token field"
                )
            expires_in = float(
                body.get("expires_in", _DEFAULT_TOKEN_LIFETIME_S)
            )
            self._token = token
            self._token_expires_at = now + expires_in
            return token


def get_token_within(transport: object, timeout_s: float) -> str:
    """``transport.get_token`` bounded by ``timeout_s`` on the real transport.

    Stand-ins (tests, a customer's own object) keep their plain
    ``get_token()``.
    """
    if isinstance(transport, ObservationTransport):
        return transport.get_token(timeout_s=timeout_s)
    return transport.get_token()  # type: ignore[attr-defined]


def _parse_retry_after(value: Optional[str], max_s: float) -> Optional[float]:
    """Retry-After as delta-seconds or an HTTP-date; None when absent or unreadable."""
    if value is None:
        return None
    v = value.strip()
    if not v:
        return None
    if re.fullmatch(r"\d+(\.\d+)?", v):
        secs = float(v)
    else:
        try:
            secs = parsedate_to_datetime(v).timestamp() - time.time()
        except (TypeError, ValueError, IndexError):
            return None
    return min(max(0.0, secs), max_s)
