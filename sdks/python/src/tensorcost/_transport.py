"""Fire-and-forget HTTP transport for observation events.

Design:

* A bounded queue + ``ThreadPoolExecutor`` drains observations off the
  customer's hot path. Customer's call returns the moment the underlying
  provider responds; the observation POST is enqueued and shipped from
  a worker thread.
* An observation that can't be delivered because TensorCost is down
  (network error, timeout, 5xx, 429, a token exchange that failed the same
  way) goes into a bounded in-memory retry queue, and the queue is replayed
  by the next delivery that succeeds. When the queue is full the oldest
  entry is dropped and ``dropped_observations`` counts it. A 4xx answer is
  not an outage and is never retried. The queue lives in this process: what
  is still in it when the process exits is lost.
* Nothing here blocks the customer's call. ``post()`` only enqueues; when
  more than ``_MAX_QUEUE_DEPTH`` deliveries are in flight, the oldest one
  that hasn't started moves to the retry queue instead.
* A short-lived JWT is fetched once via
  ``POST /api/inference-proxy/sdk-token/exchange`` using the long-lived
  API key.
  The token is cached in-memory and refreshed when within 60s of expiry.
* All errors during posting are swallowed when ``fail_open`` is True
  (the default) and logged via ``logging.warning``.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as _FutureTimeout
from typing import Any, Optional

import httpx

from ._config import MissingConfigError
from ._errors import (
    TensorCostKeyRejectedError,
    TensorCostNetworkError,
    TensorCostProxyError,
    TensorCostQuotaError,
    TensorCostTimeoutError,
)
from ._version import SDK_VERSION as _USER_AGENT

logger = logging.getLogger("tensorcost")

# Token is treated as expired this many seconds before its real expiry.
_TOKEN_REFRESH_LEEWAY_S = 60.0
# Default lifetime to assume if the server response doesn't carry one.
_DEFAULT_TOKEN_LIFETIME_S = 15 * 60.0
# Max in-flight observations before the oldest unstarted one is parked in
# the retry queue.
_MAX_QUEUE_DEPTH = 1024
# Observations kept for replay while TensorCost is unreachable. The oldest is
# dropped first. A few thousand rows of metadata is well under a megabyte.
_RETRY_QUEUE_MAX = 5000
# One warning per this many dropped observations, so an outage can't flood
# the log.
_DROP_WARN_EVERY = 500

# What one delivery attempt came to.
_SENT, _RETRY, _GIVE_UP = "sent", "retry", "give_up"


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

        self._client = http_client or httpx.Client(timeout=timeout_s)
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="tensorcost-obs",
        )

        # Tracks pending futures so we can park the oldest under back-pressure.
        self._pending: list = []
        self._pending_obs: dict[Any, dict] = {}
        self._pending_lock = threading.Lock()

        # Observations waiting for TensorCost to come back.
        self._retry: deque[dict] = deque()
        self._retry_lock = threading.Lock()
        self._replaying = False
        self.dropped_observations = 0

        # Short-lived JWT cache.
        self._token: Optional[str] = None
        self._token_expires_at: float = 0.0
        self._token_lock = threading.Lock()
        # The one token exchange a bounded get_token() waits on, and when it
        # started (see get_token).
        self._exchange: Optional[tuple[Future, float]] = None
        self._exchange_lock = threading.Lock()

    @property
    def queued_observations(self) -> int:
        """Observations waiting in the retry queue for TensorCost to answer."""
        with self._retry_lock:
            return len(self._retry)

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
        parked: list[dict] = []
        with self._pending_lock:
            self._reap_locked()
            # Back-pressure: the oldest delivery that hasn't started waits in
            # the retry queue instead.
            while len(self._pending) >= _MAX_QUEUE_DEPTH:
                oldest = self._pending.pop(0)
                obs = self._pending_obs.pop(oldest, None)
                if oldest.cancel() and obs is not None:
                    parked.append(obs)
            try:
                fut = self._executor.submit(self._deliver, observation)
                self._pending.append(fut)
                self._pending_obs[fut] = observation
            except RuntimeError:
                # Executor shut down; ignore.
                pass
        for obs in parked:
            self._queue_for_retry(obs)

    def _reap_locked(self) -> None:
        live = []
        for f in self._pending:
            if f.done():
                self._pending_obs.pop(f, None)
            else:
                live.append(f)
        self._pending = live

    def flush(self, timeout_s: float = 5.0) -> None:
        """Wait for in-flight observation POSTs (Lambda shutdown)."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            with self._pending_lock:
                self._reap_locked()
                if not self._pending:
                    return
            time.sleep(0.01)

    def close(self) -> None:
        """Best-effort flush + shutdown of the worker pool."""
        self.flush(timeout_s=5.0)
        try:
            self._executor.shutdown(wait=True)
        finally:
            try:
                self._client.close()
            except Exception:  # noqa: BLE001
                pass

    # ------------------------------------------------------------------
    # Internal

    def _deliver(self, observation: dict) -> None:
        outcome = self._send(observation)
        if outcome == _SENT:
            self._replay()
        elif outcome == _RETRY:
            self._queue_for_retry(observation)

    def _send(self, observation: dict) -> str:
        """One delivery attempt. Never raises when fail_open (the default)."""
        try:
            token = self._get_token()
            url = f"{self._base_url}/api/inference-proxy/observation"
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "User-Agent": _USER_AGENT,
            }
            if self._team_id:
                headers["x-tc-team-id"] = self._team_id
            resp = self._client.post(url, json=observation, headers=headers)
        except (
            TensorCostNetworkError,
            TensorCostTimeoutError,
            TensorCostProxyError,
            TensorCostQuotaError,
            httpx.TransportError,
            OSError,
        ) as exc:
            # TensorCost is unreachable or overloaded: keep it for later.
            logger.warning("tensorcost: observation delivery failed: %s", exc)
            return _RETRY
        except Exception as exc:  # noqa: BLE001
            # A rejected key, no tenant id, a malformed answer: retrying
            # would get the same answer.
            if not self._fail_open:
                raise
            logger.warning("tensorcost: observation delivery failed: %s", exc)
            return _GIVE_UP
        if resp.status_code >= 500 or resp.status_code in (408, 429):
            logger.warning(
                "tensorcost: observation POST failed status=%s; kept for retry",
                resp.status_code,
            )
            return _RETRY
        if resp.status_code >= 400:
            logger.warning(
                "tensorcost: observation POST failed status=%s",
                resp.status_code,
            )
            return _GIVE_UP
        return _SENT

    def _queue_for_retry(self, observation: dict) -> None:
        with self._retry_lock:
            if len(self._retry) >= _RETRY_QUEUE_MAX:
                self._retry.popleft()
                self.dropped_observations += 1
                if self.dropped_observations % _DROP_WARN_EVERY == 1:
                    logger.warning(
                        "tensorcost: TensorCost has been unreachable for a while; the retry "
                        "queue is full (%d observations) and the oldest are being dropped "
                        "(%d so far)",
                        _RETRY_QUEUE_MAX,
                        self.dropped_observations,
                    )
            self._retry.append(observation)

    def _replay(self) -> None:
        """After a delivery worked, send what an outage left in the queue.

        One replayer at a time, oldest first; it stops at the first attempt
        that fails the same way and leaves the rest queued.
        """
        with self._retry_lock:
            if self._replaying or not self._retry:
                return
            self._replaying = True
        try:
            while True:
                with self._retry_lock:
                    if not self._retry:
                        return
                    observation = self._retry.popleft()
                outcome = self._send(observation)
                if outcome == _RETRY:
                    with self._retry_lock:
                        self._retry.appendleft(observation)
                    return
        finally:
            with self._retry_lock:
                self._replaying = False

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
                f"tensorcost: token-exchange did not answer within {timeout_s}s",
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
