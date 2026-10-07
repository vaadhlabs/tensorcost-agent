"""Fire-and-forget HTTP transport for observation events.

Design:

* A bounded queue + ``ThreadPoolExecutor`` drains observations off the
  customer's hot path. Customer's call returns the moment the underlying
  provider responds; the observation POST is enqueued and shipped from
  a worker thread.
* If the queue saturates (TensorCost is down or slow), we drop the
  oldest observation rather than block. Losing data is preferable to
  slowing down the customer's request loop.
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
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import httpx

from ._version import SDK_VERSION as _USER_AGENT

logger = logging.getLogger("tensorcost")

# Token is treated as expired this many seconds before its real expiry.
_TOKEN_REFRESH_LEEWAY_S = 60.0
# Default lifetime to assume if the server response doesn't carry one.
_DEFAULT_TOKEN_LIFETIME_S = 15 * 60.0
# Max in-flight observations before we start dropping oldest.
_MAX_QUEUE_DEPTH = 1024


class ObservationTransport:
    """Thread-safe fire-and-forget poster for observation events."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        *,
        tenant_id: Optional[str] = None,
        timeout_s: float = 2.0,
        fail_open: bool = True,
        max_workers: int = 4,
        http_client: Optional[httpx.Client] = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._tenant_id = tenant_id
        self._timeout_s = timeout_s
        self._fail_open = fail_open

        self._client = http_client or httpx.Client(timeout=timeout_s)
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="tensorcost-obs",
        )

        # Tracks pending futures so we can drop oldest under back-pressure.
        self._pending: list = []
        self._pending_lock = threading.Lock()

        # Short-lived JWT cache.
        self._token: Optional[str] = None
        self._token_expires_at: float = 0.0
        self._token_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API

    def post(self, observation: dict) -> None:
        """Enqueue an observation for fire-and-forget delivery.

        Returns immediately. Errors are logged, never raised (when
        ``fail_open`` is True, which is the default).
        """
        with self._pending_lock:
            # Reap completed futures.
            self._pending = [f for f in self._pending if not f.done()]
            # Apply back-pressure: drop oldest if saturated.
            while len(self._pending) >= _MAX_QUEUE_DEPTH:
                dropped = self._pending.pop(0)
                dropped.cancel()
                logger.warning(
                    "tensorcost: observation queue saturated; dropping oldest"
                )
            try:
                fut = self._executor.submit(self._deliver, observation)
                self._pending.append(fut)
            except RuntimeError:
                # Executor shut down; ignore.
                pass

    def flush(self, timeout_s: float = 5.0) -> None:
        """Wait for in-flight observation POSTs (Lambda shutdown)."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            with self._pending_lock:
                self._pending = [f for f in self._pending if not f.done()]
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
        try:
            token = self._get_token()
            url = f"{self._base_url}/api/inference-proxy/observation"
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "User-Agent": _USER_AGENT,
            }
            resp = self._client.post(url, json=observation, headers=headers)
            if resp.status_code >= 400:
                logger.warning(
                    "tensorcost: observation POST failed status=%s",
                    resp.status_code,
                )
        except Exception as exc:  # noqa: BLE001
            if not self._fail_open:
                raise
            logger.warning("tensorcost: observation delivery failed: %s", exc)

    def get_token(self) -> str:
        """Return a valid short-lived JWT for proxy/admit/sdk-layer calls."""
        return self._get_token()

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
                raise RuntimeError(
                    "tensorcost: tenant_id required for token exchange "
                    "(pass tenant_id=... to wrap() or set TENSORCOST_TENANT_ID)"
                )
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
