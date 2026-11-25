"""Interpreter-exit hook for observation transports.

``atexit`` runs after the main thread returns and before daemon threads are
torn down, so the transport's workers are still there to send. Each
registered transport with ``flush_on_exit`` drains for what is left of a
shared EXIT_FLUSH_S budget; after that, whatever is still in memory goes to
the spool when one is configured, and is counted and reported otherwise.

atexit does not run when the process is killed by a signal it does not
handle. Python's default SIGTERM is such a kill; servers that turn SIGTERM
into a normal shutdown (gunicorn, uvicorn, most frameworks) do run it.
Serverless functions are frozen rather than exited, so they should call
``flush_observations()`` themselves.
"""

from __future__ import annotations

import atexit
import threading
import time
import weakref
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from ._transport import ObservationTransport

EXIT_FLUSH_S = 2.0

_transports: "weakref.WeakSet[ObservationTransport]" = weakref.WeakSet()
_lock = threading.Lock()
_installed = False


def register_exit_hooks(transport: "ObservationTransport") -> None:
    global _installed
    with _lock:
        _transports.add(transport)
        if not _installed:
            _installed = True
            atexit.register(_run_exit_hooks)


def _run_exit_hooks() -> None:
    deadline = time.monotonic() + EXIT_FLUSH_S
    for t in list(_transports):
        try:
            if t._flush_on_exit:
                t.flush(timeout_s=max(0.0, deadline - time.monotonic()), _for_exit=True)
            t._spill_remaining()
        except Exception:  # noqa: BLE001 — never raise from an exit hook
            pass


def _registered() -> List["ObservationTransport"]:
    return list(_transports)


def _reset() -> None:
    """Test hook: forget registered transports (the atexit hook stays)."""
    with _lock:
        _transports.clear()
