"""Opt-in disk spool for observations the in-memory buffer could not keep.

Same layout as the Node SDK's ``spool.ts``, so either SDK can replay the
other's files: one JSONL file per segment, one observation body per line,
the exact bytes that would have been POSTed (a replay resends the same body
and correlation_id, and ingestion dedupes it).

    tc-obs-<ms>-<pid>-<rand>-<seq>.open    being written by a live process
    tc-obs-<ms>-<pid>-<rand>-<seq>.jsonl   sealed (rotated, or the writer exited)
    <either>.claimed-<pid>-<rand>          taken by a process replaying it

A writer only appends to its own ``.open`` segment and seals it at the
segment size or at exit. Replay takes sealed segments, plus ``.open`` and
``.claimed`` ones untouched for ``_STALE_S`` (their process died). Claiming
is a rename, so two processes sharing a directory never both take a file;
the worst case is a duplicate line, which dedupe absorbs.

The directory is kept under ``max_bytes`` by deleting the oldest sealed
segments; their lines are reported through ``on_lost``.

All methods are called from the transport's worker threads or the exit
hook, never from ``post()``, and are serialised by the spool's own lock.
"""

from __future__ import annotations

import os
import secrets
import threading
import time
from typing import Callable, List, Optional

DEFAULT_SPOOL_MAX_BYTES = 64 * 1024 * 1024
DEFAULT_SPOOL_SEGMENT_BYTES = 1024 * 1024
_STALE_S = 5 * 60.0
_PREFIX = "tc-obs-"


class Spool:
    def __init__(
        self,
        directory: str,
        *,
        max_bytes: Optional[int] = None,
        segment_bytes: Optional[int] = None,
        on_lost: Optional[Callable[[int, str], None]] = None,
    ) -> None:
        self.dir = directory
        self._max_bytes = max_bytes or DEFAULT_SPOOL_MAX_BYTES
        self._segment_bytes = segment_bytes or DEFAULT_SPOOL_SEGMENT_BYTES
        self._on_lost = on_lost or (lambda n, reason: None)
        self._instance = f"{os.getpid()}-{secrets.token_hex(4)}"
        self._seq = 0
        self._open_path: Optional[str] = None
        self._open_bytes = 0
        self._total: Optional[int] = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Writing

    def append(self, bodies: List[bytes], *, seal: bool = False) -> None:
        """Append bodies, one per line. Failures are reported, never raised."""
        with self._lock:
            try:
                if bodies:
                    os.makedirs(self.dir, exist_ok=True)
                    if self._total is None:
                        self._total = self._measure()
                    data = b"\n".join(bodies) + b"\n"
                    self._make_room(len(data))
                    with open(self._current_path(), "ab") as f:
                        f.write(data)
                    self._open_bytes += len(data)
                    self._total += len(data)
                    if self._open_bytes >= self._segment_bytes:
                        self._seal()
                if seal:
                    self._seal()
            except Exception as exc:  # noqa: BLE001
                self._on_lost(len(bodies), f"spool write failed: {exc}")

    # ------------------------------------------------------------------
    # Replay

    def claim_next(self) -> Optional[List[bytes]]:
        """Take the oldest segment left by an earlier run (or a dead
        process), delete it, and return its lines. None when there is none."""
        try:
            names = sorted(n for n in os.listdir(self.dir) if n.startswith(_PREFIX))
        except OSError:
            return None
        now = time.time()
        for name in names:
            full = os.path.join(self.dir, name)
            if full == self._open_path:
                continue
            if not name.endswith(".jsonl"):
                try:
                    st = os.stat(full)
                except OSError:
                    continue
                # rename() bumps ctime, so a fresh claim is never mistaken for stale.
                if now - max(st.st_mtime, st.st_ctime) < _STALE_S:
                    continue
            base = name.split(".claimed-")[0]
            claimed = os.path.join(self.dir, f"{base}.claimed-{self._instance}")
            try:
                os.rename(full, claimed)
            except OSError:
                continue  # someone else got it
            try:
                with open(claimed, "rb") as f:
                    data = f.read()
                os.unlink(claimed)
            except OSError:
                continue
            with self._lock:
                if self._total is not None:
                    self._total = max(0, self._total - len(data))
            return [ln for ln in data.split(b"\n") if ln]
        return None

    # ------------------------------------------------------------------
    # Internal (lock held)

    def _current_path(self) -> str:
        if self._open_path is None:
            self._seq += 1
            stamp = str(int(time.time() * 1000)).zfill(15)
            self._open_path = os.path.join(
                self.dir, f"{_PREFIX}{stamp}-{self._instance}-{self._seq:06d}.open"
            )
            self._open_bytes = 0
        return self._open_path

    def _seal(self) -> None:
        if self._open_path is None:
            return
        src, self._open_path = self._open_path, None
        try:
            os.rename(src, src[: -len(".open")] + ".jsonl")
        except OSError:
            pass  # a stale-segment claim may have taken it

    def _measure(self) -> int:
        total = 0
        for name in os.listdir(self.dir):
            if name.startswith(_PREFIX):
                try:
                    total += os.stat(os.path.join(self.dir, name)).st_size
                except OSError:
                    pass
        return total

    def _sealed_oldest_first(self) -> List[str]:
        return [
            os.path.join(self.dir, n)
            for n in sorted(os.listdir(self.dir))
            if n.startswith(_PREFIX) and n.endswith(".jsonl")
            and os.path.join(self.dir, n) != self._open_path
        ]

    def _make_room(self, incoming: int) -> None:
        assert self._total is not None
        for _ in range(2):
            if self._total + incoming <= self._max_bytes:
                return
            for victim in self._sealed_oldest_first():
                if self._total + incoming <= self._max_bytes:
                    return
                try:
                    with open(victim, "rb") as f:
                        data = f.read()
                    os.unlink(victim)
                except OSError:
                    continue  # raced with a claim
                self._total = max(0, self._total - len(data))
                self._on_lost(data.count(b"\n"), "spool full; oldest segment removed")
            # Only our own segment is left to give up; seal it so it can go.
            self._seal()
        if self._total + incoming > self._max_bytes:
            raise OSError(f"spool directory is over its {self._max_bytes}-byte cap")

    def _after_fork(self) -> None:
        """In a forked child: a fresh lock and our own segment names."""
        self._lock = threading.Lock()
        self._instance = f"{os.getpid()}-{secrets.token_hex(4)}"
        self._open_path = None
        self._open_bytes = 0
