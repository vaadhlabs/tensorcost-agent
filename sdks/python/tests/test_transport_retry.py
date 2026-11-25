"""Delivery guarantees for observations: retry with backoff, Retry-After,
no retry on caller errors, a bounded buffer that drops the oldest, flush on
exit, and the optional disk spool. Mirrors the Node SDK's
transport-retry.test.ts."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import textwrap
import threading
import time
from email.utils import formatdate
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest

from tensorcost import _exit
from tensorcost._transport import ObservationTransport


def obs(cid: str) -> dict:
    return {
        "sdk_version": "tensorcost-python/test",
        "provider": "openai",
        "model": "gpt-4o-mini",
        "input_tokens": 10,
        "output_tokens": 20,
        "status": "success",
        "correlation_id": cid,
    }


class Server:
    """httpx handler: token exchange always works; each observation POST
    takes the next reply from the script (the last one repeats)."""

    def __init__(self, script):
        self.script = list(script)
        self.posts: list[tuple[bytes, float]] = []
        self.lock = threading.Lock()
        self.in_flight = 0
        self.max_in_flight = 0
        self.release = threading.Event()

    def set_script(self, script) -> None:
        with self.lock:
            self.script = list(script)

    def ids(self) -> list[str]:
        return [json.loads(b)["correlation_id"] for b, _ in self.posts]

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/sdk-token/exchange"):
            return httpx.Response(200, json={"token": "tok", "expires_in": 3600})
        with self.lock:
            self.posts.append((request.content, time.monotonic()))
            reply = self.script.pop(0) if len(self.script) > 1 else self.script[0]
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            if reply == "hang":
                self.release.wait(10)
                raise httpx.ReadTimeout("timed out", request=request)
            if isinstance(reply, Exception):
                raise reply
            if isinstance(reply, httpx.Response):
                return reply
            return httpx.Response(reply)
        finally:
            with self.lock:
                self.in_flight -= 1


def make(server: Server, **kwargs) -> ObservationTransport:
    client = httpx.Client(transport=httpx.MockTransport(server), timeout=2.0)
    kwargs.setdefault("backoff_base_s", 0.05)
    kwargs.setdefault("random", lambda: 1.0)
    return ObservationTransport(
        base_url="https://api.test",
        api_key="k",
        tenant_id="t-1",
        http_client=client,
        **kwargs,
    )


def wait_until(pred, timeout_s: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.005)
    return pred()


def failed_once(server: Server, t: ObservationTransport) -> bool:
    """The first POST has been answered and put back in the retry buffer."""
    return wait_until(
        lambda: len(server.posts) == 1
        and t.stats().in_flight == 0
        and t.stats().buffered == 1
    )


@pytest.fixture(autouse=True)
def _clean_exit_hooks():
    yield
    _exit._reset()


# ----------------------------------------------------------------------
# Retry with backoff


def test_retry_after_503_delivers_exactly_once():
    s = Server([503, 202])
    t = make(s)
    t.post(obs("a"))
    assert wait_until(lambda: len(s.posts) == 2)
    time.sleep(0.3)
    assert len(s.posts) == 2
    st = t.stats()
    assert (st.buffered, st.in_flight, st.dropped) == (0, 0, 0)
    t.close()


def test_retry_resends_same_body_and_correlation_id():
    s = Server([503, httpx.ConnectError("reset"), 202])
    t = make(s)
    t.post(obs("corr-1"))
    assert wait_until(lambda: len(s.posts) == 3)
    assert len({b for b, _ in s.posts}) == 1
    assert s.ids() == ["corr-1"] * 3
    t.close()


def test_429_honours_retry_after():
    s = Server([httpx.Response(429, headers={"Retry-After": "1"}), 202])
    t = make(s)
    t.post(obs("a"))
    assert wait_until(lambda: len(s.posts) == 1)
    # The 50 ms backoff would have fired many times over by now.
    time.sleep(0.6)
    assert len(s.posts) == 1
    assert wait_until(lambda: len(s.posts) == 2, 2.0)
    assert s.posts[1][1] - s.posts[0][1] >= 0.95
    t.close()


def test_retry_after_as_http_date():
    when = formatdate(time.time() + 2, usegmt=True)
    s = Server([httpx.Response(429, headers={"Retry-After": when}), 202])
    t = make(s)
    t.post(obs("a"))
    assert wait_until(lambda: len(s.posts) == 1)
    time.sleep(0.5)
    assert len(s.posts) == 1
    assert wait_until(lambda: len(s.posts) == 2, 3.0)
    t.close()


@pytest.mark.parametrize("status", [400, 401, 403])
def test_caller_errors_are_not_retried(status, caplog):
    s = Server([status])
    t = make(s)
    with caplog.at_level(logging.WARNING, logger="tensorcost"):
        t.post(obs("a"))
        assert wait_until(lambda: len(s.posts) == 1)
        time.sleep(0.3)
    assert len(s.posts) == 1
    assert f"tensorcost: observation POST failed status={status}" in caplog.text
    assert t.stats().buffered == 0
    t.close()


def test_network_errors_and_timeouts_are_retried():
    s = Server([httpx.ConnectError("refused"), httpx.ReadTimeout("slow"), 202])
    t = make(s)
    t.post(obs("a"))
    assert wait_until(lambda: len(s.posts) == 3)
    assert wait_until(lambda: t.stats().buffered == 0 and t.stats().in_flight == 0)
    t.close()


def test_token_exchange_outage_is_retried():
    calls = {"exchange": 0, "posts": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/sdk-token/exchange"):
            calls["exchange"] += 1
            if calls["exchange"] == 1:
                return httpx.Response(503)
            return httpx.Response(200, json={"token": "tok"})
        calls["posts"] += 1
        return httpx.Response(202)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    t = ObservationTransport(
        base_url="https://api.test", api_key="k", tenant_id="t",
        http_client=client, backoff_base_s=0.05,
    )
    t.post(obs("a"))
    assert wait_until(lambda: calls["posts"] == 1)
    assert calls["exchange"] == 2
    t.close()


def test_backoff_doubles_and_caps():
    s = Server([503])
    t = make(s, backoff_base_s=0.05, backoff_max_s=0.2)
    t.post(obs("a"))
    assert wait_until(lambda: len(s.posts) >= 6, 5.0)
    times = [at for _, at in s.posts[:6]]
    gaps = [times[i + 1] - times[i] for i in range(5)]
    wants = [0.05, 0.1, 0.2, 0.2, 0.2]
    for i, want in enumerate(wants):
        assert want - 0.01 <= gaps[i] <= want + 0.1, gaps
    s.set_script([202])
    t.close()


def test_one_probe_at_a_time_while_failing():
    s = Server([httpx.Response(429, headers={"Retry-After": "30"})])
    t = make(s)
    t.post(obs("first"))
    assert failed_once(s, t)
    for i in range(20):
        t.post(obs(f"o{i}"))
    time.sleep(0.3)
    assert len(s.posts) == 1  # the 20 wait rather than pile on
    s.set_script([202])
    t.flush(timeout_s=3.0)
    assert sorted(s.ids()[1:]) == sorted(["first"] + [f"o{i}" for i in range(20)])
    assert t.stats().buffered == 0
    t.close()


def test_gives_up_after_retry_horizon():
    s = Server([503])
    t = make(s, max_retry_age_s=0.3)
    t.post(obs("a"))
    assert wait_until(lambda: t.stats().dropped == 1, 3.0)
    sent = len(s.posts)
    time.sleep(0.3)
    assert len(s.posts) == sent
    assert t.stats().buffered == 0
    t.close()


def test_post_stays_non_blocking_while_busy():
    s = Server(["hang"])
    t = make(s)
    start = time.perf_counter()
    for i in range(2000):
        t.post(obs(f"o{i}"))
    assert time.perf_counter() - start < 0.5
    s.set_script([202])
    s.release.set()
    t.close()


# ----------------------------------------------------------------------
# Bounded buffer


def test_full_buffer_drops_oldest_and_counts():
    s = Server([httpx.Response(429, headers={"Retry-After": "30"})])
    t = make(s, max_buffered_observations=3)
    t.post(obs("a"))
    assert failed_once(s, t)
    for cid in "bcdef":
        t.post(obs(cid))
    st = t.stats()
    assert (st.buffered, st.dropped) == (3, 3)
    assert t.dropped_count == 3
    s.set_script([202])
    t.flush(timeout_s=3.0)
    assert sorted(s.ids()[1:]) == ["d", "e", "f"]
    t.close()


def test_buffer_capped_by_bytes():
    s = Server([httpx.Response(429, headers={"Retry-After": "30"})])
    size = len(json.dumps(obs("x0"), separators=(",", ":")))
    t = make(s, max_buffered_bytes=size * 2 + 1)
    t.post(obs("x0"))
    assert failed_once(s, t)
    for i in range(1, 5):
        t.post(obs(f"x{i}"))
    st = t.stats()
    assert (st.buffered, st.dropped) == (2, 3)
    t.close()


def test_drop_warning_at_most_once_a_minute(caplog):
    s = Server([httpx.Response(429, headers={"Retry-After": "30"})])
    t = make(s, max_buffered_observations=1)
    with caplog.at_level(logging.WARNING, logger="tensorcost"):
        t.post(obs("a"))
        assert failed_once(s, t)
        for i in range(100):
            t.post(obs(f"o{i}"))
    drops = [r for r in caplog.records if "dropped" in r.getMessage()]
    assert len(drops) == 1
    assert t.stats().dropped == 100
    t.close()


# ----------------------------------------------------------------------
# flush


def test_flush_drains_queue_and_retry_buffer():
    s = Server([httpx.Response(429, headers={"Retry-After": "30"}), 202])
    t = make(s)
    t.post(obs("buffered"))
    assert failed_once(s, t)
    t.post(obs("fresh"))
    start = time.monotonic()
    t.flush(timeout_s=3.0)
    assert time.monotonic() - start < 1.0
    assert s.ids() == ["buffered", "buffered", "fresh"]
    st = t.stats()
    assert (st.buffered, st.in_flight) == (0, 0)
    t.close()


def test_flush_returns_early_when_endpoint_still_down():
    s = Server([503])
    t = make(s, backoff_base_s=1.0)
    t.post(obs("a"))
    start = time.monotonic()
    t.flush(timeout_s=1.0)  # the next retry (1 s) falls past the deadline
    assert time.monotonic() - start < 0.5
    assert t.stats().buffered == 1
    s.set_script([202])
    t.close()


# ----------------------------------------------------------------------
# Flush on exit


def test_exit_hook_drains_queue_and_retry_buffer():
    s = Server([httpx.Response(429, headers={"Retry-After": "30"}), 202])
    t = make(s, flush_on_exit=True)
    t.post(obs("retrying"))
    assert failed_once(s, t)
    t.post(obs("queued"))
    _exit._run_exit_hooks()
    assert s.ids() == ["retrying", "retrying", "queued"]
    st = t.stats()
    assert (st.buffered, st.in_flight, st.dropped) == (0, 0, 0)


def test_exit_hook_never_holds_past_the_cap(caplog):
    s = Server(["hang"])
    t = make(s, flush_on_exit=True)
    t.post(obs("a"))
    assert wait_until(lambda: len(s.posts) == 1)
    start = time.monotonic()
    with caplog.at_level(logging.WARNING, logger="tensorcost"):
        _exit._run_exit_hooks()
    assert time.monotonic() - start < _exit.EXIT_FLUSH_S + 0.5
    assert t.stats().dropped == 1
    assert "1 observation(s) were not delivered" in caplog.text
    s.release.set()


def test_flush_on_exit_off_is_not_registered():
    s = Server([202])
    t = make(s, flush_on_exit=False)
    assert t not in _exit._registered()
    t.close()


class _Ingest(BaseHTTPRequestHandler):
    posts: list[str] = []
    count = 0

    def do_POST(self):  # noqa: N802
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        if self.path.endswith("/sdk-token/exchange"):
            payload = b'{"token":"t","expires_in":900}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        type(self).count += 1
        type(self).posts.append(json.loads(body)["correlation_id"])
        self.send_response(503 if type(self).count == 1 else 202)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, *args):  # silence
        pass


def test_real_process_exit_delivers_what_was_retrying():
    """A child process posts, gets a 503, and returns from main. The atexit
    hook must deliver the retry before the interpreter goes away."""
    _Ingest.posts, _Ingest.count = [], 0
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Ingest)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        port = srv.server_address[1]
        script = textwrap.dedent(
            f"""
            from tensorcost._transport import ObservationTransport
            t = ObservationTransport(
                base_url="http://127.0.0.1:{port}", api_key="k", tenant_id="t",
                flush_on_exit=True,
            )
            t.post({{"correlation_id": "c1"}})
            """
        )
        src = str(Path(__file__).resolve().parents[1] / "src")
        env = {**os.environ, "PYTHONPATH": src}
        start = time.monotonic()
        out = subprocess.run(
            [sys.executable, "-c", script], env=env, capture_output=True, timeout=20
        )
        assert out.returncode == 0, out.stderr.decode()
        assert time.monotonic() - start < 10
        assert _Ingest.posts == ["c1", "c1"]
    finally:
        srv.shutdown()


# ----------------------------------------------------------------------
# Disk spool


def _spool_lines(d: Path) -> list[str]:
    lines: list[str] = []
    for f in sorted(d.iterdir()):
        lines += [ln for ln in f.read_text().split("\n") if ln]
    return lines


def test_spool_round_trips_overflow_and_exit_leftovers(tmp_path):
    down = Server([httpx.Response(429, headers={"Retry-After": "30"})])
    a = make(down, max_buffered_observations=2, spool_dir=str(tmp_path), flush_on_exit=True)
    a.post(obs("o1"))
    assert failed_once(down, a)
    for cid in ["o2", "o3", "o4", "o5"]:
        a.post(obs(cid))
    a.spool_idle()
    st = a.stats()
    assert (st.buffered, st.spooled, st.dropped) == (2, 3, 0)

    a._spill_remaining()  # what the exit hook does once the drain is over
    st = a.stats()
    assert (st.buffered, st.spooled, st.dropped) == (0, 5, 0)
    lines = _spool_lines(tmp_path)
    assert sorted(json.loads(ln)["correlation_id"] for ln in lines) == ["o1", "o2", "o3", "o4", "o5"]
    assert down.posts[0][0].decode() in lines  # byte-for-byte the body that was sent
    _exit._reset()

    up = Server([202])
    b = make(up, spool_dir=str(tmp_path))
    b.replay_done()
    b.flush(timeout_s=3.0)
    assert sorted(up.ids()) == ["o1", "o2", "o3", "o4", "o5"]
    assert list(tmp_path.iterdir()) == []
    b.close()


def test_spool_stays_under_its_size_cap(tmp_path):
    down = Server([httpx.Response(429, headers={"Retry-After": "30"})])
    size = len(json.dumps(obs("s000"), separators=(",", ":"))) + 1
    t = make(
        down,
        max_buffered_observations=1,
        spool_dir=str(tmp_path),
        spool_max_bytes=size * 10,
        spool_segment_bytes=size * 3,
    )
    t.post(obs("s000"))
    assert failed_once(down, t)
    for i in range(1, 41):
        t.post(obs(f"s{i:03d}"))
        t.spool_idle()
    total = sum(f.stat().st_size for f in tmp_path.iterdir())
    assert total <= size * 10
    kept = [json.loads(ln)["correlation_id"] for ln in _spool_lines(tmp_path)]
    assert "s039" in kept
    assert "s001" not in kept
    assert t.stats().dropped > 0
    t.close()


def test_spool_format_matches_node(tmp_path):
    """One observation body per line, so either SDK can replay the other's."""
    (tmp_path / "tc-obs-000001731628800000-123-abcd-000001.jsonl").write_text(
        json.dumps(obs("from-node"), separators=(",", ":")) + "\n"
    )
    up = Server([202])
    t = make(up, spool_dir=str(tmp_path))
    t.replay_done()
    t.flush(timeout_s=3.0)
    assert up.ids() == ["from-node"]
    t.close()


# ----------------------------------------------------------------------
# wrap() configuration


def test_config_flush_on_exit_default_and_env(monkeypatch):
    from tensorcost._config import resolve_config

    assert resolve_config(api_key="k", tenant_id="t").flush_on_exit is True
    assert resolve_config(api_key="k", tenant_id="t", flush_on_exit=False).flush_on_exit is False
    monkeypatch.setenv("TENSORCOST_FLUSH_ON_EXIT", "0")
    assert resolve_config(api_key="k", tenant_id="t").flush_on_exit is False
    monkeypatch.setenv("TENSORCOST_FLUSH_ON_EXIT", "false")
    assert resolve_config(api_key="k", tenant_id="t").flush_on_exit is False


def test_config_spool_dir_from_env(monkeypatch):
    from tensorcost._config import resolve_config

    assert resolve_config(api_key="k", tenant_id="t").spool_dir is None
    monkeypatch.setenv("TENSORCOST_SPOOL_DIR", "/var/spool/tc")
    assert resolve_config(api_key="k", tenant_id="t").spool_dir == "/var/spool/tc"
    assert resolve_config(api_key="k", tenant_id="t", spool_dir="/x").spool_dir == "/x"


def test_shared_runtime_gets_buffer_limits_and_exit_flush():
    import tensorcost
    from tensorcost import _runtime
    from tensorcost._config import resolve_config

    config = resolve_config(
        api_key="k", tenant_id="t", base_url="https://api.test",
        max_buffered_observations=5,
    )
    # conftest stubs get_shared_runtime; build the real transport the way it does.
    transport = _runtime.new_transport(config)
    _runtime._runtimes["k"] = (transport, None)  # type: ignore[assignment]
    try:
        assert transport._flush_on_exit is True
        assert transport in _exit._registered()
        with transport._cond:
            for i in range(20):
                transport._enqueue_locked(transport._entry(b"{}", float(i)))
        assert (transport.stats().buffered, transport.stats().dropped) == (5, 15)
        stats = tensorcost.observation_stats()
        assert (stats.buffered, stats.dropped) == (5, 15)
    finally:
        _runtime._runtimes.clear()


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs fork")
def test_forked_child_still_delivers():
    """gunicorn --preload: the transport has sent from the parent, then the
    worker forks. The child's posts must still go out."""
    _Ingest.posts, _Ingest.count = [], 1  # no 503 this time
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Ingest)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        t = ObservationTransport(
            base_url=f"http://127.0.0.1:{srv.server_address[1]}", api_key="k", tenant_id="t"
        )
        t.post({"correlation_id": "parent"})
        t.flush(timeout_s=3.0)
        pid = os.fork()
        if pid == 0:  # child
            code = 1
            try:
                t.post({"correlation_id": "child"})
                t.flush(timeout_s=3.0)
                code = 0 if t.stats().buffered == 0 and t.stats().in_flight == 0 else 2
            finally:
                os._exit(code)
        _, status = os.waitpid(pid, 0)
        assert os.WEXITSTATUS(status) == 0
        assert wait_until(lambda: "child" in _Ingest.posts)
        assert _Ingest.posts == ["parent", "child"]
    finally:
        srv.shutdown()
