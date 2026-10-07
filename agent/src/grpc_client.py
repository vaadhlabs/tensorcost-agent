"""
gRPC bidirectional streaming client — TensorCost agent v1 protocol.

Targets `tensorcost.agent.v1.AgentService.Connect` (per
apps-new/packages/contracts/proto/agent.proto). Replaces the legacy
`gpu_agent.AgentService.AgentStream` client wholesale; nothing about
the wire shape, auth, or message envelope is shared with the prior
implementation.

Wire flow:
  1. Client opens Connect() bidi stream.
  2. First upstream message: signed AgentHello (HMAC-SHA256 over
     tenant_id || "\\n" || agent_id || "\\n" || nonce || "\\n" || be8(ts)).
     Server SETNX-rejects replayed nonces and bcrypt-rejects key_id misses.
  3. Server replies with ServerAccepted{session_id, heartbeat_interval_seconds}.
  4. Client streams MetricBatch / InstanceUpdate / CommandResult messages.
     Server pushes InstanceCommand / ServerPing.
  5. On any stream error (unauth, network, server restart) we close,
     wait an exponentially-backed-off interval, and reopen with a fresh
     hello.

Auth state — `key_id` and `hmac_pepper` (the per-agent signing-key bytes
provisioning handed us) live in instance attributes only. They are NEVER
written to logs; the few `logger.info` lines below redact deliberately.
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
import sys
import threading
from typing import Any, Awaitable, Callable, Dict, List, Optional, Union

import grpc

# Expose the vendored proto stubs even when the agent is launched outside
# its src/ directory — the legacy client did the same trick.
_proto_dir = os.path.join(os.path.dirname(__file__), "proto")
if _proto_dir not in sys.path:
    sys.path.insert(0, _proto_dir)

import agent_pb2  # type: ignore  # noqa: E402
import agent_pb2_grpc  # type: ignore  # noqa: E402

from auth.hmac import sign_agent_hello  # noqa: E402
from transport.tls_pinning import (  # noqa: E402
    PinMismatchError,
    TlsConfig,
    assert_chain_pinned,
    extract_chain_from_sslsocket,
)

logger = logging.getLogger(__name__)

# Reconnect-with-backoff parameters. Identical to the legacy values so
# operator dashboards / runbooks still apply.
_BACKOFF_BASE = 1.0
_BACKOFF_MULTIPLIER = 2.0
_BACKOFF_CAP = 60.0
_BACKOFF_JITTER = 0.1

_AGENT_VERSION = "unified-gpu-agent/2.0.0"
_HELLO_BUILDER = sign_agent_hello  # rebindable for tests


# PEP 604 `X | Y` union syntax is only supported as a runtime
# expression on Python 3.10+. The runtime image is python:3.11-slim
# (post-bump 2026-04-29), but a `Union[...]` form works on 3.9-3.13
# alike — no reason to gate the type alias on a Python floor when
# the typing-module form is exactly equivalent at runtime.
CommandHandler = Callable[[Dict[str, Any]], Union[Awaitable[Dict[str, Any]], Dict[str, Any]]]


def _resolve_tls(use_tls: Optional[bool], grpc_target: str) -> bool:
    """TLS auto-detect.

    - Caller-explicit `use_tls=True/False` wins.
    - `TC_AGENT_TLS=true|1|false|0` env var overrides when the caller
      passed None.
    - Otherwise: TLS unless the target is loopback (localhost,
      127.0.0.1, ::1, 0.0.0.0). Customer agents virtually always want
      TLS — gateway is on a public NLB with an ACM cert — and only
      local-loop tests against a plaintext gateway should fall through
      to insecure_channel().
    """
    if use_tls is not None:
        return use_tls
    flag = os.getenv("TC_AGENT_TLS", "").strip().lower()
    if flag in ("true", "1", "yes"):
        return True
    if flag in ("false", "0", "no"):
        return False
    # 'auto' (or unset): pick by target shape.
    host = grpc_target.split(":", 1)[0].strip().lower()
    is_loopback = host in ("localhost", "127.0.0.1", "0.0.0.0", "::1", "[::1]")
    return not is_loopback


class GrpcClient:
    """gRPC client for the new TensorCost AgentService.Connect bidi stream.

    Only ever holds one open stream at a time. Background asyncio task
    runs in its own thread so the existing synchronous `schedule`-driven
    main loop continues unaffected. `is_connected` reflects the latest
    stream state for HTTP-fallback routing.

    Args:
        grpc_target: host:port, e.g. `gateway.dev.tensorcost.com:443`.
        agent_id: hostname (matches `gpu.agent.hostname` server-side).
        tenant_id: tenant UUID.
        key_id: short stable key identifier (matches `gpu.agent.key_id`).
        hmac_pepper: hex-encoded 32-byte signing key handed by provisioning.
        on_command: async or sync callback the stream invokes when the
            server pushes an InstanceCommand. Receives the cmd as a dict
            and must return `{"status": "succeeded"|"failed"|"in_progress",
            "error"?: str}`.
        use_tls / ca_cert_path: TLS config.
    """

    def __init__(
        self,
        *,
        grpc_target: str,
        agent_id: str,
        tenant_id: str,
        key_id: str,
        hmac_pepper: str,
        on_command: Optional[CommandHandler] = None,
        use_tls: Optional[bool] = None,
        ca_cert_path: Optional[str] = None,
        tls_config: Optional[TlsConfig] = None,
        instance_id: Optional[str] = None,
    ):
        if not key_id:
            raise ValueError("key_id is required (set AGENT_KEY_ID)")
        if not hmac_pepper:
            raise ValueError("hmac_pepper is required (set AGENT_HMAC_PEPPER)")

        self.grpc_target = grpc_target
        self.agent_id = agent_id
        self.tenant_id = tenant_id
        # Per-host instance identity. Forwarded as the `x-tc-instance-id`
        # gRPC metadata header on every Heartbeat RPC so the SaaS can route
        # the beat to the gpu.agent_instance row for this specific container.
        # The proto AgentHeartbeat message doesn't carry the field yet —
        # metadata keeps us wire-compatible with older server builds that
        # will simply ignore the unknown header.
        self.instance_id: Optional[str] = instance_id
        # Auth state — keep in memory only; never log.
        self._key_id = key_id
        self._hmac_pepper = hmac_pepper
        self.on_command = on_command
        # TLS auto-detect: caller can force True/False; if None we pick
        # based on target shape — TLS unless target is a loopback host.
        # Customer agents almost always want TLS (gateway is on a public
        # NLB with an ACM cert); local-loop tests against a plaintext
        # local gateway are the only insecure case.
        self.use_tls = _resolve_tls(use_tls, grpc_target)
        # Resolved TLS hardening config (pins, CA bundle, mTLS). When
        # not None, _create_channel + _verify_pin_via_probe consume it.
        self._tls_config = tls_config
        # Legacy single-CA knob still honoured. If both ca_cert_path
        # and tls_config.ca_bundle_path are set, tls_config wins.
        if tls_config is not None and tls_config.ca_bundle_path:
            self.ca_cert_path = tls_config.ca_bundle_path
        else:
            self.ca_cert_path = ca_cert_path

        self._connected = False
        self._connected_lock = threading.Lock()
        self._consecutive_failures = 0
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        # Async outbound queue (created inside the loop thread).
        self._send_queue: Optional[asyncio.Queue] = None
        # Pre-accept gate: non-hello messages are queued from the moment
        # the caller calls send_metrics / send_instance_update, but the
        # iterator yields them ONLY after the server's ServerAccepted
        # arrives. Without this gate, the server's onAgentMessage handler
        # silently drops every message that lands during the ~1.8s HMAC
        # verify window. Created inside the loop thread.
        self._accepted_event: Optional[asyncio.Event] = None
        # Most recent ServerAccepted info — surfaced for tests / logs.
        self.session_id: Optional[str] = None
        self.heartbeat_interval_seconds: Optional[int] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        with self._connected_lock:
            return self._connected

    def _set_connected(self, value: bool) -> None:
        with self._connected_lock:
            self._connected = value

    def start(self) -> None:
        """Spin up the background asyncio loop + reconnect task."""
        if self._thread and self._thread.is_alive():
            logger.warning("gRPC client already running")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._thread_main, daemon=True, name="grpc-client"
        )
        self._thread.start()
        # No secret material in this log line.
        logger.info("gRPC client started target=%s key_id=%s", self.grpc_target, self._key_id)

    def stop(self) -> None:
        self._stop_event.set()
        loop = self._loop
        if loop is not None:
            loop.call_soon_threadsafe(loop.stop)
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        self._set_connected(False)
        logger.info("gRPC client stopped")

    def send_metrics(self, points: List[Dict[str, Any]]) -> None:
        """Enqueue a MetricBatch with the given points.

        Each point is a dict with the new field names: `instance_id`,
        `ts_unix_ms`, `util_pct`, `mem_pct`, `temp_c`, plus the optional
        identity pair `cloud_id` + `cloud_provider`. The legacy field
        names (`gpu_utilization`, `timestamp`, …) are NOT accepted —
        callers must rename before invoking.

        Server identity model (apps-new agent.proto): each point ships
        EITHER a UUID `instance_id` (HTTP-sync admin tools) OR
        `(cloud_id, cloud_provider)` (cloud-discovered agents like this
        one). Sending neither is a server-side drop; sending both is
        allowed (UUID wins). Customer agents always populate the cloud
        pair — main.py discovers them via IMDS at startup.
        """
        if not points:
            return
        batch = agent_pb2.MetricBatch()
        for p in points:
            mp = batch.points.add()
            mp.instance_id = str(p.get("instance_id", ""))
            mp.ts_unix_ms = int(p.get("ts_unix_ms", 0))
            mp.util_pct = float(p.get("util_pct", 0.0) or 0.0)
            mp.mem_pct = float(p.get("mem_pct", 0.0) or 0.0)
            mp.temp_c = float(p.get("temp_c", 0.0) or 0.0)
            mp.cloud_id = str(p.get("cloud_id", ""))
            mp.cloud_provider = str(p.get("cloud_provider", ""))
            mp.gpu_index = int(p.get("gpu_index", 0) or 0)
            # power_w: 0.0 is the proto3 default, treated server-side as
            # "not available" (server writes NULL when the sync service sees
            # 0.0). Agents that genuinely draw 0 W don't exist in practice
            # (datacenter GPUs idle at ~50 W). The key may be absent from
            # `p` when the cloud-monitor path (no NVML) produces the dict.
            raw_pw = p.get("power_w")
            if raw_pw is not None:
                mp.power_w = float(raw_pw)
            # mem_bw_pct: absent when the cloud-monitor path (no NVML) produces
            # the dict, or when the GPU driver doesn't expose utilization rates.
            # 0.0 proto3 default is treated server-side as "not reported" (NULL
            # stored, same pattern as power_w). Only set if the key is present
            # and non-None so the field stays at 0.0 (omitted) for non-NVML
            # callers and the server's sentinel check applies.
            raw_mbw = p.get("mem_bw_pct")
            if raw_mbw is not None:
                mp.mem_bw_pct = float(raw_mbw)
            # ecc_errors_total: cumulative uncorrected ECC count. 0 is a valid
            # value (no errors since boot) — proto3 zero-default (0) maps to
            # "no errors" in the DB (sync service stores int64 zero-default as 0,
            # not NULL). Only set when the key is explicitly present; absent key
            # means the agent predates migration 024 or the GPU doesn't support ECC.
            raw_ecc = p.get("ecc_errors_total")
            if raw_ecc is not None:
                mp.ecc_errors_total = int(raw_ecc)
            # memory_used_mb: 0.0 IS a valid value (freshly-started GPU can have
            # 0 MB used). Only set when the key is present; absent = agent predating
            # migration 025 or NVML memory info unavailable.
            raw_mem_used = p.get("memory_used_mb")
            if raw_mem_used is not None:
                mp.memory_used_mb = float(raw_mem_used)
            # memory_total_mb: 0.0 is NOT a valid value (any real GPU has positive
            # capacity). Set when present; the server applies > 0 sentinel to
            # store NULL when it receives 0.0 (same as power_w).
            raw_mem_total = p.get("memory_total_mb")
            if raw_mem_total is not None:
                mp.memory_total_mb = float(raw_mem_total)
            raw_clock = p.get("clock_mhz")
            if raw_clock is not None:
                mp.clock_mhz = int(raw_clock)
            raw_mem_clock = p.get("memory_clock_mhz")
            if raw_mem_clock is not None:
                mp.memory_clock_mhz = int(raw_mem_clock)
            raw_nvlink = p.get("nvlink_links_degraded")
            if raw_nvlink is not None:
                mp.nvlink_links_degraded = int(raw_nvlink)
        self._enqueue(agent_pb2.AgentMessage(metric_batch=batch))

    def send_instance_update(
        self,
        instance_id: str = "",
        status: str = "",
        tags: Optional[Dict[str, str]] = None,
        cloud_id: str = "",
        cloud_provider: str = "",
        region: str = "",
        mig_enabled: bool = False,
        mig_partitions: Optional[List[Dict[str, object]]] = None,
    ) -> None:
        """Enqueue a single InstanceUpdate.

        Send EITHER `instance_id` (UUID) OR `(cloud_id, cloud_provider)`
        — see send_metrics for the identity model. Existing callers
        passing only `instance_id` keep working; new callers should
        pass the cloud-id pair so the server's UPSERT branch creates
        the gpu.instance row on first sight.

        MIG fields are omitted unless `mig_partitions` is a non-empty
        list — proto3 has no bool presence and false/[] would wipe inventory.
        """
        upd = agent_pb2.InstanceUpdate(
            instance_id=instance_id,
            status=status,
            cloud_id=cloud_id,
            cloud_provider=cloud_provider,
            region=region or "",
        )
        if tags:
            for k, v in tags.items():
                upd.tags[str(k)] = str(v)
        if mig_partitions:
            upd.mig_enabled = bool(mig_enabled)
            for part in mig_partitions:
                snap = upd.mig_partitions.add()
                snap.partition_id = str(part.get("partition_id") or "")
                snap.profile = str(part.get("profile") or "unknown")
                snap.memory_mb = int(part.get("memory_mb") or 0)
                snap.compute_slice = int(part.get("compute_slice") or 0)
                snap.gpu_index = int(part.get("gpu_index") or 0)
        self._enqueue(agent_pb2.AgentMessage(instance_update=upd))

    def send_command_result(
        self, command_id: str, status: str, error: str = ""
    ) -> None:
        """Enqueue a CommandResult — the server's response to a previously
        pushed InstanceCommand."""
        cr = agent_pb2.CommandResult(
            command_id=command_id, status=status, error=error or ""
        )
        self._enqueue(agent_pb2.AgentMessage(command_result=cr))

    # ------------------------------------------------------------------
    # Internals: thread + asyncio loop
    # ------------------------------------------------------------------

    def _thread_main(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._run_with_reconnect())
        except Exception as e:
            logger.error("gRPC client thread crashed: %s", e)
        finally:
            try:
                loop.close()
            except Exception:
                pass
            self._loop = None

    async def _run_with_reconnect(self) -> None:
        """Outer loop: reconnect with exponential backoff until stopped."""
        # Create the queue + accept-gate inside the loop where they'll
        # be consumed.
        self._send_queue = asyncio.Queue(maxsize=1000)
        self._accepted_event = asyncio.Event()
        while not self._stop_event.is_set():
            try:
                await self._run_single_stream()
            except asyncio.CancelledError:
                break
            except PinMismatchError as e:
                # Pin mismatch is a security-relevant signal — log at
                # ERROR so it shows up in operator dashboards. Reconnect
                # backoff still applies; if the gateway cert legitimately
                # rotated, the operator must update AGENT_TLS_PIN_SHA256_GRPC.
                logger.error("gRPC TLS pin mismatch: %s", e)
            except Exception as e:
                logger.warning("gRPC stream loop error: %s", e)
            self._set_connected(False)
            self._consecutive_failures += 1
            backoff = min(
                _BACKOFF_BASE * (_BACKOFF_MULTIPLIER ** (self._consecutive_failures - 1)),
                _BACKOFF_CAP,
            )
            jitter = random.uniform(0, _BACKOFF_JITTER * backoff)
            wait_time = backoff + jitter
            logger.info("gRPC reconnecting in %.1fs", wait_time)
            try:
                # Allow stop() to wake us.
                await asyncio.wait_for(asyncio.sleep(wait_time), timeout=wait_time + 1)
            except asyncio.TimeoutError:
                pass
            if self._stop_event.is_set():
                break

    def _create_channel(self) -> grpc.aio.Channel:
        options = [
            ("grpc.keepalive_time_ms", 30000),
            ("grpc.keepalive_timeout_ms", 10000),
            ("grpc.keepalive_permit_without_calls", True),
            ("grpc.http2.max_pings_without_data", 0),
        ]
        if self.use_tls:
            # Pin probe FIRST. gRPC's credential API accepts a CA bundle
            # and mTLS material directly, but it does NOT expose the
            # peer chain to user code. We do a stdlib TLS probe against
            # the same host:port to validate the SPKI pin (RFC 7469
            # §2.4) before the gRPC channel is opened. This runs at
            # most once per reconnect — same cadence as the prior
            # `ssl_channel_credentials()` call. On pin mismatch we
            # raise; the reconnect loop catches, backs off, and re-probes.
            self._verify_pin_via_probe()

            root_certificates: Optional[bytes] = None
            if self.ca_cert_path:
                with open(self.ca_cert_path, "rb") as f:
                    root_certificates = f.read()

            private_key: Optional[bytes] = None
            certificate_chain: Optional[bytes] = None
            cfg = self._tls_config
            if cfg is not None and cfg.mtls_enabled:
                # Native mTLS: gRPC reads these directly. Same files
                # the HTTPS client uses; agent presents one identity
                # on both planes.
                private_key = cfg.mtls_key_pem
                certificate_chain = cfg.mtls_cert_pem

            creds = grpc.ssl_channel_credentials(
                root_certificates=root_certificates,
                private_key=private_key,
                certificate_chain=certificate_chain,
            )
            return grpc.aio.secure_channel(self.grpc_target, creds, options=options)
        return grpc.aio.insecure_channel(self.grpc_target, options=options)

    def _verify_pin_via_probe(self) -> None:
        """Do a stdlib TLS handshake against `self.grpc_target` and
        verify that the peer chain matches one of the configured gRPC
        SPKI pins. No-op when pinning is not configured.

        Why a side probe? gRPC's `ssl_channel_credentials` does the
        right verification but does not surface the chain to Python.
        Rather than ship a custom transport, we open a plain TLS
        socket to the same host:port — using the same root store the
        gRPC channel will use — extract the verified chain via
        `SSLSocket.get_verified_chain()`, and check the pin. Any chain
        the operator's CA bundle accepts here is also what gRPC will
        accept; chains rotate at the gateway, not per-RPC, so this
        probe represents the same trust decision the gRPC channel
        will make.
        """
        import socket as _socket  # noqa: WPS433
        import ssl as _ssl  # noqa: WPS433

        cfg = self._tls_config
        if cfg is None or not cfg.grpc_pinning_enabled:
            return

        host, _, port_str = self.grpc_target.partition(":")
        port = int(port_str) if port_str else 443

        if cfg.ca_bundle_path:
            ctx = _ssl.create_default_context(cafile=cfg.ca_bundle_path)
        else:
            ctx = _ssl.create_default_context()
        if cfg.mtls_enabled:
            ctx.load_cert_chain(
                certfile=cfg.mtls_cert_path,
                keyfile=cfg.mtls_key_path,
            )
        # Normal verification — pinning is layered on top.
        ctx.check_hostname = True
        ctx.verify_mode = _ssl.CERT_REQUIRED

        with _socket.create_connection((host, port), timeout=10) as raw:
            with ctx.wrap_socket(raw, server_hostname=host) as sslsock:
                chain = extract_chain_from_sslsocket(sslsock)
        assert_chain_pinned(
            chain, cfg.grpc_pins, peer_label=f"gRPC gateway {host}:{port}"
        )

    async def _run_single_stream(self) -> None:
        """Open one stream — send hello, multiplex sends + server messages."""
        channel = self._create_channel()
        # Reset the accept gate for this connection. Any messages
        # enqueued during the previous stream's death + this one's
        # handshake will be held by _outbound_iterator until the
        # server emits ServerAccepted again.
        if self._accepted_event is None:
            self._accepted_event = asyncio.Event()
        else:
            self._accepted_event.clear()
        heartbeat_task: Optional[asyncio.Task] = None
        try:
            stub = agent_pb2_grpc.AgentServiceStub(channel)
            # grpc.aio bidi: pass an async iterator for the request stream.
            request_iter = self._outbound_iterator()
            response_stream = stub.Connect(request_iter)

            # Mark connected as soon as the stream is open. Hello is the
            # first thing the iterator yields; if it gets rejected the
            # response_stream will surface UNAUTHENTICATED below.
            self._set_connected(True)
            self._consecutive_failures = 0
            logger.info("gRPC stream connected target=%s tls=%s", self.grpc_target, self.use_tls)

            # Periodic unary Heartbeat keeps gpu.agent.last_heartbeat_at
            # fresh even during idle periods (no metric ticks). Started
            # in parallel with the response read; spins waiting for the
            # accepted-gate to unblock so we don't beat against an
            # un-handshaked stream.
            heartbeat_task = asyncio.create_task(self._heartbeat_loop(stub))

            async for server_msg in response_stream:
                if self._stop_event.is_set():
                    break
                await self._handle_server_message(server_msg)
        except grpc.aio.AioRpcError as e:
            code = e.code()
            if code in (
                grpc.StatusCode.UNAUTHENTICATED,
                grpc.StatusCode.PERMISSION_DENIED,
            ):
                # Auth failures usually mean wrong key_id / pepper / clock skew.
                # Sleep longer here than network errors — retrying every 1s
                # against a permission denial just spams the auth log.
                logger.error(
                    "gRPC auth rejected (%s): %s. Check AGENT_KEY_ID and "
                    "AGENT_HMAC_PEPPER, and that the agent row exists in gpu.agent.",
                    code,
                    e.details(),
                )
                await asyncio.sleep(60)
            else:
                logger.warning("gRPC error %s: %s", code, e.details())
        finally:
            if heartbeat_task is not None:
                heartbeat_task.cancel()
                try:
                    await heartbeat_task
                except (asyncio.CancelledError, Exception):
                    pass
            try:
                await channel.close()
            except Exception:
                pass

    async def _outbound_iterator(self):
        """Yield AgentMessages: hello first, wait for ServerAccepted,
        then drain queue items.

        The pre-accept wait is critical. Server-side `onAgentMessage`
        drops every non-hello message it receives before the async
        HMAC verify completes (~1.8s on TLS). Without the gate, a
        caller that calls send_metrics()/send_instance_update()
        synchronously after start() has its first-burst writes
        silently lost — visible as "agent connected, heartbeat fine,
        but no rows in gpu.instance".
        """
        hello = _HELLO_BUILDER(
            tenant_id=self.tenant_id,
            agent_id=self.agent_id,
            key_id=self._key_id,
            hmac_pepper=self._hmac_pepper,
            version=_AGENT_VERSION,
        )
        proto_hello = agent_pb2.AgentHello(
            tenant_id=hello.tenant_id,
            agent_id=hello.agent_id,
            version=hello.version,
            key_id=hello.key_id,
            nonce=hello.nonce,
            timestamp_unix=hello.timestamp_unix,
            hmac=hello.hmac,
        )
        yield agent_pb2.AgentMessage(hello=proto_hello)

        # Block here until the server emits ServerAccepted. Bounded by
        # stop_event polling so stop() unblocks promptly.
        while not self._stop_event.is_set():
            assert self._accepted_event is not None
            try:
                await asyncio.wait_for(self._accepted_event.wait(), timeout=2.0)
                break
            except asyncio.TimeoutError:
                continue
        if self._stop_event.is_set():
            return

        # Drain the queue until stopped. We use a short timeout so the
        # iterator exits promptly when stop() is called.
        while not self._stop_event.is_set():
            try:
                assert self._send_queue is not None
                msg = await asyncio.wait_for(self._send_queue.get(), timeout=5.0)
            except asyncio.TimeoutError:
                continue
            if msg is None:
                # Poison pill: clean shutdown.
                return
            yield msg

    def _enqueue(self, msg: "agent_pb2.AgentMessage") -> None:
        """Thread-safe handoff into the async queue. Drops oldest on overflow."""
        loop = self._loop
        if loop is None or self._send_queue is None:
            # Not running — drop silently. The HTTP fallback in main.py
            # picks up the slack when the gRPC stream is unavailable.
            return

        def _put() -> None:
            assert self._send_queue is not None
            if self._send_queue.full():
                try:
                    self._send_queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                logger.warning("gRPC outbound queue full — dropped oldest message")
            try:
                self._send_queue.put_nowait(msg)
            except asyncio.QueueFull:
                pass

        loop.call_soon_threadsafe(_put)

    async def _handle_server_message(self, msg: "agent_pb2.ServerMessage") -> None:
        which = msg.WhichOneof("payload")
        if which == "accepted":
            self.session_id = msg.accepted.session_id
            self.heartbeat_interval_seconds = msg.accepted.heartbeat_interval_seconds
            logger.info(
                "gRPC accepted session_id=%s heartbeat=%ds",
                self.session_id,
                self.heartbeat_interval_seconds,
            )
            # Open the pre-accept gate so _outbound_iterator can start
            # draining queued metric/instance messages.
            if self._accepted_event is not None:
                self._accepted_event.set()
        elif which == "command":
            await self._handle_command(msg.command)
        elif which == "ping":
            # Server ping — purely a keepalive, no response required.
            logger.debug("gRPC server ping ts=%d", msg.ping.ts_unix_ms)
        else:
            logger.warning("Unknown server message payload: %s", which)

    async def _heartbeat_loop(self, stub: "agent_pb2_grpc.AgentServiceStub") -> None:
        """Periodic unary Heartbeat RPC.

        Server bumps `gpu.agent.last_heartbeat_at` on each successful
        Heartbeat call. Without this, an idle agent (no metric ticks
        for ~30s) would appear "stale" in the UI even though the gRPC
        stream is healthy. Default cadence comes from ServerAccepted
        (typically 30s); we wait for the accept-gate before starting
        so we don't fire against an un-handshaked stream.
        """
        # Block until the stream is accepted.
        if self._accepted_event is not None:
            try:
                await self._accepted_event.wait()
            except asyncio.CancelledError:
                return
        # Default to 30s if the server didn't supply one (shouldn't
        # happen in practice — ServerAccepted always sets the field).
        interval = float(self.heartbeat_interval_seconds or 30)
        try:
            while not self._stop_event.is_set():
                await asyncio.sleep(interval)
                if self._stop_event.is_set():
                    return
                try:
                    import os as _os
                    import time as _time
                    import base64 as _b64
                    from auth.hmac import compute_agent_hello_hmac as _compute_hmac

                    # 2026-05-17 GPC-W2 hardening — the server now requires an
                    # HMAC on every unary Heartbeat. We reuse the same canonical
                    # derivation as the AgentHello handshake
                    # (`tenantId || "\n" || agentId || "\n" || nonce || "\n" ||
                    # be8(timestampUnix)`) and pass the inputs via gRPC metadata
                    # rather than extending the AgentHeartbeat proto — wire-compat
                    # with older servers that ignore unknown metadata. The proto
                    # message itself is unchanged.
                    ts_unix = int(_time.time())  # SECONDS, matches verifier window
                    nonce = _os.urandom(16)
                    hmac_bytes = _compute_hmac(
                        secret=self._hmac_pepper,
                        tenant_id=self.tenant_id,
                        agent_id=self.agent_id,
                        nonce=nonce,
                        timestamp_unix=ts_unix,
                    )
                    metadata_list = [
                        ("x-tc-key-id", self._key_id),
                        ("x-tc-nonce", _b64.b64encode(nonce).decode("ascii")),
                        ("x-tc-ts", str(ts_unix)),
                        ("x-tc-hmac", _b64.b64encode(hmac_bytes).decode("ascii")),
                    ]
                    # instance_id is plumbed as a metadata header so the
                    # SaaS can route the heartbeat to the per-host
                    # gpu.agent_instance row without a proto schema change.
                    # Servers that don't know the header ignore it silently.
                    if self.instance_id:
                        metadata_list.append(("x-tc-instance-id", self.instance_id))
                    metadata = tuple(metadata_list)
                    beat = agent_pb2.AgentHeartbeat(
                        tenant_id=self.tenant_id,
                        agent_id=self.agent_id,
                        ts_unix_ms=int(_time.time() * 1000),
                    )
                    await stub.Heartbeat(beat, timeout=5.0, metadata=metadata)
                    logger.debug("gRPC heartbeat sent (hmac-signed)")
                except (grpc.aio.AioRpcError, asyncio.TimeoutError) as e:
                    # A heartbeat failure on its own doesn't tear down
                    # the stream — the bidi stream's own error path
                    # will surface the real problem if the stream is
                    # actually dead. Just log and try again next tick.
                    logger.warning("gRPC heartbeat failed: %s", e)
        except asyncio.CancelledError:
            return

    async def _handle_command(self, cmd: "agent_pb2.InstanceCommand") -> None:
        logger.info(
            "gRPC command received command_id=%s instance_id=%s type=%s",
            cmd.command_id,
            cmd.instance_id,
            cmd.type,
        )
        if not self.on_command:
            self.send_command_result(cmd.command_id, "failed", "no handler")
            return

        params = dict(cmd.params)
        cloud_provider = cmd.cloud_provider or params.get("cloud_provider", "")
        region = cmd.region or params.get("region", "")
        if cloud_provider:
            params.setdefault("cloud_provider", cloud_provider)
        if region:
            params.setdefault("region", region)
        cmd_dict = {
            "command_id": cmd.command_id,
            "instance_id": cmd.instance_id,
            "type": cmd.type,
            "params": params,
            "cloud_provider": cloud_provider,
            "region": region,
        }
        try:
            result = self.on_command(cmd_dict)
            if asyncio.iscoroutine(result):
                result = await result
            status = result.get("status", "succeeded") if isinstance(result, dict) else "succeeded"
            error = result.get("error", "") if isinstance(result, dict) else ""
            self.send_command_result(cmd.command_id, status, error)
        except Exception as e:
            logger.error("command handler raised: %s", e)
            self.send_command_result(cmd.command_id, "failed", str(e))
