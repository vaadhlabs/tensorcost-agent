"""Tests for the rewritten src/grpc_client.py — focuses on the hello
message shape and the send-method serialisation. The gRPC stream
lifecycle itself is exercised by an integration smoke against a real
gpu-service in CI, not here.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "proto"))


def _import_proto_safely():
    """Import the generated proto stubs, surviving the per-cloud test
    modules that stub `google` / `google.cloud` as bare Mocks. The real
    google.protobuf is required by agent_pb2; if a stub is in
    sys.modules from a sibling test, drop just the stub-shaped entries
    before importing — and restore them afterward so subsequent tests
    that depended on them still see the Mock."""
    import importlib
    saved = {}
    for k in list(sys.modules):
        if k == "google" or k.startswith("google."):
            mod = sys.modules[k]
            # A real package has a list __path__; a Mock doesn't.
            if not hasattr(mod, "__path__") or not isinstance(mod.__path__, list):
                saved[k] = sys.modules.pop(k)
    try:
        import agent_pb2 as _pb2  # noqa: F401
        return _pb2
    finally:
        # Restore Mock stubs so tests that come after us aren't broken.
        for k, v in saved.items():
            sys.modules.setdefault(k, v)


agent_pb2 = _import_proto_safely()  # noqa: F401

from grpc_client import GrpcClient  # noqa: E402
from auth.hmac import compute_agent_hello_hmac  # noqa: E402


@pytest.fixture
def client_kwargs():
    return dict(
        grpc_target="localhost:50051",
        agent_id="dev-agent-1",
        tenant_id="00000000-0000-0000-0000-000000000001",
        key_id="dev-key-id",
        hmac_pepper="deadbeef" * 8,
    )


class TestConstruction:
    def test_requires_key_id(self, client_kwargs):
        client_kwargs["key_id"] = ""
        with pytest.raises(ValueError, match="key_id"):
            GrpcClient(**client_kwargs)

    def test_requires_hmac_pepper(self, client_kwargs):
        client_kwargs["hmac_pepper"] = ""
        with pytest.raises(ValueError, match="hmac_pepper"):
            GrpcClient(**client_kwargs)

    def test_starts_disconnected(self, client_kwargs):
        c = GrpcClient(**client_kwargs)
        assert not c.is_connected


class TestHelloConstruction:
    """Build the same proto AgentHello the outbound iterator would emit
    on stream open, and verify the HMAC re-computes to the same digest
    the server would produce. This is the wire-compat smoke test."""

    def test_outbound_iterator_emits_signed_hello_first(self, client_kwargs, monkeypatch):
        from datetime import datetime, timezone

        # Pin the timestamp so we can re-derive the expected HMAC.
        fixed_now = 1700000000

        from auth import hmac as hmac_mod

        def fake_sign(*, tenant_id, agent_id, key_id, hmac_pepper, version, now=None):
            return hmac_mod.sign_agent_hello(
                tenant_id=tenant_id,
                agent_id=agent_id,
                key_id=key_id,
                hmac_pepper=hmac_pepper,
                version=version,
                now=fixed_now,
            )

        # Patch the rebindable builder so the hello carries fixed_now.
        import grpc_client as gc_mod
        monkeypatch.setattr(gc_mod, "_HELLO_BUILDER", fake_sign)

        c = GrpcClient(**client_kwargs)
        # Fake the loop-side queue so the iterator can be driven directly.
        import asyncio
        c._send_queue = asyncio.Queue()
        c._stop_event.set()  # Make iterator exit after the hello + first poll.

        async def drive():
            messages = []
            async for m in c._outbound_iterator():
                messages.append(m)
            return messages

        messages = asyncio.run(drive())
        assert len(messages) >= 1
        hello_msg = messages[0]
        assert hello_msg.WhichOneof("payload") == "hello"
        h = hello_msg.hello
        assert h.tenant_id == client_kwargs["tenant_id"]
        assert h.agent_id == client_kwargs["agent_id"]
        assert h.key_id == client_kwargs["key_id"]
        assert h.timestamp_unix == fixed_now
        assert len(h.nonce) == 16
        assert len(h.hmac) == 32
        # Verify the wire HMAC by re-deriving it with the same pepper.
        expected = compute_agent_hello_hmac(
            secret=client_kwargs["hmac_pepper"],
            tenant_id=h.tenant_id,
            agent_id=h.agent_id,
            nonce=h.nonce,
            timestamp_unix=h.timestamp_unix,
        )
        assert h.hmac == expected


class TestMetricSerialisation:
    def test_send_metrics_uses_new_field_names(self, client_kwargs, monkeypatch):
        c = GrpcClient(**client_kwargs)
        captured = []
        monkeypatch.setattr(c, "_enqueue", lambda m: captured.append(m))
        c.send_metrics([
            {
                "instance_id": "uuid-1",
                "ts_unix_ms": 1700000000123,
                "util_pct": 75.5,
                "mem_pct": 60.0,
                "temp_c": 65.0,
            },
        ])
        assert len(captured) == 1
        msg = captured[0]
        assert msg.WhichOneof("payload") == "metric_batch"
        assert len(msg.metric_batch.points) == 1
        p = msg.metric_batch.points[0]
        assert p.instance_id == "uuid-1"
        assert p.ts_unix_ms == 1700000000123
        assert p.util_pct == 75.5
        assert p.mem_pct == 60.0
        assert p.temp_c == 65.0

    def test_send_metrics_noop_for_empty_list(self, client_kwargs, monkeypatch):
        c = GrpcClient(**client_kwargs)
        captured = []
        monkeypatch.setattr(c, "_enqueue", lambda m: captured.append(m))
        c.send_metrics([])
        assert captured == []

    def test_send_metrics_tolerates_missing_fields(self, client_kwargs, monkeypatch):
        """Missing keys default to 0 / "" — never crash on partial input."""
        c = GrpcClient(**client_kwargs)
        captured = []
        monkeypatch.setattr(c, "_enqueue", lambda m: captured.append(m))
        c.send_metrics([{"instance_id": "i", "ts_unix_ms": 1}])
        assert len(captured) == 1
        p = captured[0].metric_batch.points[0]
        assert p.util_pct == 0.0
        assert p.mem_pct == 0.0
        assert p.temp_c == 0.0


class TestInstanceUpdate:
    def test_send_instance_update_includes_tags(self, client_kwargs, monkeypatch):
        c = GrpcClient(**client_kwargs)
        captured = []
        monkeypatch.setattr(c, "_enqueue", lambda m: captured.append(m))
        c.send_instance_update("uuid-1", "running", {"team": "ml"})
        msg = captured[0]
        assert msg.WhichOneof("payload") == "instance_update"
        assert msg.instance_update.instance_id == "uuid-1"
        assert msg.instance_update.status == "running"
        assert dict(msg.instance_update.tags) == {"team": "ml"}

    def test_send_instance_update_includes_region(self, client_kwargs, monkeypatch):
        c = GrpcClient(**client_kwargs)
        captured = []
        monkeypatch.setattr(c, "_enqueue", lambda m: captured.append(m))
        c.send_instance_update(
            cloud_id="i-abc",
            cloud_provider="aws",
            region="us-west-2",
            status="running",
        )
        upd = captured[0].instance_update
        assert upd.cloud_id == "i-abc"
        assert upd.cloud_provider == "aws"
        assert upd.region == "us-west-2"


class TestCommandHandling:
    def test_handle_command_hoists_cloud_provider_and_region(self, client_kwargs):
        c = GrpcClient(**client_kwargs)
        received = {}

        def on_command(cmd):
            received.update(cmd)
            return {"status": "succeeded"}

        c.on_command = on_command
        cmd = agent_pb2.InstanceCommand(
            command_id="cmd-1",
            instance_id="i-abc",
            type="stop",
            cloud_provider="aws",
            region="eu-west-1",
        )
        import asyncio
        asyncio.run(c._handle_command(cmd))
        assert received["cloud_provider"] == "aws"
        assert received["region"] == "eu-west-1"
        assert received["params"]["cloud_provider"] == "aws"
        assert received["params"]["region"] == "eu-west-1"


class TestCommandResult:
    def test_send_command_result_shape(self, client_kwargs, monkeypatch):
        c = GrpcClient(**client_kwargs)
        captured = []
        monkeypatch.setattr(c, "_enqueue", lambda m: captured.append(m))
        c.send_command_result("cmd-1", "succeeded")
        msg = captured[0]
        assert msg.WhichOneof("payload") == "command_result"
        assert msg.command_result.command_id == "cmd-1"
        assert msg.command_result.status == "succeeded"
        assert msg.command_result.error == ""

    def test_send_command_result_with_error(self, client_kwargs, monkeypatch):
        c = GrpcClient(**client_kwargs)
        captured = []
        monkeypatch.setattr(c, "_enqueue", lambda m: captured.append(m))
        c.send_command_result("cmd-1", "failed", "boom")
        assert captured[0].command_result.error == "boom"
