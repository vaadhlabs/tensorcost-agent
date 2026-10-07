"""Tests for src/auth/hmac.py — must agree byte-for-byte with the
TypeScript implementation in apps-new/packages/agent-sdk/src/hmac.ts and
the verifier at backend/services/gpu-service/src/grpc/agent-auth/
hmac-verifier.service.ts. Failing here = handshake will be rejected
on the wire."""

import os
import sys
import time
import hashlib
import hmac as _hmac
import struct

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from auth.hmac import (  # noqa: E402
    compute_agent_hello_hmac,
    fresh_nonce,
    sign_agent_hello,
)


# Frozen test vector — independent re-implementation so we can detect drift
# in the helper without trusting the helper to verify itself.
def _expected_canonical(tenant_id: str, agent_id: str, nonce: bytes, ts: int) -> bytes:
    return (
        tenant_id.encode("utf-8")
        + b"\n"
        + agent_id.encode("utf-8")
        + b"\n"
        + nonce
        + b"\n"
        + struct.pack(">q", ts)
    )


class TestComputeAgentHelloHmac:
    def test_deterministic_for_fixed_inputs(self):
        """Same inputs → same output. Hex secret, fixed nonce, fixed ts."""
        secret_hex = "deadbeef" * 8  # 32 bytes
        nonce = b"\x01" * 16
        ts = 1_700_000_000
        out1 = compute_agent_hello_hmac(
            secret=secret_hex,
            tenant_id="tenant-1",
            agent_id="agent-1",
            nonce=nonce,
            timestamp_unix=ts,
        )
        out2 = compute_agent_hello_hmac(
            secret=secret_hex,
            tenant_id="tenant-1",
            agent_id="agent-1",
            nonce=nonce,
            timestamp_unix=ts,
        )
        assert out1 == out2
        assert len(out1) == 32

    def test_matches_independent_recomputation(self):
        """Cross-check against an inline HMAC re-derivation — guards against
        any future canonical-string change going un-noticed."""
        secret = bytes.fromhex("a" * 64)
        nonce = b"\x42" * 16
        ts = 1_700_000_000
        out = compute_agent_hello_hmac(
            secret=secret,
            tenant_id="t",
            agent_id="a",
            nonce=nonce,
            timestamp_unix=ts,
        )
        canonical = _expected_canonical("t", "a", nonce, ts)
        expected = _hmac.new(secret, canonical, hashlib.sha256).digest()
        assert out == expected

    def test_signature_changes_when_nonce_changes(self):
        secret = b"\x10" * 32
        ts = 1_700_000_000
        out_a = compute_agent_hello_hmac(
            secret=secret, tenant_id="t", agent_id="a",
            nonce=b"\x01" * 16, timestamp_unix=ts,
        )
        out_b = compute_agent_hello_hmac(
            secret=secret, tenant_id="t", agent_id="a",
            nonce=b"\x02" * 16, timestamp_unix=ts,
        )
        assert out_a != out_b

    def test_signature_changes_when_timestamp_changes(self):
        secret = b"\x10" * 32
        nonce = b"\x01" * 16
        out_a = compute_agent_hello_hmac(
            secret=secret, tenant_id="t", agent_id="a",
            nonce=nonce, timestamp_unix=1_700_000_000,
        )
        out_b = compute_agent_hello_hmac(
            secret=secret, tenant_id="t", agent_id="a",
            nonce=nonce, timestamp_unix=1_700_000_001,
        )
        assert out_a != out_b

    def test_accepts_bytes_or_hex_secret(self):
        raw = bytes.fromhex("ab" * 32)
        out_hex = compute_agent_hello_hmac(
            secret="ab" * 32, tenant_id="t", agent_id="a",
            nonce=b"\x00" * 16, timestamp_unix=1,
        )
        out_bytes = compute_agent_hello_hmac(
            secret=raw, tenant_id="t", agent_id="a",
            nonce=b"\x00" * 16, timestamp_unix=1,
        )
        assert out_hex == out_bytes

    def test_rejects_invalid_secret_type(self):
        with pytest.raises(TypeError):
            compute_agent_hello_hmac(
                secret=12345,  # type: ignore[arg-type]
                tenant_id="t", agent_id="a",
                nonce=b"", timestamp_unix=1,
            )

    def test_canonical_uses_big_endian_int64_timestamp(self):
        """Timestamp is BE int64 — verify by replicating the struct.pack."""
        secret = b"\x00" * 32
        nonce = b""
        ts = 0x0102_0304_0506_0708
        out = compute_agent_hello_hmac(
            secret=secret, tenant_id="", agent_id="",
            nonce=nonce, timestamp_unix=ts,
        )
        canonical = b"" + b"\n" + b"" + b"\n" + nonce + b"\n" + struct.pack(">q", ts)
        expected = _hmac.new(secret, canonical, hashlib.sha256).digest()
        assert out == expected


class TestFreshNonce:
    def test_returns_16_bytes_by_default(self):
        n = fresh_nonce()
        assert isinstance(n, bytes)
        assert len(n) == 16

    def test_two_calls_produce_different_nonces(self):
        # 1-in-2^128 chance of collision — practically impossible.
        assert fresh_nonce() != fresh_nonce()


class TestSignAgentHello:
    def test_builds_a_populated_hello(self):
        hello = sign_agent_hello(
            tenant_id="tenant-1",
            agent_id="agent-1",
            key_id="dev-key-id",
            hmac_pepper="ab" * 32,
            now=1_700_000_000,
        )
        assert hello.tenant_id == "tenant-1"
        assert hello.agent_id == "agent-1"
        assert hello.key_id == "dev-key-id"
        assert hello.timestamp_unix == 1_700_000_000
        assert isinstance(hello.nonce, bytes) and len(hello.nonce) == 16
        assert isinstance(hello.hmac, bytes) and len(hello.hmac) == 32

    def test_signature_verifies_via_local_recompute(self):
        """The HMAC carried on the hello must equal a fresh recompute over
        the SAME canonical bytes — what the server does on receipt."""
        hello = sign_agent_hello(
            tenant_id="t", agent_id="a", key_id="k",
            hmac_pepper="cd" * 32, now=42,
        )
        recomputed = compute_agent_hello_hmac(
            secret="cd" * 32,
            tenant_id="t",
            agent_id="a",
            nonce=hello.nonce,
            timestamp_unix=hello.timestamp_unix,
        )
        assert hello.hmac == recomputed

    def test_now_defaults_to_current_time(self):
        before = int(time.time())
        hello = sign_agent_hello(
            tenant_id="t", agent_id="a", key_id="k",
            hmac_pepper="00" * 32,
        )
        after = int(time.time())
        # Server allows ±300s skew — we just want to confirm we used wall-clock.
        assert before <= hello.timestamp_unix <= after

    def test_nonce_changes_per_call_for_replay_safety(self):
        """Server SETNX-rejects reused (tenant,key,nonce) — clients MUST
        roll a fresh nonce every hello."""
        h1 = sign_agent_hello(
            tenant_id="t", agent_id="a", key_id="k",
            hmac_pepper="00" * 32, now=1,
        )
        h2 = sign_agent_hello(
            tenant_id="t", agent_id="a", key_id="k",
            hmac_pepper="00" * 32, now=1,
        )
        assert h1.nonce != h2.nonce
        assert h1.hmac != h2.hmac
