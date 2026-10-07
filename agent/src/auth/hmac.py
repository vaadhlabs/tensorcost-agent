"""
HMAC-SHA256 helpers for the AgentHello handshake.

Mirrors `apps-new/packages/agent-sdk/src/hmac.ts` byte-for-byte. The
server-side verifier (gpu-service `HmacVerifierService`) recomputes the
identical canonical string and compares in constant time, so any drift
between this file and the TypeScript counterpart will immediately
manifest as `agent hello rejected: HMAC mismatch` on the gateway side.

Canonical signed-over bytes:
    tenantId || "\\n" || agentId || "\\n" || nonce || "\\n" || be8(timestampUnix)

Key model (matches the TS docstring): the agent holds the already-derived
signing key bytes (`hmac_pepper` env var, hex-encoded) along with a
`key_id` that selects which `gpu.agent` row the server should look up.
The pepper proper never leaves the server; provisioning hands the agent
the derived signingKey blob directly so this side never has to know
the pepper. See AGENT_COMPATIBILITY.md §6 for the full provisioning flow.
"""

from __future__ import annotations

import hmac as _hmac
import hashlib
import os
import struct
import time
from dataclasses import dataclass
from typing import Optional, Union


__all__ = [
    "compute_agent_hello_hmac",
    "fresh_nonce",
    "sign_agent_hello",
    "SignedHello",
]


def _coerce_secret(secret: Union[str, bytes, bytearray]) -> bytes:
    """Accept hex-string OR raw-bytes secrets, return bytes."""
    if isinstance(secret, (bytes, bytearray)):
        return bytes(secret)
    if isinstance(secret, str):
        # Strict hex decoding — fail loud if the operator pasted the wrong thing.
        return bytes.fromhex(secret)
    raise TypeError(f"secret must be str (hex) or bytes, got {type(secret).__name__}")


def fresh_nonce(num_bytes: int = 16) -> bytes:
    """16 cryptographically-random bytes — the anti-replay nonce."""
    return os.urandom(num_bytes)


def compute_agent_hello_hmac(
    *,
    secret: Union[str, bytes, bytearray],
    tenant_id: str,
    agent_id: str,
    nonce: bytes,
    timestamp_unix: int,
) -> bytes:
    """
    Compute HMAC-SHA256 over the canonical byte string:
      tenantId || "\\n" || agentId || "\\n" || nonce || "\\n" || be8(timestampUnix)

    Returns the 32-byte digest. The server compares this against
    HMAC-SHA256(pepper-derived signing key, canonical) under
    `timingSafeEqual`.
    """
    secret_bytes = _coerce_secret(secret)
    ts_buf = struct.pack(">q", int(timestamp_unix))  # signed big-endian int64
    canonical = b"".join(
        [
            tenant_id.encode("utf-8"),
            b"\n",
            agent_id.encode("utf-8"),
            b"\n",
            nonce,
            b"\n",
            ts_buf,
        ]
    )
    return _hmac.new(secret_bytes, canonical, hashlib.sha256).digest()


@dataclass(frozen=True)
class SignedHello:
    """A populated AgentHello ready for the wire (proto-agnostic POD).

    The gRPC client builds the proto message from these fields; the test
    suite asserts on the dataclass directly so we don't need a proto
    runtime in the unit tests. The proto's bytes-typed `nonce` and
    `hmac` fields take the raw bytes here.
    """

    tenant_id: str
    agent_id: str
    version: str
    key_id: str
    nonce: bytes
    timestamp_unix: int
    hmac: bytes


def sign_agent_hello(
    *,
    tenant_id: str,
    agent_id: str,
    key_id: str,
    hmac_pepper: Union[str, bytes, bytearray],
    version: str = "unified-gpu-agent/2.0.0",
    now: Optional[int] = None,
) -> SignedHello:
    """
    Build a populated AgentHello with a fresh nonce + signature.

    Parameters
    ----------
    tenant_id, agent_id, key_id
        Identity fields. `agent_id` must equal the `gpu.agent.hostname`
        the operator registered server-side. `key_id` selects the row.
    hmac_pepper
        The 32-byte signing-key blob handed out at provisioning time.
        Accept hex string (preferred — env-var friendly) or raw bytes.
        Despite the name this is NOT the server-side pepper itself; it's
        the per-agent derived key (`HMAC(pepper, plaintext_secret)`)
        that provisioning emitted. We retain the variable name for
        operator continuity with the dev-agent docs.
    version
        Agent version string surfaced in the hello.
    now
        Unix seconds — overridable for deterministic tests.
    """
    ts = int(now if now is not None else time.time())
    nonce = fresh_nonce()
    digest = compute_agent_hello_hmac(
        secret=hmac_pepper,
        tenant_id=tenant_id,
        agent_id=agent_id,
        nonce=nonce,
        timestamp_unix=ts,
    )
    return SignedHello(
        tenant_id=tenant_id,
        agent_id=agent_id,
        version=version,
        key_id=key_id,
        nonce=nonce,
        timestamp_unix=ts,
        hmac=digest,
    )
