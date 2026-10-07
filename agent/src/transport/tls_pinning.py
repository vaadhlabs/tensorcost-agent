"""
TLS certificate pinning + optional mTLS — §8.2 Blocker #6.

Background
----------
The agent talks to two backend planes:

  * HTTPS  → `BACKEND_API_URL` (gateway.<env>.tensorcost.com / 443)
  * gRPC   → `GRPC_TARGET`     (grpc.<env>.tensorcost.com / 443)

Both used `requests` defaults / `grpc.ssl_channel_credentials()` with
the system trust store. A binary that can stop / terminate customer
EC2 instances should not silently trust *every* CA in the host's
bundle. A misissued cert from any one of ~150 CAs would let an
on-path attacker MITM the metric stream and (worse) the actuator
command stream.

Posture
-------
This module is **opt-in**. Default deploys (no pin env set) keep the
prior behaviour — system roots, no pinning — because customer
environments routinely run TLS-terminating corporate proxies and a
default-on pin would brick the first deploy. Operators turn pinning
on by setting one or more of:

  AGENT_TLS_PIN_SHA256_HTTPS   comma-separated base64 SHA-256(SPKI)
                                hashes; HTTPS chain must match ≥1.
  AGENT_TLS_PIN_SHA256_GRPC    same shape, applied to gRPC.
  AGENT_TLS_CA_BUNDLE          PEM bundle that REPLACES system roots
                                on both planes (BYOC topology where
                                the customer's internal CA issues
                                the gateway cert).
  AGENT_MTLS_CLIENT_CERT       client cert PEM path (mTLS).
  AGENT_MTLS_CLIENT_KEY        client key PEM path (mTLS).

Pin format follows RFC 7469 §2.4: base64( sha256( DER(SPKI) ) ).
Pinning is layered on top of normal chain validation — we never skip
chain verification. The chain must validate AND at least one cert in
the validated chain must match a configured pin.

Boot-time validation
--------------------
`load_tls_config_or_die()` is called from main.py before either
client is constructed. It:

  * parses the four+ env vars,
  * fails fast (ConfigError) on bad base64, missing files,
    half-set mTLS pair, or http://-scheme `BACKEND_API_URL` when
    pinning is configured,
  * returns a frozen `TlsConfig` the HTTPS adapter + gRPC channel
    factory consume.

What this does NOT close
------------------------
- mTLS for the BYOC topology requires the customer to provision a
  client cert offline; this module loads it but does not generate
  or rotate it.
- gRPC SPKI pinning is enforced against the verified leaf chain
  exposed by the underlying SSL socket; if a customer terminates
  TLS at a sidecar (stunnel etc.) the pin is checked against the
  sidecar's cert, not the gateway's. That's the same edge the HTTPS
  path has and is documented in the README.
"""

from __future__ import annotations

import base64
import logging
import os
import ssl
from dataclasses import dataclass, field
from typing import FrozenSet, List, Optional, Tuple
from urllib.parse import urlparse

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization

logger = logging.getLogger(__name__)


class ConfigError(ValueError):
    """Raised at boot when TLS / mTLS env config is malformed."""


# ---------------------------------------------------------------------------
# Env parsing
# ---------------------------------------------------------------------------

_ENV_PIN_HTTPS = "AGENT_TLS_PIN_SHA256_HTTPS"
_ENV_PIN_GRPC = "AGENT_TLS_PIN_SHA256_GRPC"
_ENV_CA_BUNDLE = "AGENT_TLS_CA_BUNDLE"
_ENV_MTLS_CERT = "AGENT_MTLS_CLIENT_CERT"
_ENV_MTLS_KEY = "AGENT_MTLS_CLIENT_KEY"


def _parse_pin_list(raw: Optional[str], var_name: str) -> FrozenSet[bytes]:
    """Parse a comma-separated list of base64-encoded SHA-256 hashes
    into a frozen set of raw 32-byte digests. Empty / unset -> empty set
    (= pinning OFF for that channel)."""
    if not raw or not raw.strip():
        return frozenset()
    out: List[bytes] = []
    for part in raw.split(","):
        token = part.strip()
        if not token:
            continue
        try:
            digest = base64.b64decode(token, validate=True)
        except (ValueError, base64.binascii.Error) as e:
            raise ConfigError(
                f"{var_name}: pin {token!r} is not valid base64: {e}"
            ) from e
        if len(digest) != 32:
            raise ConfigError(
                f"{var_name}: pin {token!r} decodes to {len(digest)} bytes, "
                f"expected 32 (SHA-256)"
            )
        out.append(digest)
    return frozenset(out)


@dataclass(frozen=True)
class TlsConfig:
    """Resolved TLS posture for both HTTPS and gRPC paths."""

    https_pins: FrozenSet[bytes] = field(default_factory=frozenset)
    grpc_pins: FrozenSet[bytes] = field(default_factory=frozenset)
    ca_bundle_path: Optional[str] = None
    ca_bundle_pem: Optional[bytes] = None
    mtls_cert_path: Optional[str] = None
    mtls_key_path: Optional[str] = None
    mtls_cert_pem: Optional[bytes] = None
    mtls_key_pem: Optional[bytes] = None

    @property
    def https_pinning_enabled(self) -> bool:
        return bool(self.https_pins)

    @property
    def grpc_pinning_enabled(self) -> bool:
        return bool(self.grpc_pins)

    @property
    def mtls_enabled(self) -> bool:
        return bool(self.mtls_cert_path and self.mtls_key_path)

    @property
    def any_hardening(self) -> bool:
        return (
            self.https_pinning_enabled
            or self.grpc_pinning_enabled
            or self.ca_bundle_path is not None
            or self.mtls_enabled
        )


def load_tls_config_or_die(
    *,
    backend_api_url: Optional[str] = None,
    grpc_target: Optional[str] = None,
    env: Optional[dict] = None,
) -> TlsConfig:
    """Build a `TlsConfig` from env vars. Raises `ConfigError` on any
    misconfiguration so the agent fails at boot rather than at first
    request."""
    e = env if env is not None else os.environ

    https_pins = _parse_pin_list(e.get(_ENV_PIN_HTTPS), _ENV_PIN_HTTPS)
    grpc_pins = _parse_pin_list(e.get(_ENV_PIN_GRPC), _ENV_PIN_GRPC)

    ca_bundle_path = (e.get(_ENV_CA_BUNDLE) or "").strip() or None
    ca_bundle_pem: Optional[bytes] = None
    if ca_bundle_path:
        if not os.path.isfile(ca_bundle_path):
            raise ConfigError(
                f"{_ENV_CA_BUNDLE}={ca_bundle_path}: file not found"
            )
        try:
            with open(ca_bundle_path, "rb") as f:
                ca_bundle_pem = f.read()
        except OSError as ex:
            raise ConfigError(
                f"{_ENV_CA_BUNDLE}={ca_bundle_path}: cannot read: {ex}"
            ) from ex
        if b"-----BEGIN CERTIFICATE-----" not in ca_bundle_pem:
            raise ConfigError(
                f"{_ENV_CA_BUNDLE}={ca_bundle_path}: no PEM CERTIFICATE block"
            )

    mtls_cert = (e.get(_ENV_MTLS_CERT) or "").strip() or None
    mtls_key = (e.get(_ENV_MTLS_KEY) or "").strip() or None
    if bool(mtls_cert) ^ bool(mtls_key):
        raise ConfigError(
            f"mTLS requires both {_ENV_MTLS_CERT} and {_ENV_MTLS_KEY}; "
            f"only one was set. Refusing to continue with single-sided TLS."
        )
    mtls_cert_pem: Optional[bytes] = None
    mtls_key_pem: Optional[bytes] = None
    if mtls_cert and mtls_key:
        for path, label in ((mtls_cert, _ENV_MTLS_CERT), (mtls_key, _ENV_MTLS_KEY)):
            if not os.path.isfile(path):
                raise ConfigError(f"{label}={path}: file not found")
        try:
            with open(mtls_cert, "rb") as f:
                mtls_cert_pem = f.read()
            with open(mtls_key, "rb") as f:
                mtls_key_pem = f.read()
        except OSError as ex:
            raise ConfigError(f"mTLS file read failed: {ex}") from ex

    cfg = TlsConfig(
        https_pins=https_pins,
        grpc_pins=grpc_pins,
        ca_bundle_path=ca_bundle_path,
        ca_bundle_pem=ca_bundle_pem,
        mtls_cert_path=mtls_cert,
        mtls_key_path=mtls_key,
        mtls_cert_pem=mtls_cert_pem,
        mtls_key_pem=mtls_key_pem,
    )

    # If pinning is configured, make sure the URL/target the operator
    # gave us is actually a TLS endpoint. Pinning over plaintext is a
    # configuration error, not a useful weakening.
    if cfg.https_pinning_enabled and backend_api_url:
        scheme = urlparse(backend_api_url).scheme.lower()
        if scheme != "https":
            raise ConfigError(
                f"{_ENV_PIN_HTTPS} is set but BACKEND_API_URL={backend_api_url!r} "
                f"is not https://. Pinning over plaintext would silently no-op."
            )
    if cfg.grpc_pinning_enabled and grpc_target:
        # gRPC targets are host:port — there is no scheme. The agent's
        # TC_AGENT_TLS gate handles plaintext gRPC; if pinning is on,
        # plaintext gRPC must be off.
        tls_flag = (e.get("TC_AGENT_TLS", "") or "").strip().lower()
        if tls_flag in ("false", "0", "no"):
            raise ConfigError(
                f"{_ENV_PIN_GRPC} is set but TC_AGENT_TLS={tls_flag!r} forces "
                f"plaintext gRPC. Pinning over plaintext would silently no-op."
            )

    if cfg.any_hardening:
        logger.info(
            "TLS hardening: https_pins=%d grpc_pins=%d ca_bundle=%s mtls=%s",
            len(cfg.https_pins),
            len(cfg.grpc_pins),
            "yes" if cfg.ca_bundle_path else "no",
            "yes" if cfg.mtls_enabled else "no",
        )
    return cfg


# ---------------------------------------------------------------------------
# SPKI hashing + chain validation
# ---------------------------------------------------------------------------

def spki_sha256(cert_der: bytes) -> bytes:
    """Compute SHA-256 of the cert's SubjectPublicKeyInfo, per RFC 7469
    §2.4. Input is the cert in DER form. Returns the raw 32-byte digest."""
    cert = x509.load_der_x509_certificate(cert_der)
    spki_der = cert.public_key().public_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    h = hashes.Hash(hashes.SHA256())
    h.update(spki_der)
    return h.finalize()


def spki_sha256_b64(cert_der: bytes) -> str:
    """Same as `spki_sha256` but returns base64 — matches the env-var
    format an operator would type."""
    return base64.b64encode(spki_sha256(cert_der)).decode("ascii")


def chain_matches_any_pin(
    chain_der: List[bytes], pins: FrozenSet[bytes]
) -> Tuple[bool, Optional[bytes]]:
    """True iff at least one cert in `chain_der` has an SPKI hash in
    `pins`. Returns (matched, matching_digest_or_None) so callers can
    log the matching pin on success or the leaf SPKI on failure."""
    if not pins:
        return True, None
    for cert_der in chain_der:
        try:
            digest = spki_sha256(cert_der)
        except Exception:  # noqa: BLE001 — bad cert, skip
            continue
        if digest in pins:
            return True, digest
    return False, None


def extract_chain_from_sslsocket(sslsock: ssl.SSLSocket) -> List[bytes]:
    """Return the validated peer cert chain in DER form, leaf first.

    Uses `SSLSocket.get_verified_chain()` (Python 3.10+). Falls back
    to the leaf cert if the verified chain is unavailable (some
    OpenSSL builds return an empty list) so a misconfigured stack
    fails closed on pin mismatch rather than silently allowing the
    connection."""
    chain: List[bytes] = []
    getter = getattr(sslsock, "get_verified_chain", None)
    if getter is not None:
        try:
            verified = getter() or []
            for c in verified:
                # Each element is a Certificate object exposing
                # public_bytes() in py 3.10+.
                if hasattr(c, "public_bytes"):
                    chain.append(c.public_bytes(ssl.Encoding.DER))
                elif isinstance(c, (bytes, bytearray)):
                    chain.append(bytes(c))
        except Exception as e:  # noqa: BLE001
            logger.debug("get_verified_chain failed: %s", e)
    if not chain:
        leaf = sslsock.getpeercert(binary_form=True)
        if leaf:
            chain.append(leaf)
    return chain


class PinMismatchError(Exception):
    """Raised when the peer chain is valid but does not contain a
    pinned SPKI. Surfaces as a connection-level failure on both planes."""


def assert_chain_pinned(
    chain_der: List[bytes],
    pins: FrozenSet[bytes],
    *,
    peer_label: str,
) -> None:
    """Raise `PinMismatchError` if no cert in `chain_der` matches any
    pin. No-op if `pins` is empty."""
    if not pins:
        return
    matched, _digest = chain_matches_any_pin(chain_der, pins)
    if not matched:
        leaf_b64 = spki_sha256_b64(chain_der[0]) if chain_der else "<empty>"
        raise PinMismatchError(
            f"TLS pin mismatch for {peer_label}: peer chain "
            f"(leaf SPKI sha256={leaf_b64}) matched none of "
            f"{len(pins)} configured pin(s)"
        )


# ---------------------------------------------------------------------------
# requests / urllib3 adapter
# ---------------------------------------------------------------------------
#
# `requests` runs cert verification through urllib3's SSLContext, but
# it does NOT expose the peer chain to user code. To check SPKI pins we
# subclass `HTTPSConnection` and check the chain inside connect(),
# *after* normal verification has succeeded — i.e. layered on top, not
# as a replacement.
#
# We register the subclass per-pool by overriding the adapter's
# `init_poolmanager` to pass a custom pool that uses our connection
# class. This is the same pattern `requests-toolbelt` uses for its
# fingerprinted-adapter helpers.

def build_pinned_https_adapter(cfg: TlsConfig):  # pragma: no cover for import-error path
    """Construct a `requests.adapters.HTTPAdapter` that enforces
    `cfg.https_pins` on every connection. Importing `requests` is
    deferred so this module is unit-testable without network deps."""
    import requests  # noqa: WPS433
    from requests.adapters import HTTPAdapter
    from urllib3 import PoolManager
    from urllib3.connection import HTTPSConnection
    from urllib3.connectionpool import HTTPSConnectionPool

    pins = cfg.https_pins
    peer_label = "HTTPS gateway"

    class _PinningHTTPSConnection(HTTPSConnection):
        def connect(self) -> None:
            super().connect()
            sock = self.sock
            if not isinstance(sock, ssl.SSLSocket):
                # Plaintext or proxy-tunneled non-TLS — pin can't apply.
                # If pinning is on we should never land here because
                # load_tls_config_or_die rejected http:// URLs. Be defensive.
                raise PinMismatchError(
                    f"{peer_label}: socket is not TLS; refusing to send "
                    f"with pinning enabled"
                )
            chain = extract_chain_from_sslsocket(sock)
            assert_chain_pinned(chain, pins, peer_label=peer_label)

    class _PinningHTTPSConnectionPool(HTTPSConnectionPool):
        ConnectionCls = _PinningHTTPSConnection

    class _PinningPoolManager(PoolManager):
        def _new_pool(self, scheme, host, port, request_context=None):
            if scheme == "https":
                kwargs = dict(self.connection_pool_kw)
                if request_context is not None:
                    # Carry over per-request overrides (timeout, etc.)
                    for k, v in request_context.items():
                        if k not in ("scheme", "host", "port"):
                            kwargs[k] = v
                return _PinningHTTPSConnectionPool(host, port=port, **kwargs)
            return super()._new_pool(scheme, host, port, request_context)

    class PinningHTTPAdapter(HTTPAdapter):
        def init_poolmanager(self, connections, maxsize, block=False, **pool_kwargs):  # noqa: D401
            self.poolmanager = _PinningPoolManager(
                num_pools=connections,
                maxsize=maxsize,
                block=block,
                **pool_kwargs,
            )

    return PinningHTTPAdapter()
