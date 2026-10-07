"""
Tests for `src/transport/tls_pinning.py` — §8.2 Blocker #6.

Scope: env-var parsing, SPKI hash computation, chain validation, and
mTLS / CA-bundle config plumbing. We do NOT exercise a live TLS
handshake here — that's brittle for unit tests. The HTTPS adapter and
the gRPC pin-probe are exercised via their plain helpers
(`assert_chain_pinned`, `extract_chain_from_sslsocket`) — wiring at
call sites is unit-tested by checking the env-driven config it
produces.
"""

from __future__ import annotations

import base64
import os
import sys

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from transport.tls_pinning import (  # noqa: E402
    ConfigError,
    PinMismatchError,
    TlsConfig,
    assert_chain_pinned,
    chain_matches_any_pin,
    load_tls_config_or_die,
    spki_sha256,
    spki_sha256_b64,
)


# ---------------------------------------------------------------------------
# Test cert factory
# ---------------------------------------------------------------------------

def _make_self_signed(cn: str = "test.tensorcost.invalid"):
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, cn),
    ])
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.now(timezone.utc) - timedelta(days=1))
        .not_valid_after(datetime.now(timezone.utc) + timedelta(days=30))
        .sign(key, hashes.SHA256())
    )
    der = cert.public_bytes(serialization.Encoding.DER)
    pem = cert.public_bytes(serialization.Encoding.PEM)
    key_pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption(),
    )
    return der, pem, key_pem


# ---------------------------------------------------------------------------
# SPKI hash computation
# ---------------------------------------------------------------------------

class TestSpkiHash:
    def test_digest_matches_openssl_recipe(self):
        der, _, _ = _make_self_signed()
        # The function under test must compute sha256 of the SPKI DER.
        # Recompute with the canonical recipe and assert equality.
        cert = x509.load_der_x509_certificate(der)
        spki_der = cert.public_key().public_bytes(
            encoding=serialization.Encoding.DER,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        h = hashes.Hash(hashes.SHA256())
        h.update(spki_der)
        expected = h.finalize()

        assert spki_sha256(der) == expected
        assert spki_sha256_b64(der) == base64.b64encode(expected).decode("ascii")

    def test_two_certs_with_same_pubkey_share_pin(self):
        # Same key in two certs (e.g. cert rotation that re-uses key) —
        # SPKI pin must match both.
        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        certs = []
        for cn in ("a.invalid", "b.invalid"):
            name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, cn)])
            c = (
                x509.CertificateBuilder()
                .subject_name(name)
                .issuer_name(name)
                .public_key(key.public_key())
                .serial_number(x509.random_serial_number())
                .not_valid_before(datetime.now(timezone.utc) - timedelta(days=1))
                .not_valid_after(datetime.now(timezone.utc) + timedelta(days=30))
                .sign(key, hashes.SHA256())
            )
            certs.append(c.public_bytes(serialization.Encoding.DER))
        assert spki_sha256(certs[0]) == spki_sha256(certs[1])


# ---------------------------------------------------------------------------
# Chain matching
# ---------------------------------------------------------------------------

class TestChainMatching:
    def test_empty_pin_set_means_off(self):
        der, _, _ = _make_self_signed()
        matched, _ = chain_matches_any_pin([der], frozenset())
        assert matched is True  # no pins -> trivially passes

    def test_match_on_leaf(self):
        der, _, _ = _make_self_signed()
        pin = spki_sha256(der)
        matched, hit = chain_matches_any_pin([der], frozenset({pin}))
        assert matched is True
        assert hit == pin

    def test_match_on_intermediate(self):
        leaf, _, _ = _make_self_signed("leaf")
        intermediate, _, _ = _make_self_signed("intermediate")
        # Chain is leaf-first; pinning the intermediate must match.
        pin = spki_sha256(intermediate)
        matched, hit = chain_matches_any_pin(
            [leaf, intermediate], frozenset({pin})
        )
        assert matched is True
        assert hit == pin

    def test_no_match_raises(self):
        leaf, _, _ = _make_self_signed("leaf")
        other, _, _ = _make_self_signed("rogue")
        wrong_pin = spki_sha256(other)
        with pytest.raises(PinMismatchError) as ei:
            assert_chain_pinned(
                [leaf], frozenset({wrong_pin}), peer_label="test peer"
            )
        # Useful operator log: must include the leaf SPKI we saw,
        # which is what they need to copy if the cert legitimately rotated.
        assert "test peer" in str(ei.value)
        assert spki_sha256_b64(leaf) in str(ei.value)


# ---------------------------------------------------------------------------
# Env parsing
# ---------------------------------------------------------------------------

class TestEnvParsing:
    def test_unset_returns_empty_config(self, monkeypatch):
        for v in (
            "AGENT_TLS_PIN_SHA256_HTTPS",
            "AGENT_TLS_PIN_SHA256_GRPC",
            "AGENT_TLS_CA_BUNDLE",
            "AGENT_MTLS_CLIENT_CERT",
            "AGENT_MTLS_CLIENT_KEY",
        ):
            monkeypatch.delenv(v, raising=False)
        cfg = load_tls_config_or_die()
        assert cfg.https_pinning_enabled is False
        assert cfg.grpc_pinning_enabled is False
        assert cfg.mtls_enabled is False
        assert cfg.ca_bundle_path is None
        assert cfg.any_hardening is False

    def test_single_pin_parses(self, monkeypatch):
        der, _, _ = _make_self_signed()
        b64 = spki_sha256_b64(der)
        monkeypatch.setenv("AGENT_TLS_PIN_SHA256_HTTPS", b64)
        cfg = load_tls_config_or_die()
        assert cfg.https_pinning_enabled is True
        assert cfg.grpc_pinning_enabled is False
        assert spki_sha256(der) in cfg.https_pins

    def test_multiple_pins_with_whitespace(self, monkeypatch):
        ders = [_make_self_signed(f"c{i}.invalid")[0] for i in range(3)]
        b64s = [spki_sha256_b64(d) for d in ders]
        monkeypatch.setenv(
            "AGENT_TLS_PIN_SHA256_GRPC",
            f" {b64s[0]} , {b64s[1]},,{b64s[2]} ",
        )
        cfg = load_tls_config_or_die()
        assert len(cfg.grpc_pins) == 3
        for d in ders:
            assert spki_sha256(d) in cfg.grpc_pins

    def test_invalid_base64_fails_at_boot(self, monkeypatch):
        monkeypatch.setenv(
            "AGENT_TLS_PIN_SHA256_HTTPS", "not!valid!base64!"
        )
        with pytest.raises(ConfigError) as ei:
            load_tls_config_or_die()
        assert "AGENT_TLS_PIN_SHA256_HTTPS" in str(ei.value)

    def test_wrong_length_pin_fails(self, monkeypatch):
        # 16-byte digest base64-encoded — wrong length.
        bad = base64.b64encode(b"x" * 16).decode("ascii")
        monkeypatch.setenv("AGENT_TLS_PIN_SHA256_GRPC", bad)
        with pytest.raises(ConfigError) as ei:
            load_tls_config_or_die()
        assert "32" in str(ei.value)

    def test_https_pin_with_http_url_fails(self, monkeypatch):
        der, _, _ = _make_self_signed()
        monkeypatch.setenv("AGENT_TLS_PIN_SHA256_HTTPS", spki_sha256_b64(der))
        with pytest.raises(ConfigError) as ei:
            load_tls_config_or_die(backend_api_url="http://gateway.example/")
        assert "https" in str(ei.value).lower()

    def test_grpc_pin_with_plaintext_flag_fails(self, monkeypatch):
        der, _, _ = _make_self_signed()
        monkeypatch.setenv("AGENT_TLS_PIN_SHA256_GRPC", spki_sha256_b64(der))
        monkeypatch.setenv("TC_AGENT_TLS", "false")
        with pytest.raises(ConfigError):
            load_tls_config_or_die(grpc_target="grpc.example:443")


# ---------------------------------------------------------------------------
# CA bundle
# ---------------------------------------------------------------------------

class TestCABundle:
    def test_missing_file_fails(self, monkeypatch, tmp_path):
        monkeypatch.setenv(
            "AGENT_TLS_CA_BUNDLE", str(tmp_path / "does-not-exist.pem")
        )
        with pytest.raises(ConfigError) as ei:
            load_tls_config_or_die()
        assert "not found" in str(ei.value)

    def test_non_pem_file_fails(self, monkeypatch, tmp_path):
        f = tmp_path / "junk.pem"
        f.write_bytes(b"this is not a certificate")
        monkeypatch.setenv("AGENT_TLS_CA_BUNDLE", str(f))
        with pytest.raises(ConfigError) as ei:
            load_tls_config_or_die()
        assert "PEM" in str(ei.value)

    def test_valid_bundle_loaded(self, monkeypatch, tmp_path):
        _, pem, _ = _make_self_signed("ca.invalid")
        f = tmp_path / "bundle.pem"
        f.write_bytes(pem)
        monkeypatch.setenv("AGENT_TLS_CA_BUNDLE", str(f))
        cfg = load_tls_config_or_die()
        assert cfg.ca_bundle_path == str(f)
        assert cfg.ca_bundle_pem == pem


# ---------------------------------------------------------------------------
# mTLS
# ---------------------------------------------------------------------------

class TestMTLS:
    def test_only_cert_set_fails(self, monkeypatch, tmp_path):
        _, pem, key_pem = _make_self_signed()
        cert_f = tmp_path / "cert.pem"
        cert_f.write_bytes(pem)
        monkeypatch.setenv("AGENT_MTLS_CLIENT_CERT", str(cert_f))
        monkeypatch.delenv("AGENT_MTLS_CLIENT_KEY", raising=False)
        with pytest.raises(ConfigError) as ei:
            load_tls_config_or_die()
        msg = str(ei.value)
        assert "AGENT_MTLS_CLIENT_KEY" in msg
        assert "single-sided" in msg or "both" in msg

    def test_only_key_set_fails(self, monkeypatch, tmp_path):
        _, pem, key_pem = _make_self_signed()
        key_f = tmp_path / "key.pem"
        key_f.write_bytes(key_pem)
        monkeypatch.setenv("AGENT_MTLS_CLIENT_KEY", str(key_f))
        monkeypatch.delenv("AGENT_MTLS_CLIENT_CERT", raising=False)
        with pytest.raises(ConfigError):
            load_tls_config_or_die()

    def test_missing_cert_file_fails(self, monkeypatch, tmp_path):
        _, _, key_pem = _make_self_signed()
        key_f = tmp_path / "key.pem"
        key_f.write_bytes(key_pem)
        monkeypatch.setenv(
            "AGENT_MTLS_CLIENT_CERT", str(tmp_path / "missing.pem")
        )
        monkeypatch.setenv("AGENT_MTLS_CLIENT_KEY", str(key_f))
        with pytest.raises(ConfigError) as ei:
            load_tls_config_or_die()
        assert "not found" in str(ei.value)

    def test_both_set_loads(self, monkeypatch, tmp_path):
        _, pem, key_pem = _make_self_signed()
        cert_f = tmp_path / "cert.pem"
        key_f = tmp_path / "key.pem"
        cert_f.write_bytes(pem)
        key_f.write_bytes(key_pem)
        monkeypatch.setenv("AGENT_MTLS_CLIENT_CERT", str(cert_f))
        monkeypatch.setenv("AGENT_MTLS_CLIENT_KEY", str(key_f))
        cfg = load_tls_config_or_die()
        assert cfg.mtls_enabled is True
        assert cfg.mtls_cert_path == str(cert_f)
        assert cfg.mtls_key_path == str(key_f)
        assert cfg.mtls_cert_pem == pem
        assert cfg.mtls_key_pem == key_pem


# ---------------------------------------------------------------------------
# HttpSyncClient wiring
# ---------------------------------------------------------------------------

class TestHttpSyncClientWiring:
    """Verify that TlsConfig settings flow through to the requests.Session."""

    def test_no_tls_config_means_default_session(self):
        from transport.http_client import HttpSyncClient
        c = HttpSyncClient(base_url="https://x.invalid", api_key="k")
        kwargs = c._request_kwargs()
        assert kwargs == {}  # no verify=, no cert= overrides

    def test_ca_bundle_sets_verify(self, tmp_path):
        from transport.http_client import HttpSyncClient
        _, pem, _ = _make_self_signed("ca.invalid")
        ca = tmp_path / "ca.pem"
        ca.write_bytes(pem)
        cfg = TlsConfig(ca_bundle_path=str(ca), ca_bundle_pem=pem)
        c = HttpSyncClient(
            base_url="https://x.invalid", api_key="k", tls_config=cfg,
        )
        assert c._request_kwargs()["verify"] == str(ca)

    def test_mtls_sets_cert(self, tmp_path):
        from transport.http_client import HttpSyncClient
        _, pem, key_pem = _make_self_signed()
        cert_f = tmp_path / "cert.pem"
        key_f = tmp_path / "key.pem"
        cert_f.write_bytes(pem)
        key_f.write_bytes(key_pem)
        cfg = TlsConfig(
            mtls_cert_path=str(cert_f),
            mtls_key_path=str(key_f),
            mtls_cert_pem=pem,
            mtls_key_pem=key_pem,
        )
        c = HttpSyncClient(
            base_url="https://x.invalid", api_key="k", tls_config=cfg,
        )
        assert c._request_kwargs()["cert"] == (str(cert_f), str(key_f))

    def test_pinning_mounts_adapter(self, monkeypatch):
        from transport.http_client import HttpSyncClient
        der, _, _ = _make_self_signed()
        cfg = TlsConfig(https_pins=frozenset({spki_sha256(der)}))
        c = HttpSyncClient(
            base_url="https://x.invalid", api_key="k", tls_config=cfg,
        )
        # The session should have a custom adapter mounted on https://.
        adapter = c._session.get_adapter("https://x.invalid/")
        # Default HTTPAdapter class name is "HTTPAdapter"; ours is
        # "PinningHTTPAdapter" (defined inside build_pinned_https_adapter).
        assert "Pinning" in type(adapter).__name__
