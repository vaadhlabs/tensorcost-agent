"""Tests for A-04 chargeback attribution — customer + feature."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from tensorcost import wrap
from tensorcost._applied_mode import build_proxy_headers
from tensorcost._config import MissingConfigError, resolve_config
from tensorcost._providers import _openai
from tensorcost._proxy_client import proxy_request


class TestCustomerFeatureConfig:
    def test_wrap_time_customer_and_feature(self, monkeypatch):
        monkeypatch.setenv("TENSORCOST_API_KEY", "k")
        cfg = resolve_config(customer="acme", feature="summarizer")
        assert cfg.customer == "acme"
        assert cfg.feature == "summarizer"

    def test_env_vars_resolve(self, monkeypatch):
        monkeypatch.setenv("TENSORCOST_API_KEY", "k")
        monkeypatch.setenv("TENSORCOST_CUSTOMER", "env-customer")
        monkeypatch.setenv("TENSORCOST_FEATURE", "env-feature")
        cfg = resolve_config()
        assert cfg.customer == "env-customer"
        assert cfg.feature == "env-feature"

    def test_blank_collapses_to_none(self, monkeypatch):
        monkeypatch.setenv("TENSORCOST_API_KEY", "k")
        cfg = resolve_config(customer="   ", feature="")
        assert cfg.customer is None
        assert cfg.feature is None

    def test_max_length_128(self, monkeypatch):
        monkeypatch.setenv("TENSORCOST_API_KEY", "k")
        with pytest.raises(MissingConfigError, match="customer"):
            resolve_config(customer="x" * 129)


class TestCustomerFeatureObservation:
    def test_openai_observation_carries_customer_and_feature(
        self, openai_client, recording_transport
    ):
        _openai.install(
            openai_client,
            recording_transport,
            tenant_id=None,
            environment=None,
            connection_id=None,
            customer="acme-corp",
            feature="support-bot",
        )
        openai_client.chat.completions.create(
            model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}]
        )
        assert len(recording_transport.observations) == 1
        obs = recording_transport.observations[0]
        assert obs["customer"] == "acme-corp"
        assert obs["feature"] == "support-bot"

    def test_absent_when_unconfigured(self, openai_client, recording_transport):
        _openai.install(
            openai_client,
            recording_transport,
            tenant_id=None,
            environment=None,
            connection_id=None,
        )
        openai_client.chat.completions.create(
            model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}]
        )
        obs = recording_transport.observations[0]
        assert "customer" not in obs
        assert "feature" not in obs


class TestCustomerFeatureProxyHeaders:
    def test_build_proxy_headers_includes_chargeback_fields(self):
        headers = build_proxy_headers(
            provider_url="https://api.openai.com",
            provider_auth="Bearer sk-abc",
            customer="c-1",
            feature="f-1",
        )
        assert headers["x-tc-customer"] == "c-1"
        assert headers["x-tc-feature"] == "f-1"

    def test_proxy_request_forwards_customer_and_feature(self):
        import httpx

        captured: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["headers"] = dict(request.headers)
            return httpx.Response(
                200,
                json={"usage": {"prompt_tokens": 1, "completion_tokens": 2}},
            )

        client = httpx.Client(transport=httpx.MockTransport(handler))
        proxy_request(
            proxy_url="https://proxy.example.com",
            bearer_token="jwt",
            provider_url="https://api.openai.com",
            provider_auth="Bearer sk-test",
            body={"model": "gpt-4o-mini", "messages": []},
            model="gpt-4o-mini",
            provider="openai",
            operation="chat.completions",
            correlation_id="corr-1",
            environment=None,
            customer="proxy-customer",
            feature="proxy-feature",
            http_client=client,
        )
        assert captured["headers"]["x-tc-customer"] == "proxy-customer"
        assert captured["headers"]["x-tc-feature"] == "proxy-feature"


class TestWrapCustomerFeature:
    def test_wrap_passes_customer_feature_to_proxy_request(
        self, openai_client, monkeypatch
    ):
        openai_client.api_key = "sk-test"
        openai_client.base_url = "https://api.openai.com/v1"
        monkeypatch.setenv("TENSORCOST_API_KEY", "k")
        wrap(
            openai_client,
            api_key="k",
            tenant_id="t-1",
            customer="wrap-customer",
            feature="wrap-feature",
            applied_mode=True,
            proxy_url="https://proxy.example.com",
        )
        with patch("tensorcost._providers._openai.proxy_request") as mock_proxy:
            mock_proxy.return_value = (
                {"usage": {"prompt_tokens": 1, "completion_tokens": 2}},
                None,
            )
            openai_client._tensorcost_transport._get_token = lambda: "fake-jwt"
            openai_client.chat.completions.create(
                model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}]
            )
            call_kwargs = mock_proxy.call_args[1]
            assert call_kwargs["customer"] == "wrap-customer"
            assert call_kwargs["feature"] == "wrap-feature"
        openai_client._tensorcost_transport.close()
