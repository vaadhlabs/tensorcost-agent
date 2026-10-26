"""Applied-mode (Layer 2) tests.

Covers:
  - applied_mode=False (default) — no drift from observe-only behaviour.
  - applied_mode=True + proxy_url set explicitly — request goes to proxy,
    both custom headers present, original provider URL preserved.
  - applied_mode=True + proxy_url from TENSORCOST_PROXY_URL env var.
  - applied_mode=True + neither kwarg nor env — MissingConfigError at wrap time.
  - OpenAI provider: correct auth shape (Authorization: Bearer sk-...).
  - Anthropic provider: correct auth shape (x-api-key <key>); body sent verbatim
    (no OpenAI translation); x-tc-provider-url is base URL only.
  - Proxy response forwarded verbatim to caller (no response rewriting).
  - Telemetry (observation POST) fires in both observe-only and applied paths.
  - Proxy failure falls back to direct provider call (fail-open).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest

from tensorcost import MissingConfigError, wrap
from tensorcost._applied_mode import (
    PROXY_CHAT_PATH,
    anthropic_provider_auth,
    anthropic_provider_base_url,
    build_proxy_headers,
    openai_provider_auth,
    openai_provider_base_url,
)
from tensorcost._config import resolve_config
from tensorcost._errors import TensorCostProxyError
from tensorcost._providers import _anthropic, _openai


# ---------------------------------------------------------------------------
# Config resolution tests
# ---------------------------------------------------------------------------


class TestAppliedModeConfig:
    def test_applied_mode_false_by_default(self, monkeypatch):
        monkeypatch.setenv("TENSORCOST_API_KEY", "k")
        cfg = resolve_config()
        assert cfg.applied_mode is False
        assert cfg.proxy_url is None

    def test_applied_mode_true_with_explicit_proxy_url(self, monkeypatch):
        monkeypatch.setenv("TENSORCOST_API_KEY", "k")
        cfg = resolve_config(
            applied_mode=True,
            proxy_url="https://proxy.example.com",
        )
        assert cfg.applied_mode is True
        assert cfg.proxy_url == "https://proxy.example.com"

    def test_applied_mode_true_proxy_url_trailing_slash_stripped(self, monkeypatch):
        monkeypatch.setenv("TENSORCOST_API_KEY", "k")
        cfg = resolve_config(
            applied_mode=True,
            proxy_url="https://proxy.example.com/",
        )
        assert cfg.proxy_url == "https://proxy.example.com"

    def test_applied_mode_true_proxy_url_from_env(self, monkeypatch):
        monkeypatch.setenv("TENSORCOST_API_KEY", "k")
        monkeypatch.setenv("TENSORCOST_PROXY_URL", "https://env-proxy.example.com")
        cfg = resolve_config(applied_mode=True)
        assert cfg.applied_mode is True
        assert cfg.proxy_url == "https://env-proxy.example.com"

    def test_applied_mode_true_no_proxy_url_raises_at_config(self, monkeypatch):
        monkeypatch.setenv("TENSORCOST_API_KEY", "k")
        with pytest.raises(MissingConfigError, match="proxy_url"):
            resolve_config(applied_mode=True)

    def test_applied_mode_true_no_proxy_url_raises_at_wrap_time(
        self, openai_client
    ):
        """The error surfaces from wrap(), not from the first call."""
        with pytest.raises(MissingConfigError, match="proxy_url"):
            wrap(openai_client, api_key="k", applied_mode=True)

    def test_applied_mode_false_proxy_url_still_stored(self, monkeypatch):
        """Providing proxy_url when applied_mode=False is allowed — no error,
        and proxy_url is stored (caller might flip mode later)."""
        monkeypatch.setenv("TENSORCOST_API_KEY", "k")
        cfg = resolve_config(
            applied_mode=False,
            proxy_url="https://proxy.example.com",
        )
        assert cfg.applied_mode is False
        assert cfg.proxy_url == "https://proxy.example.com"


# ---------------------------------------------------------------------------
# Applied-mode helper tests
# ---------------------------------------------------------------------------


class TestAppliedModeHelpers:
    def test_build_proxy_headers_required_only(self):
        headers = build_proxy_headers(
            provider_url="https://api.openai.com",
            provider_auth="Bearer sk-abc",
        )
        assert headers["x-tc-provider-url"] == "https://api.openai.com"
        assert headers["x-tc-provider-auth"] == "Bearer sk-abc"
        assert "x-tc-correlation-id" not in headers
        assert "x-tc-environment" not in headers

    def test_build_proxy_headers_with_optional_fields(self):
        headers = build_proxy_headers(
            provider_url="https://api.openai.com",
            provider_auth="Bearer sk-abc",
            correlation_id="corr-123",
            environment="staging",
        )
        assert headers["x-tc-correlation-id"] == "corr-123"
        assert headers["x-tc-environment"] == "staging"

    def test_openai_provider_auth_bearer_prefix(self):
        client = MagicMock()
        client.api_key = "sk-test-key"
        assert openai_provider_auth(client) == "Bearer sk-test-key"

    def test_openai_provider_auth_missing_key(self):
        client = MagicMock(spec=[])  # no api_key attribute
        assert openai_provider_auth(client) == ""

    def test_openai_provider_base_url_strips_version(self):
        client = MagicMock()
        client.base_url = "https://api.openai.com/v1"
        assert openai_provider_base_url(client) == "https://api.openai.com"

    def test_openai_provider_base_url_no_version_suffix(self):
        client = MagicMock()
        client.base_url = "https://api.openai.com"
        assert openai_provider_base_url(client) == "https://api.openai.com"

    def test_openai_provider_base_url_fallback(self):
        client = MagicMock(spec=[])  # no base_url
        assert openai_provider_base_url(client) == "https://api.openai.com"

    def test_anthropic_provider_auth_shape(self):
        client = MagicMock()
        client.api_key = "sk-ant-test"
        assert anthropic_provider_auth(client) == "x-api-key sk-ant-test"

    def test_anthropic_provider_auth_missing_key(self):
        client = MagicMock(spec=[])
        assert anthropic_provider_auth(client) == ""

    def test_anthropic_provider_base_url_strips_trailing_slash(self):
        client = MagicMock()
        client.base_url = "https://api.anthropic.com/"
        assert anthropic_provider_base_url(client) == "https://api.anthropic.com"

    def test_anthropic_provider_base_url_fallback(self):
        client = MagicMock(spec=[])
        assert anthropic_provider_base_url(client) == "https://api.anthropic.com"

    def test_proxy_chat_path_constant(self):
        assert PROXY_CHAT_PATH == "/api/inference-proxy/v1/chat/completions"


# ---------------------------------------------------------------------------
# Shared response fixtures
# ---------------------------------------------------------------------------


def _proxy_success_response() -> dict:
    """OpenAI-shaped proxy success response (returned directly by proxy_request)."""
    return {
        "id": "chatcmpl-proxy",
        "object": "chat.completion",
        "choices": [{"message": {"role": "assistant", "content": "hello"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20},
    }


def _anthropic_proxy_success_response() -> dict:
    """Anthropic-shaped proxy success response (returned directly by proxy_request)."""
    return {
        "id": "msg_proxy",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "hello"}],
        "model": "claude-3-5-sonnet-20241022",
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 20},
    }


# ---------------------------------------------------------------------------
# OpenAI applied-mode routing tests
# ---------------------------------------------------------------------------


class TestOpenAIAppliedMode:
    """OpenAI-specific applied-mode routing."""

    def _make_client(self, api_key: str = "sk-test", base_url: str = "https://api.openai.com/v1") -> Any:
        client = MagicMock()
        client.api_key = api_key
        client.base_url = base_url
        client.chat = MagicMock()
        client.chat.completions = MagicMock()
        client.chat.completions.create = MagicMock(
            return_value=MagicMock(
                usage=MagicMock(prompt_tokens=12, completion_tokens=34),
            )
        )
        client.completions = MagicMock()
        client.completions.create = MagicMock(
            return_value=MagicMock(
                usage=MagicMock(prompt_tokens=5, completion_tokens=7),
            )
        )
        return client

    def test_observe_only_no_proxy_call(self, recording_transport):
        """applied_mode=False: no proxy involved, original method called directly."""
        client = self._make_client()
        sentinel = client.chat.completions.create.return_value

        _openai.install(
            client, recording_transport,
            tenant_id=None, environment=None, connection_id=None,
            applied_mode=False, proxy_url=None,
        )
        result = client.chat.completions.create(
            model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}]
        )

        assert result is sentinel
        assert len(recording_transport.observations) == 1
        obs = recording_transport.observations[0]
        assert obs["status"] == "success"
        assert obs["input_tokens"] == 12
        assert obs["output_tokens"] == 34

    def test_applied_mode_request_goes_to_proxy(self, recording_transport):
        """proxy_request is called with the correct provider_url and provider_auth."""
        client = self._make_client(api_key="sk-real-key", base_url="https://api.openai.com/v1")

        _openai.install(
            client, recording_transport,
            tenant_id="t-1", environment=None, connection_id=None,
            applied_mode=True, proxy_url="https://proxy.tensorcost.com",
        )

        with patch("tensorcost._providers._openai.proxy_request") as mock_proxy:
            mock_proxy.return_value = (_proxy_success_response(), None)
            client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "hi"}],
            )

        mock_proxy.assert_called_once()
        call_kwargs = mock_proxy.call_args[1]
        assert call_kwargs["proxy_url"] == "https://proxy.tensorcost.com"
        assert call_kwargs["provider_url"] == "https://api.openai.com"
        assert call_kwargs["provider_auth"] == "Bearer sk-real-key"

    def test_applied_mode_proxy_url_from_env(self, monkeypatch, recording_transport):
        """TENSORCOST_PROXY_URL env var is picked up by resolve_config."""
        monkeypatch.setenv("TENSORCOST_PROXY_URL", "https://env-proxy.example.com")
        cfg = resolve_config(api_key="k", applied_mode=True)
        assert cfg.proxy_url == "https://env-proxy.example.com"

    def test_applied_mode_response_forwarded_verbatim(self, recording_transport):
        """The proxy response dict is returned as-is; no structure rewriting."""
        expected_response = _proxy_success_response()
        client = self._make_client()

        _openai.install(
            client, recording_transport,
            tenant_id=None, environment=None, connection_id=None,
            applied_mode=True, proxy_url="https://proxy.example.com",
        )

        with patch("tensorcost._providers._openai.proxy_request") as mock_proxy:
            mock_proxy.return_value = (expected_response, None)
            result = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "hi"}],
            )

        assert result == expected_response

    def test_applied_mode_telemetry_still_fires(self, recording_transport):
        """Observation POST fires even when the request goes through the proxy."""
        client = self._make_client()

        _openai.install(
            client, recording_transport,
            tenant_id="t-2", environment=None, connection_id=None,
            applied_mode=True, proxy_url="https://proxy.example.com",
        )

        with patch("tensorcost._providers._openai.proxy_request") as mock_proxy:
            mock_proxy.return_value = (_proxy_success_response(), None)
            client.chat.completions.create(
                model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}]
            )

        assert len(recording_transport.observations) == 1
        obs = recording_transport.observations[0]
        assert obs["status"] == "success"
        assert "tenant_id" not in obs

    def test_applied_mode_proxy_failure_falls_back_to_direct(self, recording_transport):
        """If the proxy errors, the SDK calls the original provider and still posts telemetry."""
        client = self._make_client()
        sentinel = client.chat.completions.create.return_value

        _openai.install(
            client, recording_transport,
            tenant_id=None, environment=None, connection_id=None,
            applied_mode=True, proxy_url="https://proxy.example.com",
        )

        with patch("tensorcost._providers._openai.proxy_request") as mock_proxy:
            # proxy_request only ever raises typed errors; an outage is
            # TensorCostProxyError (1.5.0: raw exceptions don't fall back).
            mock_proxy.side_effect = TensorCostProxyError("proxy down")
            result = client.chat.completions.create(
                model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}]
            )

        # Falls back to the original mock return value.
        assert result is sentinel
        # Telemetry still fires after the fallback path.
        assert len(recording_transport.observations) == 1

    def test_applied_mode_legacy_completions_not_proxied(self, recording_transport):
        """Legacy completions.create bypasses applied-mode (v1 scope is chat only)."""
        client = self._make_client()
        sentinel = client.completions.create.return_value

        _openai.install(
            client, recording_transport,
            tenant_id=None, environment=None, connection_id=None,
            applied_mode=True, proxy_url="https://proxy.example.com",
        )

        # No proxy patch — the call must go directly to the original method.
        result = client.completions.create(
            model="text-davinci-003", prompt="hello"
        )
        assert result is sentinel
        assert len(recording_transport.observations) == 1
        assert recording_transport.observations[0]["operation"] == "completions"

    def test_applied_mode_x_tc_provider_url_preserved(self, recording_transport):
        """The provider URL in the proxy call matches the client's configured base URL,
        not the proxy URL."""
        client = self._make_client(api_key="sk-azure", base_url="https://my-azure.openai.azure.com/openai")

        _openai.install(
            client, recording_transport,
            tenant_id=None, environment=None, connection_id=None,
            applied_mode=True, proxy_url="https://proxy.example.com",
        )

        with patch("tensorcost._providers._openai.proxy_request") as mock_proxy:
            mock_proxy.return_value = (_proxy_success_response(), None)
            client.chat.completions.create(
                model="gpt-4o", messages=[{"role": "user", "content": "hi"}]
            )

        call_kwargs = mock_proxy.call_args[1]
        # Azure URL: /openai suffix is not a version path so it's kept as-is.
        # The key assertion is that it is NOT the proxy URL.
        assert call_kwargs["provider_url"] != "https://proxy.example.com"
        assert "azure" in call_kwargs["provider_url"]


# ---------------------------------------------------------------------------
# Anthropic applied-mode routing tests
# ---------------------------------------------------------------------------


class TestAnthropicAppliedMode:
    def _make_client(self, api_key: str = "sk-ant-test") -> Any:
        client = MagicMock()
        client.api_key = api_key
        client.base_url = "https://api.anthropic.com"
        client.messages = MagicMock()
        client.messages.create = MagicMock(
            return_value=MagicMock(
                usage=MagicMock(input_tokens=21, output_tokens=43),
            )
        )
        return client

    def test_observe_only_direct_call(self, recording_transport):
        client = self._make_client()
        sentinel = client.messages.create.return_value

        _anthropic.install(
            client, recording_transport,
            tenant_id=None, environment=None, connection_id=None,
            applied_mode=False, proxy_url=None,
        )
        result = client.messages.create(
            model="claude-3-5-sonnet-20241022", max_tokens=10,
            messages=[{"role": "user", "content": "hi"}],
        )

        assert result is sentinel
        assert len(recording_transport.observations) == 1

    # 1.5.0: the proxy has no Anthropic Messages route, so applied mode /
    # steer / route for an Anthropic client runs as govern. The old
    # proxy-path tests here asserted header shapes on a call that, against a
    # real proxy, could only fail open around TensorCost.

    def test_applied_mode_is_pinned_to_govern(self, recording_transport, caplog):
        client = self._make_client()
        sentinel = client.messages.create.return_value
        _anthropic._reset_pin_warning()
        with caplog.at_level("WARNING", logger="tensorcost"):
            _anthropic.install(
                client, recording_transport,
                tenant_id=None, environment=None, connection_id=None,
                applied_mode=True, proxy_url="https://proxy.example.com",
            )
            _anthropic.install(
                self._make_client(), recording_transport,
                tenant_id=None, environment=None, connection_id=None,
                max_layer="route", proxy_url="https://proxy.example.com",
            )

        with patch("tensorcost._providers._anthropic.proxy_request") as mock_proxy, patch(
            "tensorcost._providers._anthropic.admit_request",
            return_value=(True, "layer=govern; action=allow"),
        ) as mock_admit:
            result = client.messages.create(
                model="claude-3-5-sonnet-20241022", max_tokens=10,
                messages=[{"role": "user", "content": "hi"}],
            )

        assert result is sentinel
        mock_proxy.assert_not_called()
        assert mock_admit.call_count == 1
        assert mock_admit.call_args.kwargs["metadata"]["provider"] == "anthropic"
        pin = [r for r in caplog.records if "Anthropic" in r.getMessage()]
        assert len(pin) == 1
        assert "runs in 'govern'" in pin[0].getMessage()
        assert len(recording_transport.observations) == 1

    def test_pinned_govern_budget_refusal_is_terminal(self, recording_transport):
        from tensorcost import TensorCostBudgetHardCapError, TensorCostModelGovernanceDeniedError

        client = self._make_client()
        direct = client.messages.create
        _anthropic.install(
            client, recording_transport,
            tenant_id=None, environment=None, connection_id=None,
            max_layer="route", proxy_url="https://proxy.example.com",
        )
        calls = []

        def handler(request):
            calls.append(str(request.url))
            return httpx.Response(
                403,
                json={"error": {"code": "BUDGET_HARD_CAP", "budget_id": "b1", "burn_pct": 101}},
            )

        import tensorcost._admit_client as admit_mod

        real = admit_mod.admit_request

        def via_mock_transport(**kw):
            return real(**kw, http_client=httpx.Client(transport=httpx.MockTransport(handler)))

        with patch("tensorcost._providers._anthropic.admit_request", side_effect=via_mock_transport):
            with pytest.raises(TensorCostBudgetHardCapError) as info:
                client.messages.create(model="claude-3-5-sonnet-20241022", max_tokens=10, messages=[])

        assert not isinstance(info.value, TensorCostModelGovernanceDeniedError)
        assert info.value.code == "BUDGET_HARD_CAP"
        assert info.value.budget_id == "b1"
        assert "hard cap is reached" in str(info.value)
        assert len(calls) == 1
        direct.assert_not_called()


class TestWrapAppliedMode:
    def test_wrap_applied_mode_false_unchanged_behaviour(self, openai_client):
        """wrap() with applied_mode=False: method is wrapped but no proxy involved."""
        result = wrap(
            openai_client,
            api_key="k",
            base_url="https://api.tensorcost.com",
            applied_mode=False,
        )
        assert result is openai_client
        assert getattr(openai_client.chat.completions.create, "__tensorcost_wrapped__", False)
        openai_client._tensorcost_transport.close()

    def test_wrap_applied_mode_true_with_proxy_url(self, openai_client):
        """wrap() with applied_mode=True + explicit proxy_url: no error at wrap time."""
        result = wrap(
            openai_client,
            api_key="k",
            tenant_id="t-1",
            base_url="https://api.tensorcost.com",
            applied_mode=True,
            proxy_url="https://proxy.example.com",
        )
        assert result is openai_client
        openai_client._tensorcost_transport.close()

    def test_wrap_applied_mode_true_env_proxy_url(self, openai_client, monkeypatch):
        """wrap() picks up TENSORCOST_PROXY_URL from env."""
        monkeypatch.setenv("TENSORCOST_PROXY_URL", "https://env-proxy.example.com")
        result = wrap(
            openai_client,
            api_key="k",
            tenant_id="t-1",
            base_url="https://api.tensorcost.com",
            applied_mode=True,
        )
        assert result is openai_client
        openai_client._tensorcost_transport.close()

    def test_wrap_applied_mode_true_no_proxy_url_raises(self, openai_client):
        """Missing proxy URL raises at wrap time, not at first call."""
        with pytest.raises(MissingConfigError, match="proxy_url"):
            wrap(
                openai_client,
                api_key="k",
                tenant_id="t-1",
                base_url="https://api.tensorcost.com",
                applied_mode=True,
            )

    def test_wrap_applied_mode_anthropic(self, anthropic_client):
        """Anthropic client also gets applied-mode wiring."""
        result = wrap(
            anthropic_client,
            api_key="k",
            tenant_id="t-1",
            base_url="https://api.tensorcost.com",
            applied_mode=True,
            proxy_url="https://proxy.example.com",
        )
        assert result is anthropic_client
        anthropic_client._tensorcost_transport.close()
