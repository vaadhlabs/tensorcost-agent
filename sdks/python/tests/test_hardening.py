"""End-to-end hardening integration tests via wrap().

These tests exercise the full wrap() → install() → proxy_request() chain
by patching proxy_request at the module-level import in the provider wrappers.
They verify that RetryConfig, timeout_s, on_lifecycle_event, fail_open_enabled,
and circuit breaker wiring all reach the proxy call correctly.

The ObservationTransport._get_token() call inside the applied-mode path makes a
real HTTP call to exchange the API key for a JWT.  We stub it out here so tests
run offline.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from tensorcost import wrap
from tensorcost._errors import TensorCostProxyError
from tensorcost._retry import RetryConfig
from tensorcost._telemetry import LifecycleEvent


def _stub_token(client: Any) -> None:
    """Stub _get_token() on the transport that wrap() attached to *client*."""
    client._tensorcost_transport._get_token = lambda: "fake-jwt"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_openai_client(api_key: str = "sk-test") -> Any:
    client = MagicMock()
    client.api_key = api_key
    client.base_url = "https://api.openai.com/v1"
    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    client.chat.completions.create = MagicMock(
        return_value=MagicMock(
            usage=MagicMock(prompt_tokens=10, completion_tokens=5),
        )
    )
    client.completions = MagicMock()
    client.completions.create = MagicMock(
        return_value=MagicMock(
            usage=MagicMock(prompt_tokens=3, completion_tokens=2),
        )
    )
    return client


def _make_anthropic_client(api_key: str = "sk-ant-test") -> Any:
    client = MagicMock()
    client.api_key = api_key
    client.base_url = "https://api.anthropic.com"
    # Suppress MagicMock's auto-creation of client.chat so detect_provider
    # correctly identifies this as an Anthropic client (not OpenAI).
    client.chat = None
    client.messages = MagicMock()
    client.messages.create = MagicMock(
        return_value=MagicMock(
            usage=MagicMock(input_tokens=8, output_tokens=4),
        )
    )
    return client


def _openai_proxy_response() -> dict:
    return {
        "id": "chatcmpl-test",
        "choices": [],
        "usage": {"prompt_tokens": 5, "completion_tokens": 10},
    }


def _anthropic_proxy_response() -> dict:
    return {
        "id": "msg_test",
        "usage": {"input_tokens": 6, "output_tokens": 12},
    }


# ---------------------------------------------------------------------------
# OpenAI hardening via wrap()
# ---------------------------------------------------------------------------


class TestOpenAIHardeningViaWrap:
    def test_proxy_success_returns_response(self):
        client = _make_openai_client()
        expected = _openai_proxy_response()

        wrap(
            client,
            api_key="k",
            base_url="https://api.tensorcost.com",
            applied_mode=True,
            proxy_url="https://proxy.example.com",
        )
        _stub_token(client)
        try:
            with patch("tensorcost._providers._openai.proxy_request") as mock:
                mock.return_value = (expected, None)
                result = client.chat.completions.create(
                    model="gpt-4o-mini", messages=[]
                )
            assert result == expected
        finally:
            client._tensorcost_transport.close()

    def test_retry_config_forwarded_to_proxy_request(self):
        client = _make_openai_client()
        captured_kwargs: list[dict] = []

        def fake_proxy_request(**kwargs: Any) -> tuple[dict, None]:
            captured_kwargs.append(kwargs)
            return _openai_proxy_response(), None

        retry = RetryConfig(max_attempts=3, base_delay_ms=100.0, max_delay_ms=1000.0)
        wrap(
            client,
            api_key="k",
            base_url="https://api.tensorcost.com",
            applied_mode=True,
            proxy_url="https://proxy.example.com",
            retry=retry,
        )
        _stub_token(client)
        try:
            with patch("tensorcost._providers._openai.proxy_request", side_effect=fake_proxy_request):
                client.chat.completions.create(model="gpt-4o-mini", messages=[])
            assert len(captured_kwargs) == 1
            assert captured_kwargs[0]["retry"].max_attempts == 3
        finally:
            client._tensorcost_transport.close()

    def test_timeout_forwarded_to_proxy_request(self):
        client = _make_openai_client()
        captured: list[dict] = []

        def fake_proxy(**kwargs: Any) -> tuple[dict, None]:
            captured.append(kwargs)
            return _openai_proxy_response(), None

        wrap(
            client,
            api_key="k",
            base_url="https://api.tensorcost.com",
            applied_mode=True,
            proxy_url="https://proxy.example.com",
            timeout_s=120.0,
        )
        _stub_token(client)
        try:
            with patch("tensorcost._providers._openai.proxy_request", side_effect=fake_proxy):
                client.chat.completions.create(model="gpt-4o-mini", messages=[])
            assert captured[0]["timeout_s"] == 120.0
        finally:
            client._tensorcost_transport.close()

    def test_lifecycle_callback_forwarded(self):
        client = _make_openai_client()
        events: list[LifecycleEvent] = []

        wrap(
            client,
            api_key="k",
            base_url="https://api.tensorcost.com",
            applied_mode=True,
            proxy_url="https://proxy.example.com",
            on_lifecycle_event=events.append,
        )
        _stub_token(client)
        received_cb = []

        def capturing_proxy(**kwargs: Any) -> tuple[dict, None]:
            received_cb.append(kwargs.get("on_lifecycle_event"))
            return _openai_proxy_response(), None

        try:
            with patch("tensorcost._providers._openai.proxy_request", side_effect=capturing_proxy):
                client.chat.completions.create(model="gpt-4o-mini", messages=[])
            # The callback that proxy_request received is a bound method of
            # `events`.  Call it with a sentinel to confirm it appends.
            assert len(received_cb) == 1
            cb = received_cb[0]
            assert cb is not None
            cb("test-event")
            assert events == ["test-event"]
        finally:
            client._tensorcost_transport.close()

    def test_fail_open_routes_direct_when_proxy_fails(self):
        client = _make_openai_client()
        direct_sentinel = client.chat.completions.create.return_value

        wrap(
            client,
            api_key="k",
            base_url="https://api.tensorcost.com",
            applied_mode=True,
            proxy_url="https://proxy.example.com",
            fail_open_enabled=True,
        )
        _stub_token(client)
        try:
            with patch("tensorcost._providers._openai.proxy_request") as mock:
                mock.side_effect = TensorCostProxyError("proxy down", attempt=1)
                result = client.chat.completions.create(model="gpt-4o-mini", messages=[])
            assert result is direct_sentinel
        finally:
            client._tensorcost_transport.close()

    def test_fail_open_disabled_raises_on_proxy_failure(self):
        client = _make_openai_client()

        wrap(
            client,
            api_key="k",
            base_url="https://api.tensorcost.com",
            applied_mode=True,
            proxy_url="https://proxy.example.com",
            fail_open_enabled=False,
        )
        _stub_token(client)
        try:
            with patch("tensorcost._providers._openai.proxy_request") as mock:
                mock.side_effect = TensorCostProxyError("proxy down", attempt=1)
                with pytest.raises(TensorCostProxyError):
                    client.chat.completions.create(model="gpt-4o-mini", messages=[])
        finally:
            client._tensorcost_transport.close()

    def test_circuit_attached_to_client(self):
        client = _make_openai_client()
        wrap(
            client,
            api_key="k",
            base_url="https://api.tensorcost.com",
            applied_mode=True,
            proxy_url="https://proxy.example.com",
        )
        try:
            assert hasattr(client, "_tensorcost_circuit")
            assert client._tensorcost_circuit.state == "CLOSED"
        finally:
            client._tensorcost_transport.close()


# ---------------------------------------------------------------------------
# Anthropic hardening via wrap()
# ---------------------------------------------------------------------------


class TestAnthropicHardeningViaWrap:
    def test_proxy_success_returns_response(self):
        client = _make_anthropic_client()
        expected = _anthropic_proxy_response()

        wrap(
            client,
            api_key="k",
            base_url="https://api.tensorcost.com",
            applied_mode=True,
            proxy_url="https://proxy.example.com",
        )
        _stub_token(client)
        try:
            with patch("tensorcost._providers._anthropic.proxy_request") as mock:
                mock.return_value = (expected, None)
                result = client.messages.create(
                    model="claude-3-5-sonnet-20241022", max_tokens=10, messages=[]
                )
            assert result == expected
        finally:
            client._tensorcost_transport.close()

    def test_fail_open_routes_direct_when_proxy_fails(self):
        client = _make_anthropic_client()
        direct_sentinel = client.messages.create.return_value

        wrap(
            client,
            api_key="k",
            base_url="https://api.tensorcost.com",
            applied_mode=True,
            proxy_url="https://proxy.example.com",
            fail_open_enabled=True,
        )
        _stub_token(client)
        try:
            with patch("tensorcost._providers._anthropic.proxy_request") as mock:
                mock.side_effect = TensorCostProxyError("proxy down", attempt=1)
                result = client.messages.create(
                    model="claude-3-5-sonnet-20241022", max_tokens=10, messages=[]
                )
            assert result is direct_sentinel
        finally:
            client._tensorcost_transport.close()

    def test_fail_open_disabled_raises_on_proxy_failure(self):
        client = _make_anthropic_client()

        wrap(
            client,
            api_key="k",
            base_url="https://api.tensorcost.com",
            applied_mode=True,
            proxy_url="https://proxy.example.com",
            fail_open_enabled=False,
        )
        _stub_token(client)
        try:
            with patch("tensorcost._providers._anthropic.proxy_request") as mock:
                mock.side_effect = TensorCostProxyError("proxy down", attempt=1)
                with pytest.raises(TensorCostProxyError):
                    client.messages.create(
                        model="claude-3-5-sonnet-20241022", max_tokens=10, messages=[]
                    )
        finally:
            client._tensorcost_transport.close()
