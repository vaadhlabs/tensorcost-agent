from unittest.mock import MagicMock

import pytest

from tensorcost._providers import _openai


def test_chat_completions_create_returns_underlying_response(
    openai_client, recording_transport
):
    sentinel = openai_client.chat.completions.create.return_value
    _openai.install(openai_client, recording_transport, tenant_id="t-1", environment=None, connection_id=None)

    result = openai_client.chat.completions.create(
        model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}]
    )

    assert result is sentinel


def test_chat_completions_observation_shape(openai_client, recording_transport):
    _openai.install(openai_client, recording_transport, tenant_id="t-1", environment=None, connection_id=None)
    openai_client.chat.completions.create(
        model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}]
    )

    assert len(recording_transport.observations) == 1
    obs = recording_transport.observations[0]
    assert obs["provider"] == "openai"
    assert obs["operation"] == "chat.completions"
    assert obs["model"] == "gpt-4o-mini"
    assert obs["status"] == "success"
    assert obs["input_tokens"] == 12
    assert obs["output_tokens"] == 34
    assert "tenant_id" not in obs
    assert obs["sdk_version"] == "tensorcost-python/1.3.0"
    # Phase A4 batch 4 — connection_id absent when wrap() got no
    # connection_id; backend then resolves env via the SDK envelope's
    # `environment` field or falls back to 'production'.
    assert "connection_id" not in obs
    assert obs["correlation_id"]
    assert obs["error_message"] is None
    assert obs["cost_usd_cents"] is None


def test_legacy_completions_wrapped(openai_client, recording_transport):
    _openai.install(openai_client, recording_transport, tenant_id=None, environment=None, connection_id=None)
    openai_client.completions.create(model="text-davinci-003", prompt="hi")

    assert len(recording_transport.observations) == 1
    obs = recording_transport.observations[0]
    assert obs["operation"] == "completions"
    assert obs["input_tokens"] == 5
    assert obs["output_tokens"] == 7


def test_provider_error_records_error_observation_and_reraises(
    openai_client, recording_transport
):
    openai_client.chat.completions.create = MagicMock(
        side_effect=RuntimeError("boom")
    )
    _openai.install(openai_client, recording_transport, tenant_id=None, environment=None, connection_id=None)

    with pytest.raises(RuntimeError, match="boom"):
        openai_client.chat.completions.create(model="gpt-4o", messages=[])

    assert len(recording_transport.observations) == 1
    obs = recording_transport.observations[0]
    assert obs["status"] == "error"
    assert "RuntimeError: boom" in obs["error_message"]


def test_tensorcost_down_does_not_break_customer_call(openai_client):
    """If the transport.post call itself blows up, the customer still
    gets their response."""

    class ExplodingTransport:
        def post(self, _):
            raise RuntimeError("transport disaster")

    sentinel = openai_client.chat.completions.create.return_value
    _openai.install(openai_client, ExplodingTransport(), tenant_id=None, environment=None, connection_id=None)

    result = openai_client.chat.completions.create(
        model="gpt-4o", messages=[]
    )
    assert result is sentinel


def test_install_is_idempotent(openai_client, recording_transport):
    _openai.install(openai_client, recording_transport, tenant_id=None, environment=None, connection_id=None)
    method_after_first = openai_client.chat.completions.create
    _openai.install(openai_client, recording_transport, tenant_id=None, environment=None, connection_id=None)
    assert openai_client.chat.completions.create is method_after_first


def test_connection_id_stamped_on_envelope(openai_client, recording_transport):
    _openai.install(
        openai_client,
        recording_transport,
        tenant_id="t-1",
        environment=None,
        connection_id="conn-py-77",
    )
    openai_client.chat.completions.create(
        model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}]
    )
    assert len(recording_transport.observations) == 1
    assert recording_transport.observations[0]["connection_id"] == "conn-py-77"


def test_usage_dict_response_supported(recording_transport):
    """Some response types expose ``usage`` as a plain dict."""

    client = MagicMock()
    client.chat.completions.create = MagicMock(
        return_value={"usage": {"prompt_tokens": 99, "completion_tokens": 11}}
    )
    client.completions = None  # only patch chat
    _openai.install(client, recording_transport, tenant_id=None, environment=None, connection_id=None)
    client.chat.completions.create(model="gpt-4o", messages=[])

    obs = recording_transport.observations[0]
    assert obs["input_tokens"] == 99
    assert obs["output_tokens"] == 11
