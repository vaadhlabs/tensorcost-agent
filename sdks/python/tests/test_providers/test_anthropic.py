from unittest.mock import MagicMock

import pytest

from tensorcost._providers import _anthropic


def test_messages_create_returns_underlying_response(
    anthropic_client, recording_transport
):
    sentinel = anthropic_client.messages.create.return_value
    _anthropic.install(anthropic_client, recording_transport, tenant_id="t-1", environment=None, connection_id=None)

    result = anthropic_client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=10,
        messages=[{"role": "user", "content": "hi"}],
    )
    assert result is sentinel


def test_messages_observation_shape(anthropic_client, recording_transport):
    _anthropic.install(anthropic_client, recording_transport, tenant_id="t-1", environment=None, connection_id=None)
    anthropic_client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=10,
        messages=[{"role": "user", "content": "hi"}],
    )

    assert len(recording_transport.observations) == 1
    obs = recording_transport.observations[0]
    assert obs["provider"] == "anthropic"
    assert obs["operation"] == "messages"
    assert obs["model"] == "claude-3-5-sonnet-20241022"
    assert obs["status"] == "success"
    assert obs["input_tokens"] == 21
    assert obs["output_tokens"] == 43
    assert "tenant_id" not in obs
    assert obs["sdk_version"] == "tensorcost-python/1.3.0"
    assert obs["correlation_id"]
    assert "connection_id" not in obs


def test_connection_id_stamped_on_envelope(anthropic_client, recording_transport):
    _anthropic.install(
        anthropic_client,
        recording_transport,
        tenant_id="t-1",
        environment=None,
        connection_id="conn-anth-py",
    )
    anthropic_client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=10,
        messages=[{"role": "user", "content": "hi"}],
    )
    assert len(recording_transport.observations) == 1
    assert recording_transport.observations[0]["connection_id"] == "conn-anth-py"


def test_provider_error_records_and_reraises(
    anthropic_client, recording_transport
):
    anthropic_client.messages.create = MagicMock(
        side_effect=RuntimeError("nope")
    )
    _anthropic.install(anthropic_client, recording_transport, tenant_id=None, environment=None, connection_id=None)

    with pytest.raises(RuntimeError, match="nope"):
        anthropic_client.messages.create(model="claude-3-opus", messages=[])

    obs = recording_transport.observations[0]
    assert obs["status"] == "error"
    assert "RuntimeError: nope" in obs["error_message"]


def test_tensorcost_down_does_not_break_customer_call(anthropic_client):
    class ExplodingTransport:
        def post(self, _):
            raise RuntimeError("transport disaster")

    sentinel = anthropic_client.messages.create.return_value
    _anthropic.install(anthropic_client, ExplodingTransport(), tenant_id=None, environment=None, connection_id=None)

    result = anthropic_client.messages.create(
        model="claude-3-opus", messages=[]
    )
    assert result is sentinel


def test_install_is_idempotent(anthropic_client, recording_transport):
    _anthropic.install(anthropic_client, recording_transport, tenant_id=None, environment=None, connection_id=None)
    method_after_first = anthropic_client.messages.create
    _anthropic.install(anthropic_client, recording_transport, tenant_id=None, environment=None, connection_id=None)
    assert anthropic_client.messages.create is method_after_first
