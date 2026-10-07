import pytest

from tensorcost._config import MissingConfigError
from tensorcost._providers import _vertex


def test_generate_content_returns_underlying_response(vertex_client, recording_transport):
    sentinel = vertex_client.models.generate_content.return_value
    _vertex.install(vertex_client, recording_transport, tenant_id="t-1", environment=None, connection_id=None)

    result = vertex_client.models.generate_content(
        model="gemini-2.0-flash",
        contents="hello",
    )
    assert result is sentinel


def test_generate_content_observation_shape(vertex_client, recording_transport):
    _vertex.install(vertex_client, recording_transport, tenant_id="t-1", environment=None, connection_id=None)
    vertex_client.models.generate_content(model="gemini-2.0-flash", contents="hello")

    assert len(recording_transport.observations) == 1
    obs = recording_transport.observations[0]
    assert obs["provider"] == "vertex"
    assert obs["operation"] == "vertex.generate_content"
    assert obs["model"] == "gemini-2.0-flash"
    assert obs["status"] == "success"
    assert obs["input_tokens"] == 42
    assert obs["output_tokens"] == 17
    assert obs["sdk_version"] == "tensorcost-python/1.3.0"
    assert obs["correlation_id"]


def test_generate_content_positional_model(vertex_client, recording_transport):
    _vertex.install(vertex_client, recording_transport, tenant_id=None, environment=None, connection_id=None)
    vertex_client.models.generate_content("gemini-1.5-pro", contents="hi")

    obs = recording_transport.observations[0]
    assert obs["model"] == "gemini-1.5-pro"


def test_generate_content_dict_usage_metadata(vertex_client, recording_transport):
    vertex_client.models.generate_content.return_value = {
        "usage_metadata": {"prompt_token_count": 10, "candidates_token_count": 5},
    }
    _vertex.install(vertex_client, recording_transport, tenant_id=None, environment=None, connection_id=None)
    vertex_client.models.generate_content(model="gemini-2.0-flash")

    obs = recording_transport.observations[0]
    assert obs["input_tokens"] == 10
    assert obs["output_tokens"] == 5


def test_connection_id_stamped_on_envelope(vertex_client, recording_transport):
    _vertex.install(
        vertex_client,
        recording_transport,
        tenant_id="t-1",
        environment=None,
        connection_id="conn-vertex-py",
    )
    vertex_client.models.generate_content(model="gemini-2.0-flash")
    assert recording_transport.observations[0]["connection_id"] == "conn-vertex-py"


def test_generate_content_error_records_and_reraises(vertex_client, recording_transport):
    boom = RuntimeError("ResourceExhausted: quota exceeded")
    vertex_client.models.generate_content.side_effect = boom
    _vertex.install(vertex_client, recording_transport, tenant_id=None, environment=None, connection_id=None)

    with pytest.raises(RuntimeError, match="ResourceExhausted"):
        vertex_client.models.generate_content(model="gemini-2.0-flash")

    obs = recording_transport.observations[0]
    assert obs["status"] == "error"
    assert "ResourceExhausted" in obs["error_message"]
    assert obs["input_tokens"] is None
    assert obs["output_tokens"] is None


def test_tensorcost_down_does_not_break_customer_call(vertex_client):
    class ExplodingTransport:
        def post(self, _):
            raise RuntimeError("transport disaster")

    sentinel = vertex_client.models.generate_content.return_value
    _vertex.install(vertex_client, ExplodingTransport(), tenant_id=None, environment=None, connection_id=None)

    result = vertex_client.models.generate_content(model="gemini-2.0-flash")
    assert result is sentinel


def test_install_is_idempotent(vertex_client, recording_transport):
    _vertex.install(vertex_client, recording_transport, tenant_id=None, environment=None, connection_id=None)
    method_after_first = vertex_client.models.generate_content
    _vertex.install(vertex_client, recording_transport, tenant_id=None, environment=None, connection_id=None)
    assert vertex_client.models.generate_content is method_after_first


def test_generate_content_stream_emits_placeholder_observation(vertex_client, recording_transport):
    sentinel = vertex_client.models.generate_content_stream.return_value
    _vertex.install(vertex_client, recording_transport, tenant_id=None, environment=None, connection_id="conn-456")
    result = vertex_client.models.generate_content_stream(model="gemini-2.0-flash", contents="stream me")

    assert len(recording_transport.observations) == 1
    obs = recording_transport.observations[0]
    assert obs["operation"] == "vertex.generate_content_stream"
    assert obs["input_tokens"] is None
    assert obs["output_tokens"] is None
    assert obs["status"] == "success"
    assert "streaming" in obs["error_message"]
    assert obs["connection_id"] == "conn-456"
    assert result is sentinel


def test_generate_content_stream_error_records_and_reraises(vertex_client, recording_transport):
    vertex_client.models.generate_content_stream.side_effect = RuntimeError("Unavailable")
    _vertex.install(vertex_client, recording_transport, tenant_id=None, environment=None, connection_id=None)

    with pytest.raises(RuntimeError, match="Unavailable"):
        vertex_client.models.generate_content_stream(model="gemini-2.0-flash")

    obs = recording_transport.observations[0]
    assert obs["status"] == "error"
    assert "Unavailable" in obs["error_message"]


def test_install_raises_on_applied_mode(vertex_client, recording_transport):
    with pytest.raises(MissingConfigError, match="not supported for Google Vertex"):
        _vertex.install(
            vertex_client,
            recording_transport,
            tenant_id=None,
            environment=None,
            connection_id=None,
            applied_mode=True,
        )
    assert vertex_client.models.generate_content.__class__.__name__ == "MagicMock"
