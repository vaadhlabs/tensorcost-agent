import io
import json

import pytest

from tensorcost._config import MissingConfigError
from tensorcost._providers import _bedrock


def test_invoke_model_returns_underlying_response(bedrock_client, recording_transport):
    sentinel = bedrock_client.invoke_model.return_value
    _bedrock.install(bedrock_client, recording_transport, tenant_id="t-1", environment=None, connection_id=None)

    result = bedrock_client.invoke_model(
        modelId="anthropic.claude-3-haiku-20240307-v1:0",
        body=json.dumps({"messages": []}),
    )
    assert result is sentinel


def test_invoke_model_anthropic_shape_observation(bedrock_client, recording_transport):
    _bedrock.install(bedrock_client, recording_transport, tenant_id="t-1", environment=None, connection_id=None)
    bedrock_client.invoke_model(
        modelId="anthropic.claude-3-haiku-20240307-v1:0",
        body=json.dumps({"messages": []}),
    )

    assert len(recording_transport.observations) == 1
    obs = recording_transport.observations[0]
    assert obs["provider"] == "bedrock"
    assert obs["operation"] == "bedrock.invoke_model"
    assert obs["model"] == "anthropic.claude-3-haiku-20240307-v1:0"
    assert obs["status"] == "success"
    assert obs["input_tokens"] == 15
    assert obs["output_tokens"] == 30
    assert "tenant_id" not in obs
    assert obs["sdk_version"] == "tensorcost-python/1.3.0"
    assert obs["correlation_id"]
    assert "connection_id" not in obs


def test_invoke_model_body_is_still_readable_after_observation(bedrock_client, recording_transport):
    """We drain response['body'] to extract tokens; the customer must
    still be able to read the same bytes afterward."""
    _bedrock.install(bedrock_client, recording_transport, tenant_id=None, environment=None, connection_id=None)
    result = bedrock_client.invoke_model(modelId="anthropic.claude-3-haiku-20240307-v1:0")

    replayed = json.loads(result["body"].read())
    assert replayed == {"usage": {"input_tokens": 15, "output_tokens": 30}}


def test_invoke_model_nova_titan_shape(bedrock_client, recording_transport):
    bedrock_client.invoke_model.return_value = {
        "body": io.BytesIO(
            json.dumps(
                {
                    "output": {"message": {"content": []}},
                    "usage": {"inputTokens": 100, "outputTokens": 55},
                }
            ).encode("utf-8")
        )
    }
    _bedrock.install(bedrock_client, recording_transport, tenant_id=None, environment=None, connection_id=None)
    bedrock_client.invoke_model(modelId="amazon.nova-micro-v1:0")

    obs = recording_transport.observations[0]
    assert obs["input_tokens"] == 100
    assert obs["output_tokens"] == 55
    assert obs["model"] == "amazon.nova-micro-v1:0"


def test_invoke_model_unrecognised_usage_shape_emits_none(bedrock_client, recording_transport):
    bedrock_client.invoke_model.return_value = {
        "body": io.BytesIO(json.dumps({"usage": {"weird_field": 1}}).encode("utf-8"))
    }
    _bedrock.install(bedrock_client, recording_transport, tenant_id=None, environment=None, connection_id=None)
    bedrock_client.invoke_model(modelId="some.unknown-model")

    obs = recording_transport.observations[0]
    assert obs["input_tokens"] is None
    assert obs["output_tokens"] is None
    assert obs["status"] == "success"


def test_converse_observation_shape(bedrock_client, recording_transport):
    _bedrock.install(bedrock_client, recording_transport, tenant_id=None, environment="staging", connection_id=None)
    bedrock_client.converse(
        modelId="anthropic.claude-3-5-sonnet-20241022-v2:0",
        messages=[{"role": "user", "content": [{"text": "hi"}]}],
    )

    assert len(recording_transport.observations) == 1
    obs = recording_transport.observations[0]
    assert obs["operation"] == "bedrock.converse"
    assert obs["input_tokens"] == 12
    assert obs["output_tokens"] == 24
    assert obs["environment"] == "staging"


def test_connection_id_stamped_on_envelope(bedrock_client, recording_transport):
    _bedrock.install(
        bedrock_client,
        recording_transport,
        tenant_id="t-1",
        environment=None,
        connection_id="conn-bedrock-py",
    )
    bedrock_client.converse(modelId="anthropic.claude-3-5-sonnet-20241022-v2:0")
    assert recording_transport.observations[0]["connection_id"] == "conn-bedrock-py"


def test_invoke_model_error_records_and_reraises(bedrock_client, recording_transport):
    boom = RuntimeError("ThrottlingException: too many requests")
    bedrock_client.invoke_model.side_effect = boom
    _bedrock.install(bedrock_client, recording_transport, tenant_id=None, environment=None, connection_id=None)

    with pytest.raises(RuntimeError, match="ThrottlingException"):
        bedrock_client.invoke_model(modelId="amazon.nova-micro-v1:0")

    obs = recording_transport.observations[0]
    assert obs["status"] == "error"
    assert "ThrottlingException" in obs["error_message"]
    assert obs["input_tokens"] is None
    assert obs["output_tokens"] is None


def test_converse_error_records_and_reraises(bedrock_client, recording_transport):
    bedrock_client.converse.side_effect = RuntimeError("nope")
    _bedrock.install(bedrock_client, recording_transport, tenant_id=None, environment=None, connection_id=None)

    with pytest.raises(RuntimeError, match="nope"):
        bedrock_client.converse(modelId="anthropic.claude-3-opus")

    obs = recording_transport.observations[0]
    assert obs["status"] == "error"
    assert "RuntimeError: nope" in obs["error_message"]


def test_tensorcost_down_does_not_break_customer_call(bedrock_client):
    class ExplodingTransport:
        def post(self, _):
            raise RuntimeError("transport disaster")

    sentinel = bedrock_client.invoke_model.return_value
    _bedrock.install(bedrock_client, ExplodingTransport(), tenant_id=None, environment=None, connection_id=None)

    result = bedrock_client.invoke_model(modelId="anthropic.claude-3-opus")
    assert result is sentinel


def test_install_is_idempotent(bedrock_client, recording_transport):
    _bedrock.install(bedrock_client, recording_transport, tenant_id=None, environment=None, connection_id=None)
    method_after_first = bedrock_client.invoke_model
    _bedrock.install(bedrock_client, recording_transport, tenant_id=None, environment=None, connection_id=None)
    assert bedrock_client.invoke_model is method_after_first


# ── Streaming — placeholder observation ──────────────────────────────────────


def test_invoke_model_stream_emits_placeholder_observation(bedrock_client, recording_transport):
    sentinel = bedrock_client.invoke_model_with_response_stream.return_value
    _bedrock.install(bedrock_client, recording_transport, tenant_id=None, environment=None, connection_id="conn-456")
    result = bedrock_client.invoke_model_with_response_stream(modelId="amazon.nova-pro-v1:0")

    assert len(recording_transport.observations) == 1
    obs = recording_transport.observations[0]
    assert obs["operation"] == "bedrock.invoke_model_stream"
    assert obs["input_tokens"] is None
    assert obs["output_tokens"] is None
    assert obs["status"] == "success"
    assert "streaming" in obs["error_message"]
    assert obs["connection_id"] == "conn-456"
    # We never touch the stream object itself.
    assert result is sentinel


def test_converse_stream_emits_placeholder_observation(bedrock_client, recording_transport):
    _bedrock.install(bedrock_client, recording_transport, tenant_id=None, environment=None, connection_id=None)
    bedrock_client.converse_stream(modelId="anthropic.claude-3-5-sonnet-20241022-v2:0")

    assert len(recording_transport.observations) == 1
    obs = recording_transport.observations[0]
    assert obs["operation"] == "bedrock.converse_stream"
    assert obs["input_tokens"] is None


def test_converse_stream_error_records_and_reraises(bedrock_client, recording_transport):
    bedrock_client.converse_stream.side_effect = RuntimeError("ServiceUnavailableException")
    _bedrock.install(bedrock_client, recording_transport, tenant_id=None, environment=None, connection_id=None)

    with pytest.raises(RuntimeError, match="ServiceUnavailableException"):
        bedrock_client.converse_stream(modelId="amazon.nova-micro-v1:0")

    obs = recording_transport.observations[0]
    assert obs["status"] == "error"
    assert "ServiceUnavailableException" in obs["error_message"]


# ── Applied mode guard ────────────────────────────────────────────────────────


def test_install_raises_on_applied_mode(bedrock_client, recording_transport):
    with pytest.raises(MissingConfigError, match="not supported for AWS Bedrock"):
        _bedrock.install(
            bedrock_client,
            recording_transport,
            tenant_id=None,
            environment=None,
            connection_id=None,
            applied_mode=True,
        )
    # Nothing should have been patched.
    assert bedrock_client.invoke_model.__class__.__name__ == "MagicMock"


def test_invoke_model_llama_shape_observation(bedrock_client, recording_transport):
    body = json.dumps(
        {"generation": "hi", "prompt_token_count": 31, "generation_token_count": 12}
    ).encode("utf-8")
    bedrock_client.invoke_model.return_value = {"body": io.BytesIO(body)}
    _bedrock.install(bedrock_client, recording_transport, tenant_id=None, environment=None, connection_id=None)
    bedrock_client.invoke_model(modelId="us.meta.llama3-1-8b-instruct-v1:0", body="{}")

    obs = recording_transport.observations[0]
    assert obs["input_tokens"] == 31
    assert obs["output_tokens"] == 12


def test_invoke_model_prefers_bedrock_token_headers(bedrock_client, recording_transport):
    body = json.dumps({"outputs": [{"text": "hi"}]}).encode("utf-8")
    bedrock_client.invoke_model.return_value = {
        "body": io.BytesIO(body),
        "ResponseMetadata": {
            "HTTPHeaders": {
                "x-amzn-bedrock-input-token-count": "55",
                "x-amzn-bedrock-output-token-count": "9",
            }
        },
    }
    _bedrock.install(bedrock_client, recording_transport, tenant_id=None, environment=None, connection_id=None)
    result = bedrock_client.invoke_model(modelId="mistral.mistral-large-2402-v1:0", body="{}")

    obs = recording_transport.observations[0]
    assert obs["input_tokens"] == 55
    assert obs["output_tokens"] == 9
    # Headers path must not drain the body the customer reads next.
    assert json.loads(result["body"].read())["outputs"][0]["text"] == "hi"
