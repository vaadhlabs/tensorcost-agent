import pytest

from tensorcost import MissingConfigError, wrap, __version__
from tensorcost._providers._detect import UnsupportedClientError


def test_version_string():
    assert __version__ == "1.3.0"


def test_wrap_returns_same_object(openai_client):
    result = wrap(openai_client, api_key="k", base_url="https://x")
    assert result is openai_client
    result._tensorcost_transport.close()


def test_wrap_marks_methods_as_wrapped(openai_client):
    wrap(openai_client, api_key="k", base_url="https://x")
    try:
        assert getattr(
            openai_client.chat.completions.create,
            "__tensorcost_wrapped__",
            False,
        ) is True
    finally:
        openai_client._tensorcost_transport.close()


def test_wrap_is_idempotent(openai_client):
    wrap(openai_client, api_key="k", base_url="https://x")
    first_method = openai_client.chat.completions.create
    try:
        wrap(openai_client, api_key="k", base_url="https://x")
        assert openai_client.chat.completions.create is first_method
    finally:
        openai_client._tensorcost_transport.close()


def test_wrap_rejects_unknown_client():
    class Mystery:
        pass

    with pytest.raises(UnsupportedClientError):
        wrap(Mystery(), api_key="k", base_url="https://x")


def test_wrap_requires_api_key(openai_client):
    with pytest.raises(MissingConfigError):
        wrap(openai_client)


def test_wrap_reads_env_vars(openai_client, monkeypatch):
    monkeypatch.setenv("TENSORCOST_API_KEY", "from-env")
    wrap(openai_client, base_url="https://x")
    try:
        # Confirm wrapping happened (proxy for "config resolved").
        assert getattr(
            openai_client.chat.completions.create,
            "__tensorcost_wrapped__",
            False,
        )
    finally:
        openai_client._tensorcost_transport.close()


def test_wrap_bedrock_observe_only_succeeds(bedrock_client):
    result = wrap(bedrock_client, api_key="k", base_url="https://x")
    try:
        assert result is bedrock_client
        assert getattr(
            bedrock_client.invoke_model, "__tensorcost_wrapped__", False
        ) is True
        assert getattr(
            bedrock_client.converse, "__tensorcost_wrapped__", False
        ) is True
    finally:
        bedrock_client._tensorcost_transport.close()


def test_wrap_bedrock_applied_mode_raises(bedrock_client):
    with pytest.raises(MissingConfigError, match="not supported for bedrock"):
        wrap(
            bedrock_client,
            api_key="k",
            base_url="https://x",
            applied_mode=True,
            proxy_url="https://proxy.local",
        )


def test_wrap_vertex_observe_only_succeeds(vertex_client):
    result = wrap(vertex_client, api_key="k", base_url="https://x")
    try:
        assert result is vertex_client
        assert getattr(
            vertex_client.models.generate_content, "__tensorcost_wrapped__", False
        ) is True
        assert getattr(
            vertex_client.models.generate_content_stream, "__tensorcost_wrapped__", False
        ) is True
    finally:
        vertex_client._tensorcost_transport.close()


def test_wrap_vertex_applied_mode_raises(vertex_client):
    with pytest.raises(MissingConfigError, match="not supported for vertex"):
        wrap(
            vertex_client,
            api_key="k",
            base_url="https://x",
            applied_mode=True,
            proxy_url="https://proxy.local",
        )
