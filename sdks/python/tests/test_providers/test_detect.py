import pytest

from tensorcost._providers._detect import (
    UnsupportedClientError,
    detect_provider,
)


def test_detect_openai(openai_client):
    assert detect_provider(openai_client) == "openai"


def test_detect_anthropic(anthropic_client):
    assert detect_provider(anthropic_client) == "anthropic"


def test_detect_bedrock(bedrock_client):
    assert detect_provider(bedrock_client) == "bedrock"


def test_detect_vertex(vertex_client):
    assert detect_provider(vertex_client) == "vertex"


def test_detect_gemini_dev_not_vertex(gemini_dev_client):
    with pytest.raises(UnsupportedClientError):
        detect_provider(gemini_dev_client)


def test_detect_bedrock_falls_back_to_duck_typing_without_meta():
    """A minimal fake with no botocore `meta` object should still be
    detected via invoke_model + converse callables."""

    class NakedBedrock:
        def invoke_model(self, **kwargs):
            ...

        def converse(self, **kwargs):
            ...

    assert detect_provider(NakedBedrock()) == "bedrock"


def test_detect_unknown_raises():
    class Naked:
        pass

    with pytest.raises(UnsupportedClientError):
        detect_provider(Naked())


def test_anthropic_with_chat_attribute_is_not_anthropic(openai_client):
    """Defensive — if a hybrid client carries both ``chat`` and
    ``messages``, OpenAI wins (it's the canonical shape)."""
    assert detect_provider(openai_client) == "openai"
