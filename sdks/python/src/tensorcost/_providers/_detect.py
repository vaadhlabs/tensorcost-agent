"""Provider detection via duck-typing.

We deliberately avoid ``isinstance`` checks against the OpenAI /
Anthropic / boto3 SDK classes so that the SDK has zero hard imports of
those packages. A customer who uses only OpenAI shouldn't need
``anthropic`` or ``boto3`` installed (and vice versa).

Bedrock detection prefers ``client.meta.service_model.service_name``
(set by botocore for every low-level client, e.g. ``"bedrock-runtime"``)
because it's the most precise discriminator available and doesn't depend
on any particular method being mocked out. We fall back to a duck-typed
check (``invoke_model`` + ``converse`` callables, no ``chat``/``messages``
shape) for lightweight test doubles that don't carry a real botocore
``meta`` object.
"""

from __future__ import annotations

from typing import Any


class UnsupportedClientError(TypeError):
    """Raised by :func:`detect_provider` when the client shape is not
    recognized as an OpenAI, Anthropic, Bedrock-runtime, or Vertex client."""


def _looks_like_bedrock(client: Any) -> bool:
    meta = getattr(client, "meta", None)
    service_model = getattr(meta, "service_model", None) if meta is not None else None
    service_name = getattr(service_model, "service_name", None)
    if service_name == "bedrock-runtime":
        return True

    # Fallback duck-typing for fakes/mocks without a botocore `meta`
    # object: require both call methods and the absence of the
    # OpenAI/Anthropic shapes so we don't collide with hybrid test doubles.
    has_invoke = callable(getattr(client, "invoke_model", None))
    has_converse = callable(getattr(client, "converse", None))
    return (
        has_invoke
        and has_converse
        and getattr(client, "chat", None) is None
        and getattr(client, "messages", None) is None
    )


def _looks_like_openai(client: Any) -> bool:
    chat = getattr(client, "chat", None)
    completions = getattr(chat, "completions", None) if chat is not None else None
    return callable(getattr(completions, "create", None))


def _looks_like_vertex(client: Any) -> bool:
    vertexai_flag = getattr(client, "vertexai", False) is True
    api_client = getattr(client, "_api_client", None)
    if not vertexai_flag and api_client is not None:
        vertexai_flag = getattr(api_client, "vertexai", False) is True
    if not vertexai_flag:
        return False
    models = getattr(client, "models", None)
    return callable(getattr(models, "generate_content", None))


def _looks_like_anthropic(client: Any) -> bool:
    # Anthropic clients expose ``client.messages.create``; we additionally
    # require the *absence* of ``client.chat`` to disambiguate from OpenAI.
    if getattr(client, "chat", None) is not None:
        return False
    messages = getattr(client, "messages", None)
    return callable(getattr(messages, "create", None))


def detect_provider(client: Any) -> str:
    """Return ``"openai"``, ``"anthropic"``, ``"bedrock"``, or ``"vertex"``
    based on duck-typing.

    Raises :class:`UnsupportedClientError` if no shape matches.
    """
    # Checked first: Bedrock clients have a generic enough shape (plain
    # bound methods, no nested `.chat`/`.messages` namespace) that we want
    # the most specific signal (service_name) to win before falling
    # through to the looser OpenAI/Anthropic checks.
    if _looks_like_bedrock(client):
        return "bedrock"
    if _looks_like_vertex(client):
        return "vertex"
    if _looks_like_openai(client):
        return "openai"
    if _looks_like_anthropic(client):
        return "anthropic"
    raise UnsupportedClientError(
        "tensorcost.wrap(): client does not look like an OpenAI, Anthropic, "
        "Bedrock-runtime, or Vertex AI client. Supported: openai>=1.0, "
        "anthropic>=0.20, boto3 bedrock-runtime clients (observe-only), "
        "google-genai Client with vertexai=True (observe-only)."
    )
