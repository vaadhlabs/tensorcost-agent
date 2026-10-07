"""Shared test fixtures."""

from __future__ import annotations

import io
import json
import threading
from typing import Any
from unittest.mock import MagicMock

import pytest


class FakeOpenAIClient:
    """Mimics the surface of ``openai.OpenAI`` we monkey-patch."""

    def __init__(self) -> None:
        self.chat = MagicMock()
        self.chat.completions = MagicMock()
        self.chat.completions.create = MagicMock(
            return_value=MagicMock(
                usage=MagicMock(prompt_tokens=12, completion_tokens=34),
            )
        )
        self.completions = MagicMock()
        self.completions.create = MagicMock(
            return_value=MagicMock(
                usage=MagicMock(prompt_tokens=5, completion_tokens=7),
            )
        )


class FakeAnthropicClient:
    """Mimics the surface of ``anthropic.Anthropic``."""

    def __init__(self) -> None:
        self.messages = MagicMock()
        self.messages.create = MagicMock(
            return_value=MagicMock(
                usage=MagicMock(input_tokens=21, output_tokens=43),
            )
        )


@pytest.fixture
def openai_client() -> FakeOpenAIClient:
    return FakeOpenAIClient()


@pytest.fixture
def anthropic_client() -> FakeAnthropicClient:
    return FakeAnthropicClient()


class _FakeServiceModel:
    service_name = "bedrock-runtime"


class _FakeMeta:
    service_model = _FakeServiceModel()


class FakeBedrockClient:
    """Mimics the surface of a boto3 ``bedrock-runtime`` client we touch:
    ``meta.service_model.service_name`` (detection) plus the
    ``invoke_model`` / ``converse`` / streaming call methods.

    ``invoke_model``'s response ``body`` mimics botocore's single-read
    ``StreamingBody`` closely enough for our purposes: an ``io.BytesIO``
    supports exactly one meaningful ``.read()`` before it's drained.
    """

    def __init__(self) -> None:
        self.meta = _FakeMeta()
        invoke_body = json.dumps(
            {"usage": {"input_tokens": 15, "output_tokens": 30}}
        ).encode("utf-8")
        self.invoke_model = MagicMock(
            return_value={
                "body": io.BytesIO(invoke_body),
                "contentType": "application/json",
            }
        )
        self.converse = MagicMock(
            return_value={
                "output": {"message": {"role": "assistant", "content": []}},
                "usage": {"inputTokens": 12, "outputTokens": 24},
            }
        )
        self.invoke_model_with_response_stream = MagicMock(
            return_value={"body": iter([])}
        )
        self.converse_stream = MagicMock(return_value={"stream": iter([])})


@pytest.fixture
def bedrock_client() -> FakeBedrockClient:
    return FakeBedrockClient()


class _FakeUsageMetadata:
    prompt_token_count = 42
    candidates_token_count = 17


class FakeVertexClient:
    """Mimics ``google.genai.Client(vertexai=True)`` surface we patch."""

    def __init__(self) -> None:
        self.vertexai = True
        self.models = MagicMock()
        self.models.generate_content = MagicMock(
            return_value=MagicMock(usage_metadata=_FakeUsageMetadata())
        )
        self.models.generate_content_stream = MagicMock(
            return_value=iter([])
        )


class FakeGeminiDevClient:
    """Gemini Developer API client — must NOT detect as Vertex."""

    def __init__(self) -> None:
        self.vertexai = False
        self.models = MagicMock()
        self.models.generate_content = MagicMock(
            return_value=MagicMock(usage_metadata=_FakeUsageMetadata())
        )


@pytest.fixture
def vertex_client() -> FakeVertexClient:
    return FakeVertexClient()


@pytest.fixture
def gemini_dev_client() -> FakeGeminiDevClient:
    return FakeGeminiDevClient()


class RecordingTransport:
    """Drop-in replacement for ObservationTransport that records calls
    synchronously (no thread pool). Used to assert payload shape."""

    def __init__(self) -> None:
        self.observations: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def post(self, observation: dict[str, Any]) -> None:
        with self._lock:
            self.observations.append(observation)

    def _get_token(self) -> str:
        """Stub used by applied-mode tests that call transport._get_token()."""
        return "fake-test-jwt"

    def get_token(self) -> str:
        return self._get_token()

    def close(self) -> None:  # pragma: no cover - parity with real transport
        pass


@pytest.fixture
def recording_transport() -> RecordingTransport:
    return RecordingTransport()


@pytest.fixture(autouse=True)
def _passthrough_sdk_layer(
    monkeypatch: pytest.MonkeyPatch,
    recording_transport: RecordingTransport,
) -> None:
    """Avoid live sdk-layer fetches in unit tests."""
    from tensorcost._runtime import _runtimes
    from tensorcost._sdk_layer import PassthroughSdkLayer

    _runtimes.clear()

    def fake_shared_runtime(_config):
        return recording_transport, PassthroughSdkLayer()

    monkeypatch.setattr(
        "tensorcost._wrap.get_shared_runtime",
        fake_shared_runtime,
    )
    monkeypatch.setattr(
        "tensorcost._runtime.get_shared_runtime",
        fake_shared_runtime,
    )
    monkeypatch.setattr(
        "tensorcost._wrap.warmup_shared_runtime",
        lambda _config: None,
    )


@pytest.fixture(autouse=True)
def _scrub_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure tests start with a clean env so config resolution is
    deterministic. Individual tests opt back in by setting env vars."""
    for var in (
        "TENSORCOST_API_KEY",
        "TENSORCOST_BASE_URL",
        "TENSORCOST_TENANT_ID",
        "TENSORCOST_PROXY_URL",
        "TENSORCOST_CUSTOMER",
        "TENSORCOST_FEATURE",
    ):
        monkeypatch.delenv(var, raising=False)
