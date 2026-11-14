"""billing_account on SDK observations (1.7.0).

Per provider: the header (or URL, or credentials) is there -> the field is
sent; it isn't -> no field; reading it raises -> no field and the caller
gets exactly what the provider returned.

The Anthropic / OpenAI cases run a real httpx.Client over a MockTransport,
because the headers come off httpx's response hook; the fakes only stand in
for the provider SDK's resource classes.
"""

from __future__ import annotations

from typing import Any, Callable
from unittest.mock import MagicMock, patch

import httpx
import pytest

from tensorcost import _billing_account as ba
from tensorcost._providers import _anthropic, _bedrock, _openai, _vertex

MSG = {"id": "msg_1", "usage": {"input_tokens": 3, "output_tokens": 4}}
CHAT = {"id": "chatcmpl-1", "usage": {"prompt_tokens": 1, "completion_tokens": 2}}


class _APIStatusError(Exception):
    """Shaped like anthropic / openai APIStatusError: keeps the httpx response."""

    def __init__(self, response: httpx.Response) -> None:
        super().__init__(f"status {response.status_code}")
        self.response = response


def _http(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


class _Resource:
    def __init__(self, client: "_FakeClient", path: str) -> None:
        self._c, self._path = client, path

    def create(self, **kwargs: Any) -> Any:
        r = self._c._client.post(self._c.base_url + self._path, json=kwargs)
        if r.status_code >= 400:
            raise _APIStatusError(r)
        return r.json()


class _FakeClient:
    """anthropic.Anthropic / openai.OpenAI as far as the wrapper sees them:
    ``base_url``, ``_client`` (the httpx.Client) and the resource methods."""

    def __init__(self, base_url: str, http: httpx.Client) -> None:
        self.base_url = base_url
        self.api_key = "sk-test"
        self._client = http
        self.messages = _Resource(self, "/v1/messages")
        self.chat = MagicMock()
        self.chat.completions = _Resource(self, "/chat/completions")
        self.completions = None


def _anthropic_client(headers: dict, status: int = 200, base_url: str = "https://api.anthropic.com") -> _FakeClient:
    return _FakeClient(base_url, _http(lambda req: httpx.Response(status, json=MSG, headers=headers)))


def _openai_client(headers: dict, status: int = 200, base_url: str = "https://api.openai.com/v1") -> _FakeClient:
    return _FakeClient(base_url, _http(lambda req: httpx.Response(status, json=CHAT, headers=headers)))


def _install_anthropic(c: Any, t: Any) -> None:
    _anthropic.install(c, t, tenant_id=None, environment=None, connection_id=None)


def _install_openai(c: Any, t: Any) -> None:
    _openai.install(c, t, tenant_id=None, environment=None, connection_id=None)


# ── The rules ─────────────────────────────────────────────────────────────


def test_rules_match_the_backend_contract():
    h = httpx.Headers
    assert ba.billing_account_of_call("https://api.anthropic.com", h({"anthropic-organization-id": " org-a "})) == "org-a"
    assert (
        ba.billing_account_of_call(
            "https://api.openai.com/v1", h({"openai-organization": "org-o", "openai-project": "proj_1"})
        )
        == "org-o"
    )
    assert ba.billing_account_of_call("https://api.openai.com/v1", {"OpenAI-Organization": "org-rec"}) == "org-rec"
    assert ba.billing_account_of_call("https://MyRes.openai.azure.com/openai/", None) == "myres"
    assert ba.billing_account_of_call("https://r2.cognitiveservices.azure.com/", None) == "r2"
    assert ba.billing_account_of_call("https://r3.services.ai.azure.com/models", None) == "r3"
    assert (
        ba.billing_account_of_call(
            "https://us-central1-aiplatform.googleapis.com/v1/projects/my-proj/locations/us-central1/x", None
        )
        == "my-proj"
    )
    # A host we don't know names nothing, even with the header.
    assert ba.billing_account_of_call("https://gateway.corp/v1", h({"openai-organization": "org-o"})) is None
    assert ba.billing_account_of_call("not a url", None) is None
    assert ba.billing_account_of_call(None, None) is None
    assert ba.normalize_billing_account("x" * 129) is None
    assert ba.normalize_billing_account("   ") is None


# ── Anthropic ─────────────────────────────────────────────────────────────


def test_anthropic_sends_the_organization_header(recording_transport):
    c = _anthropic_client({"anthropic-organization-id": "org-anth"})
    _install_anthropic(c, recording_transport)
    assert c.messages.create(model="claude-sonnet-4-5", max_tokens=8, messages=[]) == MSG
    assert recording_transport.observations[0]["billing_account"] == "org-anth"


def test_anthropic_no_header_no_field(recording_transport):
    c = _anthropic_client({"request-id": "req_1"})
    _install_anthropic(c, recording_transport)
    c.messages.create(model="claude-sonnet-4-5", max_tokens=8, messages=[])
    assert "billing_account" not in recording_transport.observations[0]


def test_anthropic_error_response_is_labelled(recording_transport):
    c = _anthropic_client({"anthropic-organization-id": "org-anth"}, status=529)
    _install_anthropic(c, recording_transport)
    with pytest.raises(_APIStatusError):
        c.messages.create(model="claude-sonnet-4-5", max_tokens=8, messages=[])
    obs = recording_transport.observations[0]
    assert obs["status"] == "error"
    assert obs["billing_account"] == "org-anth"


def test_anthropic_gateway_base_url_no_field(recording_transport):
    c = _anthropic_client({"anthropic-organization-id": "org-anth"}, base_url="https://llm-gateway.corp")
    _install_anthropic(c, recording_transport)
    c.messages.create(model="claude-sonnet-4-5", max_tokens=8, messages=[])
    assert "billing_account" not in recording_transport.observations[0]


class _ThrowingHeaders:
    def get(self, name: str) -> Any:
        raise RuntimeError("header read blew up")


class _HookedClient(httpx.Client):
    """Runs the response hooks itself with a response whose headers raise."""

    def post(self, url: Any, **kwargs: Any) -> Any:  # type: ignore[override]
        resp = MagicMock()
        resp.headers = _ThrowingHeaders()
        resp.status_code = 200
        resp.json.return_value = MSG
        for hook in self.event_hooks["response"]:
            hook(resp)
        return resp


def test_anthropic_header_read_that_raises_costs_the_field_not_the_call(recording_transport):
    c = _FakeClient("https://api.anthropic.com", _HookedClient())
    _install_anthropic(c, recording_transport)
    assert c.messages.create(model="claude-sonnet-4-5", max_tokens=8, messages=[]) == MSG
    obs = recording_transport.observations[0]
    assert obs["status"] == "success"
    assert "billing_account" not in obs


def test_hook_is_added_once_and_is_inert_outside_a_wrapped_call(recording_transport):
    c = _anthropic_client({"anthropic-organization-id": "org-anth"})
    _install_anthropic(c, recording_transport)
    _install_anthropic(c, recording_transport)
    hooks = [h for h in c._client.event_hooks["response"] if getattr(h, "__tensorcost_billing__", False)]
    assert len(hooks) == 1
    # A request through the same httpx client but outside the wrapper.
    assert c._client.get("https://api.anthropic.com/v1/models").status_code == 200


def test_hook_is_never_added_to_an_async_client(recording_transport):
    c = _FakeClient("https://api.anthropic.com", httpx.AsyncClient())  # type: ignore[arg-type]
    ba.install_response_hook(c)
    assert c._client.event_hooks["response"] == []


# ── OpenAI / Azure OpenAI ─────────────────────────────────────────────────


def test_openai_sends_the_organization_not_the_project(recording_transport):
    c = _openai_client({"openai-organization": "org-oai", "openai-project": "proj_1"})
    _install_openai(c, recording_transport)
    assert c.chat.completions.create(model="gpt-4o", messages=[]) == CHAT
    assert recording_transport.observations[0]["billing_account"] == "org-oai"


def test_openai_no_header_no_field(recording_transport):
    c = _openai_client({})
    _install_openai(c, recording_transport)
    c.chat.completions.create(model="gpt-4o", messages=[])
    assert "billing_account" not in recording_transport.observations[0]


def test_openai_header_read_that_raises_costs_the_field_not_the_call(recording_transport):
    http = _HookedClient()
    c = _FakeClient("https://api.openai.com/v1", http)
    _install_openai(c, recording_transport)
    assert c.chat.completions.create(model="gpt-4o", messages=[]) == MSG
    assert "billing_account" not in recording_transport.observations[0]


def test_openai_429_is_labelled(recording_transport):
    c = _openai_client({"openai-organization": "org-oai"}, status=429)
    _install_openai(c, recording_transport)
    with pytest.raises(_APIStatusError):
        c.chat.completions.create(model="gpt-4o", messages=[])
    assert recording_transport.observations[0]["billing_account"] == "org-oai"


def test_openai_call_the_proxy_made_carries_no_field(recording_transport):
    c = _openai_client({"openai-organization": "org-oai"})
    _openai.install(
        c, recording_transport, tenant_id="t-1", environment=None, connection_id=None,
        applied_mode=True, proxy_url="https://proxy.tensorcost.com",
    )
    with patch("tensorcost._providers._openai.proxy_request") as mock_proxy:
        mock_proxy.return_value = (CHAT, None)
        assert c.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}]) == CHAT
    mock_proxy.assert_called_once()
    assert len(recording_transport.observations) == 1
    assert "billing_account" not in recording_transport.observations[0]


def test_azure_names_the_resource_from_the_host(recording_transport):
    c = _openai_client({}, base_url="https://Contoso-EastUS.openai.azure.com/openai")
    _install_openai(c, recording_transport)
    c.chat.completions.create(model="gpt-4o", messages=[])
    obs = recording_transport.observations[0]
    assert obs["provider"] == "azure_openai"
    assert obs["billing_account"] == "contoso-eastus"


def test_azure_base_url_that_raises_costs_the_field_not_the_call(recording_transport):
    class Raising(_FakeClient):
        # Readable at install (provider detection reads it there), raises after.
        armed = False

        @property
        def base_url(self) -> str:  # type: ignore[override]
            if self.armed:
                raise RuntimeError("no")
            return "https://x.openai.azure.com/openai"

        @base_url.setter
        def base_url(self, v: str) -> None:
            pass

    c = Raising("https://x.openai.azure.com/openai", _http(lambda req: httpx.Response(200, json=CHAT)))
    c.chat.completions = MagicMock()
    c.chat.completions.create = MagicMock(return_value=CHAT)
    _install_openai(c, recording_transport)
    c.armed = True
    assert c.chat.completions.create(model="gpt-4o", messages=[]) == CHAT
    assert "billing_account" not in recording_transport.observations[0]


# ── Vertex ────────────────────────────────────────────────────────────────


def _genai(vertexai: bool, project: Any, api_key: Any = None, base_url: Any = "https://us-central1-aiplatform.googleapis.com/") -> Any:
    """google.genai.Client as far as the wrapper reads it."""
    client = MagicMock()
    client._api_client = MagicMock()
    client._api_client.vertexai = vertexai
    client._api_client.project = project
    client._api_client.location = "us-central1"
    client._api_client.api_key = api_key
    client._api_client._http_options = MagicMock(base_url=base_url, api_version="v1beta1")
    client.models.generate_content = MagicMock(return_value=MagicMock())
    client.models.generate_content_stream = MagicMock(return_value=iter([]))
    return client


def test_vertex_sends_the_project_from_the_url(recording_transport):
    c = _genai(True, "my-proj")
    assert ba.vertex_call_url(c) == (
        "https://us-central1-aiplatform.googleapis.com/v1beta1/projects/my-proj/locations/us-central1/publishers/google/models"
    )
    sentinel = c.models.generate_content.return_value
    _vertex.install(c, recording_transport, tenant_id=None, environment=None, connection_id=None)
    assert c.models.generate_content(model="gemini-2.0-flash", contents="x") is sentinel
    c.models.generate_content_stream(model="gemini-2.0-flash", contents="x")
    assert [o["billing_account"] for o in recording_transport.observations] == ["my-proj", "my-proj"]


def test_vertex_express_mode_and_gemini_api_no_field(recording_transport):
    for c in (_genai(True, None, api_key="k"), _genai(False, None, api_key="k")):
        recording_transport.observations.clear()
        _vertex.install(c, recording_transport, tenant_id=None, environment=None, connection_id=None)
        c.models.generate_content(model="gemini-2.0-flash", contents="x")
        assert "billing_account" not in recording_transport.observations[0]


def test_vertex_url_read_that_raises_costs_the_field_not_the_call(recording_transport):
    c = _genai(True, "my-proj")
    type(c._api_client).project = property(lambda self: (_ for _ in ()).throw(RuntimeError("no")))
    sentinel = c.models.generate_content.return_value
    _vertex.install(c, recording_transport, tenant_id=None, environment=None, connection_id=None)
    assert c.models.generate_content(model="gemini-2.0-flash", contents="x") is sentinel
    obs = recording_transport.observations[0]
    assert obs["status"] == "success"
    assert "billing_account" not in obs


# ── Bedrock ───────────────────────────────────────────────────────────────


def _bedrock_client(creds: Any) -> Any:
    client = MagicMock()
    client._request_signer = MagicMock()
    client._request_signer._credentials = creds
    client.converse = MagicMock(return_value={"usage": {"inputTokens": 1, "outputTokens": 2}})
    for attr in ("invoke_model", "invoke_model_with_response_stream", "converse_stream"):
        setattr(client, attr, None)
    return client


def test_bedrock_sends_the_account_the_credentials_carry(recording_transport):
    Credentials = getattr(pytest.importorskip("botocore.credentials"), "Credentials")

    c = _bedrock_client(Credentials("AKIA", "secret", account_id="123456789012"))
    sentinel = c.converse.return_value
    _bedrock.install(c, recording_transport, tenant_id=None, environment=None, connection_id=None)
    assert c.converse(modelId="anthropic.claude-3") is sentinel
    assert recording_transport.observations[0]["billing_account"] == "123456789012"


def test_bedrock_refreshable_credentials_are_read_without_a_refresh(recording_transport):
    RefreshableCredentials = getattr(pytest.importorskip("botocore.credentials"), "RefreshableCredentials")

    def refresh() -> dict:
        raise AssertionError("reading the account must not refresh credentials")

    creds = RefreshableCredentials(
        "AKIA", "secret", "tok", expiry_time=MagicMock(), refresh_using=refresh, method="test",
        account_id="210987654321",
    )
    c = _bedrock_client(creds)
    _bedrock.install(c, recording_transport, tenant_id=None, environment=None, connection_id=None)
    c.converse(modelId="m")
    assert recording_transport.observations[0]["billing_account"] == "210987654321"


def test_bedrock_no_account_or_a_bad_one_no_field(recording_transport):
    Credentials = getattr(pytest.importorskip("botocore.credentials"), "Credentials")

    for creds in (Credentials("AKIA", "secret"), Credentials("AKIA", "secret", account_id="1234"), None):
        recording_transport.observations.clear()
        c = _bedrock_client(creds)
        _bedrock.install(c, recording_transport, tenant_id=None, environment=None, connection_id=None)
        c.converse(modelId="m")
        assert "billing_account" not in recording_transport.observations[0]


def test_bedrock_credentials_read_that_raises_costs_the_field_not_the_call(recording_transport):
    class Raising:
        @property
        def __dict__(self) -> Any:  # type: ignore[override]
            raise RuntimeError("no")

    c = _bedrock_client(Raising())
    sentinel = c.converse.return_value
    _bedrock.install(c, recording_transport, tenant_id=None, environment=None, connection_id=None)
    assert c.converse(modelId="m") is sentinel
    obs = recording_transport.observations[0]
    assert obs["status"] == "success"
    assert "billing_account" not in obs
