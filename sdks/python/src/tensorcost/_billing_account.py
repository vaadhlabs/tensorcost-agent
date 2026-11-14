"""The account the provider billed a call under, as the provider names it.

TensorCost lets a provider's discount reach the counted figures only for a
provider-day where every metered call names its billing account, and one call
without it keeps the whole day at the metered price. Proxied calls are
labelled by the proxy; this is the same label for the calls the SDK makes
itself, sent on the observation as ``billing_account``.

The formats and the host rules are the backend's, copied from
``@tensorcost/contracts`` billing-account.ts (``billingAccountOfCall``), and
must stay identical to it: the two sides only meet if both write the same
string.

  anthropic     ``anthropic-organization-id`` response header
  openai        ``openai-organization`` response header (not the project)
  azure_openai  the resource: the endpoint host's first label, lower-cased
  vertex        the project in ``/projects/<id>/`` of the request URL
  bedrock       the 12-digit AWS account of the credentials that signed it

How the headers are read: the Anthropic and OpenAI clients send through an
``httpx.Client`` (``httpx2.Client`` from anthropic 2.x and openai 3.x, same
API), and it runs its response event hooks for every response
(error responses and streams too, before the body is read). One hook per
httpx client copies the headers into a slot the wrapper opens around its own
call, so the call path, its arguments and what it returns are untouched and
no request is added. Outside a wrapped call the slot is empty and the hook
does nothing.

Every reader here returns None rather than raise. A missing or unreadable
header costs the call its label, never the call.
"""

from __future__ import annotations

import contextvars
import re
from typing import Any, Mapping, Optional
from urllib.parse import unquote, urlsplit

_MAX_LEN = 128


def normalize_billing_account(v: Any) -> Optional[str]:
    """Trimmed, at most 128 characters, or None (the proxy's DTO caps it at 128)."""
    if not isinstance(v, str):
        return None
    s = v.strip()
    return s if 0 < len(s) <= _MAX_LEN else None


def _header(headers: Any, name: str) -> Optional[str]:
    if headers is None:
        return None
    # httpx.Headers and most mappings are case-insensitive through .get();
    # a plain dict isn't, so fall back to a scan.
    get = getattr(headers, "get", None)
    if callable(get):
        v = get(name)
        if v is not None:
            return normalize_billing_account(v)
    if isinstance(headers, Mapping):
        want = name.lower()
        for k, v in headers.items():
            if isinstance(k, str) and k.lower() == want:
                return normalize_billing_account(v[0] if isinstance(v, (list, tuple)) and v else v)
    return None


def _is_host(host: str, domain: str) -> bool:
    return host == domain or host.endswith("." + domain)


_VERTEX_PROJECT = re.compile(r"/projects/([^/]+)/")


def billing_account_of_call(url: Optional[str], headers: Any) -> Optional[str]:
    """The billing account a call went to, from its URL and the provider's
    response headers. None where neither names one, for a host we don't know
    (a gateway, the TensorCost proxy itself), and when reading raises."""
    if not isinstance(url, str) or not url:
        return None
    try:
        parts = urlsplit(url)
        if not parts.scheme or not parts.hostname:
            return None
        host = parts.hostname.lower()
        if _is_host(host, "anthropic.com"):
            return _header(headers, "anthropic-organization-id")
        if _is_host(host, "openai.com"):
            return _header(headers, "openai-organization")
        if (
            host.endswith(".openai.azure.com")
            or host.endswith(".cognitiveservices.azure.com")
            or host.endswith(".services.ai.azure.com")
        ):
            return normalize_billing_account(host.split(".")[0])
        if host == "aiplatform.googleapis.com" or host.endswith("-aiplatform.googleapis.com"):
            m = _VERTEX_PROJECT.search(parts.path)
            return normalize_billing_account(unquote(m.group(1))) if m else None
        return None
    except Exception:  # noqa: BLE001
        return None


def aws_account_id(v: Any) -> Optional[str]:
    """A 12-digit AWS account id, or None."""
    if not isinstance(v, str):
        return None
    s = v.strip()
    return s if len(s) == 12 and s.isdigit() else None


# ── Response headers for Anthropic / OpenAI ───────────────────────────────


class Capture:
    """Where the hook leaves the last response's headers for one wrapped call."""

    __slots__ = ("headers",)

    def __init__(self) -> None:
        self.headers: Any = None


_current: "contextvars.ContextVar[Optional[Capture]]" = contextvars.ContextVar(
    "tensorcost_billing_capture", default=None
)


def _response_hook(response: Any) -> None:
    try:
        cap = _current.get()
        if cap is not None:
            # Retries overwrite it, so the slot holds the response the call
            # ended on.
            cap.headers = response.headers
    except Exception:  # noqa: BLE001
        pass


setattr(_response_hook, "__tensorcost_billing__", True)


def _is_sync_httpx_client(http: Any) -> bool:
    """An ``httpx.Client`` or an ``httpx2.Client`` (anthropic 2.x and openai
    3.x send through httpx2, same hook API). Never an AsyncClient: it awaits
    its hooks, and a sync function there would break every call."""
    names = [(c.__module__.split(".")[0], c.__name__) for c in type(http).__mro__]
    if any(name == "AsyncClient" for _, name in names):
        return False
    return any(mod in ("httpx", "httpx2") and name == "Client" for mod, name in names)


def install_response_hook(client: Any) -> None:
    """Adds the header hook to the client's sync httpx / httpx2 Client, once."""
    try:
        http = getattr(client, "_client", None)
        if http is None or not _is_sync_httpx_client(http):
            return
        hooks = http.event_hooks
        response_hooks = hooks.get("response")
        if not isinstance(response_hooks, list):
            return
        if any(getattr(h, "__tensorcost_billing__", False) for h in response_hooks):
            return
        response_hooks.append(_response_hook)
    except Exception:  # noqa: BLE001
        pass


class capturing:
    """``with capturing(cap): call()`` lets the hook fill ``cap`` for this call only.

    Never swallows or changes the call's exception.
    """

    __slots__ = ("_cap", "_token")

    def __init__(self, cap: Capture) -> None:
        self._cap = cap
        self._token: Any = None

    def __enter__(self) -> Capture:
        try:
            self._token = _current.set(self._cap)
        except Exception:  # noqa: BLE001
            self._token = None
        return self._cap

    def __exit__(self, *exc: Any) -> bool:
        if self._token is not None:
            try:
                _current.reset(self._token)
            except Exception:  # noqa: BLE001
                pass
        return False


def client_base_url(client: Any) -> Optional[str]:
    """The client's base URL (anthropic, openai and AzureOpenAI expose ``base_url``)."""
    try:
        v = getattr(client, "base_url", None)
        return str(v) if v is not None else None
    except Exception:  # noqa: BLE001
        return None


def account_of_http_call(client: Any, cap: Capture, exc: Optional[BaseException] = None) -> Optional[str]:
    """The billing account of a call an Anthropic / OpenAI client made: from
    the URL (Azure), the headers the hook saw, or on an APIStatusError its
    ``response.headers``."""
    try:
        url = client_base_url(client)
        headers = cap.headers
        if headers is None and exc is not None:
            headers = getattr(getattr(exc, "response", None), "headers", None)
        return billing_account_of_call(url, headers)
    except Exception:  # noqa: BLE001
        return None


# ── Vertex ────────────────────────────────────────────────────────────────


def vertex_call_url(client: Any) -> Optional[str]:
    """The URL a google-genai Vertex client sends generate_content to,
    ``<base_url>/<api_version>/projects/<project>/locations/<location>/...``.
    None for the Gemini API, Vertex express mode (an API key, no project in
    the path) and anything that isn't that client."""
    try:
        api = getattr(client, "_api_client", None)
        if api is None or getattr(api, "vertexai", False) is not True:
            return None
        if getattr(api, "api_key", None):
            return None
        project = getattr(api, "project", None)
        if not isinstance(project, str) or not project:
            return None
        location = getattr(api, "location", None) or "global"
        http_options = getattr(api, "_http_options", None)
        base = getattr(http_options, "base_url", None)
        version = getattr(http_options, "api_version", None)
        if not isinstance(base, str) or not base:
            return None
        parts = [base.rstrip("/")]
        if isinstance(version, str) and version:
            parts.append(version)
        parts.append(f"projects/{project}/locations/{location}/publishers/google/models")
        return "/".join(parts)
    except Exception:  # noqa: BLE001
        return None


# ── Bedrock ───────────────────────────────────────────────────────────────


def bedrock_account_of_client(client: Any) -> Optional[str]:
    """The AWS account of the credentials a boto3 client signs with.

    botocore keeps it on the credentials when it knows it (AWS_ACCOUNT_ID,
    aws_account_id in the shared config, SSO, assumed roles). Read from the
    instance's own attributes: ``RefreshableCredentials.account_id`` is a
    property that may refresh, which can mean a network call. No STS call
    is made; plain access keys without an account leave the field out.
    """
    try:
        signer = getattr(client, "_request_signer", None)
        creds = getattr(signer, "_credentials", None)
        if creds is None:
            return None
        attrs = vars(creds)
        return aws_account_id(attrs.get("_account_id") or attrs.get("account_id"))
    except Exception:  # noqa: BLE001
        return None


def field(account: Optional[str]) -> dict:
    """``{"billing_account": ...}`` when there is one, ``{}`` otherwise."""
    return {"billing_account": account} if account else {}
