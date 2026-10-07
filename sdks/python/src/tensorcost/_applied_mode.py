"""Applied-mode (Layer 2) HTTP routing helpers.

When applied_mode=True, chat/completions calls are forwarded through the
TensorCost inference-proxy rather than going straight to the upstream
provider. The proxy decides per-request whether to route or pass through;
the SDK is not involved in that decision.

The two required proxy headers:
  x-tc-provider-url   The original provider's base URL (e.g. "https://api.openai.com").
  x-tc-provider-auth  The auth header value the SDK would have sent to the provider
                      (e.g. "Bearer sk-..." for OpenAI, "sk-ant-..." for Anthropic).

The proxy is OpenAI-API-compatible — the request body is forwarded verbatim,
and the response body is returned verbatim to the caller. The customer's
parsing code sees no difference.

This module is an internal helper; nothing here is part of the public API.
"""

from __future__ import annotations

from typing import Any

import httpx


# Proxy path appended to proxy_url when routing chat/completions.
PROXY_CHAT_PATH = "/api/inference-proxy/v1/chat/completions"


def build_proxy_headers(
    provider_url: str,
    provider_auth: str,
    correlation_id: str | None = None,
    environment: str | None = None,
    customer: str | None = None,
    feature: str | None = None,
    agent_id: str | None = None,
    workflow_id: str | None = None,
) -> dict[str, str]:
    """Return the custom headers the proxy controller requires.

    Only x-tc-provider-url and x-tc-provider-auth are mandatory; the
    correlation and environment headers are added when present so the
    proxy can tie the request to the observation record and use the
    environment for policy matching.
    """
    headers: dict[str, str] = {
        "x-tc-provider-url": provider_url,
        "x-tc-provider-auth": provider_auth,
    }
    if correlation_id:
        headers["x-tc-correlation-id"] = correlation_id
    if environment:
        headers["x-tc-environment"] = environment
    if customer:
        headers["x-tc-customer"] = customer
    if feature:
        headers["x-tc-feature"] = feature
    if agent_id:
        headers["x-tc-agent-id"] = agent_id
    if workflow_id:
        headers["x-tc-workflow-id"] = workflow_id
    return headers


def openai_provider_auth(client: Any) -> str:
    """Extract the Authorization header value for an OpenAI-style client.

    OpenAI uses "Bearer <api_key>" as the Authorization header.  We
    read the key from the client object rather than re-capturing a
    closure variable so that key rotation on the client instance is
    respected automatically.
    """
    # openai.OpenAI stores the key as client.api_key.
    api_key = getattr(client, "api_key", None)
    if api_key:
        return f"Bearer {api_key}"
    # Fallback: return empty string; the proxy will 400 on an absent auth.
    return ""


def openai_provider_base_url(client: Any) -> str:
    """Return the upstream base URL for an OpenAI-style client.

    openai.OpenAI stores it as client.base_url (an httpx.URL).  We
    coerce to str and strip trailing slash so the proxy gets a clean URL.
    """
    base = getattr(client, "base_url", None)
    if base is not None:
        url = str(base).rstrip("/")
        # openai-python sets base_url to "https://api.openai.com/v1" by
        # default. The proxy expects the provider's plain base (no path),
        # because the proxy appends "/v1/chat/completions" itself.
        # Strip any trailing path component that looks like an API version
        # (i.e. ends with /v1, /v2, etc.) so x-tc-provider-url is always
        # the scheme+host portion. The proxy will receive
        # "https://api.openai.com" and know what path to append.
        import re
        url = re.sub(r"/v\d+$", "", url)
        return url
    return "https://api.openai.com"


def anthropic_provider_auth(client: Any) -> str:
    """Extract the x-api-key value for an Anthropic client.

    Anthropic's SDK uses "x-api-key: <key>" rather than Bearer.
    The proxy is responsible for forwarding this in the right header;
    the SDK just surfaces the raw key.  We prefix with the literal
    header name so the proxy can reconstruct the Authorization shape.
    """
    api_key = getattr(client, "api_key", None)
    if api_key:
        # Encode as "x-api-key <key>" so the proxy can distinguish this
        # from OpenAI-style Bearer tokens and set the right header on the
        # upstream call.
        return f"x-api-key {api_key}"
    return ""


def anthropic_provider_base_url(client: Any) -> str:
    """Return the upstream base URL for an Anthropic client."""
    base = getattr(client, "base_url", None)
    if base is not None:
        return str(base).rstrip("/")
    return "https://api.anthropic.com"


def route_via_proxy(
    proxy_url: str,
    bearer_token: str,
    provider_url: str,
    provider_auth: str,
    body: dict[str, Any],
    timeout_s: float = 30.0,
    correlation_id: str | None = None,
    environment: str | None = None,
) -> httpx.Response:
    """POST *body* to the inference-proxy and return the raw httpx.Response.

    This is a synchronous blocking call — it runs on the customer's
    calling thread, same as the original provider call would have.

    The caller is responsible for forwarding the response status and body
    back to the customer's code.
    """
    url = proxy_url.rstrip("/") + PROXY_CHAT_PATH
    headers = {
        "Authorization": f"Bearer {bearer_token}",
        "Content-Type": "application/json",
        **build_proxy_headers(
            provider_url=provider_url,
            provider_auth=provider_auth,
            correlation_id=correlation_id,
            environment=environment,
        ),
    }
    with httpx.Client(timeout=timeout_s) as client:
        return client.post(url, json=body, headers=headers)
