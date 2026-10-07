"""Configuration resolution for the SDK.

Resolution order:
  1. Explicit kwargs to ``wrap()``
  2. Environment variables (TENSORCOST_API_KEY, TENSORCOST_BASE_URL,
     TENSORCOST_TENANT_ID, TENSORCOST_PROXY_URL)
  3. Defaults (only for base_url)

If ``api_key`` is still missing after resolution, ``MissingConfigError``
is raised. This is the *one* case the SDK does not fail open: it means
the SDK was never configured at all.

If ``applied_mode=True`` but no proxy URL is available (from kwarg or
``TENSORCOST_PROXY_URL``), ``MissingConfigError`` is raised at wrap time
rather than on the first request — misconfigured applied-mode fails
loudly so the problem surfaces in development, not in production.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, Optional

from ._layer import (
    PROXY_HEADERS_TIMEOUT_S,
    ControlLayer,
    needs_proxy_url,
    resolve_max_layer,
)
from ._retry import RetryConfig, DEFAULT_RETRY_CONFIG
from ._telemetry import LifecycleEventCallback


DEFAULT_BASE_URL = "https://api.tensorcost.com"


class MissingConfigError(RuntimeError):
    """Raised when required configuration (api_key or proxy_url) is missing."""


@dataclass(frozen=True)
class ResolvedConfig:
    api_key: str
    base_url: str
    tenant_id: Optional[str]
    # Phase A4 batch 3 — resolved environment tag. ``None`` when the
    # customer did not configure one; the SDK omits the ``environment``
    # field from the envelope in that case so the backend's default
    # ('production') applies.
    environment: Optional[str]
    # Phase A4 batch 4 — resolved provider-connection identifier.
    # ``None`` when the customer did not configure one; the SDK omits
    # the ``connection_id`` field from the envelope in that case.
    connection_id: Optional[str]
    fail_open: bool
    max_layer: ControlLayer
    # Deprecated — derived from max_layer in (steer, route).
    applied_mode: bool
    # Resolved proxy base URL. Only meaningful when applied_mode is True.
    # Points to the TensorCost deployment (e.g. "https://api.my-instance.tensorcost.com").
    # The SDK appends /api/inference-proxy/v1/chat/completions to form the target URL.
    proxy_url: Optional[str]

    # Hardening fields — all have safe defaults in resolve_config().
    retry: RetryConfig = field(default_factory=lambda: DEFAULT_RETRY_CONFIG)
    timeout_s: float = 60.0
    headers_timeout_s: float = PROXY_HEADERS_TIMEOUT_S
    idle_timeout_s: float = 30.0
    on_lifecycle_event: Optional[LifecycleEventCallback] = None
    fail_open_enabled: bool = True
    provider_api_key: Optional[str] = None
    # Routing-policy scope headers (x-tc-application / x-tc-tags).
    application: Optional[str] = None
    tags: tuple[str, ...] = ()
    # A-04 chargeback — end-customer cohort and product feature.
    customer: Optional[str] = None
    feature: Optional[str] = None
    # Agent-run attribution — workflow rollup in TensorCost Agent runs.
    agent_id: Optional[str] = None
    workflow_id: Optional[str] = None
    team_id: Optional[str] = None


def _resolve_attribution_field(raw: Optional[str], field_name: str) -> Optional[str]:
    if raw is None:
        return None
    trimmed = raw.strip()
    if not trimmed:
        return None
    if len(trimmed) > 128:
        raise MissingConfigError(
            f"TensorCost {field_name} must be 128 characters or fewer."
        )
    return trimmed


def resolve_config(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    tenant_id: Optional[str] = None,
    environment: Optional[str] = None,
    connection_id: Optional[str] = None,
    fail_open: bool = True,
    max_layer: Optional[ControlLayer] = None,
    applied_mode: bool = False,
    proxy_url: Optional[str] = None,
    # Hardening options.
    retry: Optional[RetryConfig] = None,
    timeout_s: float = 60.0,
    headers_timeout_s: float = PROXY_HEADERS_TIMEOUT_S,
    idle_timeout_s: float = 30.0,
    on_lifecycle_event: Optional[LifecycleEventCallback] = None,
    fail_open_enabled: bool = True,
    provider_api_key: Optional[str] = None,
    application: Optional[str] = None,
    tags: Optional[list[str] | str] = None,
    customer: Optional[str] = None,
    feature: Optional[str] = None,
    agent_id: Optional[str] = None,
    workflow_id: Optional[str] = None,
    team_id: Optional[str] = None,
) -> ResolvedConfig:
    """Resolve final config values, env vars filling in any blanks."""
    final_api_key = api_key or os.environ.get("TENSORCOST_API_KEY")
    final_base_url = (
        base_url
        or os.environ.get("TENSORCOST_BASE_URL")
        or DEFAULT_BASE_URL
    )
    final_tenant_id = tenant_id or os.environ.get("TENSORCOST_TENANT_ID")

    # Phase A4 batch 3 — environment tag. Trimmed; capped at 64 chars
    # per project_environment_scoping.md. An empty / whitespace-only
    # value collapses to None so the SDK omits the field from the
    # envelope and the backend's column default ('production') applies.
    raw_env = environment or os.environ.get("TENSORCOST_ENVIRONMENT")
    final_environment: Optional[str] = None
    if raw_env is not None:
        trimmed = raw_env.strip()
        if len(trimmed) > 64:
            raise MissingConfigError(
                "TensorCost environment must be 64 characters or fewer."
            )
        final_environment = trimmed if trimmed else None

    # Phase A4 batch 4 — provider-connection id. Same trim+cap rules as
    # environment; collapsing blank to ``None`` so the SDK omits the
    # field when not configured.
    raw_conn = connection_id or os.environ.get("TENSORCOST_CONNECTION_ID")
    final_connection_id: Optional[str] = None
    if raw_conn is not None:
        trimmed = raw_conn.strip()
        if len(trimmed) > 64:
            raise MissingConfigError(
                "TensorCost connection_id must be 64 characters or fewer."
            )
        final_connection_id = trimmed if trimmed else None

    if not final_api_key:
        raise MissingConfigError(
            "TensorCost api_key not found. Pass api_key=... to wrap() or "
            "set the TENSORCOST_API_KEY environment variable."
        )

    # Resolve proxy_url. Only required when applied_mode is True.
    final_proxy_url = (
        proxy_url or os.environ.get("TENSORCOST_PROXY_URL") or None
    )
    if final_proxy_url is not None:
        final_proxy_url = final_proxy_url.rstrip("/")

    final_max_layer = resolve_max_layer(
        max_layer=max_layer,
        applied_mode=applied_mode,
    )
    final_applied_mode = needs_proxy_url(final_max_layer)

    if needs_proxy_url(final_max_layer) and not final_proxy_url:
        raise MissingConfigError(
            f"max_layer='{final_max_layer}' requires a proxy URL. Pass proxy_url=... "
            "to wrap() or set the TENSORCOST_PROXY_URL environment variable."
        )

    raw_app = application or os.environ.get("TC_APPLICATION") or os.environ.get(
        "TENSORCOST_APPLICATION"
    )
    final_application: Optional[str] = None
    if raw_app is not None:
        trimmed = raw_app.strip()
        final_application = trimmed if trimmed else None

    raw_tags = tags if tags is not None else (
        os.environ.get("TC_TAGS") or os.environ.get("TENSORCOST_TAGS")
    )
    final_tags: tuple[str, ...] = ()
    if isinstance(raw_tags, (list, tuple)):
        final_tags = tuple(str(x).strip() for x in raw_tags if str(x).strip())
    elif isinstance(raw_tags, str) and raw_tags.strip():
        final_tags = tuple(
            part.strip() for part in raw_tags.split(",") if part.strip()
        )

    final_customer = _resolve_attribution_field(
        customer or os.environ.get("TENSORCOST_CUSTOMER"),
        "customer",
    )
    final_feature = _resolve_attribution_field(
        feature or os.environ.get("TENSORCOST_FEATURE"),
        "feature",
    )
    final_agent_id = _resolve_attribution_field(
        agent_id or os.environ.get("TENSORCOST_AGENT_ID"),
        "agent_id",
    )
    final_workflow_id = _resolve_attribution_field(
        workflow_id or os.environ.get("TENSORCOST_WORKFLOW_ID"),
        "workflow_id",
    )
    raw_team = team_id or os.environ.get("TENSORCOST_TEAM_ID")
    final_team_id: Optional[str] = None
    if raw_team is not None:
        trimmed = raw_team.strip()
        final_team_id = trimmed if trimmed else None

    return ResolvedConfig(
        api_key=final_api_key,
        base_url=final_base_url.rstrip("/"),
        tenant_id=final_tenant_id,
        environment=final_environment,
        connection_id=final_connection_id,
        fail_open=fail_open,
        max_layer=final_max_layer,
        applied_mode=final_applied_mode,
        proxy_url=final_proxy_url,
        retry=retry if retry is not None else DEFAULT_RETRY_CONFIG,
        timeout_s=timeout_s,
        headers_timeout_s=headers_timeout_s,
        idle_timeout_s=idle_timeout_s,
        on_lifecycle_event=on_lifecycle_event,
        fail_open_enabled=fail_open_enabled,
        provider_api_key=provider_api_key,
        application=final_application,
        tags=final_tags,
        customer=final_customer,
        feature=final_feature,
        agent_id=final_agent_id,
        workflow_id=final_workflow_id,
        team_id=final_team_id,
    )
