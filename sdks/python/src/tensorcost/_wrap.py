"""The :func:`wrap` entry point — the entire public API of this SDK."""

from __future__ import annotations

from typing import Literal, Optional, TypeVar

from ._attribution import attach_with_meta
from ._circuit import CircuitBreaker
from ._config import MissingConfigError, resolve_config
from ._layer import PROXY_HEADERS_TIMEOUT_S, ControlLayer, needs_proxy_url
from ._providers import _anthropic, _bedrock, _openai, _vertex
from ._providers._detect import detect_provider
from ._retry import RetryConfig
from ._runtime import get_shared_runtime, warmup_shared_runtime
from ._telemetry import LifecycleEventCallback

T = TypeVar("T")


def _steer_route_error(provider: str, max_layer: str) -> MissingConfigError:
    return MissingConfigError(
        f"TensorCost max_layer '{max_layer}' is not supported for {provider} clients. "
        "Bedrock and Vertex use provider-native signing that cannot be intercepted at "
        "the proxy layer. Use max_layer 'govern' or lower for telemetry, or wrap an "
        "OpenAI / Anthropic client for steer/route.",
    )


def wrap(
    client: T,
    *,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    tenant_id: Optional[str] = None,
    environment: Optional[str] = None,
    connection_id: Optional[str] = None,
    application: Optional[str] = None,
    tags: Optional[list[str] | str] = None,
    customer: Optional[str] = None,
    feature: Optional[str] = None,
    agent_id: Optional[str] = None,
    workflow_id: Optional[str] = None,
    team_id: Optional[str] = None,
    fail_open: bool = True,
    max_layer: Optional[ControlLayer] = None,
    applied_mode: bool = False,
    proxy_url: Optional[str] = None,
    retry: Optional[RetryConfig] = None,
    timeout_s: float = 60.0,
    headers_timeout_s: float = PROXY_HEADERS_TIMEOUT_S,
    idle_timeout_s: float = 30.0,
    on_lifecycle_event: Optional[LifecycleEventCallback] = None,
    fail_open_enabled: bool = True,
    provider_api_key: Optional[str] = None,
) -> T:
    """Wrap an OpenAI, Anthropic, Bedrock-runtime, or Vertex AI client."""
    config = resolve_config(
        api_key=api_key,
        base_url=base_url,
        tenant_id=tenant_id,
        environment=environment,
        connection_id=connection_id,
        application=application,
        tags=tags,
        customer=customer,
        feature=feature,
        agent_id=agent_id,
        workflow_id=workflow_id,
        team_id=team_id,
        fail_open=fail_open,
        max_layer=max_layer,
        applied_mode=applied_mode,
        proxy_url=proxy_url,
        retry=retry,
        timeout_s=timeout_s,
        headers_timeout_s=headers_timeout_s,
        idle_timeout_s=idle_timeout_s,
        on_lifecycle_event=on_lifecycle_event,
        fail_open_enabled=fail_open_enabled,
        provider_api_key=provider_api_key,
    )

    if config.max_layer == "off":
        return client

    provider = detect_provider(client)

    if needs_proxy_url(config.max_layer) and provider in ("bedrock", "vertex"):
        raise _steer_route_error(provider, config.max_layer)

    transport, sdk_layer = get_shared_runtime(config)
    warmup_shared_runtime(config)

    circuit = CircuitBreaker()

    common = dict(
        transport=transport,
        tenant_id=config.tenant_id,
        environment=config.environment,
        connection_id=config.connection_id,
        application=config.application,
        tags=list(config.tags),
        max_layer=config.max_layer,
        base_url=config.base_url,
        sdk_layer=sdk_layer,
        proxy_url=config.proxy_url,
        retry=config.retry,
        timeout_s=config.timeout_s,
        headers_timeout_s=config.headers_timeout_s,
        on_lifecycle_event=config.on_lifecycle_event,
        fail_open_enabled=config.fail_open_enabled,
        circuit=circuit,
        customer=config.customer,
        feature=config.feature,
        agent_id=config.agent_id,
        workflow_id=config.workflow_id,
        team_id=config.team_id,
    )

    if provider == "openai":
        _openai.install(client, **common)
        attach_with_meta(client)
    elif provider == "anthropic":
        _anthropic.install(client, **common)
        attach_with_meta(client)
    elif provider == "bedrock":
        _bedrock.install(
            client,
            transport,
            config.tenant_id,
            config.environment,
            config.connection_id,
            applied_mode=config.applied_mode,
            customer=config.customer,
            feature=config.feature,
            agent_id=config.agent_id,
            workflow_id=config.workflow_id,
        )
    elif provider == "vertex":
        _vertex.install(
            client,
            transport,
            config.tenant_id,
            config.environment,
            config.connection_id,
            applied_mode=config.applied_mode,
            customer=config.customer,
            feature=config.feature,
            agent_id=config.agent_id,
            workflow_id=config.workflow_id,
        )

    setattr(client, "_tensorcost_transport", transport)
    setattr(client, "_tensorcost_circuit", circuit)
    return client
