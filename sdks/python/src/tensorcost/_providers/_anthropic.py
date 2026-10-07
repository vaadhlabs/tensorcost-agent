"""Anthropic client wrapping.

Intercepts ``client.messages.create``. Same shape and guarantees as the
OpenAI wrapper: metadata-only capture, fail-open on observation errors.

Applied-mode hardening (v0.4.0): messages.create calls route through the
proxy-client module, which handles retries, timeouts, typed errors,
telemetry hooks, and the fail-open circuit breaker.
"""

from __future__ import annotations

import functools
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from .._admit_client import admit_request
from .._compliance_guard import (
    assert_team_scope,
    compliance_stamp_from,
    enforce_in_process_dlp,
)
from .._dlp import extract_anthropic_messages_text
from .._errors import (
    TensorCostComplianceDeniedError,
    TensorCostComplianceTeamMismatchError,
    TensorCostProviderError,
    TensorCostProxyError,
)
from .._attribution import resolve_scope_field
from .._applied_mode import (
    anthropic_provider_auth,
    anthropic_provider_base_url,
)
from .._circuit import CircuitBreaker
from .._decision import decision_metadata
from .._layer import (
    PROXY_HEADERS_TIMEOUT_S,
    ControlLayer,
    code_layer_for_install,
    needs_proxy_url,
)
from .._models import Observation
from .._proxy_client import proxy_request
from .._retry import DEFAULT_RETRY_CONFIG, RetryConfig
from .._sdk_layer import PassthroughSdkLayer, SdkLayerClient
from .._telemetry import LifecycleEventCallback, OnFallbackEvent, emit_event
from .._transport import ObservationTransport
from .._version import SDK_VERSION as _SDK_VERSION
_log = logging.getLogger("tensorcost")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _extract_usage(response: Any) -> tuple[int | None, int | None]:
    usage = getattr(response, "usage", None)
    if usage is None and isinstance(response, dict):
        usage = response.get("usage")
    if usage is None:
        return None, None
    if isinstance(usage, dict):
        return usage.get("input_tokens"), usage.get("output_tokens")
    return (
        getattr(usage, "input_tokens", None),
        getattr(usage, "output_tokens", None),
    )


def _post_obs(
    transport: ObservationTransport,
    model: str,
    tenant_id: str | None,
    environment: str | None,
    connection_id: str | None,
    correlation_id: str,
    request_at: str,
    input_tokens: int | None,
    output_tokens: int | None,
    status: str,
    error_message: str | None,
    customer: str | None = None,
    feature: str | None = None,
    agent_id: str | None = None,
    workflow_id: str | None = None,
) -> None:
    obs: dict = {
        "sdk_version": _SDK_VERSION,
        "provider": "anthropic",
        "model": model,
        "operation": "messages",
        "request_at": request_at,
        "response_at": _now_iso(),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd_cents": None,
        "status": status,
        "error_message": error_message,
        "correlation_id": correlation_id,
        # tenant_id comes from the short-lived JWT — ObservationDto
        # forbidNonWhitelisted rejects it on the wire.
        **({"environment": environment} if environment is not None else {}),
        **({"connection_id": connection_id} if connection_id is not None else {}),
        **({"customer": customer} if customer is not None else {}),
        **({"feature": feature} if feature is not None else {}),
        **({"agent_id": agent_id} if agent_id is not None else {}),
        **({"workflow_id": workflow_id} if workflow_id is not None else {}),
    }
    try:
        transport.post(obs)
    except Exception:  # noqa: BLE001
        pass


def _wrap_method(
    bound_method: Any,
    *,
    transport: ObservationTransport,
    tenant_id: str | None,
    environment: str | None,
    connection_id: str | None,
    application: Optional[str] = None,
    tags: Optional[list[str]] = None,
    max_layer: ControlLayer = "observe",
    applied_mode: bool = False,
    proxy_url: Optional[str] = None,
    base_url: str = "https://api.tensorcost.com",
    sdk_layer: Optional[SdkLayerClient] = None,
    client_ref: Optional[Any] = None,
    retry: RetryConfig = DEFAULT_RETRY_CONFIG,
    timeout_s: float = 60.0,
    headers_timeout_s: float = PROXY_HEADERS_TIMEOUT_S,
    on_lifecycle_event: Optional[LifecycleEventCallback] = None,
    fail_open_enabled: bool = True,
    circuit: Optional[CircuitBreaker] = None,
    customer: Optional[str] = None,
    feature: Optional[str] = None,
    agent_id: Optional[str] = None,
    workflow_id: Optional[str] = None,
    team_id: Optional[str] = None,
) -> Any:
    @functools.wraps(bound_method)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        eff_agent_id = resolve_scope_field(client_ref, "agent_id", agent_id)
        eff_workflow_id = resolve_scope_field(client_ref, "workflow_id", workflow_id)
        eff_customer = resolve_scope_field(client_ref, "customer", customer)
        eff_feature = resolve_scope_field(client_ref, "feature", feature)
        request_at = _now_iso()
        model = kwargs.get("model") or (args[0] if args else "unknown")
        correlation_id = str(uuid.uuid4())

        layer_client = sdk_layer or PassthroughSdkLayer()
        effective_layer = layer_client.effective_layer(max_layer)

        compliance_stamp: dict[str, Any] = {}
        try:
            assert_team_scope(client_team_id=team_id)
            compliance_snap = None
            if isinstance(layer_client, SdkLayerClient):
                compliance_snap = layer_client.get_compliance_snapshot(effective_layer)
            first_arg = args[0] if args else kwargs
            dlp_result = enforce_in_process_dlp(
                extract_anthropic_messages_text(first_arg),
                compliance_snap,
            )
            compliance_stamp = compliance_stamp_from(compliance_snap, dlp_result)
        except (TensorCostComplianceDeniedError, TensorCostComplianceTeamMismatchError):
            raise
        except Exception:
            pass

        if effective_layer == "govern":
            try:
                bearer_token = transport.get_token()
            except Exception:
                bearer_token = None
            if bearer_token is None:
                allowed, decision_header = True, None
            else:
                allowed, decision_header = admit_request(
                    base_url=base_url,
                    bearer_token=bearer_token,
                    max_layer=effective_layer,
                    metadata={
                        "provider": "anthropic",
                        "model": str(model),
                        "operation": "messages",
                        "correlation_id": correlation_id,
                        **({"environment": environment} if environment else {}),
                        **({"connection_id": connection_id} if connection_id else {}),
                        **({"agent_id": eff_agent_id} if eff_agent_id else {}),
                        **({"workflow_id": eff_workflow_id} if eff_workflow_id else {}),
                        **({"customer": eff_customer} if eff_customer else {}),
                        **({"feature": eff_feature} if eff_feature else {}),
                        **({"application": application} if application else {}),
                        **({"tags": ",".join(tags)} if tags else {}),
                        **({"team_id": team_id} if team_id else {}),
                    },
                    transport=transport,
                    fail_open=True,
                )
            if not allowed:
                raise TensorCostProviderError(
                    "tensorcost: request refused at govern layer",
                    status=403,
                    attempt=1,
                )

        if needs_proxy_url(effective_layer) and proxy_url:
            provider_url = anthropic_provider_base_url(client_ref) if client_ref else "https://api.anthropic.com"
            provider_auth = anthropic_provider_auth(client_ref) if client_ref else ""

            body: dict[str, Any] = dict(kwargs)
            body["model"] = str(model)

            is_streaming = body.get("stream") is True

            proxy_failed = False
            proxy_error: Optional[Exception] = None
            proxy_result: Optional[Any] = None

            circuit_was_open = circuit is not None and circuit.should_bypass()
            should_try_proxy = not circuit_was_open
            if circuit_was_open and circuit is not None:
                should_try_proxy = circuit.should_probe()

            if should_try_proxy:
                try:
                    bearer_token = transport.get_token()
                    if is_streaming:
                        from .._proxy_client import proxy_request_stream

                        proxy_result, decision_header = proxy_request_stream(
                            proxy_url=proxy_url,
                            bearer_token=bearer_token,
                            provider_url=provider_url,
                            provider_auth=provider_auth,
                            body=body,
                            model=str(model),
                            provider="anthropic",
                            operation="messages",
                            correlation_id=correlation_id,
                            environment=environment,
                            application=application,
                            tags=tags,
                            customer=eff_customer,
                            feature=eff_feature,
                            agent_id=eff_agent_id,
                            workflow_id=eff_workflow_id,
                            retry=retry,
                            timeout_s=timeout_s,
                            headers_timeout_s=headers_timeout_s,
                            on_lifecycle_event=on_lifecycle_event,
                            circuit=circuit,
                            max_layer=effective_layer,
                            fail_open_fast=fail_open_enabled,
                        )
                    else:
                        proxy_result, decision_header = proxy_request(
                            proxy_url=proxy_url,
                            bearer_token=bearer_token,
                            provider_url=provider_url,
                            provider_auth=provider_auth,
                            body=body,
                            model=str(model),
                            provider="anthropic",
                            operation="messages",
                            correlation_id=correlation_id,
                            environment=environment,
                            application=application,
                            tags=tags,
                            customer=eff_customer,
                            feature=eff_feature,
                            agent_id=eff_agent_id,
                            workflow_id=eff_workflow_id,
                            retry=retry,
                            timeout_s=timeout_s,
                            headers_timeout_s=headers_timeout_s,
                            on_lifecycle_event=on_lifecycle_event,
                            circuit=circuit,
                            max_layer=effective_layer,
                            fail_open_fast=fail_open_enabled,
                        )
                    decision_meta = decision_metadata(decision_header)
                except Exception as exc:  # noqa: BLE001
                    from .._proxy_client import should_fall_back_to_provider

                    if not should_fall_back_to_provider(exc):
                        raise
                    proxy_failed = True
                    proxy_error = exc
                finally:
                    if circuit is not None:
                        circuit.release_stuck_probe()
            else:
                proxy_failed = True

            if proxy_failed:
                if not fail_open_enabled:
                    tc_err = (
                        proxy_error
                        if isinstance(proxy_error, TensorCostProxyError)
                        else TensorCostProxyError(
                            str(proxy_error) if proxy_error else "proxy unavailable",
                            root_cause=proxy_error if isinstance(proxy_error, Exception) else None,
                            attempt=1,
                        )
                    )
                    _post_obs(
                        transport, str(model), tenant_id, environment,
                        connection_id, correlation_id, request_at, None, None,
                        "error", f"{type(tc_err).__name__}: {tc_err}",
                        customer=eff_customer, feature=eff_feature,
                        agent_id=eff_agent_id, workflow_id=eff_workflow_id,
                    )
                    raise tc_err

                emit_event(
                    on_lifecycle_event,
                    OnFallbackEvent(
                        kind="on_fallback",
                        provider="anthropic",
                        model=str(model),
                        operation="messages",
                        attempt_number=1,
                        elapsed_ms=0.0,
                        consecutive_failures=circuit.failures if circuit else 0,
                    ),
                )
                try:
                    direct_response = bound_method(*args, **kwargs)
                    input_tokens, output_tokens = _extract_usage(direct_response)
                    _post_obs(
                        transport, str(model), tenant_id, environment,
                        connection_id, correlation_id, request_at, input_tokens,
                        output_tokens, "success", None,
                        customer=eff_customer, feature=eff_feature,
                        agent_id=eff_agent_id, workflow_id=eff_workflow_id,
                    )
                    return direct_response
                except Exception as exc:
                    _post_obs(
                        transport, str(model), tenant_id, environment,
                        connection_id, correlation_id, request_at, None, None,
                        "error", f"{type(exc).__name__}: {exc}",
                        customer=eff_customer, feature=eff_feature,
                        agent_id=eff_agent_id, workflow_id=eff_workflow_id,
                    )
                    raise

            # Proxy success.
            resp_dict = proxy_result if isinstance(proxy_result, dict) else {}
            usage = resp_dict.get("usage") if isinstance(resp_dict, dict) else None
            input_tokens = usage.get("input_tokens") if isinstance(usage, dict) else None
            output_tokens = usage.get("output_tokens") if isinstance(usage, dict) else None
            _post_obs(
                transport, str(model), tenant_id, environment,
                connection_id, correlation_id, request_at, input_tokens,
                output_tokens, "success", None,
                customer=eff_customer, feature=eff_feature,
                agent_id=eff_agent_id, workflow_id=eff_workflow_id,
            )
            return proxy_result

        # Observe-only path.
        try:
            response = bound_method(*args, **kwargs)
        except Exception as exc:
            _post_obs(
                transport, str(model), tenant_id, environment,
                connection_id, correlation_id, request_at, None, None,
                "error", f"{type(exc).__name__}: {exc}",
                customer=eff_customer, feature=eff_feature,
                agent_id=eff_agent_id, workflow_id=eff_workflow_id,
            )
            raise

        input_tokens, output_tokens = _extract_usage(response)
        _post_obs(
            transport, str(model), tenant_id, environment,
            connection_id, correlation_id, request_at, input_tokens,
            output_tokens, "success", None,
            customer=eff_customer, feature=eff_feature,
            agent_id=eff_agent_id, workflow_id=eff_workflow_id,
        )
        return response

    setattr(wrapper, "__tensorcost_wrapped__", True)
    return wrapper


def install(
    client: Any,
    transport: ObservationTransport,
    tenant_id: str | None,
    environment: str | None,
    connection_id: str | None,
    application: Optional[str] = None,
    tags: Optional[list[str]] = None,
    max_layer: ControlLayer = "observe",
    applied_mode: bool = False,
    proxy_url: Optional[str] = None,
    base_url: str = "https://api.tensorcost.com",
    sdk_layer: Optional[SdkLayerClient] = None,
    retry: RetryConfig = DEFAULT_RETRY_CONFIG,
    timeout_s: float = 60.0,
    headers_timeout_s: float = PROXY_HEADERS_TIMEOUT_S,
    on_lifecycle_event: Optional[LifecycleEventCallback] = None,
    fail_open_enabled: bool = True,
    circuit: Optional[CircuitBreaker] = None,
    customer: Optional[str] = None,
    feature: Optional[str] = None,
    agent_id: Optional[str] = None,
    workflow_id: Optional[str] = None,
    team_id: Optional[str] = None,
) -> None:
    """Monkey-patch ``client.messages.create`` in place."""
    code_layer = code_layer_for_install(max_layer, applied_mode)
    messages = getattr(client, "messages", None)
    if messages is None or not callable(getattr(messages, "create", None)):
        return
    original = messages.create
    if getattr(original, "__tensorcost_wrapped__", False):
        return
    messages.create = _wrap_method(
        original,
        transport=transport,
        tenant_id=tenant_id,
        environment=environment,
        application=application,
        tags=tags,
        connection_id=connection_id,
        max_layer=code_layer,
        applied_mode=applied_mode,
        proxy_url=proxy_url,
        base_url=base_url,
        sdk_layer=sdk_layer,
        client_ref=client,
        retry=retry,
        timeout_s=timeout_s,
        headers_timeout_s=headers_timeout_s,
        on_lifecycle_event=on_lifecycle_event,
        fail_open_enabled=fail_open_enabled,
        circuit=circuit,
        customer=customer,
        feature=feature,
        agent_id=agent_id,
        workflow_id=workflow_id,
        team_id=team_id,
    )
