"""OpenAI client wrapping.

Intercepts:
  * ``client.chat.completions.create``
  * ``client.completions.create`` (legacy text completions; v0 still ships)

We capture only metadata (model, message count, token usage) — never
prompt content. Customers' prompts stay on their machine.

Applied-mode hardening (v0.4.0): chat/completions calls route through the
proxy-client module, which handles retries, timeouts, typed errors,
telemetry hooks, and the fail-open circuit breaker.
"""

from __future__ import annotations

import functools
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from .._admit_client import ADMIT_UNAVAILABLE_METADATA, AdmitOutcome, admit_call
from .._compliance_guard import (
    assert_team_scope,
    compliance_stamp_from,
    enforce_in_process_dlp,
)
from .._dlp import extract_openai_messages_text
from .._errors import (
    TensorCostComplianceDeniedError,
    TensorCostComplianceTeamMismatchError,
)
from .._attribution import resolve_scope_field
from .._applied_mode import (
    openai_provider_auth,
    openai_provider_base_url,
    openai_provider_for,
)
from .._circuit import CircuitBreaker
from .._errors import TensorCostProxyError, TensorCostRefusedError, TensorCostTimeoutError
from .._budget_signals import report_would_refuse
from .._request_options import split_request_options
from .._typed_response import UNTYPED, replay_through_client, route_stream_result
from .._usage import (
    can_observe_stream,
    observe_stream,
    openai_chunk_id,
    openai_chunk_usage,
    parsed_for_usage,
    response_id_field,
    response_id_of,
)
from .._layer import (
    ADMIT_TIMEOUT_S,
    PROXY_HEADERS_TIMEOUT_S,
    ControlLayer,
    code_layer_for_install,
    needs_proxy_url,
)
from .._models import Observation
from .._proxy_client import proxy_request
from .._retry import DEFAULT_RETRY_CONFIG, RetryConfig
from .._sdk_layer import PassthroughSdkLayer, SdkLayerClient, resolve_layer
from .._telemetry import LifecycleEventCallback, OnFallbackEvent, emit_event
from .._transport import ObservationTransport, get_token_within
from .._observation_scope import observation_scope
from .._provider_errors import provider_native_error
from .._version import SDK_VERSION as _SDK_VERSION
from .. import _billing_account as _ba
_log = logging.getLogger("tensorcost")


# Below this much of the admit budget left after the token exchange, the
# POST isn't attempted: it could only time out.
_MIN_ADMIT_S = 0.05


def _create_of(client: Any) -> Any:
    return client.chat.completions.create


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _extract_usage(response: Any) -> tuple[int | None, int | None]:
    usage = getattr(response, "usage", None)
    if usage is None and isinstance(response, dict):
        usage = response.get("usage")
    if usage is None:
        return None, None
    if isinstance(usage, dict):
        return usage.get("prompt_tokens"), usage.get("completion_tokens")
    return (
        getattr(usage, "prompt_tokens", None),
        getattr(usage, "completion_tokens", None),
    )


def _post_obs(
    transport: ObservationTransport,
    model: str,
    operation: str,
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
    application: Optional[str] = None,
    tags: Optional[list] = None,
    billing_account: Optional[str] = None,
    provider: str = "openai",
    metadata: Optional[dict] = None,
    provider_response_id: Optional[str] = None,
) -> None:
    obs: dict = {
        "sdk_version": _SDK_VERSION,
        "provider": provider,
        "model": model,
        "operation": operation,
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
        # Routing-policy scope from wrap(application=, tags=); empty while
        # _observation_scope.SEND is off (1.5.0), see that module.
        **observation_scope(application, tags),
        # Only on calls the SDK made itself; the proxy labels its own.
        **_ba.field(billing_account),
        # admit_unavailable: this call went ahead without an /admit answer.
        **({"metadata": dict(metadata)} if metadata else {}),
        # The provider's id for the call: the same call seen by the proxy, a
        # gateway or a cost import is matched to this row on it.
        **response_id_field(provider_response_id),
    }
    try:
        transport.post(obs)
    except Exception:  # noqa: BLE001
        pass


def _wrap_method(
    bound_method: Any,
    *,
    operation: str,
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
    # azure_openai for an AzureOpenAI client: on observations and the govern
    # check. The proxy request keeps "openai", which picks the proxy's route.
    provider_name = openai_provider_for(client_ref)

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
        effective_layer, layer_decayed = resolve_layer(layer_client, max_layer)

        compliance_stamp: dict[str, Any] = {}
        if operation == "chat.completions":
            try:
                assert_team_scope(client_team_id=team_id)
                compliance_snap = None
                if isinstance(layer_client, SdkLayerClient):
                    compliance_snap = layer_client.get_compliance_snapshot(effective_layer)
                dlp_result = enforce_in_process_dlp(
                    extract_openai_messages_text(args),
                    compliance_snap,
                    args=args,
                )
                compliance_stamp = compliance_stamp_from(compliance_snap, dlp_result)
            except (TensorCostComplianceDeniedError, TensorCostComplianceTeamMismatchError):
                raise
            except Exception:
                pass

        # Set when the call goes ahead without an /admit answer; the call's
        # own observation then says so and keeps its agent, team and usage.
        obs_metadata: Optional[dict] = None
        if effective_layer == "govern" and operation == "chat.completions":
            from .._proxy_client import should_fall_back_to_provider

            # Token exchange and /admit together get ADMIT_TIMEOUT_S.
            admit_started = time.monotonic()
            try:
                bearer_token: Optional[str] = get_token_within(transport, ADMIT_TIMEOUT_S)
            except Exception as exc:  # noqa: BLE001
                # Same rule as the proxy path: go around TensorCost only when
                # it is unavailable, and only with fail_open_enabled.
                if not (fail_open_enabled and should_fall_back_to_provider(exc)):
                    raise
                bearer_token = None
            remaining = ADMIT_TIMEOUT_S - (time.monotonic() - admit_started)
            if bearer_token is None or remaining < _MIN_ADMIT_S:
                if bearer_token is not None and not fail_open_enabled:
                    raise TensorCostTimeoutError(
                        f"tensorcost: admit budget ({ADMIT_TIMEOUT_S}s) spent on the token exchange",
                        "total",
                    )
                outcome = AdmitOutcome(True, None, unavailable=True)
            else:
                outcome = admit_call(
                    base_url=base_url,
                    bearer_token=bearer_token,
                    max_layer=effective_layer,
                    metadata={
                        "provider": provider_name,
                        "model": str(model),
                        "operation": operation,
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
                    fail_open=fail_open_enabled,
                    timeout_s=remaining,
                )
            if not outcome.allowed:
                # admit_call raises every refusal typed by its code; this
                # is only a guard. It used to be a TensorCostProviderError.
                raise TensorCostRefusedError(
                    "tensorcost: request refused at govern layer",
                    status=403,
                    attempt=1,
                )
            if outcome.unavailable:
                obs_metadata = dict(ADMIT_UNAVAILABLE_METADATA)
            report_would_refuse(
                on_lifecycle_event, outcome.would_refuse,
                provider=provider_name, model=str(model), operation=operation,
            )

        def post(
            input_tokens: Optional[int],
            output_tokens: Optional[int],
            error: Optional[BaseException] = None,
            billing_account: Optional[str] = None,
            response_id: Optional[str] = None,
        ) -> None:
            _post_obs(
                transport, str(model), operation, tenant_id, environment,
                connection_id, correlation_id, request_at, input_tokens, output_tokens,
                "error" if error is not None else "success",
                f"{type(error).__name__}: {error}" if error is not None else None,
                customer=eff_customer, feature=eff_feature,
                agent_id=eff_agent_id, workflow_id=eff_workflow_id,
                application=application, tags=tags,
                billing_account=billing_account,
                provider=provider_name,
                metadata=obs_metadata,
                provider_response_id=response_id,
            )

        def fallback(reason: str) -> None:
            emit_event(
                on_lifecycle_event,
                OnFallbackEvent(
                    kind="on_fallback",
                    provider="openai",
                    model=str(model),
                    operation=operation,
                    attempt_number=1,
                    elapsed_ms=0.0,
                    consecutive_failures=circuit.failures if circuit else 0,
                    reason=reason,  # type: ignore[arg-type]
                ),
            )

        def call_provider() -> Any:
            """The customer's call, straight to the provider, observed."""
            _cap = _ba.Capture()
            try:
                with _ba.capturing(_cap):
                    response = bound_method(*args, **kwargs)
            except Exception as exc:
                post(None, None, exc, _ba.account_of_http_call(client_ref, _cap, exc))
                raise
            billing_account = _ba.account_of_http_call(client_ref, _cap)
            if kwargs.get("stream") is True and can_observe_stream(response):
                return observe_stream(
                    response, openai_chunk_usage, openai_chunk_id,
                    lambda i, o, err, rid: post(i, o, err, billing_account, rid),
                )
            parsed = parsed_for_usage(response)
            input_tokens, output_tokens = _extract_usage(parsed)
            post(input_tokens, output_tokens, None, billing_account, response_id_of(parsed))
            return response

        wants_proxy = (
            needs_proxy_url(max_layer) and bool(proxy_url) and operation == "chat.completions"
        )
        if wants_proxy and needs_proxy_url(effective_layer):
            return route(post, fallback, call_provider, args, kwargs, model, correlation_id,
                         eff_customer, eff_feature, eff_agent_id, eff_workflow_id,
                         effective_layer)

        if wants_proxy and layer_decayed:
            # Route was asked for, but the published layer couldn't be had,
            # so the client decayed to govern and goes to the provider
            # directly. A refusal from /admit has already raised above.
            fallback("layer_decayed")
        return call_provider()

    def route(
        post: Any, fallback: Any, call_provider: Any, args: tuple, kwargs: dict, model: Any,
        correlation_id: str, eff_customer: Optional[str], eff_feature: Optional[str],
        eff_agent_id: Optional[str], eff_workflow_id: Optional[str],
        effective_layer: ControlLayer,
    ) -> Any:
        from .._proxy_client import proxy_request_stream, should_fall_back_to_provider

        provider_url = openai_provider_base_url(client_ref) if client_ref else "https://api.openai.com"
        provider_auth = openai_provider_auth(client_ref) if client_ref else ""

        # The body openai would send: request options become headers, query
        # and timeout, extra_body is merged in (_request_options).
        opts = split_request_options(kwargs)
        body: dict[str, Any] = opts.body
        body["model"] = str(model)
        call_timeout_s = opts.timeout_s or timeout_s
        is_streaming = body.get("stream") is True

        proxy_error: Optional[Exception] = None
        proxy_result: Any = None
        sink: list = []
        fallback_reason: Optional[str] = None

        circuit_was_open = circuit is not None and circuit.should_bypass()
        should_try_proxy = not circuit_was_open
        if circuit_was_open and circuit is not None:
            should_try_proxy = circuit.should_probe()

        if should_try_proxy:
            try:
                bearer_token = get_token_within(transport, min(headers_timeout_s, call_timeout_s))
                common = dict(
                    proxy_url=proxy_url,
                    bearer_token=bearer_token,
                    provider_url=provider_url,
                    provider_auth=provider_auth,
                    body=body,
                    model=str(model),
                    provider="openai",
                    operation=operation,
                    correlation_id=correlation_id,
                    environment=environment,
                    application=application,
                    tags=tags,
                    customer=eff_customer,
                    feature=eff_feature,
                    agent_id=eff_agent_id,
                    workflow_id=eff_workflow_id,
                    retry=retry,
                    timeout_s=call_timeout_s,
                    headers_timeout_s=headers_timeout_s,
                    on_lifecycle_event=on_lifecycle_event,
                    circuit=circuit,
                    max_layer=effective_layer,
                    fail_open_fast=fail_open_enabled,
                    extra_headers=opts.headers,
                    extra_query=opts.query,
                )
                if is_streaming:
                    proxy_result, _decision = proxy_request_stream(**common, raw=True)
                else:
                    proxy_result, _decision = proxy_request(**common, response_sink=sink)
            except Exception as exc:  # noqa: BLE001
                if not should_fall_back_to_provider(exc):
                    native = provider_native_error(client_ref, exc, provider_url)
                    if native is exc:
                        raise
                    raise native from exc
                fallback_reason = "proxy_unavailable"
                proxy_error = exc
            finally:
                if circuit is not None:
                    circuit.release_stuck_probe()
        else:
            fallback_reason = "circuit_open"

        if fallback_reason is not None:
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
                post(None, None, tc_err)
                raise tc_err
            # Fail-open: say so, then call the provider directly.
            fallback(fallback_reason)
            return call_provider()

        if is_streaming:
            return route_stream_result(
                client_ref, _create_of, proxy_result, openai_chunk_usage, openai_chunk_id,
                lambda i, o, err, rid: post(i, o, err, None, rid), args, kwargs,
            )

        # Proxy success: openai's own type for it (ChatCompletion, or the
        # raw-response wrapper under with_raw_response).
        resp_dict = proxy_result if isinstance(proxy_result, dict) else {}
        input_tokens, output_tokens = _extract_usage(resp_dict)
        post(input_tokens, output_tokens, None, None, response_id_of(resp_dict))
        resp = sink[0] if sink else None
        typed = replay_through_client(
            client_ref, _create_of, args, kwargs,
            status_code=resp.status_code if resp is not None else 200,
            headers=resp.headers if resp is not None else {"content-type": "application/json"},
            content=resp.content if resp is not None else json.dumps(resp_dict).encode(),
        )
        return proxy_result if typed is UNTYPED else typed

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
    """Monkey-patch the OpenAI client's ``create`` methods in place."""
    code_layer = code_layer_for_install(max_layer, applied_mode)
    chat = getattr(client, "chat", None)
    completions = getattr(chat, "completions", None) if chat is not None else None
    if completions is not None and callable(getattr(completions, "create", None)):
        original = completions.create
        if not getattr(original, "__tensorcost_wrapped__", False):
            completions.create = _wrap_method(
                original,
                operation="chat.completions",
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

    legacy = getattr(client, "completions", None)
    if legacy is not None and callable(getattr(legacy, "create", None)):
        original = legacy.create
        if not getattr(original, "__tensorcost_wrapped__", False):
            legacy.create = _wrap_method(
                original,
                operation="completions",
                transport=transport,
                tenant_id=tenant_id,
                environment=environment,
                application=application,
                tags=tags,
                connection_id=connection_id,
                # Legacy completions endpoint not in scope for applied-mode v1.
                applied_mode=False,
                proxy_url=None,
                client_ref=client,
                retry=retry,
                timeout_s=timeout_s,
                on_lifecycle_event=on_lifecycle_event,
                fail_open_enabled=fail_open_enabled,
                circuit=circuit,
                customer=customer,
                feature=feature,
                agent_id=agent_id,
                workflow_id=workflow_id,
            )
    # Response headers for billing_account (_billing_account.py).
    _ba.install_response_hook(client)
