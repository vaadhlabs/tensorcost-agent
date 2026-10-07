"""Google Vertex AI (google-genai Client, vertexai=True) — observe-only.

The unified ``google.genai.Client`` with ``vertexai=True`` exposes
``client.models.generate_content`` and ``generate_content_stream``. We
monkey-patch those methods on the ``models`` namespace — the same
technique used for OpenAI / Anthropic nested call sites.

Supported operations:
  * ``models.generate_content``        -> operation "vertex.generate_content"
  * ``models.generate_content_stream`` -> operation "vertex.generate_content_stream"

For streaming calls we emit an observation with input_tokens=None,
output_tokens=None, and a note in error_message that per-chunk token
accounting is not yet implemented. We never consume the returned stream.

Token-count extraction (generate_content):
  ``response.usage_metadata.prompt_token_count`` ->
  ``input_tokens``; ``candidates_token_count`` -> ``output_tokens``.

Applied mode:
  Not supported. Vertex uses Application Default Credentials and
  Google-managed transport that cannot be re-routed like HTTP Bearer
  ``base_url`` proxy for OpenAI/Anthropic.
  ``wrap(vertex_client, applied_mode=True)`` raises
  ``MissingConfigError`` immediately, both from :func:`install` here and
  (earlier, so this is defense-in-depth) from ``wrap()`` itself.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from .._models import Observation
from .._transport import ObservationTransport
from .._version import SDK_VERSION as _SDK_VERSION

_APPLIED_MODE_ERROR = (
    "TensorCost applied mode is not supported for Google Vertex AI clients. "
    "Vertex uses Application Default Credentials and Google-managed transport "
    "that cannot be intercepted at the proxy layer. Omit applied_mode (or pass "
    "applied_mode=False) to continue with observe-only mode, or use an OpenAI "
    "/ Anthropic client instead."
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve_model(*args: Any, **kwargs: Any) -> str:
    model = kwargs.get("model")
    if model is None and args:
        model = args[0]
    return str(model) if model is not None else "unknown"


def _extract_generate_content_tokens(response: Any) -> "tuple[int | None, int | None]":
    usage = getattr(response, "usage_metadata", None)
    if usage is None and isinstance(response, dict):
        usage = response.get("usage_metadata")
    if usage is None:
        return None, None

    if isinstance(usage, dict):
        return usage.get("prompt_token_count"), usage.get("candidates_token_count")

    prompt = getattr(usage, "prompt_token_count", None)
    candidates = getattr(usage, "candidates_token_count", None)
    return prompt, candidates


def _post_obs(
    transport: ObservationTransport,
    model: str,
    operation: str,
    tenant_id: Optional[str],
    environment: Optional[str],
    connection_id: Optional[str],
    correlation_id: str,
    request_at: str,
    input_tokens: Optional[int],
    output_tokens: Optional[int],
    status: str,
    error_message: Optional[str],
    customer: Optional[str] = None,
    feature: Optional[str] = None,
    agent_id: Optional[str] = None,
    workflow_id: Optional[str] = None,
) -> None:
    obs: Observation = {
        "sdk_version": _SDK_VERSION,
        "provider": "vertex",
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
    operation: str,
    transport: ObservationTransport,
    tenant_id: Optional[str],
    environment: Optional[str],
    connection_id: Optional[str],
    token_extractor: Optional[Callable[[Any], "tuple[int | None, int | None]"]],
    is_streaming: bool,
    customer: Optional[str] = None,
    feature: Optional[str] = None,
    agent_id: Optional[str] = None,
    workflow_id: Optional[str] = None,
) -> Any:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        request_at = _now_iso()
        model = _resolve_model(*args, **kwargs)
        correlation_id = str(uuid.uuid4())

        try:
            response = bound_method(*args, **kwargs)
        except Exception as exc:
            _post_obs(
                transport, model, operation, tenant_id, environment,
                connection_id, correlation_id, request_at, None, None,
                "error", f"{type(exc).__name__}: {exc}",
                customer=customer, feature=feature,
                agent_id=agent_id, workflow_id=workflow_id,
            )
            raise

        if is_streaming:
            _post_obs(
                transport, model, operation, tenant_id, environment,
                connection_id, correlation_id, request_at, None, None,
                "success",
                "streaming: per-chunk token accounting not yet implemented",
                customer=customer, feature=feature,
                agent_id=agent_id, workflow_id=workflow_id,
            )
            return response

        input_tokens, output_tokens = (
            token_extractor(response) if token_extractor else (None, None)
        )
        _post_obs(
            transport, model, operation, tenant_id, environment,
            connection_id, correlation_id, request_at, input_tokens,
            output_tokens, "success", None,
            customer=customer, feature=feature,
            agent_id=agent_id, workflow_id=workflow_id,
        )
        return response

    setattr(wrapper, "__tensorcost_wrapped__", True)
    return wrapper


_METHODS = (
    ("generate_content", "vertex.generate_content", _extract_generate_content_tokens, False),
    ("generate_content_stream", "vertex.generate_content_stream", None, True),
)


def install(
    client: Any,
    transport: ObservationTransport,
    tenant_id: Optional[str],
    environment: Optional[str],
    connection_id: Optional[str],
    applied_mode: bool = False,
    customer: Optional[str] = None,
    feature: Optional[str] = None,
    agent_id: Optional[str] = None,
    workflow_id: Optional[str] = None,
) -> None:
    """Monkey-patch a ``google.genai.Client`` (Vertex mode) ``models`` namespace.

    Raises ``MissingConfigError`` if ``applied_mode=True`` — Vertex does
    not support applied mode (see module docstring).
    """
    if applied_mode:
        from .._config import MissingConfigError

        raise MissingConfigError(_APPLIED_MODE_ERROR)

    models = getattr(client, "models", None)
    if models is None:
        return

    for attr, operation, extractor, is_streaming in _METHODS:
        original = getattr(models, attr, None)
        if original is None or not callable(original):
            continue
        if getattr(original, "__tensorcost_wrapped__", False):
            continue
        setattr(
            models,
            attr,
            _wrap_method(
                original,
                operation=operation,
                transport=transport,
                tenant_id=tenant_id,
                environment=environment,
                connection_id=connection_id,
                token_extractor=extractor,
                is_streaming=is_streaming,
                customer=customer,
                feature=feature,
                agent_id=agent_id,
                workflow_id=workflow_id,
            ),
        )
