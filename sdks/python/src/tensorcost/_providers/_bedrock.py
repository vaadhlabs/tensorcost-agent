"""AWS Bedrock (boto3) client wrapping — observe-only.

Bedrock differs fundamentally from OpenAI and Anthropic: there is no HTTP
client we can point at a proxy baseURL, and there is no single
``messages.create``-style hook. A ``boto3.client("bedrock-runtime")``
instance exposes plain bound methods generated at runtime by botocore
(``invoke_model``, ``converse``, and their streaming counterparts), so we
monkey-patch those directly — the same technique used for
``client.chat.completions.create`` / ``client.messages.create`` in the
OpenAI / Anthropic wrappers, just applied to a different set of methods.

Supported operations (non-streaming):
  * ``invoke_model``  -> operation "bedrock.invoke_model"
  * ``converse``      -> operation "bedrock.converse"

Streaming operations (token accounting deferred, mirrors the Node SDK):
  * ``invoke_model_with_response_stream`` -> "bedrock.invoke_model_stream"
  * ``converse_stream``                   -> "bedrock.converse_stream"

For streaming calls we emit an observation with input_tokens=None,
output_tokens=None, and a note in error_message that per-chunk token
accounting is not yet implemented. We never touch the returned stream
object itself, so the customer's iteration over it is completely
unaffected.

Token-count extraction (InvokeModel):
  boto3's ``invoke_model`` response ``body`` is a single-read
  ``botocore.response.StreamingBody``. To read it for token counts
  without breaking the customer's own ``response["body"].read()`` call,
  we drain it once here and replace ``response["body"]`` with an
  in-memory ``io.BytesIO`` positioned at the start, exposing the same
  ``read()`` interface with the identical bytes.

  We probe two common body shapes, in order:
    x-amzn-bedrock-input/output-token-count response headers first (every
    model family). Without them, body shapes:
    1. Anthropic-on-Bedrock: body JSON -> usage.input_tokens / output_tokens
    2. Amazon Nova / Titan:  body JSON -> usage.inputTokens / outputTokens
    3. Meta Llama:           body JSON -> prompt_token_count / generation_token_count
  If neither matches we emit None/None and log a warning.

Token-count extraction (Converse):
  The Converse API standardises usage across model families:
  response["usage"]["inputTokens"] / ["outputTokens"].

Applied mode:
  Not supported. There is no HTTP baseURL to redirect for a boto3
  client — request signing (SigV4) and transport happen inside
  botocore. ``wrap(bedrock_client, applied_mode=True)`` raises
  ``MissingConfigError`` immediately, both from :func:`install` here and
  (earlier, so this is defense-in-depth) from ``wrap()`` itself.
"""

from __future__ import annotations

import io
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from .._models import Observation
from .._transport import ObservationTransport

from .._version import SDK_VERSION as _SDK_VERSION
_log = logging.getLogger("tensorcost")

_APPLIED_MODE_ERROR = (
    "TensorCost applied mode is not supported for AWS Bedrock clients. "
    "Bedrock uses SigV4 signing that cannot be intercepted at the proxy "
    "layer. Omit applied_mode (or pass applied_mode=False) to continue "
    "with observe-only mode, or use an OpenAI / Anthropic client instead."
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _tokens_from_headers(response: Any) -> "tuple[int | None, int | None] | None":
    """Bedrock sets these headers on every InvokeModel response, any model."""
    meta = response.get("ResponseMetadata") if isinstance(response, dict) else None
    headers = meta.get("HTTPHeaders") if isinstance(meta, dict) else None
    if not isinstance(headers, dict):
        return None

    def read(name: str) -> "int | None":
        raw = headers.get(name)
        try:
            n = int(raw) if raw is not None else None
        except (TypeError, ValueError):
            return None
        return n if n is not None and n >= 0 else None

    inp = read("x-amzn-bedrock-input-token-count")
    out = read("x-amzn-bedrock-output-token-count")
    return None if inp is None and out is None else (inp, out)


def _extract_invoke_model_tokens(response: Any) -> "tuple[int | None, int | None]":
    if not isinstance(response, dict):
        return None, None
    from_headers = _tokens_from_headers(response)
    if from_headers is not None:
        return from_headers
    body = response.get("body")
    if body is None or not callable(getattr(body, "read", None)):
        return None, None

    try:
        raw = body.read()
    except Exception:  # noqa: BLE001
        return None, None

    # Replace with a replayable in-memory buffer so the customer can still
    # call response["body"].read() exactly as with the original
    # single-read StreamingBody.
    response["body"] = io.BytesIO(raw)

    try:
        text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        parsed = json.loads(text)
    except Exception:  # noqa: BLE001
        _log.warning(
            "tensorcost: could not parse Bedrock InvokeModel response body "
            "for token counts"
        )
        return None, None

    # Meta Llama: top-level prompt_token_count / generation_token_count.
    if isinstance(parsed, dict) and (
        "prompt_token_count" in parsed or "generation_token_count" in parsed
    ):
        return parsed.get("prompt_token_count"), parsed.get("generation_token_count")

    usage = parsed.get("usage") if isinstance(parsed, dict) else None
    if not isinstance(usage, dict):
        _log.warning(
            "tensorcost: Bedrock InvokeModel response has no usage field — "
            "token counts will be missing. If this is a new model family, "
            "please open an issue at https://github.com/vaadhlabs/tensorcost."
        )
        return None, None

    # Anthropic-on-Bedrock: usage.input_tokens / output_tokens
    if "input_tokens" in usage or "output_tokens" in usage:
        return usage.get("input_tokens"), usage.get("output_tokens")

    # Amazon Nova / Titan: usage.inputTokens / outputTokens
    if "inputTokens" in usage or "outputTokens" in usage:
        return usage.get("inputTokens"), usage.get("outputTokens")

    _log.warning(
        "tensorcost: Bedrock InvokeModel usage shape not recognised — "
        "token counts will be missing. Known shapes: Anthropic "
        "(usage.input_tokens/output_tokens), Nova/Titan "
        "(usage.inputTokens/outputTokens)."
    )
    return None, None


def _extract_converse_tokens(response: Any) -> "tuple[int | None, int | None]":
    if not isinstance(response, dict):
        return None, None
    usage = response.get("usage")
    if not isinstance(usage, dict):
        return None, None
    return usage.get("inputTokens"), usage.get("outputTokens")


_METHODS = (
    ("invoke_model", "bedrock.invoke_model", _extract_invoke_model_tokens, False),
    ("converse", "bedrock.converse", _extract_converse_tokens, False),
    ("invoke_model_with_response_stream", "bedrock.invoke_model_stream", None, True),
    ("converse_stream", "bedrock.converse_stream", None, True),
)


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
        "provider": "bedrock",
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
        model = kwargs.get("modelId", "unknown")
        correlation_id = str(uuid.uuid4())

        try:
            response = bound_method(*args, **kwargs)
        except Exception as exc:
            _post_obs(
                transport, str(model), operation, tenant_id, environment,
                connection_id, correlation_id, request_at, None, None,
                "error", f"{type(exc).__name__}: {exc}",
                customer=customer, feature=feature,
                agent_id=agent_id, workflow_id=workflow_id,
            )
            raise

        if is_streaming:
            # We never touch the returned stream — token accounting across
            # chunks is a follow-up workstream. The placeholder note lets
            # operators distinguish this from a genuine extraction gap.
            _post_obs(
                transport, str(model), operation, tenant_id, environment,
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
            transport, str(model), operation, tenant_id, environment,
            connection_id, correlation_id, request_at, input_tokens,
            output_tokens, "success", None,
            customer=customer, feature=feature,
            agent_id=agent_id, workflow_id=workflow_id,
        )
        return response

    setattr(wrapper, "__tensorcost_wrapped__", True)
    return wrapper


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
    """Monkey-patch a boto3 ``bedrock-runtime`` client's call methods.

    Raises ``MissingConfigError`` if ``applied_mode=True`` — Bedrock does
    not support applied mode (see module docstring). ``wrap()`` already
    checks this before calling into this module; the check here is
    defense-in-depth for callers that invoke this module directly.
    """
    if applied_mode:
        # Imported lazily to avoid a module-level cycle (``_config`` has no
        # reason to import provider modules, but keeping the import local
        # here documents that this is the one place _bedrock needs it).
        from .._config import MissingConfigError

        raise MissingConfigError(_APPLIED_MODE_ERROR)

    for attr, operation, extractor, is_streaming in _METHODS:
        original = getattr(client, attr, None)
        if original is None or not callable(original):
            continue
        if getattr(original, "__tensorcost_wrapped__", False):
            continue
        setattr(
            client,
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
