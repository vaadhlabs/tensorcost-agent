"""
LLM Tracking Middleware - Automatic Token Usage Tracking
Intercepts LLM API calls at the network/library level without code changes.

Usage:
    # Just import and enable - that's it!
    from integrations.middleware import enable_llm_tracking
    
    enable_llm_tracking()
    
    # Now all LLM calls are automatically tracked
    import openai
    response = openai.ChatCompletion.create(...)  # Automatically tracked!
"""

import os
import json
import logging
import functools
import time
from typing import Dict, Any, Optional, Callable
from datetime import datetime

from .llm_tracker import get_tracker, LLMTracker

logger = logging.getLogger(__name__)

# Global tracking state
_tracking_enabled = False
_tracker = None


def enable_llm_tracking(
    backend_api_url: Optional[str] = None,
    backend_api_key: Optional[str] = None,
    tenant_id: Optional[str] = None
):
    """
    Enable automatic LLM tracking middleware.
    This intercepts LLM API calls and tracks them automatically.
    
    Call this once at application startup - no other code changes needed!
    """
    global _tracking_enabled, _tracker
    
    if _tracking_enabled:
        logger.warning("LLM tracking already enabled")
        return
    
    _tracker = get_tracker()
    if backend_api_url:
        _tracker.backend_api_url = backend_api_url
    if backend_api_key:
        _tracker.backend_api_key = backend_api_key
    if tenant_id:
        _tracker.tenant_id = tenant_id
    
    # Install interceptors
    _install_openai_interceptor()
    _install_anthropic_interceptor()
    _install_bedrock_interceptor()
    _install_vertex_ai_interceptor()
    _install_requests_interceptor()
    _install_httpx_interceptor()
    
    _tracking_enabled = True
    logger.info("LLM tracking middleware enabled - all LLM calls will be automatically tracked")


def disable_llm_tracking():
    """Disable automatic LLM tracking"""
    global _tracking_enabled
    _tracking_enabled = False
    logger.info("LLM tracking middleware disabled")


def _install_openai_interceptor():
    """Install interceptor for OpenAI library (supports both v0.x and v1.x+)"""
    try:
        import openai

        openai_version = getattr(openai, '__version__', '0.0.0')
        major_version = int(openai_version.split('.')[0])

        if major_version >= 1:
            # OpenAI v1.x+ uses client-based API: client.chat.completions.create()
            _install_openai_v1_interceptor(openai)
        else:
            # OpenAI v0.x uses module-level API: openai.ChatCompletion.create()
            _install_openai_v0_interceptor(openai)

    except ImportError:
        logger.debug("OpenAI library not found, skipping interceptor")
    except Exception as e:
        logger.error(f"Failed to install OpenAI interceptor: {e}")


def _install_openai_v0_interceptor(openai):
    """Install interceptor for OpenAI v0.x (openai.ChatCompletion.create).

    v0 uses module-level `openai.api_base` / `openai.api_type='azure'` to
    select Azure. We read those globals at call-time to pick the right
    provider label (OpenAI vs Azure OpenAI). In v0 the globals are
    inherently process-wide, so there's no client object to inspect.
    """
    if not hasattr(openai, '_original_chat_completion_create'):
        openai._original_chat_completion_create = openai.ChatCompletion.create
        openai._original_completion_create = openai.Completion.create

        def _v0_provider():
            # api_type == 'azure' is the most explicit signal; api_base
            # containing openai.azure.com is a fallback.
            try:
                if str(getattr(openai, 'api_type', '') or '').lower() == 'azure':
                    return 'azure_openai'
                if 'openai.azure.com' in str(getattr(openai, 'api_base', '') or '').lower():
                    return 'azure_openai'
            except Exception:
                pass
            return 'openai'

        @functools.wraps(openai.ChatCompletion.create)
        def tracked_chat_completion_create(*args, **kwargs):
            start_time = time.time()
            response = openai._original_chat_completion_create(*args, **kwargs)
            latency_ms = (time.time() - start_time) * 1000
            _track_openai_response(response, kwargs.get('model', 'unknown'), latency_ms, provider=_v0_provider())
            return response

        @functools.wraps(openai.Completion.create)
        def tracked_completion_create(*args, **kwargs):
            start_time = time.time()
            response = openai._original_completion_create(*args, **kwargs)
            latency_ms = (time.time() - start_time) * 1000
            _track_openai_response(response, kwargs.get('model', 'unknown'), latency_ms, provider=_v0_provider())
            return response

        openai.ChatCompletion.create = staticmethod(tracked_chat_completion_create)
        openai.Completion.create = staticmethod(tracked_completion_create)

        logger.info("OpenAI v0.x interceptor installed")


def _install_openai_v1_interceptor(openai):
    """Install interceptor for OpenAI v1.x+ (client.chat.completions.create).

    Covers both OpenAI() and AzureOpenAI() — they share the Completions
    resource class, so a single class-level patch catches both. The
    per-call dispatcher reads self._client.base_url to pick the right
    provider label ('openai' vs 'azure_openai').
    """
    try:
        from openai.resources.chat import completions as chat_completions_mod

        original_create = chat_completions_mod.Completions.create
        if hasattr(original_create, '_gpu_dashboard_wrapped'):
            return  # Already wrapped

        @functools.wraps(original_create)
        def tracked_create(self, *args, **kwargs):
            start_time = time.time()
            response = original_create(self, *args, **kwargs)
            latency_ms = (time.time() - start_time) * 1000

            provider = _detect_openai_provider(self)

            # Check if this is a streaming response
            if kwargs.get('stream', False):
                # Wrap the generator to track on completion
                model = kwargs.get('model', 'unknown')
                response = _wrap_streaming_response(response, provider, model, start_time, _tracker)
            else:
                _track_openai_response(response, kwargs.get('model', 'unknown'), latency_ms, provider=provider)

            return response

        tracked_create._gpu_dashboard_wrapped = True
        chat_completions_mod.Completions.create = tracked_create

        logger.info("OpenAI v1.x+ interceptor installed (chat.completions.create)")

        # Also wrap async version
        try:
            from openai.resources.chat import completions as async_mod
            original_async_create = async_mod.AsyncCompletions.create
            if not hasattr(original_async_create, '_gpu_dashboard_wrapped'):
                @functools.wraps(original_async_create)
                async def tracked_async_create(self, *args, **kwargs):
                    start_time = time.time()
                    response = await original_async_create(self, *args, **kwargs)
                    latency_ms = (time.time() - start_time) * 1000
                    provider = _detect_openai_provider(self)
                    _track_openai_response(response, kwargs.get('model', 'unknown'), latency_ms, provider=provider)
                    return response

                tracked_async_create._gpu_dashboard_wrapped = True
                async_mod.AsyncCompletions.create = tracked_async_create
                logger.info("OpenAI v1.x+ async interceptor installed")
        except Exception:
            pass  # Async not available or already wrapped

    except Exception as e:
        logger.warning(f"Could not install OpenAI v1.x interceptor: {e}")


def _detect_openai_provider(completions_resource):
    """Decide 'openai' vs 'azure_openai' by inspecting the bound client's base URL.

    openai.resources.chat.completions.Completions holds a reference to its
    client on `._client`. For OpenAI proper, base_url is api.openai.com;
    for AzureOpenAI, it's <resource>.openai.azure.com. We fall back to
    'openai' if inspection fails for any reason — wrong label beats a
    crash on every call.
    """
    if completions_resource is None:
        return 'openai'
    try:
        client = getattr(completions_resource, '_client', None) or getattr(completions_resource, 'client', None)
        if client is None:
            return 'openai'
        base_url = str(getattr(client, 'base_url', '') or getattr(client, '_base_url', ''))
        if 'openai.azure.com' in base_url.lower():
            return 'azure_openai'
    except Exception:
        pass
    return 'openai'


def _install_anthropic_interceptor():
    """Install interceptor for Anthropic library (supports both sync and async)"""
    try:
        import anthropic

        # Wrap sync Messages.create
        try:
            from anthropic.resources import messages as messages_mod
            original_create = messages_mod.Messages.create
            if not hasattr(original_create, '_gpu_dashboard_wrapped'):
                @functools.wraps(original_create)
                def tracked_messages_create(self, *args, **kwargs):
                    start_time = time.time()
                    response = original_create(self, *args, **kwargs)
                    latency_ms = (time.time() - start_time) * 1000

                    # Check if this is a streaming response
                    if kwargs.get('stream', False):
                        # Wrap the stream to track on completion
                        model = kwargs.get('model', 'unknown')
                        response = _wrap_anthropic_streaming_response(response, model, start_time, _tracker)
                    else:
                        _track_anthropic_response(response, kwargs.get('model', 'unknown'), latency_ms)

                    return response

                tracked_messages_create._gpu_dashboard_wrapped = True
                messages_mod.Messages.create = tracked_messages_create
                logger.info("Anthropic sync interceptor installed")
        except Exception as e:
            # Fallback for older anthropic SDK
            try:
                if not hasattr(anthropic, '_original_messages_create'):
                    original = anthropic.Anthropic.messages.create
                    anthropic._original_messages_create = original

                    @functools.wraps(original)
                    def tracked_fallback(self, *args, **kwargs):
                        start_time = time.time()
                        response = anthropic._original_messages_create(self, *args, **kwargs)
                        latency_ms = (time.time() - start_time) * 1000
                        _track_anthropic_response(response, kwargs.get('model', 'unknown'), latency_ms)
                        return response

                    anthropic.Anthropic.messages.create = tracked_fallback
                    logger.info("Anthropic fallback interceptor installed")
            except Exception:
                pass

        # Wrap async Messages.create
        try:
            from anthropic.resources import messages as messages_mod
            original_async = messages_mod.AsyncMessages.create
            if not hasattr(original_async, '_gpu_dashboard_wrapped'):
                @functools.wraps(original_async)
                async def tracked_async_messages(self, *args, **kwargs):
                    start_time = time.time()
                    response = await original_async(self, *args, **kwargs)
                    latency_ms = (time.time() - start_time) * 1000
                    _track_anthropic_response(response, kwargs.get('model', 'unknown'), latency_ms)
                    return response

                tracked_async_messages._gpu_dashboard_wrapped = True
                messages_mod.AsyncMessages.create = tracked_async_messages
                logger.info("Anthropic async interceptor installed")
        except Exception:
            pass

    except ImportError:
        logger.debug("Anthropic library not found, skipping interceptor")
    except Exception as e:
        logger.error(f"Failed to install Anthropic interceptor: {e}")


def _install_requests_interceptor():
    """Install interceptor for requests library (catches all HTTP calls)"""
    try:
        import requests

        # Store original methods
        if not hasattr(requests, '_original_post'):
            requests._original_post = requests.post
            requests._original_request = requests.request

            # Wrap post method
            @functools.wraps(requests.post)
            def tracked_post(url, *args, **kwargs):
                start_time = time.time()
                response = requests._original_post(url, *args, **kwargs)
                latency_ms = (time.time() - start_time) * 1000
                _intercept_http_response(url, response, 'POST', latency_ms)
                return response

            # Wrap request method
            @functools.wraps(requests.request)
            def tracked_request(method, url, *args, **kwargs):
                start_time = time.time()
                response = requests._original_request(method, url, *args, **kwargs)
                latency_ms = (time.time() - start_time) * 1000
                if method.upper() == 'POST':
                    _intercept_http_response(url, response, method, latency_ms)
                return response

            # Replace methods
            requests.post = tracked_post
            requests.request = tracked_request

            logger.info("Requests interceptor installed")
    except ImportError:
        logger.debug("Requests library not found, skipping interceptor")
    except Exception as e:
        logger.error(f"Failed to install requests interceptor: {e}")


def _install_httpx_interceptor():
    """Install interceptor for httpx library"""
    try:
        import httpx
        
        # Store original methods
        if not hasattr(httpx, '_original_post'):
            httpx._original_post = httpx.post
            httpx._original_request = httpx.request
            
            # Wrap post method
            @functools.wraps(httpx.post)
            async def tracked_post_async(url, *args, **kwargs):
                response = await httpx._original_post(url, *args, **kwargs)
                _intercept_http_response(str(url), response, 'POST')
                return response
            
            def tracked_post_sync(url, *args, **kwargs):
                response = httpx._original_post(url, *args, **kwargs)
                _intercept_http_response(str(url), response, 'POST')
                return response
            
            # Wrap request method
            @functools.wraps(httpx.request)
            async def tracked_request_async(method, url, *args, **kwargs):
                response = await httpx._original_request(method, url, *args, **kwargs)
                if method.upper() == 'POST':
                    _intercept_http_response(str(url), response, method)
                return response
            
            def tracked_request_sync(method, url, *args, **kwargs):
                response = httpx._original_request(method, url, *args, **kwargs)
                if method.upper() == 'POST':
                    _intercept_http_response(str(url), response, method)
                return response
            
            # Replace methods (detect sync/async)
            try:
                import inspect
                if inspect.iscoroutinefunction(httpx.post):
                    httpx.post = tracked_post_async
                    httpx.request = tracked_request_async
                else:
                    httpx.post = tracked_post_sync
                    httpx.request = tracked_request_sync
            except:
                httpx.post = tracked_post_sync
                httpx.request = tracked_request_sync
            
            logger.info("HTTPX interceptor installed")
    except ImportError:
        logger.debug("HTTPX library not found, skipping interceptor")
    except Exception as e:
        logger.error(f"Failed to install httpx interceptor: {e}")


def _normalize_costs(costs: Any) -> Dict[str, float]:
    if isinstance(costs, dict):
        return costs
    return {"input_cost": 0.0, "output_cost": 0.0, "total_cost": 0.0}


def _wrap_streaming_response(response: Any, provider: str, model: str, start_time: float, tracker):
    """Wrap a streaming response to track tokens on completion."""
    collected_content = []
    completion_tokens = 0
    prompt_tokens = 0

    for chunk in response:
        # Yield chunk through to caller
        yield chunk

        # Accumulate for tracking
        if hasattr(chunk, 'choices') and chunk.choices:
            delta = chunk.choices[0].delta
            if hasattr(delta, 'content') and delta.content:
                collected_content.append(delta.content)

        # Check for usage in final chunk (OpenAI includes this)
        if hasattr(chunk, 'usage') and chunk.usage:
            prompt_tokens = chunk.usage.prompt_tokens or 0
            completion_tokens = chunk.usage.completion_tokens or 0

    # After stream completes, estimate if no usage provided
    if completion_tokens == 0 and collected_content:
        # Rough estimate: ~4 chars per token
        completion_tokens = len(''.join(collected_content)) // 4

    latency_ms = (time.time() - start_time) * 1000
    costs = _normalize_costs(
        tracker.calculate_costs(provider, model, prompt_tokens, completion_tokens)
    )
    tracker.track_call(
        provider=provider,
        model=model,
        usage_data={
            'input_tokens': prompt_tokens,
            'output_tokens': completion_tokens,
            **costs,
        },
        metadata={'streamed': True, 'latency_ms': latency_ms},
    )


def _track_openai_response(response: Any, model: str, latency_ms: float = None, provider: str = 'openai'):
    """Track OpenAI / Azure OpenAI API response.

    `provider` comes from _detect_openai_provider(self) on the call site. It
    decides which label goes to the backend (so Azure OpenAI spend doesn't
    silently aggregate as vanilla OpenAI) AND which price table
    calculate_costs() uses — Azure's list price differs from OpenAI direct.
    """
    if not _tracking_enabled or not _tracker:
        return

    try:
        # Handle both dict and object responses
        if isinstance(response, dict):
            usage = response.get('usage', {})
            response_id = response.get('id')
            model = response.get('model', model)
        else:
            usage = getattr(response, 'usage', {})
            response_id = getattr(response, 'id', None)
            model = getattr(response, 'model', model)

        if not usage:
            return

        # Extract token counts
        if isinstance(usage, dict):
            input_tokens = usage.get('prompt_tokens', 0)
            output_tokens = usage.get('completion_tokens', 0)
        else:
            input_tokens = getattr(usage, 'prompt_tokens', 0)
            output_tokens = getattr(usage, 'completion_tokens', 0)

        if input_tokens == 0 and output_tokens == 0:
            return

        # Calculate costs
        costs = _tracker.calculate_costs(provider, model, input_tokens, output_tokens)

        # Track the call
        _tracker.track_call(
            provider=provider,
            model=model,
            usage_data={
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                **costs
            },
            request_id=response_id,
            metadata={
                'intercepted': True,
                'method': 'middleware',
                'latency_ms': round(latency_ms, 2) if latency_ms else None
            }
        )
    except Exception as e:
        logger.debug(f"Failed to track OpenAI response: {e}")


def _wrap_anthropic_streaming_response(response: Any, model: str, start_time: float, tracker):
    """Wrap Anthropic streaming response to track tokens on completion."""
    collected_content = []
    completion_tokens = 0
    prompt_tokens = 0

    # Anthropic uses MessageStream which wraps events
    for event in response:
        # Yield event through to caller
        yield event

        # Accumulate content from content_block_delta events
        if hasattr(event, 'type') and event.type == 'content_block_delta':
            if hasattr(event, 'delta') and hasattr(event.delta, 'text'):
                collected_content.append(event.delta.text)

        # Extract usage from message_delta event
        if hasattr(event, 'type') and event.type == 'message_delta':
            if hasattr(event, 'usage'):
                completion_tokens = event.usage.output_tokens or 0

        # Or get it from the final message event
        if hasattr(event, 'type') and event.type == 'message_start':
            if hasattr(event, 'message') and hasattr(event.message, 'usage'):
                prompt_tokens = event.message.usage.input_tokens or 0

    # After stream completes, estimate if no usage provided
    if completion_tokens == 0 and collected_content:
        # Rough estimate: ~4 chars per token
        completion_tokens = len(''.join(collected_content)) // 4

    latency_ms = (time.time() - start_time) * 1000
    costs = _normalize_costs(
        tracker.calculate_costs("anthropic", model, prompt_tokens, completion_tokens)
    )
    tracker.track_call(
        provider='anthropic',
        model=model,
        usage_data={
            'input_tokens': prompt_tokens,
            'output_tokens': completion_tokens,
            **costs,
        },
        metadata={'streamed': True, 'latency_ms': latency_ms},
    )


def _track_anthropic_response(response: Any, model: str, latency_ms: float = None):
    """Track Anthropic API response"""
    if not _tracking_enabled or not _tracker:
        return

    try:
        # Handle both dict and object responses
        if isinstance(response, dict):
            usage = response.get('usage', {})
            response_id = response.get('id')
            model = response.get('model', model)
        else:
            usage = getattr(response, 'usage', {})
            response_id = getattr(response, 'id', None)
            model = getattr(response, 'model', model)

        if not usage:
            return

        # Extract token counts
        if isinstance(usage, dict):
            input_tokens = usage.get('input_tokens', 0)
            output_tokens = usage.get('output_tokens', 0)
        else:
            input_tokens = getattr(usage, 'input_tokens', 0)
            output_tokens = getattr(usage, 'output_tokens', 0)

        if input_tokens == 0 and output_tokens == 0:
            return

        # Calculate costs
        costs = _tracker.calculate_costs('anthropic', model, input_tokens, output_tokens)

        # Track the call
        _tracker.track_call(
            provider='anthropic',
            model=model,
            usage_data={
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                **costs
            },
            request_id=response_id,
            metadata={
                'intercepted': True,
                'method': 'middleware',
                'latency_ms': round(latency_ms, 2) if latency_ms else None
            }
        )
    except Exception as e:
        logger.debug(f"Failed to track Anthropic response: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# AWS Bedrock — via botocore event hooks
# ═══════════════════════════════════════════════════════════════════════════
#
# Why a dedicated interceptor (vs. the HTTP-level one below): boto3 uses
# urllib3 directly through botocore, NOT the `requests` or `httpx` libraries
# we monkey-patch for generic HTTP. So Bedrock calls slip past the HTTP
# hooks. The botocore event system is the sanctioned extension point.
#
# Two-phase hook per operation:
#   before-call  stashes modelId + start timestamp in the operation context.
#                Needed because `modelId` is in the REQUEST params, and the
#                after-call event only sees the response.
#   after-call   reads the parsed response, dispatches on model family,
#                feeds tokens to the tracker.
#
# Covered operations:
#   InvokeModel                       per-provider response shapes
#   Converse                          unified usage field
#   ConverseStream                    streaming — terminal metadata chunk
#                                     carries usage
#
# NOT YET covered: InvokeModelWithResponseStream. Each model family emits
# differently-shaped chunks and cumulative counts aren't always in the
# terminal chunk. Customers on streaming should migrate to ConverseStream
# (AWS is nudging everyone that way anyway).


def _install_bedrock_interceptor():
    """Install interceptors for AWS Bedrock via botocore event hooks."""
    try:
        import boto3  # noqa: F401 — probe import
    except ImportError:
        logger.debug("boto3 not installed, skipping Bedrock interceptor")
        return

    try:
        _bedrock_register_on_default_session()
        _bedrock_patch_session_init()
        logger.info("Bedrock interceptor installed (botocore event hooks)")
    except Exception as e:
        logger.error(f"Failed to install Bedrock interceptor: {e}")


def _bedrock_register_on_default_session():
    """Register handlers on the default boto3 session. Covers `boto3.client(...)` callers."""
    import boto3

    # Force default-session creation so subsequent `boto3.client(...)` uses it.
    if boto3.DEFAULT_SESSION is None:
        boto3.setup_default_session()
    _bedrock_register_handlers(boto3.DEFAULT_SESSION.events)


def _bedrock_patch_session_init():
    """Patch boto3.Session.__init__ so every new Session gets our handlers too."""
    import boto3.session

    if getattr(boto3.session.Session.__init__, '_gc_bedrock_patched', False):
        return  # idempotent

    _original_init = boto3.session.Session.__init__

    @functools.wraps(_original_init)
    def _patched_init(self, *args, **kwargs):
        _original_init(self, *args, **kwargs)
        try:
            _bedrock_register_handlers(self.events)
        except Exception as e:
            logger.debug(f"Failed to register bedrock handlers on new session: {e}")

    _patched_init._gc_bedrock_patched = True
    boto3.session.Session.__init__ = _patched_init


def _bedrock_register_handlers(events):
    """Register all four bedrock-runtime event handlers on a given session's event system."""
    # before-call: stash the modelId + start ts in the context dict.
    events.register(
        'before-call.bedrock-runtime.InvokeModel', _bedrock_before_call
    )
    events.register(
        'before-call.bedrock-runtime.InvokeModelWithResponseStream', _bedrock_before_call
    )
    events.register(
        'before-call.bedrock-runtime.Converse', _bedrock_before_call
    )
    events.register(
        'before-call.bedrock-runtime.ConverseStream', _bedrock_before_call
    )
    # after-call: read response, dispatch on model family, track tokens.
    events.register(
        'after-call.bedrock-runtime.InvokeModel', _bedrock_after_invoke_model
    )
    events.register(
        'after-call.bedrock-runtime.Converse', _bedrock_after_converse
    )
    events.register(
        'after-call.bedrock-runtime.ConverseStream', _bedrock_after_converse_stream
    )


def _bedrock_before_call(params, context, **_):
    """Stash modelId + start timestamp for correlation with after-call."""
    try:
        context['_gc_bedrock_model_id'] = params.get('modelId') or 'unknown'
        context['_gc_bedrock_start'] = time.time()
    except Exception:
        pass  # never let tracing break the real call


def _bedrock_after_invoke_model(http_response, parsed, context, **_):
    """InvokeModel — response body is a StreamingBody with JSON; tee + parse."""
    if not _tracking_enabled or not _tracker or not parsed:
        return
    try:
        body_bytes = _bedrock_tee_streaming_body(parsed)
        if not body_bytes:
            return
        response_json = json.loads(body_bytes)
        model_id = context.get('_gc_bedrock_model_id', 'unknown')
        input_tokens, output_tokens = _bedrock_extract_tokens(model_id, response_json)
        if input_tokens or output_tokens:
            _bedrock_record(model_id, input_tokens, output_tokens, context)
    except Exception as e:
        logger.debug(f"bedrock InvokeModel track failed: {e}")


def _bedrock_after_converse(http_response, parsed, context, **_):
    """Converse — unified response.usage across all providers."""
    if not _tracking_enabled or not _tracker or not parsed:
        return
    try:
        usage = parsed.get('usage') or {}
        input_tokens = int(usage.get('inputTokens', 0))
        output_tokens = int(usage.get('outputTokens', 0))
        if input_tokens or output_tokens:
            model_id = context.get('_gc_bedrock_model_id', 'unknown')
            _bedrock_record(model_id, input_tokens, output_tokens, context)
    except Exception as e:
        logger.debug(f"bedrock Converse track failed: {e}")


def _bedrock_after_converse_stream(http_response, parsed, context, **_):
    """ConverseStream — wrap the EventStream so the terminal metadata chunk triggers tracking."""
    if not _tracking_enabled or not _tracker or not parsed:
        return
    try:
        stream = parsed.get('stream')
        if stream is None:
            return
        model_id = context.get('_gc_bedrock_model_id', 'unknown')
        parsed['stream'] = _bedrock_wrap_stream(stream, model_id, context)
    except Exception as e:
        logger.debug(f"bedrock ConverseStream wrap failed: {e}")


def _bedrock_tee_streaming_body(parsed):
    """Read the StreamingBody once, re-wrap it so the caller can still read.

    Bedrock InvokeModel's body is single-use — if we drain it here, customer
    code gets empty bytes. Solution: read → cache → replace with a fresh
    StreamingBody backed by BytesIO.
    """
    body = parsed.get('body')
    if body is None:
        return None
    try:
        import io
        from botocore.response import StreamingBody
        data = body.read()
        parsed['body'] = StreamingBody(io.BytesIO(data), len(data))
        return data
    except Exception as e:
        logger.debug(f"bedrock tee failed: {e}")
        return None


def _bedrock_extract_tokens(model_id, response):
    """Dispatch on Bedrock model family; return (input_tokens, output_tokens).

    Unknown families return (0, 0) — caller will skip tracking. Customers on
    obscure models should migrate to the Converse API which has uniform
    usage (handled separately by _bedrock_after_converse).
    """
    mid = (model_id or '').lower()

    # ── Anthropic Claude (claude-v2, claude-3-*, claude-3-5-*, claude-3-7-*)
    if 'anthropic.claude' in mid:
        usage = response.get('usage') or {}
        return (
            int(usage.get('input_tokens', 0)),
            int(usage.get('output_tokens', 0)),
        )

    # ── Amazon Titan (Text, Embeddings)
    if mid.startswith('amazon.titan'):
        inp = int(response.get('inputTextTokenCount', 0))
        out_results = response.get('results') or []
        out = sum(int(r.get('tokenCount', 0)) for r in out_results)
        return inp, out

    # ── Meta Llama (llama2, llama3, llama3-1, llama3-2, llama3-3)
    if mid.startswith('meta.llama'):
        return (
            int(response.get('prompt_token_count', 0)),
            int(response.get('generation_token_count', 0)),
        )

    # ── Cohere Command-R / Command-R-Plus — billed_units in meta
    if mid.startswith('cohere.command-r'):
        meta = response.get('meta') or {}
        billed = meta.get('billed_units') or meta.get('tokens') or {}
        return (
            int(billed.get('input_tokens', 0)),
            int(billed.get('output_tokens', 0)),
        )

    # ── AI21 Jamba (Jurassic legacy is more complex; Jamba follows OpenAI-ish shape)
    if mid.startswith('ai21.jamba'):
        usage = response.get('usage') or {}
        return (
            int(usage.get('prompt_tokens', 0)),
            int(usage.get('completion_tokens', 0)),
        )

    # Unknown model family — no clean way to extract. Recommend Converse.
    return 0, 0


def _bedrock_wrap_stream(stream, model_id, context):
    """Wrap an EventStream so iteration end triggers token tracking.

    ConverseStream emits a terminal event with a `metadata` key containing
    usage. We yield through every event unchanged, harvest usage from the
    metadata event, and fire the tracker when iteration finishes.
    """
    def _gen():
        input_tokens = 0
        output_tokens = 0
        try:
            for event in stream:
                yield event
                # Terminal event has shape {'metadata': {'usage': {...}, ...}}
                if isinstance(event, dict) and 'metadata' in event:
                    usage = (event['metadata'] or {}).get('usage') or {}
                    input_tokens = int(usage.get('inputTokens', 0))
                    output_tokens = int(usage.get('outputTokens', 0))
        finally:
            if input_tokens or output_tokens:
                _bedrock_record(model_id, input_tokens, output_tokens, context)

    return _gen()


def _bedrock_record(model_id, input_tokens, output_tokens, context):
    """Compute cost + forward to the tracker. Never raise — tracing must never break callers."""
    try:
        start = context.get('_gc_bedrock_start')
        latency_ms = (time.time() - start) * 1000 if start else None
        costs = _tracker.calculate_costs('bedrock', model_id, input_tokens, output_tokens)
        _tracker.track_call(
            provider='bedrock',
            model=model_id,
            usage_data={
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                **costs,
            },
            metadata={
                'intercepted': True,
                'method': 'botocore-event',
                'latency_ms': round(latency_ms, 2) if latency_ms else None,
            },
        )
    except Exception as e:
        logger.debug(f"bedrock tracker record failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# Google Vertex AI / Gemini — via SDK method patching
# ═══════════════════════════════════════════════════════════════════════════
#
# Unlike Bedrock, Google's SDKs are patchable at the method level — the
# response shape is uniform (`response.usage_metadata.{prompt_token_count,
# candidates_token_count, total_token_count}`) regardless of model. So we
# don't need a per-model dispatcher; one wrapper handles every Gemini
# model from 1.0 Pro to whatever ships next.
#
# Two SDKs we care about:
#   1. vertexai.generative_models (classic Vertex SDK, most deployments)
#   2. google.genai                (unified Google GenAI SDK, 2024+)
#
# A third, google.generativeai, hits the direct Gemini API (not Vertex).
# Same response shape but different billing — we could add it in a follow-
# up with ~15 LOC; deferred since most of our customers are on Vertex.
#
# Provider labels:
#   vertex_ai        - Vertex-hosted (classic SDK, or unified SDK w/ vertexai=True)
#   google_gemini    - direct Gemini API (unified SDK w/ vertexai=False; reserved)
#
# Labels are distinct because billing + pricing differ between Vertex and
# the direct API, and customers often compare the two.


def _install_vertex_ai_interceptor():
    """Install interceptors for Google Vertex AI / Gemini SDK families."""
    _install_vertexai_classic_interceptor()
    _install_google_genai_interceptor()


def _install_vertexai_classic_interceptor():
    """Patch the classic `vertexai.generative_models.GenerativeModel` entry points."""
    try:
        from vertexai.generative_models import GenerativeModel
    except ImportError:
        logger.debug("vertexai.generative_models not installed, skipping classic Vertex interceptor")
        return
    except Exception as e:
        logger.debug(f"vertexai import errored ({e}); skipping classic Vertex interceptor")
        return

    try:
        # Sync: model.generate_content(...)
        original_sync = GenerativeModel.generate_content
        if not getattr(original_sync, '_gpu_dashboard_wrapped', False):
            @functools.wraps(original_sync)
            def tracked_sync(self, *args, **kwargs):
                start_time = time.time()
                response = original_sync(self, *args, **kwargs)
                model_name = _vertex_model_name(self)
                if kwargs.get('stream'):
                    response = _wrap_vertex_stream(response, model_name, 'vertex_ai', start_time)
                else:
                    latency_ms = (time.time() - start_time) * 1000
                    _track_vertex_response(response, model_name, 'vertex_ai', latency_ms)
                return response
            tracked_sync._gpu_dashboard_wrapped = True
            GenerativeModel.generate_content = tracked_sync
            logger.info("Vertex AI classic SDK interceptor installed (GenerativeModel.generate_content)")

        # Async: model.generate_content_async(...)
        original_async = getattr(GenerativeModel, 'generate_content_async', None)
        if original_async and not getattr(original_async, '_gpu_dashboard_wrapped', False):
            @functools.wraps(original_async)
            async def tracked_async(self, *args, **kwargs):
                start_time = time.time()
                response = await original_async(self, *args, **kwargs)
                model_name = _vertex_model_name(self)
                if kwargs.get('stream'):
                    response = _wrap_vertex_async_stream(response, model_name, 'vertex_ai', start_time)
                else:
                    latency_ms = (time.time() - start_time) * 1000
                    _track_vertex_response(response, model_name, 'vertex_ai', latency_ms)
                return response
            tracked_async._gpu_dashboard_wrapped = True
            GenerativeModel.generate_content_async = tracked_async
            logger.info("Vertex AI classic SDK async interceptor installed")

    except Exception as e:
        logger.warning(f"Could not install classic Vertex AI interceptor: {e}")


def _install_google_genai_interceptor():
    """Patch the unified google.genai SDK (Client().models.generate_content)."""
    try:
        from google.genai.models import Models
    except ImportError:
        logger.debug("google.genai not installed, skipping unified GenAI interceptor")
        return
    except Exception as e:
        logger.debug(f"google.genai import errored ({e}); skipping unified GenAI interceptor")
        return

    try:
        # Sync: client.models.generate_content(model=..., contents=...)
        original_sync = Models.generate_content
        if not getattr(original_sync, '_gpu_dashboard_wrapped', False):
            @functools.wraps(original_sync)
            def tracked_sync(self, *args, **kwargs):
                start_time = time.time()
                response = original_sync(self, *args, **kwargs)
                model_name = kwargs.get('model', 'unknown')
                provider = _google_genai_provider(self)
                latency_ms = (time.time() - start_time) * 1000
                _track_vertex_response(response, model_name, provider, latency_ms)
                return response
            tracked_sync._gpu_dashboard_wrapped = True
            Models.generate_content = tracked_sync
            logger.info("Google GenAI unified SDK interceptor installed (Models.generate_content)")

        # Streaming: client.models.generate_content_stream(...)
        original_stream = getattr(Models, 'generate_content_stream', None)
        if original_stream and not getattr(original_stream, '_gpu_dashboard_wrapped', False):
            @functools.wraps(original_stream)
            def tracked_stream(self, *args, **kwargs):
                start_time = time.time()
                stream = original_stream(self, *args, **kwargs)
                model_name = kwargs.get('model', 'unknown')
                provider = _google_genai_provider(self)
                return _wrap_vertex_stream(stream, model_name, provider, start_time)
            tracked_stream._gpu_dashboard_wrapped = True
            Models.generate_content_stream = tracked_stream
            logger.info("Google GenAI unified SDK streaming interceptor installed")

        # Async: client.aio.models.generate_content(...)
        try:
            from google.genai.models import AsyncModels
            original_async = AsyncModels.generate_content
            if not getattr(original_async, '_gpu_dashboard_wrapped', False):
                @functools.wraps(original_async)
                async def tracked_async(self, *args, **kwargs):
                    start_time = time.time()
                    response = await original_async(self, *args, **kwargs)
                    model_name = kwargs.get('model', 'unknown')
                    provider = _google_genai_provider(self)
                    latency_ms = (time.time() - start_time) * 1000
                    _track_vertex_response(response, model_name, provider, latency_ms)
                    return response
                tracked_async._gpu_dashboard_wrapped = True
                AsyncModels.generate_content = tracked_async
                logger.info("Google GenAI unified SDK async interceptor installed")
        except (ImportError, AttributeError):
            pass  # older google.genai without AsyncModels; not fatal

    except Exception as e:
        logger.warning(f"Could not install unified Google GenAI interceptor: {e}")


def _vertex_model_name(generative_model_instance):
    """Pull the model name from a vertexai.generative_models.GenerativeModel.

    The attribute has moved between SDK versions:
      - newer: ._model_name (private) or .model_name (property)
      - older: ._prediction_resource_name (parseable)
    Fall back to 'unknown' rather than raising — tracing must never break callers.
    """
    try:
        for attr in ('_model_name', 'model_name', '_prediction_resource_name'):
            v = getattr(generative_model_instance, attr, None)
            if v:
                s = str(v)
                # Resource paths look like projects/.../publishers/google/models/gemini-1.5-pro.
                # Keep just the last segment so dashboards aggregate cleanly.
                return s.rsplit('/', 1)[-1]
    except Exception:
        pass
    return 'unknown'


def _google_genai_provider(models_instance):
    """Decide 'vertex_ai' vs 'google_gemini' for a google.genai.models.Models instance.

    google.genai.Client(vertexai=True, ...) wires into Vertex; without
    vertexai=True it targets the direct Gemini API (different billing,
    different auth). The flag ends up on the underlying api_client.
    """
    try:
        api_client = getattr(models_instance, '_api_client', None)
        if api_client is None:
            return 'vertex_ai'  # conservative default — most customers here are on Vertex
        if getattr(api_client, 'vertexai', False):
            return 'vertex_ai'
        return 'google_gemini'
    except Exception:
        return 'vertex_ai'


def _track_vertex_response(response, model, provider, latency_ms=None):
    """Record a single Gemini response's token usage + cost.

    Uniform shape across all Gemini models (1.0, 1.5, 2.0):
        response.usage_metadata.prompt_token_count     -> input tokens
        response.usage_metadata.candidates_token_count -> output tokens
    Dict responses (rare — some wrappers unwrap to dict) also supported.
    """
    if not _tracking_enabled or not _tracker or response is None:
        return
    try:
        usage = getattr(response, 'usage_metadata', None)
        if usage is None and isinstance(response, dict):
            usage = response.get('usage_metadata') or response.get('usageMetadata')
        if not usage:
            return
        if isinstance(usage, dict):
            input_tokens = int(usage.get('prompt_token_count') or usage.get('promptTokenCount') or 0)
            output_tokens = int(usage.get('candidates_token_count') or usage.get('candidatesTokenCount') or 0)
        else:
            input_tokens = int(getattr(usage, 'prompt_token_count', 0) or 0)
            output_tokens = int(getattr(usage, 'candidates_token_count', 0) or 0)
        if input_tokens == 0 and output_tokens == 0:
            return
        costs = _tracker.calculate_costs(provider, model, input_tokens, output_tokens)
        _tracker.track_call(
            provider=provider,
            model=model,
            usage_data={
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                **costs,
            },
            metadata={
                'intercepted': True,
                'method': 'middleware',
                'latency_ms': round(latency_ms, 2) if latency_ms else None,
            }
        )
    except Exception as e:
        logger.debug(f"Failed to track Vertex response: {e}")


def _wrap_vertex_stream(stream, model, provider, start_time):
    """Wrap a sync Vertex streaming response — track usage from the last chunk."""
    def _gen():
        last_chunk_with_usage = None
        try:
            for chunk in stream:
                yield chunk
                # Usage metadata typically appears on the final chunk only;
                # we conservatively overwrite each time so we always have
                # the latest.
                if getattr(chunk, 'usage_metadata', None):
                    last_chunk_with_usage = chunk
        finally:
            if last_chunk_with_usage is not None:
                latency_ms = (time.time() - start_time) * 1000
                _track_vertex_response(last_chunk_with_usage, model, provider, latency_ms)
    return _gen()


def _wrap_vertex_async_stream(stream, model, provider, start_time):
    """Wrap an async Vertex streaming response — same semantics as sync."""
    async def _agen():
        last_chunk_with_usage = None
        try:
            async for chunk in stream:
                yield chunk
                if getattr(chunk, 'usage_metadata', None):
                    last_chunk_with_usage = chunk
        finally:
            if last_chunk_with_usage is not None:
                latency_ms = (time.time() - start_time) * 1000
                _track_vertex_response(last_chunk_with_usage, model, provider, latency_ms)
    return _agen()


def _intercept_http_response(url: str, response: Any, method: str, latency_ms: float = None):
    """Intercept HTTP responses and detect LLM API calls"""
    if not _tracking_enabled or not _tracker:
        return

    try:
        url_str = str(url).lower()

        # Detect OpenAI API calls
        if 'api.openai.com' in url_str or 'openai.azure.com' in url_str:
            _intercept_openai_http(url, response, latency_ms)

        # Detect Anthropic API calls
        elif 'api.anthropic.com' in url_str:
            _intercept_anthropic_http(url, response, latency_ms)

        # Detect AWS Bedrock calls
        elif 'bedrock' in url_str and 'amazonaws.com' in url_str:
            _intercept_bedrock_http(url, response, latency_ms)

        # Detect Azure OpenAI calls
        elif 'openai.azure.com' in url_str:
            _intercept_azure_openai_http(url, response, latency_ms)

    except Exception as e:
        logger.debug(f"Failed to intercept HTTP response: {e}")


def _intercept_openai_http(url: str, response: Any, latency_ms: float = None):
    """Intercept OpenAI HTTP response"""
    try:
        if response.status_code != 200:
            return

        # Parse response
        if hasattr(response, 'json'):
            data = response.json()
        elif hasattr(response, 'text'):
            data = json.loads(response.text)
        else:
            return

        if 'usage' not in data:
            return

        usage = data.get('usage', {})
        input_tokens = usage.get('prompt_tokens', 0)
        output_tokens = usage.get('completion_tokens', 0)
        model = data.get('model', 'unknown')
        response_id = data.get('id')

        if input_tokens == 0 and output_tokens == 0:
            return

        # Determine provider
        provider = 'azure_openai' if 'openai.azure.com' in str(url).lower() else 'openai'

        # Calculate costs
        costs = _tracker.calculate_costs(provider, model, input_tokens, output_tokens)

        # Track the call
        _tracker.track_call(
            provider=provider,
            model=model,
            usage_data={
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                **costs
            },
            request_id=response_id,
            metadata={
                'intercepted': True,
                'method': 'http_interceptor',
                'url': str(url),
                'latency_ms': round(latency_ms, 2) if latency_ms else None
            }
        )
    except Exception as e:
        logger.debug(f"Failed to intercept OpenAI HTTP: {e}")


def _intercept_anthropic_http(url: str, response: Any, latency_ms: float = None):
    """Intercept Anthropic HTTP response"""
    try:
        if response.status_code != 200:
            return

        # Parse response
        if hasattr(response, 'json'):
            data = response.json()
        elif hasattr(response, 'text'):
            data = json.loads(response.text)
        else:
            return

        if 'usage' not in data:
            return

        usage = data.get('usage', {})
        input_tokens = usage.get('input_tokens', 0)
        output_tokens = usage.get('output_tokens', 0)
        model = data.get('model', 'unknown')
        response_id = data.get('id')

        if input_tokens == 0 and output_tokens == 0:
            return

        # Calculate costs
        costs = _tracker.calculate_costs('anthropic', model, input_tokens, output_tokens)

        # Track the call
        _tracker.track_call(
            provider='anthropic',
            model=model,
            usage_data={
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                **costs
            },
            request_id=response_id,
            metadata={
                'intercepted': True,
                'method': 'http_interceptor',
                'url': str(url),
                'latency_ms': round(latency_ms, 2) if latency_ms else None
            }
        )
    except Exception as e:
        logger.debug(f"Failed to intercept Anthropic HTTP: {e}")


def _intercept_bedrock_http(url: str, response: Any, latency_ms: float = None):
    """Intercept AWS Bedrock HTTP response"""
    try:
        if response.status_code != 200:
            return

        # Parse response
        if hasattr(response, 'json'):
            data = response.json()
        elif hasattr(response, 'text'):
            data = json.loads(response.text)
        else:
            return

        # Bedrock response format varies by model
        # Extract usage from response
        usage = data.get('usage', {}) or data.get('OutputUsage', {})

        if not usage:
            return

        input_tokens = usage.get('input_tokens', 0) or usage.get('InputTokens', 0)
        output_tokens = usage.get('output_tokens', 0) or usage.get('OutputTokens', 0)

        # Try to extract model from URL or response
        model = data.get('model', 'unknown')
        if model == 'unknown':
            # Extract from URL path
            url_parts = str(url).split('/')
            for part in url_parts:
                if 'claude' in part.lower() or 'llama' in part.lower():
                    model = part
                    break

        if input_tokens == 0 and output_tokens == 0:
            return

        # Calculate costs
        costs = _tracker.calculate_costs('aws_bedrock', model, input_tokens, output_tokens)

        # Track the call
        _tracker.track_call(
            provider='aws_bedrock',
            model=model,
            usage_data={
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                **costs
            },
            metadata={
                'intercepted': True,
                'method': 'http_interceptor',
                'url': str(url),
                'latency_ms': round(latency_ms, 2) if latency_ms else None
            }
        )
    except Exception as e:
        logger.debug(f"Failed to intercept Bedrock HTTP: {e}")


def _intercept_azure_openai_http(url: str, response: Any, latency_ms: float = None):
    """Intercept Azure OpenAI HTTP response"""
    # Azure OpenAI uses same format as OpenAI
    _intercept_openai_http(url, response, latency_ms)


# Auto-enable if environment variable is set
if os.getenv('LLM_TRACKING_AUTO_ENABLE', 'false').lower() == 'true':
    enable_llm_tracking()
