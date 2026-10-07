"""
LLM Tracker - Application Integration Library
This module provides easy integration for tracking LLM token usage in your applications.

Usage:
    # For OpenAI
    from integrations.llm_tracker import track_openai_call
    
    response = openai.ChatCompletion.create(...)
    track_openai_call(response, workload_id="my-workload", project_id="project-1")
    
    # For Anthropic
    from integrations.llm_tracker import track_anthropic_call
    
    response = anthropic.messages.create(...)
    track_anthropic_call(response, workload_id="my-workload")
"""

import os
import logging
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class LLMTracker:
    """In-process LLM usage helper — does not egress to the TensorCost backend."""

    FEATURE_HEADER = 'X-TensorCost-Feature'
    FEATURE_VERSION_HEADER = 'X-TensorCost-Feature-Version'

    def __init__(
        self,
        backend_api_url: Optional[str] = None,
        backend_api_key: Optional[str] = None,
        tenant_id: Optional[str] = None,
        default_feature_tag: Optional[str] = None,
        version_resolver_fn=None,
    ):
        self.backend_api_url = backend_api_url or os.getenv('BACKEND_API_URL')
        self.backend_api_key = backend_api_key or os.getenv('BACKEND_API_KEY')
        self.tenant_id = tenant_id or os.getenv('TENANT_ID', 'default')
        self.enabled = os.getenv('LLM_TRACKING_ENABLED', 'true').lower() == 'true'
        # Fallback used when a call site doesn't tag anything — a team can set
        # default_feature_tag="search" on init so every call is attributed.
        self.default_feature_tag = default_feature_tag or os.getenv('LLM_DEFAULT_FEATURE')
        # Callable that, given the current call context, returns the feature
        # version (e.g. reads the git SHA at deploy time). Invoked lazily per
        # call to avoid baking build-time state into a long-running process.
        self.version_resolver_fn = version_resolver_fn

    def _pick_header(self, headers_in: Optional[Dict[str, str]], name: str) -> Optional[str]:
        if not headers_in:
            return None
        # HTTP headers are case-insensitive; do a tolerant lookup.
        lowered = {k.lower(): v for k, v in headers_in.items()}
        return lowered.get(name.lower())

    def _resolve_feature_tag(self, feature, headers_in) -> Optional[str]:
        if feature:
            return feature
        from_header = self._pick_header(headers_in, self.FEATURE_HEADER)
        if from_header:
            return from_header
        env_feature = os.getenv('LLM_FEATURE')
        if env_feature:
            return env_feature
        return self.default_feature_tag

    def _resolve_feature_version(self, feature_version, headers_in) -> Optional[str]:
        if feature_version:
            return feature_version
        from_header = self._pick_header(headers_in, self.FEATURE_VERSION_HEADER)
        if from_header:
            return from_header
        env_version = os.getenv('LLM_FEATURE_VERSION')
        if env_version:
            return env_version
        if callable(self.version_resolver_fn):
            try:
                return self.version_resolver_fn()
            except Exception:
                return None
        return None
    
    def track_call(
        self,
        provider: str,
        model: str,
        usage_data: Dict[str, Any],
        workload_id: Optional[str] = None,
        project_id: Optional[str] = None,
        user_id: Optional[str] = None,
        request_id: Optional[str] = None,
        session_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        feature: Optional[str] = None,
        feature_version: Optional[str] = None,
        headers_in: Optional[Dict[str, str]] = None,
    ) -> bool:
        """
        Track an LLM API call

        Args:
            provider: LLM provider (openai, anthropic, aws_bedrock, azure_openai)
            model: Model name (gpt-4, claude-3-opus, etc.)
            usage_data: Token usage data with keys: input_tokens, output_tokens, input_cost, output_cost, total_cost
            workload_id: Optional workload ID
            project_id: Optional project ID
            user_id: Optional user ID
            request_id: Optional request ID
            session_id: Optional session ID
            conversation_id: Optional conversation ID
            metadata: Optional additional metadata
            feature: Application feature tag (e.g. "search", "chatbot").
                If not set, falls back to X-tensorcost-Feature in headers_in,
                then to LLM_FEATURE env var, then to the default_feature
                configured on this tracker.
            feature_version: Version identifier for this feature (e.g. "v2").
                Same resolution order as `feature`.
            headers_in: Optional dict of inbound HTTP headers (from the calling
                service's request). Used to pluck X-tensorcost-Feature* so
                middleware-style instrumentation can tag LLM spend by feature
                without editing every call site.
        """
        if not self.enabled:
            return False

        resolved_feature = self._resolve_feature_tag(feature, headers_in)
        resolved_feature_version = self._resolve_feature_version(feature_version, headers_in)

        try:
            logger.debug(
                "LLM call recorded locally (no backend egress): %s/%s feature=%s tokens=%s+%s",
                provider,
                model,
                resolved_feature,
                usage_data.get('input_tokens', 0),
                usage_data.get('output_tokens', 0),
            )
            return True
        except Exception as e:
            logger.error(f"Failed to record LLM call locally: {e}")
            return False
    
    def calculate_costs(self, provider: str, model: str, input_tokens: int, output_tokens: int) -> Dict[str, float]:
        """Calculate costs based on provider pricing"""
        pricing = self._get_pricing(provider, model)
        
        input_cost = (input_tokens / 1000) * pricing.get('input_price_per_1k', 0)
        output_cost = (output_tokens / 1000) * pricing.get('output_price_per_1k', 0)
        total_cost = input_cost + output_cost
        
        return {
            'input_cost': round(input_cost, 6),
            'output_cost': round(output_cost, 6),
            'total_cost': round(total_cost, 6)
        }
    
    def _get_pricing(self, provider: str, model: str) -> Dict[str, float]:
        """
        Get pricing for provider/model combination.
        Checks environment variable overrides first, then falls back to built-in defaults.

        Override format: LLM_PRICING_{PROVIDER}_{MODEL}=input_per_1k,output_per_1k
        Example: LLM_PRICING_OPENAI_GPT4O=0.0025,0.01
        """
        # Check for env override
        env_key = f"LLM_PRICING_{provider.upper()}_{model.upper().replace('-', '_').replace('.', '_')}"
        env_val = os.getenv(env_key)
        if env_val:
            try:
                parts = env_val.split(',')
                return {
                    'input_price_per_1k': float(parts[0]),
                    'output_price_per_1k': float(parts[1]) if len(parts) > 1 else float(parts[0])
                }
            except (ValueError, IndexError):
                logger.warning(f"Invalid pricing override {env_key}={env_val}, using defaults")

        # Also check for a JSON config file
        config_path = os.getenv('LLM_PRICING_CONFIG')
        if config_path and os.path.exists(config_path):
            try:
                import json
                with open(config_path) as f:
                    config = json.load(f)
                pricing = config.get(provider, {}).get(model)
                if pricing:
                    return pricing
            except Exception:
                pass

        pricing_map = {
            'openai': {
                'gpt-4': {'input_price_per_1k': 0.03, 'output_price_per_1k': 0.06},
                'gpt-4-turbo': {'input_price_per_1k': 0.01, 'output_price_per_1k': 0.03},
                'gpt-4-turbo-preview': {'input_price_per_1k': 0.01, 'output_price_per_1k': 0.03},
                'gpt-4o': {'input_price_per_1k': 0.0025, 'output_price_per_1k': 0.01},
                'gpt-4o-mini': {'input_price_per_1k': 0.00015, 'output_price_per_1k': 0.0006},
                'gpt-3.5-turbo': {'input_price_per_1k': 0.0005, 'output_price_per_1k': 0.0015},
                'gpt-3.5-turbo-16k': {'input_price_per_1k': 0.003, 'output_price_per_1k': 0.004},
                'o1': {'input_price_per_1k': 0.015, 'output_price_per_1k': 0.06},
                'o1-mini': {'input_price_per_1k': 0.003, 'output_price_per_1k': 0.012},
                'o3-mini': {'input_price_per_1k': 0.0011, 'output_price_per_1k': 0.0044},
            },
            'anthropic': {
                'claude-3-opus': {'input_price_per_1k': 0.015, 'output_price_per_1k': 0.075},
                'claude-3-opus-20240229': {'input_price_per_1k': 0.015, 'output_price_per_1k': 0.075},
                'claude-3.5-sonnet': {'input_price_per_1k': 0.003, 'output_price_per_1k': 0.015},
                'claude-3-5-sonnet-20241022': {'input_price_per_1k': 0.003, 'output_price_per_1k': 0.015},
                'claude-3-sonnet': {'input_price_per_1k': 0.003, 'output_price_per_1k': 0.015},
                'claude-3.5-haiku': {'input_price_per_1k': 0.0008, 'output_price_per_1k': 0.004},
                'claude-3-haiku': {'input_price_per_1k': 0.00025, 'output_price_per_1k': 0.00125},
                'claude-3-haiku-20240307': {'input_price_per_1k': 0.00025, 'output_price_per_1k': 0.00125},
                'claude-4-opus': {'input_price_per_1k': 0.015, 'output_price_per_1k': 0.075},
                'claude-4-sonnet': {'input_price_per_1k': 0.003, 'output_price_per_1k': 0.015},
                'claude-2.1': {'input_price_per_1k': 0.008, 'output_price_per_1k': 0.024},
                'claude-2.0': {'input_price_per_1k': 0.008, 'output_price_per_1k': 0.024},
            },
            'aws_bedrock': {
                'anthropic.claude-3-opus': {'input_price_per_1k': 0.015, 'output_price_per_1k': 0.075},
                'anthropic.claude-3-sonnet': {'input_price_per_1k': 0.003, 'output_price_per_1k': 0.015},
                'anthropic.claude-3-haiku': {'input_price_per_1k': 0.00025, 'output_price_per_1k': 0.00125},
                'anthropic.claude-3-5-sonnet': {'input_price_per_1k': 0.003, 'output_price_per_1k': 0.015},
                'meta.llama3-70b-instruct': {'input_price_per_1k': 0.00265, 'output_price_per_1k': 0.0035},
                'meta.llama3-8b-instruct': {'input_price_per_1k': 0.0003, 'output_price_per_1k': 0.0006},
                'amazon.titan-text-express': {'input_price_per_1k': 0.0002, 'output_price_per_1k': 0.0006},
            },
            'azure_openai': {
                'gpt-4': {'input_price_per_1k': 0.03, 'output_price_per_1k': 0.06},
                'gpt-4o': {'input_price_per_1k': 0.0025, 'output_price_per_1k': 0.01},
                'gpt-4o-mini': {'input_price_per_1k': 0.00015, 'output_price_per_1k': 0.0006},
                'gpt-35-turbo': {'input_price_per_1k': 0.0005, 'output_price_per_1k': 0.0015},
            },
            'gcp_vertex': {
                'gemini-1.5-pro': {'input_price_per_1k': 0.00125, 'output_price_per_1k': 0.005},
                'gemini-1.5-flash': {'input_price_per_1k': 0.000075, 'output_price_per_1k': 0.0003},
                'gemini-2.0-flash': {'input_price_per_1k': 0.0001, 'output_price_per_1k': 0.0004},
            }
        }

        # Try exact match first, then partial match for model variants
        provider_pricing = pricing_map.get(provider, {})
        if model in provider_pricing:
            return provider_pricing[model]

        # Fuzzy match: try to find a base model match (e.g., 'gpt-4o-2024-05-13' matches 'gpt-4o')
        for known_model in sorted(provider_pricing.keys(), key=len, reverse=True):
            if model.startswith(known_model):
                return provider_pricing[known_model]

        return {'input_price_per_1k': 0, 'output_price_per_1k': 0}


# Global tracker instance
_tracker = None

def get_tracker() -> LLMTracker:
    """Get or create global tracker instance"""
    global _tracker
    if _tracker is None:
        _tracker = LLMTracker()
    return _tracker


# Convenience functions for common providers
def track_openai_call(
    response: Any,
    workload_id: Optional[str] = None,
    project_id: Optional[str] = None,
    user_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    feature: Optional[str] = None,
    feature_version: Optional[str] = None,
    headers_in: Optional[Dict[str, str]] = None,
) -> bool:
    """
    Track OpenAI API call
    
    Usage:
        import openai
        from integrations.llm_tracker import track_openai_call
        
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello"}]
        )
        track_openai_call(response, workload_id="my-workload")
    """
    tracker = get_tracker()
    
    # Extract usage from OpenAI response
    usage = response.get('usage', {}) if isinstance(response, dict) else getattr(response, 'usage', {})
    input_tokens = usage.get('prompt_tokens', 0) if isinstance(usage, dict) else getattr(usage, 'prompt_tokens', 0)
    output_tokens = usage.get('completion_tokens', 0) if isinstance(usage, dict) else getattr(usage, 'completion_tokens', 0)
    
    # Get model name
    model = response.get('model', 'unknown') if isinstance(response, dict) else getattr(response, 'model', 'unknown')
    
    # Calculate costs
    costs = tracker.calculate_costs('openai', model, input_tokens, output_tokens)
    
    # Get request ID
    request_id = response.get('id', None) if isinstance(response, dict) else getattr(response, 'id', None)
    
    return tracker.track_call(
        provider='openai',
        model=model,
        usage_data={
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            **costs
        },
        workload_id=workload_id,
        project_id=project_id,
        user_id=user_id,
        request_id=request_id,
        metadata=metadata,
        feature=feature,
        feature_version=feature_version,
        headers_in=headers_in,
    )


def track_anthropic_call(
    response: Any,
    workload_id: Optional[str] = None,
    project_id: Optional[str] = None,
    user_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    feature: Optional[str] = None,
    feature_version: Optional[str] = None,
    headers_in: Optional[Dict[str, str]] = None,
) -> bool:
    """
    Track Anthropic API call
    
    Usage:
        import anthropic
        from integrations.llm_tracker import track_anthropic_call
        
        client = anthropic.Anthropic()
        response = client.messages.create(
            model="claude-3-opus",
            messages=[{"role": "user", "content": "Hello"}]
        )
        track_anthropic_call(response, workload_id="my-workload")
    """
    tracker = get_tracker()
    
    # Extract usage from Anthropic response
    usage = response.get('usage', {}) if isinstance(response, dict) else getattr(response, 'usage', {})
    input_tokens = usage.get('input_tokens', 0) if isinstance(usage, dict) else getattr(usage, 'input_tokens', 0)
    output_tokens = usage.get('output_tokens', 0) if isinstance(usage, dict) else getattr(usage, 'output_tokens', 0)
    
    # Get model name
    model = response.get('model', 'unknown') if isinstance(response, dict) else getattr(response, 'model', 'unknown')
    
    # Calculate costs
    costs = tracker.calculate_costs('anthropic', model, input_tokens, output_tokens)
    
    # Get request ID
    request_id = response.get('id', None) if isinstance(response, dict) else getattr(response, 'id', None)
    
    return tracker.track_call(
        provider='anthropic',
        model=model,
        usage_data={
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            **costs
        },
        workload_id=workload_id,
        project_id=project_id,
        user_id=user_id,
        request_id=request_id,
        metadata=metadata,
        feature=feature,
        feature_version=feature_version,
        headers_in=headers_in,
    )
