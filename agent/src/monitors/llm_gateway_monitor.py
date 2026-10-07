"""
LLM Gateway Monitor

Monitors LLM API gateways/proxies that route requests across multiple providers.
Supports: LiteLLM, OpenRouter, and custom API gateways.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


class LLMGatewayMonitor:
    """
    Monitors LLM API gateways and proxies that route requests across multiple providers.

    Supports:
    - LiteLLM (self-hosted gateway)
    - OpenRouter (managed routing service)
    - Custom API gateways
    """

    REQUEST_TIMEOUT = 10  # seconds

    def __init__(self):
        """Initialize gateway monitor with configuration from environment variables."""
        self.litellm_endpoint = os.getenv("LITELLM_ENDPOINT")
        self.litellm_api_key = os.getenv("LITELLM_API_KEY")
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

        # Parse custom gateway endpoints and API keys
        custom_endpoints_str = os.getenv("CUSTOM_GATEWAY_ENDPOINTS", "")
        custom_api_keys_str = os.getenv("CUSTOM_GATEWAY_API_KEYS", "")

        self.custom_gateways: List[Tuple[str, str]] = []
        if custom_endpoints_str and custom_api_keys_str:
            endpoints = [ep.strip() for ep in custom_endpoints_str.split(",") if ep.strip()]
            api_keys = [key.strip() for key in custom_api_keys_str.split(",") if key.strip()]
            if len(endpoints) == len(api_keys):
                self.custom_gateways = list(zip(endpoints, api_keys))
            else:
                logger.warning(
                    f"Custom gateway endpoints ({len(endpoints)}) and API keys "
                    f"({len(api_keys)}) count mismatch"
                )

        self._session = self._create_session()

    def _create_session(self) -> requests.Session:
        """Create requests session with retry strategy."""
        session = requests.Session()
        retry_strategy = Retry(
            total=2,
            status_forcelist=[429, 500, 502, 503, 504],
            method_whitelist=["GET", "POST"],
            backoff_factor=0.5,
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    def get_gateway_metrics(self) -> List[Dict[str, Any]]:
        """
        Collect routing and performance metrics from all configured gateways.

        Returns:
            List of gateway metrics dictionaries containing:
            - gateway_type: Type of gateway (litellm, openrouter, custom)
            - endpoint: Gateway endpoint URL
            - total_requests_1h: Number of requests in past hour
            - total_tokens_1h: Total tokens used in past hour
            - requests_by_model: Dict of request counts per model
            - tokens_by_model: Dict of token counts per model
            - avg_latency_ms: Average request latency
            - p95_latency_ms: 95th percentile latency
            - p99_latency_ms: 99th percentile latency
            - error_rate_pct: Error rate percentage
            - rate_limit_hits_1h: Number of rate limit hits
            - fallback_count_1h: Number of fallback attempts
            - fallback_rate_pct: Fallback rate percentage
            - cost_1h: Cost in USD for past hour
            - cost_by_model: Dict of costs per model
            - active_models: List of active models
            - healthy: Overall health status
            - timestamp: Collection timestamp
        """
        metrics = []

        if self.litellm_endpoint and self.litellm_api_key:
            try:
                litellm_metrics = self._collect_litellm_metrics(
                    self.litellm_endpoint, self.litellm_api_key
                )
                if litellm_metrics:
                    metrics.append(litellm_metrics)
            except Exception as e:
                logger.error(f"Failed to collect LiteLLM metrics: {e}")

        if self.openrouter_api_key:
            try:
                openrouter_metrics = self._collect_openrouter_metrics(
                    self.openrouter_api_key
                )
                if openrouter_metrics:
                    metrics.append(openrouter_metrics)
            except Exception as e:
                logger.error(f"Failed to collect OpenRouter metrics: {e}")

        for endpoint, api_key in self.custom_gateways:
            try:
                custom_metrics = self._collect_custom_gateway_metrics(endpoint, api_key)
                if custom_metrics:
                    metrics.append(custom_metrics)
            except Exception as e:
                logger.error(f"Failed to collect custom gateway metrics for {endpoint}: {e}")

        return metrics

    def get_model_routing_stats(self) -> List[Dict[str, Any]]:
        """
        Get per-model routing statistics across all gateways.

        Returns:
            List of per-model statistics containing:
            - model_name: Name of the model
            - provider: Provider name (OpenAI, Anthropic, etc.)
            - total_requests: Total requests to this model
            - total_tokens: Total tokens consumed
            - avg_latency_ms: Average latency in milliseconds
            - error_rate_pct: Error rate percentage
            - cost_per_1k_tokens: Cost per 1000 tokens
            - success_rate_pct: Success rate percentage
            - timeout_rate_pct: Timeout rate percentage
        """
        model_stats = {}

        gateway_metrics = self.get_gateway_metrics()

        for gateway in gateway_metrics:
            requests_by_model = gateway.get("requests_by_model", {})
            tokens_by_model = gateway.get("tokens_by_model", {})
            cost_by_model = gateway.get("cost_by_model", {})

            for model, request_count in requests_by_model.items():
                if model not in model_stats:
                    model_stats[model] = {
                        "model_name": model,
                        "provider": self._extract_provider_from_model(model),
                        "total_requests": 0,
                        "total_tokens": 0,
                        "cost_usd": 0.0,
                        "error_rate_pct": 0.0,
                        "success_rate_pct": 100.0,
                        "timeout_rate_pct": 0.0,
                    }

                model_stats[model]["total_requests"] += request_count
                model_stats[model]["total_tokens"] += tokens_by_model.get(model, 0)
                model_stats[model]["cost_usd"] += cost_by_model.get(model, 0.0)

        # Convert to list and add calculated fields
        result = []
        for model, stats in model_stats.items():
            tokens = stats["total_tokens"]
            cost_per_1k = (stats["cost_usd"] / tokens * 1000) if tokens > 0 else 0.0

            result.append({
                "model_name": stats["model_name"],
                "provider": stats["provider"],
                "total_requests": stats["total_requests"],
                "total_tokens": stats["total_tokens"],
                "avg_latency_ms": 0.0,  # Aggregate from gateways if available
                "error_rate_pct": stats["error_rate_pct"],
                "cost_per_1k_tokens": round(cost_per_1k, 6),
                "success_rate_pct": stats["success_rate_pct"],
                "timeout_rate_pct": stats["timeout_rate_pct"],
            })

        return result

    def get_provider_health(self) -> List[Dict[str, Any]]:
        """
        Check health of all upstream LLM providers via gateways.

        Returns:
            List of provider health status containing:
            - provider: Provider name
            - model: Model name
            - status: Health status (healthy, degraded, down)
            - latency_ms: Current latency in milliseconds
            - error_rate_pct: Error rate percentage
            - last_success: Timestamp of last successful request
            - last_error: Timestamp of last error
        """
        health_status = []

        if self.litellm_endpoint and self.litellm_api_key:
            try:
                health = self._check_litellm_provider_health(
                    self.litellm_endpoint, self.litellm_api_key
                )
                health_status.extend(health)
            except Exception as e:
                logger.error(f"Failed to check LiteLLM provider health: {e}")

        if self.openrouter_api_key:
            try:
                health = self._check_openrouter_provider_health(
                    self.openrouter_api_key
                )
                health_status.extend(health)
            except Exception as e:
                logger.error(f"Failed to check OpenRouter provider health: {e}")

        for endpoint, api_key in self.custom_gateways:
            try:
                health = self._check_custom_gateway_provider_health(endpoint, api_key)
                health_status.extend(health)
            except Exception as e:
                logger.error(
                    f"Failed to check custom gateway provider health for {endpoint}: {e}"
                )

        return health_status

    def get_cost_optimization_insights(self) -> List[Dict[str, Any]]:
        """
        Analyze gateway routing for cost optimization opportunities.

        Returns:
            List of optimization insights containing:
            - type: Type of optimization (cheaper_alternative, latency_optimization, etc.)
            - description: Description of the opportunity
            - potential_savings_pct: Estimated percentage savings
            - current_model: Current model being used
            - recommended_model: Recommended alternative model
            - estimated_monthly_savings_usd: Estimated monthly savings
        """
        insights = []

        model_stats = self.get_model_routing_stats()

        # Group by provider to identify alternatives
        provider_models = {}
        for stat in model_stats:
            provider = stat["provider"]
            if provider not in provider_models:
                provider_models[provider] = []
            provider_models[provider].append(stat)

        # Check for cheaper alternatives
        for provider, models in provider_models.items():
            sorted_models = sorted(models, key=lambda x: x["cost_per_1k_tokens"])
            if len(sorted_models) > 1:
                cheapest = sorted_models[0]
                for model in sorted_models[1:]:
                    cost_diff = model["cost_per_1k_tokens"] - cheapest["cost_per_1k_tokens"]
                    if cost_diff > 0:
                        savings_pct = (cost_diff / model["cost_per_1k_tokens"]) * 100
                        monthly_tokens = model["total_tokens"] * 30  # Extrapolate
                        monthly_savings = (cost_diff / 1000) * monthly_tokens

                        if monthly_savings > 0.01:  # Only suggest if savings > $0.01/month
                            insights.append({
                                "type": "cheaper_alternative",
                                "description": (
                                    f"Model {cheapest['model_name']} is "
                                    f"{savings_pct:.1f}% cheaper than {model['model_name']}"
                                ),
                                "potential_savings_pct": round(savings_pct, 2),
                                "current_model": model["model_name"],
                                "recommended_model": cheapest["model_name"],
                                "estimated_monthly_savings_usd": round(monthly_savings, 2),
                            })

        return insights

    def get_rate_limit_status(self) -> List[Dict[str, Any]]:
        """
        Get rate limit status across providers.

        Returns:
            List of rate limit status containing:
            - provider: Provider name
            - model: Model name
            - requests_remaining: Requests remaining in window
            - tokens_remaining: Tokens remaining in window
            - reset_time: ISO timestamp when limit resets
            - utilization_pct: Percentage of limit utilized
        """
        rate_limits = []

        if self.litellm_endpoint and self.litellm_api_key:
            try:
                limits = self._get_litellm_rate_limits(
                    self.litellm_endpoint, self.litellm_api_key
                )
                rate_limits.extend(limits)
            except Exception as e:
                logger.error(f"Failed to get LiteLLM rate limits: {e}")

        if self.openrouter_api_key:
            try:
                limits = self._get_openrouter_rate_limits(self.openrouter_api_key)
                rate_limits.extend(limits)
            except Exception as e:
                logger.error(f"Failed to get OpenRouter rate limits: {e}")

        for endpoint, api_key in self.custom_gateways:
            try:
                limits = self._get_custom_gateway_rate_limits(endpoint, api_key)
                rate_limits.extend(limits)
            except Exception as e:
                logger.error(
                    f"Failed to get custom gateway rate limits for {endpoint}: {e}"
                )

        return rate_limits

    # Provider-specific collectors

    def _collect_litellm_metrics(self, endpoint: str, api_key: str) -> Optional[Dict[str, Any]]:
        """Collect metrics from LiteLLM gateway."""
        try:
            headers = {"Authorization": f"Bearer {api_key}"}

            # Get spending data
            spend_response = self._session.get(
                f"{endpoint}/spend/logs",
                headers=headers,
                timeout=self.REQUEST_TIMEOUT,
            )
            spend_data = spend_response.json() if spend_response.ok else {}

            # Get model info
            model_response = self._session.get(
                f"{endpoint}/model/info",
                headers=headers,
                timeout=self.REQUEST_TIMEOUT,
            )
            model_data = model_response.json() if model_response.ok else {}

            # Get health
            health_response = self._session.get(
                f"{endpoint}/health",
                headers=headers,
                timeout=self.REQUEST_TIMEOUT,
            )
            health_data = health_response.json() if health_response.ok else {}

            # Get rate limit info
            key_response = self._session.get(
                f"{endpoint}/key/info",
                headers=headers,
                timeout=self.REQUEST_TIMEOUT,
            )
            key_data = key_response.json() if key_response.ok else {}

            # Parse spending data (aggregated last 1 hour)
            requests_by_model = self._parse_litellm_spend_by_model(spend_data)

            total_requests = sum(requests_by_model.values())
            total_cost = self._calculate_litellm_cost(spend_data)

            return {
                "gateway_type": "litellm",
                "endpoint": endpoint,
                "total_requests_1h": total_requests,
                "total_tokens_1h": self._calculate_litellm_tokens(spend_data),
                "requests_by_model": requests_by_model,
                "tokens_by_model": self._parse_litellm_tokens_by_model(spend_data),
                "avg_latency_ms": self._extract_litellm_avg_latency(spend_data),
                "p95_latency_ms": self._extract_litellm_p95_latency(spend_data),
                "p99_latency_ms": self._extract_litellm_p99_latency(spend_data),
                "error_rate_pct": self._extract_litellm_error_rate(spend_data),
                "rate_limit_hits_1h": key_data.get("rate_limit_hits", 0),
                "fallback_count_1h": spend_data.get("fallback_count", 0),
                "fallback_rate_pct": self._calculate_litellm_fallback_rate(spend_data),
                "cost_1h": total_cost,
                "cost_by_model": self._parse_litellm_cost_by_model(spend_data),
                "active_models": list(model_data.get("models", {}).keys()),
                "healthy": health_data.get("status") == "healthy",
                "timestamp": datetime.utcnow().isoformat(),
            }
        except Exception as e:
            logger.error(f"Error collecting LiteLLM metrics from {endpoint}: {e}")
            return None

    def _collect_openrouter_metrics(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Collect metrics from OpenRouter."""
        try:
            headers = {"Authorization": f"Bearer {api_key}"}

            # Get auth/key info (includes rate limits and usage)
            key_response = self._session.get(
                "https://openrouter.ai/api/v1/auth/key",
                headers=headers,
                timeout=self.REQUEST_TIMEOUT,
            )
            key_data = key_response.json() if key_response.ok else {}

            # Get available models
            models_response = self._session.get(
                "https://openrouter.ai/api/v1/models",
                headers=headers,
                timeout=self.REQUEST_TIMEOUT,
            )
            models_data = models_response.json() if models_response.ok else {}

            # Parse usage data
            usage = key_data.get("data", {}).get("usage", {})

            return {
                "gateway_type": "openrouter",
                "endpoint": "https://openrouter.ai/api/v1",
                "total_requests_1h": usage.get("requests", 0),
                "total_tokens_1h": usage.get("tokens", 0),
                "requests_by_model": {},  # OpenRouter doesn't break down by model in key info
                "tokens_by_model": {},
                "avg_latency_ms": 0.0,
                "p95_latency_ms": 0.0,
                "p99_latency_ms": 0.0,
                "error_rate_pct": 0.0,
                "rate_limit_hits_1h": 0,
                "fallback_count_1h": 0,
                "fallback_rate_pct": 0.0,
                "cost_1h": float(usage.get("spent_usd", 0.0)),
                "cost_by_model": {},
                "active_models": [
                    model.get("id")
                    for model in models_data.get("data", [])
                ],
                "healthy": key_response.ok,
                "timestamp": datetime.utcnow().isoformat(),
            }
        except Exception as e:
            logger.error(f"Error collecting OpenRouter metrics: {e}")
            return None

    def _collect_custom_gateway_metrics(
        self, endpoint: str, api_key: str
    ) -> Optional[Dict[str, Any]]:
        """Collect metrics from custom API gateway."""
        try:
            headers = {"Authorization": f"Bearer {api_key}"}

            # Try to get metrics endpoint (common pattern)
            metrics_response = self._session.get(
                f"{endpoint}/metrics",
                headers=headers,
                timeout=self.REQUEST_TIMEOUT,
            )

            if not metrics_response.ok:
                logger.warning(f"Custom gateway {endpoint}/metrics returned {metrics_response.status_code}")
                return None

            metrics_data = metrics_response.json()

            return {
                "gateway_type": "custom",
                "endpoint": endpoint,
                "total_requests_1h": metrics_data.get("total_requests_1h", 0),
                "total_tokens_1h": metrics_data.get("total_tokens_1h", 0),
                "requests_by_model": metrics_data.get("requests_by_model", {}),
                "tokens_by_model": metrics_data.get("tokens_by_model", {}),
                "avg_latency_ms": metrics_data.get("avg_latency_ms", 0.0),
                "p95_latency_ms": metrics_data.get("p95_latency_ms", 0.0),
                "p99_latency_ms": metrics_data.get("p99_latency_ms", 0.0),
                "error_rate_pct": metrics_data.get("error_rate_pct", 0.0),
                "rate_limit_hits_1h": metrics_data.get("rate_limit_hits_1h", 0),
                "fallback_count_1h": metrics_data.get("fallback_count_1h", 0),
                "fallback_rate_pct": metrics_data.get("fallback_rate_pct", 0.0),
                "cost_1h": metrics_data.get("cost_1h", 0.0),
                "cost_by_model": metrics_data.get("cost_by_model", {}),
                "active_models": metrics_data.get("active_models", []),
                "healthy": metrics_data.get("healthy", True),
                "timestamp": datetime.utcnow().isoformat(),
            }
        except Exception as e:
            logger.error(f"Error collecting custom gateway metrics from {endpoint}: {e}")
            return None

    # LiteLLM helper methods

    def _parse_litellm_spend_by_model(self, spend_data: Dict[str, Any]) -> Dict[str, int]:
        """Parse request counts by model from LiteLLM spend logs."""
        by_model = {}
        for log in spend_data.get("logs", []):
            model = log.get("model", "unknown")
            by_model[model] = by_model.get(model, 0) + 1
        return by_model

    def _parse_litellm_tokens_by_model(self, spend_data: Dict[str, Any]) -> Dict[str, int]:
        """Parse token counts by model from LiteLLM spend logs."""
        by_model = {}
        for log in spend_data.get("logs", []):
            model = log.get("model", "unknown")
            tokens = log.get("total_tokens", 0)
            by_model[model] = by_model.get(model, 0) + tokens
        return by_model

    def _parse_litellm_cost_by_model(self, spend_data: Dict[str, Any]) -> Dict[str, float]:
        """Parse cost by model from LiteLLM spend logs."""
        by_model = {}
        for log in spend_data.get("logs", []):
            model = log.get("model", "unknown")
            cost = log.get("cost", 0.0)
            by_model[model] = by_model.get(model, 0.0) + cost
        return by_model

    def _calculate_litellm_cost(self, spend_data: Dict[str, Any]) -> float:
        """Calculate total cost from LiteLLM spend logs."""
        return sum(log.get("cost", 0.0) for log in spend_data.get("logs", []))

    def _calculate_litellm_tokens(self, spend_data: Dict[str, Any]) -> int:
        """Calculate total tokens from LiteLLM spend logs."""
        return sum(log.get("total_tokens", 0) for log in spend_data.get("logs", []))

    def _extract_litellm_avg_latency(self, spend_data: Dict[str, Any]) -> float:
        """Extract average latency from LiteLLM spend logs."""
        latencies = [log.get("latency_ms", 0) for log in spend_data.get("logs", [])]
        return sum(latencies) / len(latencies) if latencies else 0.0

    def _extract_litellm_p95_latency(self, spend_data: Dict[str, Any]) -> float:
        """Extract 95th percentile latency from LiteLLM spend logs."""
        latencies = sorted([log.get("latency_ms", 0) for log in spend_data.get("logs", [])])
        if not latencies:
            return 0.0
        idx = int(len(latencies) * 0.95)
        return float(latencies[idx])

    def _extract_litellm_p99_latency(self, spend_data: Dict[str, Any]) -> float:
        """Extract 99th percentile latency from LiteLLM spend logs."""
        latencies = sorted([log.get("latency_ms", 0) for log in spend_data.get("logs", [])])
        if not latencies:
            return 0.0
        idx = int(len(latencies) * 0.99)
        return float(latencies[idx])

    def _extract_litellm_error_rate(self, spend_data: Dict[str, Any]) -> float:
        """Extract error rate from LiteLLM spend logs."""
        logs = spend_data.get("logs", [])
        if not logs:
            return 0.0
        errors = sum(1 for log in logs if log.get("status") == "error")
        return (errors / len(logs)) * 100

    def _calculate_litellm_fallback_rate(self, spend_data: Dict[str, Any]) -> float:
        """Calculate fallback rate from LiteLLM spend logs."""
        logs = spend_data.get("logs", [])
        if not logs:
            return 0.0
        fallbacks = sum(1 for log in logs if log.get("fallback", False))
        return (fallbacks / len(logs)) * 100

    def _check_litellm_provider_health(
        self, endpoint: str, api_key: str
    ) -> List[Dict[str, Any]]:
        """Check health of providers via LiteLLM health endpoint."""
        try:
            headers = {"Authorization": f"Bearer {api_key}"}
            response = self._session.get(
                f"{endpoint}/health",
                headers=headers,
                timeout=self.REQUEST_TIMEOUT,
            )
            health_data = response.json() if response.ok else {}

            health_status = []
            for provider_name, provider_info in health_data.get("providers", {}).items():
                for model in provider_info.get("models", []):
                    status = "healthy"
                    if provider_info.get("status") == "degraded":
                        status = "degraded"
                    elif provider_info.get("status") == "down":
                        status = "down"

                    health_status.append({
                        "provider": provider_name,
                        "model": model,
                        "status": status,
                        "latency_ms": provider_info.get("latency_ms", 0.0),
                        "error_rate_pct": provider_info.get("error_rate_pct", 0.0),
                        "last_success": provider_info.get("last_success"),
                        "last_error": provider_info.get("last_error"),
                    })

            return health_status
        except Exception as e:
            logger.error(f"Error checking LiteLLM provider health: {e}")
            return []

    def _get_litellm_rate_limits(
        self, endpoint: str, api_key: str
    ) -> List[Dict[str, Any]]:
        """Get rate limit information from LiteLLM."""
        try:
            headers = {"Authorization": f"Bearer {api_key}"}
            response = self._session.get(
                f"{endpoint}/key/info",
                headers=headers,
                timeout=self.REQUEST_TIMEOUT,
            )
            key_data = response.json() if response.ok else {}

            rate_limits = []
            for provider_name, provider_limits in key_data.get("rate_limits", {}).items():
                reset_time = provider_limits.get("reset_time")
                requests_remaining = provider_limits.get("requests_remaining", 0)
                requests_limit = provider_limits.get("requests_limit", 1)
                utilization = ((requests_limit - requests_remaining) / requests_limit * 100) if requests_limit > 0 else 0

                rate_limits.append({
                    "provider": provider_name,
                    "model": provider_limits.get("model", "all"),
                    "requests_remaining": requests_remaining,
                    "tokens_remaining": provider_limits.get("tokens_remaining", 0),
                    "reset_time": reset_time,
                    "utilization_pct": utilization,
                })

            return rate_limits
        except Exception as e:
            logger.error(f"Error getting LiteLLM rate limits: {e}")
            return []

    # OpenRouter helper methods

    def _check_openrouter_provider_health(self, api_key: str) -> List[Dict[str, Any]]:
        """Check health of providers via OpenRouter."""
        try:
            headers = {"Authorization": f"Bearer {api_key}"}

            # Get models list which includes availability info
            response = self._session.get(
                "https://openrouter.ai/api/v1/models",
                headers=headers,
                timeout=self.REQUEST_TIMEOUT,
            )

            if not response.ok:
                return []

            models_data = response.json()
            health_status = []

            for model in models_data.get("data", []):
                # OpenRouter doesn't provide detailed health, assume healthy if present
                health_status.append({
                    "provider": self._extract_provider_from_model(model.get("id", "")),
                    "model": model.get("id", "unknown"),
                    "status": "healthy",
                    "latency_ms": 0.0,
                    "error_rate_pct": 0.0,
                    "last_success": datetime.utcnow().isoformat(),
                    "last_error": None,
                })

            return health_status
        except Exception as e:
            logger.error(f"Error checking OpenRouter provider health: {e}")
            return []

    def _get_openrouter_rate_limits(self, api_key: str) -> List[Dict[str, Any]]:
        """Get rate limit information from OpenRouter."""
        try:
            headers = {"Authorization": f"Bearer {api_key}"}
            response = self._session.get(
                "https://openrouter.ai/api/v1/auth/key",
                headers=headers,
                timeout=self.REQUEST_TIMEOUT,
            )

            if not response.ok:
                return []

            key_data = response.json()
            limits = key_data.get("data", {}).get("limits", {})

            return [{
                "provider": "openrouter",
                "model": "all",
                "requests_remaining": limits.get("requests_remaining", 0),
                "tokens_remaining": limits.get("tokens_remaining", 0),
                "reset_time": limits.get("reset_time"),
                "utilization_pct": 0.0,  # OpenRouter doesn't expose utilization
            }]
        except Exception as e:
            logger.error(f"Error getting OpenRouter rate limits: {e}")
            return []

    # Custom gateway helper methods

    def _check_custom_gateway_provider_health(
        self, endpoint: str, api_key: str
    ) -> List[Dict[str, Any]]:
        """Check health of providers via custom gateway."""
        try:
            headers = {"Authorization": f"Bearer {api_key}"}
            response = self._session.get(
                f"{endpoint}/health",
                headers=headers,
                timeout=self.REQUEST_TIMEOUT,
            )

            if not response.ok:
                return []

            health_data = response.json()
            health_status = []

            for provider_info in health_data.get("providers", []):
                health_status.append({
                    "provider": provider_info.get("name", "unknown"),
                    "model": provider_info.get("model", "all"),
                    "status": provider_info.get("status", "unknown"),
                    "latency_ms": provider_info.get("latency_ms", 0.0),
                    "error_rate_pct": provider_info.get("error_rate_pct", 0.0),
                    "last_success": provider_info.get("last_success"),
                    "last_error": provider_info.get("last_error"),
                })

            return health_status
        except Exception as e:
            logger.error(f"Error checking custom gateway provider health: {e}")
            return []

    def _get_custom_gateway_rate_limits(
        self, endpoint: str, api_key: str
    ) -> List[Dict[str, Any]]:
        """Get rate limit information from custom gateway."""
        try:
            headers = {"Authorization": f"Bearer {api_key}"}
            response = self._session.get(
                f"{endpoint}/rate_limits",
                headers=headers,
                timeout=self.REQUEST_TIMEOUT,
            )

            if not response.ok:
                return []

            limits_data = response.json()
            rate_limits = []

            for limit_info in limits_data.get("limits", []):
                rate_limits.append({
                    "provider": limit_info.get("provider", "unknown"),
                    "model": limit_info.get("model", "all"),
                    "requests_remaining": limit_info.get("requests_remaining", 0),
                    "tokens_remaining": limit_info.get("tokens_remaining", 0),
                    "reset_time": limit_info.get("reset_time"),
                    "utilization_pct": limit_info.get("utilization_pct", 0.0),
                })

            return rate_limits
        except Exception as e:
            logger.error(f"Error getting custom gateway rate limits: {e}")
            return []

    # Utility methods

    @staticmethod
    def _extract_provider_from_model(model: str) -> str:
        """Extract provider name from model identifier."""
        model_lower = model.lower()

        if "gpt" in model_lower or "openai" in model_lower:
            return "OpenAI"
        elif "claude" in model_lower or "anthropic" in model_lower:
            return "Anthropic"
        elif "gemini" in model_lower or "google" in model_lower:
            return "Google"
        elif "llama" in model_lower or "meta" in model_lower:
            return "Meta"
        elif "command" in model_lower or "cohere" in model_lower:
            return "Cohere"
        elif "palm" in model_lower:
            return "PaLM"
        else:
            return "Unknown"
