"""Comprehensive tests for LLMGatewayMonitor."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import requests

from monitors.llm_gateway_monitor import LLMGatewayMonitor


class TestLLMGatewayMonitorInitialization:
    """Test initialization with various configurations."""

    @patch('monitors.llm_gateway_monitor.LLMGatewayMonitor._create_session')
    def test_init_with_no_env_vars(self, mock_create_session):
        """Initialization with no environment variables."""
        mock_create_session.return_value = Mock(spec=requests.Session)
        with patch.dict(os.environ, {}, clear=True):
            monitor = LLMGatewayMonitor()
            assert monitor.litellm_endpoint is None
            assert monitor.litellm_api_key is None
            assert monitor.openrouter_api_key is None
            assert monitor.custom_gateways == []

    @patch('monitors.llm_gateway_monitor.LLMGatewayMonitor._create_session')
    def test_init_with_litellm_endpoint(self, mock_create_session):
        """Initialization with LiteLLM endpoint."""
        mock_create_session.return_value = Mock(spec=requests.Session)
        env_vars = {
            "LITELLM_ENDPOINT": "http://localhost:8000",
            "LITELLM_API_KEY": "test-key",
        }
        with patch.dict(os.environ, env_vars):
            monitor = LLMGatewayMonitor()
            assert monitor.litellm_endpoint == "http://localhost:8000"
            assert monitor.litellm_api_key == "test-key"

    @patch('monitors.llm_gateway_monitor.LLMGatewayMonitor._create_session')
    def test_init_with_openrouter_api_key(self, mock_create_session):
        """Initialization with OpenRouter API key."""
        mock_create_session.return_value = Mock(spec=requests.Session)
        env_vars = {"OPENROUTER_API_KEY": "or-test-key"}
        with patch.dict(os.environ, env_vars):
            monitor = LLMGatewayMonitor()
            assert monitor.openrouter_api_key == "or-test-key"

    @patch('monitors.llm_gateway_monitor.LLMGatewayMonitor._create_session')
    def test_init_with_custom_gateways(self, mock_create_session):
        """Initialization with custom gateway endpoints and keys."""
        mock_create_session.return_value = Mock(spec=requests.Session)
        env_vars = {
            "CUSTOM_GATEWAY_ENDPOINTS": "http://gateway1:8000,http://gateway2:8000",
            "CUSTOM_GATEWAY_API_KEYS": "key1,key2",
        }
        with patch.dict(os.environ, env_vars):
            monitor = LLMGatewayMonitor()
            assert len(monitor.custom_gateways) == 2
            assert monitor.custom_gateways[0] == ("http://gateway1:8000", "key1")
            assert monitor.custom_gateways[1] == ("http://gateway2:8000", "key2")

    @patch('monitors.llm_gateway_monitor.LLMGatewayMonitor._create_session')
    def test_init_custom_gateways_key_mismatch_warning(self, mock_create_session):
        """Warning when endpoint/key count mismatch."""
        mock_create_session.return_value = Mock(spec=requests.Session)
        env_vars = {
            "CUSTOM_GATEWAY_ENDPOINTS": "http://gateway1:8000,http://gateway2:8000",
            "CUSTOM_GATEWAY_API_KEYS": "key1",  # Only 1 key for 2 endpoints
        }
        with patch.dict(os.environ, env_vars):
            with patch('monitors.llm_gateway_monitor.logger') as mock_logger:
                monitor = LLMGatewayMonitor()
                assert monitor.custom_gateways == []
                mock_logger.warning.assert_called()

    @patch('monitors.llm_gateway_monitor.LLMGatewayMonitor._create_session')
    def test_init_creates_session(self, mock_create_session):
        """Initialization creates requests session."""
        mock_session = Mock(spec=requests.Session)
        mock_create_session.return_value = mock_session
        with patch.dict(os.environ, {}, clear=True):
            monitor = LLMGatewayMonitor()
            assert monitor._session is not None
            assert monitor._session is mock_session


class TestGetGatewayMetrics:
    """Test gateway metrics collection."""

    @patch('monitors.llm_gateway_monitor.LLMGatewayMonitor._create_session')
    @patch('monitors.llm_gateway_monitor.LLMGatewayMonitor._collect_litellm_metrics')
    @patch('monitors.llm_gateway_monitor.LLMGatewayMonitor._collect_openrouter_metrics')
    @patch('monitors.llm_gateway_monitor.LLMGatewayMonitor._collect_custom_gateway_metrics')
    def test_get_gateway_metrics_litellm(self, mock_custom, mock_openrouter, mock_litellm, mock_create_session):
        """Collect metrics from LiteLLM gateway."""
        mock_create_session.return_value = Mock(spec=requests.Session)
        mock_litellm.return_value = {
            "gateway_type": "litellm",
            "endpoint": "http://localhost:8000",
            "total_requests_1h": 100,
            "total_tokens_1h": 50000,
        }
        mock_openrouter.return_value = None
        mock_custom.return_value = None

        env_vars = {
            "LITELLM_ENDPOINT": "http://localhost:8000",
            "LITELLM_API_KEY": "test-key",
        }
        with patch.dict(os.environ, env_vars):
            monitor = LLMGatewayMonitor()
            metrics = monitor.get_gateway_metrics()

        assert len(metrics) == 1
        assert metrics[0]["gateway_type"] == "litellm"

    @patch('monitors.llm_gateway_monitor.LLMGatewayMonitor._create_session')
    @patch('monitors.llm_gateway_monitor.LLMGatewayMonitor._collect_openrouter_metrics')
    @patch('monitors.llm_gateway_monitor.LLMGatewayMonitor._collect_litellm_metrics')
    @patch('monitors.llm_gateway_monitor.LLMGatewayMonitor._collect_custom_gateway_metrics')
    def test_get_gateway_metrics_openrouter(self, mock_custom, mock_litellm, mock_openrouter, mock_create_session):
        """Collect metrics from OpenRouter gateway."""
        mock_create_session.return_value = Mock(spec=requests.Session)
        mock_openrouter.return_value = {
            "gateway_type": "openrouter",
            "endpoint": "https://openrouter.ai/api/v1",
            "total_requests_1h": 50,
            "total_tokens_1h": 25000,
        }
        mock_litellm.return_value = None
        mock_custom.return_value = None

        env_vars = {"OPENROUTER_API_KEY": "or-key"}
        with patch.dict(os.environ, env_vars):
            monitor = LLMGatewayMonitor()
            metrics = monitor.get_gateway_metrics()

        assert len(metrics) == 1
        assert metrics[0]["gateway_type"] == "openrouter"

    @patch('monitors.llm_gateway_monitor.LLMGatewayMonitor._create_session')
    @patch('monitors.llm_gateway_monitor.LLMGatewayMonitor._collect_custom_gateway_metrics')
    @patch('monitors.llm_gateway_monitor.LLMGatewayMonitor._collect_litellm_metrics')
    @patch('monitors.llm_gateway_monitor.LLMGatewayMonitor._collect_openrouter_metrics')
    def test_get_gateway_metrics_custom(self, mock_openrouter, mock_litellm, mock_custom, mock_create_session):
        """Collect metrics from custom gateway."""
        mock_create_session.return_value = Mock(spec=requests.Session)
        mock_custom.return_value = {
            "gateway_type": "custom",
            "endpoint": "http://custom-gateway:8000",
            "total_requests_1h": 75,
        }
        mock_litellm.return_value = None
        mock_openrouter.return_value = None

        env_vars = {
            "CUSTOM_GATEWAY_ENDPOINTS": "http://custom-gateway:8000",
            "CUSTOM_GATEWAY_API_KEYS": "custom-key",
        }
        with patch.dict(os.environ, env_vars):
            monitor = LLMGatewayMonitor()
            metrics = monitor.get_gateway_metrics()

        assert len(metrics) == 1
        assert metrics[0]["gateway_type"] == "custom"

    @patch('monitors.llm_gateway_monitor.LLMGatewayMonitor._create_session')
    @patch('monitors.llm_gateway_monitor.LLMGatewayMonitor._collect_litellm_metrics')
    def test_get_gateway_metrics_collection_error(self, mock_litellm, mock_create_session):
        """Handle collection errors gracefully."""
        mock_create_session.return_value = Mock(spec=requests.Session)
        mock_litellm.side_effect = Exception("Connection failed")

        env_vars = {
            "LITELLM_ENDPOINT": "http://localhost:8000",
            "LITELLM_API_KEY": "test-key",
        }
        with patch.dict(os.environ, env_vars):
            with patch('monitors.llm_gateway_monitor.logger'):
                monitor = LLMGatewayMonitor()
                metrics = monitor.get_gateway_metrics()

        assert metrics == []


class TestGetModelRoutingStats:
    """Test model routing statistics."""

    @patch('monitors.llm_gateway_monitor.LLMGatewayMonitor._create_session')
    @patch('monitors.llm_gateway_monitor.LLMGatewayMonitor.get_gateway_metrics')
    def test_get_model_routing_stats_aggregation(self, mock_gateway_metrics, mock_create_session):
        """Aggregate model statistics across gateways."""
        mock_create_session.return_value = Mock(spec=requests.Session)
        mock_gateway_metrics.return_value = [
            {
                "gateway_type": "litellm",
                "requests_by_model": {
                    "gpt-4": 50,
                    "gpt-3.5-turbo": 100,
                },
                "tokens_by_model": {
                    "gpt-4": 25000,
                    "gpt-3.5-turbo": 50000,
                },
                "cost_by_model": {
                    "gpt-4": 0.75,
                    "gpt-3.5-turbo": 0.25,
                },
            },
        ]

        with patch.dict(os.environ, {}, clear=True):
            monitor = LLMGatewayMonitor()
            stats = monitor.get_model_routing_stats()

        assert len(stats) == 2
        gpt4_stats = [s for s in stats if s["model_name"] == "gpt-4"][0]
        assert gpt4_stats["total_requests"] == 50
        assert gpt4_stats["total_tokens"] == 25000

    @patch('monitors.llm_gateway_monitor.LLMGatewayMonitor._create_session')
    @patch('monitors.llm_gateway_monitor.LLMGatewayMonitor.get_gateway_metrics')
    def test_get_model_routing_stats_provider_extraction(self, mock_gateway_metrics, mock_create_session):
        """Extract provider from model name."""
        mock_create_session.return_value = Mock(spec=requests.Session)
        mock_gateway_metrics.return_value = [
            {
                "requests_by_model": {
                    "gpt-4": 10,
                    "claude-2": 10,
                    "gemini-pro": 10,
                },
                "tokens_by_model": {
                    "gpt-4": 5000,
                    "claude-2": 5000,
                    "gemini-pro": 5000,
                },
                "cost_by_model": {
                    "gpt-4": 0.1,
                    "claude-2": 0.1,
                    "gemini-pro": 0.1,
                },
            },
        ]

        with patch.dict(os.environ, {}, clear=True):
            monitor = LLMGatewayMonitor()
            stats = monitor.get_model_routing_stats()

        providers = {s["model_name"]: s["provider"] for s in stats}
        assert providers["gpt-4"] == "OpenAI"
        assert providers["claude-2"] == "Anthropic"
        assert providers["gemini-pro"] == "Google"

    @patch('monitors.llm_gateway_monitor.LLMGatewayMonitor._create_session')
    @patch('monitors.llm_gateway_monitor.LLMGatewayMonitor.get_gateway_metrics')
    def test_get_model_routing_stats_cost_calculation(self, mock_gateway_metrics, mock_create_session):
        """Calculate cost per 1k tokens."""
        mock_create_session.return_value = Mock(spec=requests.Session)
        mock_gateway_metrics.return_value = [
            {
                "requests_by_model": {"model-a": 10},
                "tokens_by_model": {"model-a": 10000},
                "cost_by_model": {"model-a": 0.10},
            },
        ]

        with patch.dict(os.environ, {}, clear=True):
            monitor = LLMGatewayMonitor()
            stats = monitor.get_model_routing_stats()

        assert len(stats) == 1
        # 0.10 / 10000 * 1000 = 0.01 per 1k tokens
        assert stats[0]["cost_per_1k_tokens"] == 0.01


class TestGetProviderHealth:
    """Test provider health checks."""

    @patch('monitors.llm_gateway_monitor.LLMGatewayMonitor._create_session')
    @patch('monitors.llm_gateway_monitor.LLMGatewayMonitor._check_litellm_provider_health')
    @patch('monitors.llm_gateway_monitor.LLMGatewayMonitor._check_openrouter_provider_health')
    @patch('monitors.llm_gateway_monitor.LLMGatewayMonitor._check_custom_gateway_provider_health')
    def test_get_provider_health_litellm(self, mock_custom_health, mock_openrouter_health, mock_litellm_health, mock_create_session):
        """Collect health from LiteLLM gateway."""
        mock_create_session.return_value = Mock(spec=requests.Session)
        mock_litellm_health.return_value = [
            {
                "provider": "openai",
                "model": "gpt-4",
                "status": "healthy",
                "latency_ms": 150.0,
            },
        ]
        mock_openrouter_health.return_value = []
        mock_custom_health.return_value = []

        env_vars = {
            "LITELLM_ENDPOINT": "http://localhost:8000",
            "LITELLM_API_KEY": "test-key",
        }
        with patch.dict(os.environ, env_vars):
            monitor = LLMGatewayMonitor()
            health = monitor.get_provider_health()

        assert len(health) == 1
        assert health[0]["provider"] == "openai"
        assert health[0]["status"] == "healthy"

    @patch('monitors.llm_gateway_monitor.LLMGatewayMonitor._create_session')
    @patch('monitors.llm_gateway_monitor.LLMGatewayMonitor._check_litellm_provider_health')
    @patch('monitors.llm_gateway_monitor.LLMGatewayMonitor._check_openrouter_provider_health')
    @patch('monitors.llm_gateway_monitor.LLMGatewayMonitor._check_custom_gateway_provider_health')
    def test_get_provider_health_aggregation(self, mock_custom_health, mock_openrouter_health, mock_litellm_health, mock_create_session):
        """Aggregate health from multiple gateways."""
        mock_create_session.return_value = Mock(spec=requests.Session)
        mock_litellm_health.return_value = [
            {"provider": "openai", "model": "gpt-4", "status": "healthy"},
        ]
        mock_openrouter_health.return_value = [
            {"provider": "anthropic", "model": "claude-2", "status": "degraded"},
        ]
        mock_custom_health.return_value = []

        env_vars = {
            "LITELLM_ENDPOINT": "http://localhost:8000",
            "LITELLM_API_KEY": "test-key",
            "OPENROUTER_API_KEY": "or-key",
        }
        with patch.dict(os.environ, env_vars):
            monitor = LLMGatewayMonitor()
            health = monitor.get_provider_health()

        assert len(health) == 2


class TestGetCostOptimizationInsights:
    """Test cost optimization insight generation."""

    @patch('monitors.llm_gateway_monitor.LLMGatewayMonitor._create_session')
    @patch('monitors.llm_gateway_monitor.LLMGatewayMonitor.get_model_routing_stats')
    def test_get_cost_optimization_cheaper_alternative(self, mock_stats, mock_create_session):
        """Identify cheaper model alternatives."""
        mock_create_session.return_value = Mock(spec=requests.Session)
        mock_stats.return_value = [
            {
                "model_name": "gpt-4",
                "provider": "openai",
                "total_tokens": 100000,
                "cost_per_1k_tokens": 0.03,
            },
            {
                "model_name": "gpt-3.5-turbo",
                "provider": "openai",
                "total_tokens": 100000,
                "cost_per_1k_tokens": 0.0015,
            },
        ]

        with patch.dict(os.environ, {}, clear=True):
            monitor = LLMGatewayMonitor()
            insights = monitor.get_cost_optimization_insights()

        # Should recommend gpt-3.5-turbo as cheaper alternative
        assert len(insights) > 0
        assert any(i["type"] == "cheaper_alternative" for i in insights)

    @patch('monitors.llm_gateway_monitor.LLMGatewayMonitor._create_session')
    @patch('monitors.llm_gateway_monitor.LLMGatewayMonitor.get_model_routing_stats')
    def test_get_cost_optimization_savings_calculation(self, mock_stats, mock_create_session):
        """Calculate potential savings."""
        mock_create_session.return_value = Mock(spec=requests.Session)
        mock_stats.return_value = [
            {
                "model_name": "expensive-model",
                "provider": "test",
                "total_tokens": 100000,
                "cost_per_1k_tokens": 0.01,
            },
            {
                "model_name": "cheap-model",
                "provider": "test",
                "total_tokens": 1000,
                "cost_per_1k_tokens": 0.001,
            },
        ]

        with patch.dict(os.environ, {}, clear=True):
            monitor = LLMGatewayMonitor()
            insights = monitor.get_cost_optimization_insights()

        if insights:
            insight = insights[0]
            # Cost diff: 0.01 - 0.001 = 0.009 per 1k
            # Savings%: (0.009 / 0.01) * 100 = 90%
            # Monthly: (0.009 / 1000) * (100000 * 30) = 27 dollars
            assert insight["potential_savings_pct"] > 0


class TestCollectLiteLLMMetrics:
    """Test LiteLLM metrics collection."""

    @patch('monitors.llm_gateway_monitor.LLMGatewayMonitor._create_session')
    def test_collect_litellm_metrics_success(self, mock_create_session):
        """Successfully collect LiteLLM metrics."""
        # Mock session object
        session = Mock()

        # Mock responses
        spend_response = Mock()
        spend_response.json.return_value = {
            "logs": [
                {
                    "model": "gpt-4",
                    "total_tokens": 1000,
                    "cost": 0.03,
                    "latency_ms": 150,
                    "status": "success",
                },
            ]
        }
        spend_response.ok = True

        model_response = Mock()
        model_response.json.return_value = {
            "models": {
                "gpt-4": {"id": "gpt-4"},
                "gpt-3.5-turbo": {"id": "gpt-3.5-turbo"},
            }
        }
        model_response.ok = True

        health_response = Mock()
        health_response.json.return_value = {"status": "healthy"}
        health_response.ok = True

        key_response = Mock()
        key_response.json.return_value = {"rate_limit_hits": 0}
        key_response.ok = True

        session.get.side_effect = [spend_response, model_response, health_response, key_response]
        mock_create_session.return_value = session

        env_vars = {
            "LITELLM_ENDPOINT": "http://localhost:8000",
            "LITELLM_API_KEY": "test-key",
        }
        with patch.dict(os.environ, env_vars):
            monitor = LLMGatewayMonitor()
            metrics = monitor._collect_litellm_metrics("http://localhost:8000", "test-key")

        assert metrics is not None
        assert metrics["gateway_type"] == "litellm"
        assert metrics["total_requests_1h"] == 1
        assert metrics["total_tokens_1h"] == 1000

    @patch('monitors.llm_gateway_monitor.LLMGatewayMonitor._create_session')
    def test_collect_litellm_metrics_error(self, mock_create_session):
        """Handle LiteLLM collection errors."""
        session = Mock()
        session.get.side_effect = Exception("Connection error")
        mock_create_session.return_value = session

        env_vars = {
            "LITELLM_ENDPOINT": "http://localhost:8000",
            "LITELLM_API_KEY": "test-key",
        }
        with patch.dict(os.environ, env_vars):
            with patch('monitors.llm_gateway_monitor.logger'):
                monitor = LLMGatewayMonitor()
                metrics = monitor._collect_litellm_metrics("http://localhost:8000", "test-key")

        assert metrics is None


class TestCollectOpenRouterMetrics:
    """Test OpenRouter metrics collection."""

    @patch('monitors.llm_gateway_monitor.LLMGatewayMonitor._create_session')
    def test_collect_openrouter_metrics_success(self, mock_create_session):
        """Successfully collect OpenRouter metrics."""
        session = Mock()

        key_response = Mock()
        key_response.json.return_value = {
            "data": {
                "usage": {
                    "requests": 100,
                    "tokens": 50000,
                    "spent_usd": 1.50,
                }
            }
        }
        key_response.ok = True

        models_response = Mock()
        models_response.json.return_value = {
            "data": [
                {"id": "gpt-4"},
                {"id": "claude-2"},
            ]
        }
        models_response.ok = True

        session.get.side_effect = [key_response, models_response]
        mock_create_session.return_value = session

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "or-key"}):
            monitor = LLMGatewayMonitor()
            metrics = monitor._collect_openrouter_metrics("or-key")

        assert metrics is not None
        assert metrics["gateway_type"] == "openrouter"
        assert metrics["total_requests_1h"] == 100
        assert metrics["total_tokens_1h"] == 50000
        assert metrics["cost_1h"] == 1.50


class TestCollectCustomGatewayMetrics:
    """Test custom gateway metrics collection."""

    @patch('monitors.llm_gateway_monitor.LLMGatewayMonitor._create_session')
    def test_collect_custom_gateway_metrics_success(self, mock_create_session):
        """Successfully collect custom gateway metrics."""
        session = Mock()

        metrics_response = Mock()
        metrics_response.json.return_value = {
            "total_requests_1h": 75,
            "total_tokens_1h": 35000,
            "requests_by_model": {"model-a": 50, "model-b": 25},
            "tokens_by_model": {"model-a": 20000, "model-b": 15000},
            "avg_latency_ms": 200.0,
            "p95_latency_ms": 500.0,
            "p99_latency_ms": 750.0,
            "error_rate_pct": 0.5,
            "cost_1h": 0.50,
            "cost_by_model": {"model-a": 0.30, "model-b": 0.20},
            "active_models": ["model-a", "model-b"],
            "healthy": True,
        }
        metrics_response.ok = True

        session.get.return_value = metrics_response
        mock_create_session.return_value = session

        with patch.dict(os.environ, {}, clear=True):
            monitor = LLMGatewayMonitor()
            metrics = monitor._collect_custom_gateway_metrics("http://custom:8000", "key")

        assert metrics is not None
        assert metrics["gateway_type"] == "custom"
        assert metrics["total_requests_1h"] == 75

    @patch('monitors.llm_gateway_monitor.LLMGatewayMonitor._create_session')
    def test_collect_custom_gateway_metrics_not_found(self, mock_create_session):
        """Handle missing custom gateway metrics endpoint."""
        session = Mock()

        metrics_response = Mock()
        metrics_response.ok = False
        metrics_response.status_code = 404

        session.get.return_value = metrics_response
        mock_create_session.return_value = session

        with patch.dict(os.environ, {}, clear=True):
            with patch('monitors.llm_gateway_monitor.logger'):
                monitor = LLMGatewayMonitor()
                metrics = monitor._collect_custom_gateway_metrics("http://custom:8000", "key")

        assert metrics is None


class TestProviderHealthMethods:
    """Test provider-specific health check methods."""

    @patch('monitors.llm_gateway_monitor.LLMGatewayMonitor._create_session')
    def test_check_litellm_provider_health(self, mock_create_session):
        """Check provider health via LiteLLM."""
        session = Mock()

        health_response = Mock()
        health_response.json.return_value = {
            "providers": {
                "openai": {
                    "status": "healthy",
                    "latency_ms": 150.0,
                    "error_rate_pct": 0.1,
                    "models": ["gpt-4", "gpt-3.5-turbo"],
                    "last_success": "2024-01-01T00:00:00",
                    "last_error": None,
                }
            }
        }
        health_response.ok = True

        session.get.return_value = health_response
        mock_create_session.return_value = session

        with patch.dict(os.environ, {}, clear=True):
            monitor = LLMGatewayMonitor()
            health = monitor._check_litellm_provider_health("http://localhost:8000", "key")

        assert len(health) == 2  # 2 models
        assert health[0]["provider"] == "openai"
        assert health[0]["status"] == "healthy"


class TestExtractProviderFromModel:
    """Test provider extraction from model name."""

    def test_extract_provider_openai(self):
        """Extract OpenAI from model name."""
        assert LLMGatewayMonitor._extract_provider_from_model("gpt-4") == "OpenAI"
        assert LLMGatewayMonitor._extract_provider_from_model("gpt-3.5-turbo") == "OpenAI"
        assert LLMGatewayMonitor._extract_provider_from_model("openai/gpt-4") == "OpenAI"

    def test_extract_provider_anthropic(self):
        """Extract Anthropic from model name."""
        assert LLMGatewayMonitor._extract_provider_from_model("claude-2") == "Anthropic"
        assert LLMGatewayMonitor._extract_provider_from_model("claude-instant") == "Anthropic"

    def test_extract_provider_google(self):
        """Extract Google from model name."""
        assert LLMGatewayMonitor._extract_provider_from_model("gemini-pro") == "Google"
        assert LLMGatewayMonitor._extract_provider_from_model("google/palm") == "Google"

    def test_extract_provider_meta(self):
        """Extract Meta from model name."""
        assert LLMGatewayMonitor._extract_provider_from_model("llama-2") == "Meta"
        assert LLMGatewayMonitor._extract_provider_from_model("llama2-70b") == "Meta"

    def test_extract_provider_unknown(self):
        """Unknown provider returns Unknown."""
        assert LLMGatewayMonitor._extract_provider_from_model("custom-model") == "Unknown"
        assert LLMGatewayMonitor._extract_provider_from_model("unknown-provider") == "Unknown"


class TestGetRateLimitStatus:
    """Test rate limit status collection."""

    @patch('monitors.llm_gateway_monitor.LLMGatewayMonitor._create_session')
    @patch('monitors.llm_gateway_monitor.LLMGatewayMonitor._get_litellm_rate_limits')
    @patch('monitors.llm_gateway_monitor.LLMGatewayMonitor._get_openrouter_rate_limits')
    @patch('monitors.llm_gateway_monitor.LLMGatewayMonitor._get_custom_gateway_rate_limits')
    def test_get_rate_limit_status(self, mock_custom_limits, mock_openrouter_limits, mock_litellm_limits, mock_create_session):
        """Collect rate limit status from gateways."""
        mock_create_session.return_value = Mock(spec=requests.Session)
        mock_litellm_limits.return_value = [
            {
                "provider": "openai",
                "model": "gpt-4",
                "requests_remaining": 900,
                "tokens_remaining": 45000,
                "reset_time": "2024-01-01T01:00:00",
                "utilization_pct": 10.0,
            },
        ]
        mock_openrouter_limits.return_value = []
        mock_custom_limits.return_value = []

        env_vars = {
            "LITELLM_ENDPOINT": "http://localhost:8000",
            "LITELLM_API_KEY": "test-key",
        }
        with patch.dict(os.environ, env_vars):
            monitor = LLMGatewayMonitor()
            limits = monitor.get_rate_limit_status()

        assert len(limits) == 1
        assert limits[0]["provider"] == "openai"
        assert limits[0]["requests_remaining"] == 900


class TestCreateSession:
    """Test session creation with retry strategy."""

    @patch('monitors.llm_gateway_monitor.LLMGatewayMonitor._create_session')
    def test_create_session_returns_session(self, mock_create_session):
        """Session creation returns requests.Session object."""
        mock_session = Mock(spec=requests.Session)
        mock_create_session.return_value = mock_session

        with patch.dict(os.environ, {}, clear=True):
            monitor = LLMGatewayMonitor()
            assert isinstance(monitor._session, Mock)

    @patch('monitors.llm_gateway_monitor.LLMGatewayMonitor._create_session')
    def test_session_is_created(self, mock_create_session):
        """Session is created during initialization."""
        mock_session = Mock(spec=requests.Session)
        mock_create_session.return_value = mock_session

        with patch.dict(os.environ, {}, clear=True):
            monitor = LLMGatewayMonitor()
            # Session should be set
            assert monitor._session is mock_session


class TestLiteLLMMetricsHelpers:
    """Test LiteLLM helper methods."""

    @patch('monitors.llm_gateway_monitor.LLMGatewayMonitor._create_session')
    def test_parse_litellm_spend_by_model(self, mock_create_session):
        """Parse request counts by model."""
        mock_create_session.return_value = Mock(spec=requests.Session)

        with patch.dict(os.environ, {}, clear=True):
            monitor = LLMGatewayMonitor()

            spend_data = {
                "logs": [
                    {"model": "gpt-4", "total_tokens": 1000},
                    {"model": "gpt-4", "total_tokens": 500},
                    {"model": "claude-2", "total_tokens": 2000},
                ]
            }

            result = monitor._parse_litellm_spend_by_model(spend_data)
            assert result["gpt-4"] == 2
            assert result["claude-2"] == 1

    @patch('monitors.llm_gateway_monitor.LLMGatewayMonitor._create_session')
    def test_parse_litellm_tokens_by_model(self, mock_create_session):
        """Parse token counts by model."""
        mock_create_session.return_value = Mock(spec=requests.Session)

        with patch.dict(os.environ, {}, clear=True):
            monitor = LLMGatewayMonitor()

            spend_data = {
                "logs": [
                    {"model": "gpt-4", "total_tokens": 1000},
                    {"model": "gpt-4", "total_tokens": 500},
                    {"model": "claude-2", "total_tokens": 2000},
                ]
            }

            result = monitor._parse_litellm_tokens_by_model(spend_data)
            assert result["gpt-4"] == 1500
            assert result["claude-2"] == 2000

    @patch('monitors.llm_gateway_monitor.LLMGatewayMonitor._create_session')
    def test_parse_litellm_cost_by_model(self, mock_create_session):
        """Parse cost by model."""
        mock_create_session.return_value = Mock(spec=requests.Session)

        with patch.dict(os.environ, {}, clear=True):
            monitor = LLMGatewayMonitor()

            spend_data = {
                "logs": [
                    {"model": "gpt-4", "cost": 0.05},
                    {"model": "gpt-4", "cost": 0.03},
                    {"model": "claude-2", "cost": 0.02},
                ]
            }

            result = monitor._parse_litellm_cost_by_model(spend_data)
            assert result["gpt-4"] == 0.08
            assert result["claude-2"] == 0.02

    @patch('monitors.llm_gateway_monitor.LLMGatewayMonitor._create_session')
    def test_calculate_litellm_cost(self, mock_create_session):
        """Calculate total cost from logs."""
        mock_create_session.return_value = Mock(spec=requests.Session)

        with patch.dict(os.environ, {}, clear=True):
            monitor = LLMGatewayMonitor()

            spend_data = {
                "logs": [
                    {"cost": 0.05},
                    {"cost": 0.03},
                    {"cost": 0.02},
                ]
            }

            result = monitor._calculate_litellm_cost(spend_data)
            assert result == 0.1

    @patch('monitors.llm_gateway_monitor.LLMGatewayMonitor._create_session')
    def test_calculate_litellm_tokens(self, mock_create_session):
        """Calculate total tokens from logs."""
        mock_create_session.return_value = Mock(spec=requests.Session)

        with patch.dict(os.environ, {}, clear=True):
            monitor = LLMGatewayMonitor()

            spend_data = {
                "logs": [
                    {"total_tokens": 1000},
                    {"total_tokens": 2000},
                    {"total_tokens": 1500},
                ]
            }

            result = monitor._calculate_litellm_tokens(spend_data)
            assert result == 4500

    @patch('monitors.llm_gateway_monitor.LLMGatewayMonitor._create_session')
    def test_extract_litellm_avg_latency(self, mock_create_session):
        """Extract average latency from logs."""
        mock_create_session.return_value = Mock(spec=requests.Session)

        with patch.dict(os.environ, {}, clear=True):
            monitor = LLMGatewayMonitor()

            spend_data = {
                "logs": [
                    {"latency_ms": 100},
                    {"latency_ms": 200},
                    {"latency_ms": 300},
                ]
            }

            result = monitor._extract_litellm_avg_latency(spend_data)
            assert result == 200.0

    @patch('monitors.llm_gateway_monitor.LLMGatewayMonitor._create_session')
    def test_extract_litellm_p95_latency(self, mock_create_session):
        """Extract 95th percentile latency."""
        mock_create_session.return_value = Mock(spec=requests.Session)

        with patch.dict(os.environ, {}, clear=True):
            monitor = LLMGatewayMonitor()

            spend_data = {
                "logs": [
                    {"latency_ms": i * 10} for i in range(1, 101)
                ]
            }

            result = monitor._extract_litellm_p95_latency(spend_data)
            assert result > 0

    @patch('monitors.llm_gateway_monitor.LLMGatewayMonitor._create_session')
    def test_extract_litellm_p99_latency(self, mock_create_session):
        """Extract 99th percentile latency."""
        mock_create_session.return_value = Mock(spec=requests.Session)

        with patch.dict(os.environ, {}, clear=True):
            monitor = LLMGatewayMonitor()

            spend_data = {
                "logs": [
                    {"latency_ms": i * 10} for i in range(1, 101)
                ]
            }

            result = monitor._extract_litellm_p99_latency(spend_data)
            assert result > 0

    @patch('monitors.llm_gateway_monitor.LLMGatewayMonitor._create_session')
    def test_extract_litellm_error_rate(self, mock_create_session):
        """Extract error rate from logs."""
        mock_create_session.return_value = Mock(spec=requests.Session)

        with patch.dict(os.environ, {}, clear=True):
            monitor = LLMGatewayMonitor()

            spend_data = {
                "logs": [
                    {"status": "success"},
                    {"status": "success"},
                    {"status": "error"},
                ]
            }

            result = monitor._extract_litellm_error_rate(spend_data)
            assert result == pytest.approx(33.33, 0.1)

    @patch('monitors.llm_gateway_monitor.LLMGatewayMonitor._create_session')
    def test_calculate_litellm_fallback_rate(self, mock_create_session):
        """Calculate fallback rate from logs."""
        mock_create_session.return_value = Mock(spec=requests.Session)

        with patch.dict(os.environ, {}, clear=True):
            monitor = LLMGatewayMonitor()

            spend_data = {
                "logs": [
                    {"fallback": True},
                    {"fallback": False},
                    {"fallback": False},
                ]
            }

            result = monitor._calculate_litellm_fallback_rate(spend_data)
            assert result == pytest.approx(33.33, 0.1)


class TestOpenRouterHelpers:
    """Test OpenRouter helper methods."""

    @patch('monitors.llm_gateway_monitor.LLMGatewayMonitor._create_session')
    def test_check_openrouter_provider_health_success(self, mock_create_session):
        """Check OpenRouter provider health successfully."""
        session = Mock()

        models_response = Mock()
        models_response.json.return_value = {
            "data": [
                {"id": "gpt-4"},
                {"id": "claude-2"},
            ]
        }
        models_response.ok = True

        session.get.return_value = models_response
        mock_create_session.return_value = session

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "or-key"}):
            monitor = LLMGatewayMonitor()
            health = monitor._check_openrouter_provider_health("or-key")

        assert len(health) == 2
        assert all(h["status"] == "healthy" for h in health)

    @patch('monitors.llm_gateway_monitor.LLMGatewayMonitor._create_session')
    def test_check_openrouter_provider_health_error(self, mock_create_session):
        """Handle OpenRouter health check error."""
        session = Mock()
        session.get.side_effect = Exception("Connection error")
        mock_create_session.return_value = session

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "or-key"}):
            with patch('monitors.llm_gateway_monitor.logger'):
                monitor = LLMGatewayMonitor()
                health = monitor._check_openrouter_provider_health("or-key")

        assert health == []

    @patch('monitors.llm_gateway_monitor.LLMGatewayMonitor._create_session')
    def test_get_openrouter_rate_limits_success(self, mock_create_session):
        """Get OpenRouter rate limits successfully."""
        session = Mock()

        key_response = Mock()
        key_response.json.return_value = {
            "data": {
                "limits": {
                    "requests_remaining": 900,
                    "tokens_remaining": 45000,
                    "reset_time": "2024-01-01T01:00:00",
                }
            }
        }
        key_response.ok = True

        session.get.return_value = key_response
        mock_create_session.return_value = session

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "or-key"}):
            monitor = LLMGatewayMonitor()
            limits = monitor._get_openrouter_rate_limits("or-key")

        assert len(limits) == 1
        assert limits[0]["provider"] == "openrouter"
        assert limits[0]["requests_remaining"] == 900

    @patch('monitors.llm_gateway_monitor.LLMGatewayMonitor._create_session')
    def test_get_openrouter_rate_limits_error(self, mock_create_session):
        """Handle OpenRouter rate limits error."""
        session = Mock()
        session.get.side_effect = Exception("Connection error")
        mock_create_session.return_value = session

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "or-key"}):
            with patch('monitors.llm_gateway_monitor.logger'):
                monitor = LLMGatewayMonitor()
                limits = monitor._get_openrouter_rate_limits("or-key")

        assert limits == []


class TestCustomGatewayHelpers:
    """Test custom gateway helper methods."""

    @patch('monitors.llm_gateway_monitor.LLMGatewayMonitor._create_session')
    def test_check_custom_gateway_provider_health_success(self, mock_create_session):
        """Check custom gateway provider health successfully."""
        session = Mock()

        health_response = Mock()
        health_response.json.return_value = {
            "providers": [
                {
                    "name": "openai",
                    "model": "gpt-4",
                    "status": "healthy",
                    "latency_ms": 150.0,
                    "error_rate_pct": 0.1,
                    "last_success": "2024-01-01T00:00:00",
                    "last_error": None,
                }
            ]
        }
        health_response.ok = True

        session.get.return_value = health_response
        mock_create_session.return_value = session

        with patch.dict(os.environ, {}, clear=True):
            monitor = LLMGatewayMonitor()
            health = monitor._check_custom_gateway_provider_health("http://custom:8000", "key")

        assert len(health) == 1
        assert health[0]["provider"] == "openai"
        assert health[0]["status"] == "healthy"

    @patch('monitors.llm_gateway_monitor.LLMGatewayMonitor._create_session')
    def test_check_custom_gateway_provider_health_error(self, mock_create_session):
        """Handle custom gateway health check error."""
        session = Mock()
        session.get.side_effect = Exception("Connection error")
        mock_create_session.return_value = session

        with patch.dict(os.environ, {}, clear=True):
            with patch('monitors.llm_gateway_monitor.logger'):
                monitor = LLMGatewayMonitor()
                health = monitor._check_custom_gateway_provider_health("http://custom:8000", "key")

        assert health == []

    @patch('monitors.llm_gateway_monitor.LLMGatewayMonitor._create_session')
    def test_get_custom_gateway_rate_limits_success(self, mock_create_session):
        """Get custom gateway rate limits successfully."""
        session = Mock()

        limits_response = Mock()
        limits_response.json.return_value = {
            "limits": [
                {
                    "provider": "openai",
                    "model": "gpt-4",
                    "requests_remaining": 900,
                    "tokens_remaining": 45000,
                    "reset_time": "2024-01-01T01:00:00",
                    "utilization_pct": 10.0,
                }
            ]
        }
        limits_response.ok = True

        session.get.return_value = limits_response
        mock_create_session.return_value = session

        with patch.dict(os.environ, {}, clear=True):
            monitor = LLMGatewayMonitor()
            limits = monitor._get_custom_gateway_rate_limits("http://custom:8000", "key")

        assert len(limits) == 1
        assert limits[0]["provider"] == "openai"

    @patch('monitors.llm_gateway_monitor.LLMGatewayMonitor._create_session')
    def test_get_custom_gateway_rate_limits_error(self, mock_create_session):
        """Handle custom gateway rate limits error."""
        session = Mock()
        session.get.side_effect = Exception("Connection error")
        mock_create_session.return_value = session

        with patch.dict(os.environ, {}, clear=True):
            with patch('monitors.llm_gateway_monitor.logger'):
                monitor = LLMGatewayMonitor()
                limits = monitor._get_custom_gateway_rate_limits("http://custom:8000", "key")

        assert limits == []


class TestLiteLLMRateLimits:
    """Test LiteLLM rate limit methods."""

    @patch('monitors.llm_gateway_monitor.LLMGatewayMonitor._create_session')
    def test_get_litellm_rate_limits_success(self, mock_create_session):
        """Get LiteLLM rate limits successfully."""
        session = Mock()

        key_response = Mock()
        key_response.json.return_value = {
            "rate_limits": {
                "openai": {
                    "requests_remaining": 900,
                    "requests_limit": 1000,
                    "tokens_remaining": 45000,
                    "model": "gpt-4",
                    "reset_time": "2024-01-01T01:00:00",
                }
            }
        }
        key_response.ok = True

        session.get.return_value = key_response
        mock_create_session.return_value = session

        with patch.dict(os.environ, {}, clear=True):
            monitor = LLMGatewayMonitor()
            limits = monitor._get_litellm_rate_limits("http://localhost:8000", "key")

        assert len(limits) == 1
        assert limits[0]["provider"] == "openai"
        assert limits[0]["requests_remaining"] == 900

    @patch('monitors.llm_gateway_monitor.LLMGatewayMonitor._create_session')
    def test_get_litellm_rate_limits_error(self, mock_create_session):
        """Handle LiteLLM rate limits error."""
        session = Mock()
        session.get.side_effect = Exception("Connection error")
        mock_create_session.return_value = session

        with patch.dict(os.environ, {}, clear=True):
            with patch('monitors.llm_gateway_monitor.logger'):
                monitor = LLMGatewayMonitor()
                limits = monitor._get_litellm_rate_limits("http://localhost:8000", "key")

        assert limits == []


class TestEdgeCases:
    """Test edge cases and error conditions."""

    @patch('monitors.llm_gateway_monitor.LLMGatewayMonitor._create_session')
    def test_extract_latency_empty_logs(self, mock_create_session):
        """Handle empty logs for latency extraction."""
        mock_create_session.return_value = Mock(spec=requests.Session)

        with patch.dict(os.environ, {}, clear=True):
            monitor = LLMGatewayMonitor()

            spend_data = {"logs": []}

            avg = monitor._extract_litellm_avg_latency(spend_data)
            p95 = monitor._extract_litellm_p95_latency(spend_data)
            p99 = monitor._extract_litellm_p99_latency(spend_data)

            assert avg == 0.0
            assert p95 == 0.0
            assert p99 == 0.0

    @patch('monitors.llm_gateway_monitor.LLMGatewayMonitor._create_session')
    def test_error_rate_empty_logs(self, mock_create_session):
        """Handle empty logs for error rate calculation."""
        mock_create_session.return_value = Mock(spec=requests.Session)

        with patch.dict(os.environ, {}, clear=True):
            monitor = LLMGatewayMonitor()

            spend_data = {"logs": []}

            error_rate = monitor._extract_litellm_error_rate(spend_data)
            assert error_rate == 0.0

    @patch('monitors.llm_gateway_monitor.LLMGatewayMonitor._create_session')
    def test_fallback_rate_empty_logs(self, mock_create_session):
        """Handle empty logs for fallback rate calculation."""
        mock_create_session.return_value = Mock(spec=requests.Session)

        with patch.dict(os.environ, {}, clear=True):
            monitor = LLMGatewayMonitor()

            spend_data = {"logs": []}

            fallback_rate = monitor._calculate_litellm_fallback_rate(spend_data)
            assert fallback_rate == 0.0

    @patch('monitors.llm_gateway_monitor.LLMGatewayMonitor._create_session')
    def test_collect_litellm_metrics_with_failed_requests(self, mock_create_session):
        """Collect LiteLLM metrics when some requests fail."""
        session = Mock()

        spend_response = Mock()
        spend_response.json.return_value = {"logs": []}
        spend_response.ok = True

        model_response = Mock()
        model_response.ok = False

        health_response = Mock()
        health_response.ok = False

        key_response = Mock()
        key_response.ok = False

        session.get.side_effect = [spend_response, model_response, health_response, key_response]
        mock_create_session.return_value = session

        env_vars = {
            "LITELLM_ENDPOINT": "http://localhost:8000",
            "LITELLM_API_KEY": "test-key",
        }
        with patch.dict(os.environ, env_vars):
            monitor = LLMGatewayMonitor()
            metrics = monitor._collect_litellm_metrics("http://localhost:8000", "test-key")

        assert metrics is not None
        assert metrics["healthy"] is False

    @patch('monitors.llm_gateway_monitor.LLMGatewayMonitor._create_session')
    def test_collect_openrouter_metrics_failed_response(self, mock_create_session):
        """Handle OpenRouter response failures."""
        session = Mock()

        key_response = Mock()
        key_response.ok = False

        models_response = Mock()
        models_response.ok = False

        session.get.side_effect = [key_response, models_response]
        mock_create_session.return_value = session

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "or-key"}):
            monitor = LLMGatewayMonitor()
            metrics = monitor._collect_openrouter_metrics("or-key")

        assert metrics is not None
        assert metrics["total_requests_1h"] == 0
        assert metrics["active_models"] == []

    @patch('monitors.llm_gateway_monitor.LLMGatewayMonitor._create_session')
    def test_collect_custom_gateway_metrics_no_response(self, mock_create_session):
        """Handle missing custom gateway response."""
        session = Mock()

        metrics_response = Mock()
        metrics_response.ok = False
        metrics_response.status_code = 500

        session.get.return_value = metrics_response
        mock_create_session.return_value = session

        with patch.dict(os.environ, {}, clear=True):
            with patch('monitors.llm_gateway_monitor.logger'):
                monitor = LLMGatewayMonitor()
                metrics = monitor._collect_custom_gateway_metrics("http://custom:8000", "key")

        assert metrics is None

    @patch('monitors.llm_gateway_monitor.LLMGatewayMonitor._create_session')
    def test_extract_provider_case_insensitive(self, mock_create_session):
        """Provider extraction is case insensitive."""
        with patch.dict(os.environ, {}, clear=True):
            monitor = LLMGatewayMonitor()

            assert monitor._extract_provider_from_model("GPT-4") == "OpenAI"
            assert monitor._extract_provider_from_model("CLAUDE-2") == "Anthropic"
            assert monitor._extract_provider_from_model("GEMINI-PRO") == "Google"

    @patch('monitors.llm_gateway_monitor.LLMGatewayMonitor._create_session')
    def test_model_routing_stats_with_zero_tokens(self, mock_create_session):
        """Handle cost calculation with zero tokens."""
        mock_create_session.return_value = Mock(spec=requests.Session)

        with patch.object(LLMGatewayMonitor, 'get_gateway_metrics', return_value=[
            {
                "requests_by_model": {"model-a": 10},
                "tokens_by_model": {"model-a": 0},  # Zero tokens
                "cost_by_model": {"model-a": 0.0},
            }
        ]):
            with patch.dict(os.environ, {}, clear=True):
                monitor = LLMGatewayMonitor()
                stats = monitor.get_model_routing_stats()

                assert len(stats) == 1
                assert stats[0]["cost_per_1k_tokens"] == 0.0


@pytest.mark.unit
class TestLLMGatewayExceptionPaths:
    """Test exception handling paths."""

    @patch('monitors.llm_gateway_monitor.LLMGatewayMonitor._create_session')
    def test_get_gateway_metrics_litellm_exception(self, mock_create_session):
        """Should handle exception when collecting LiteLLM metrics."""
        mock_create_session.return_value = Mock(spec=requests.Session)

        env_vars = {
            "LITELLM_ENDPOINT": "http://localhost:8000",
            "LITELLM_API_KEY": "test-key",
        }
        with patch.dict(os.environ, env_vars):
            monitor = LLMGatewayMonitor()

            with patch.object(monitor, '_collect_litellm_metrics', side_effect=Exception('Connection failed')):
                with patch('monitors.llm_gateway_monitor.logger') as mock_logger:
                    metrics = monitor.get_gateway_metrics()

                    # Should have logged the error
                    error_calls = [c for c in mock_logger.error.call_args_list
                                 if 'LiteLLM' in str(c)]
                    assert len(error_calls) > 0

    @patch('monitors.llm_gateway_monitor.LLMGatewayMonitor._create_session')
    def test_get_gateway_metrics_openrouter_exception(self, mock_create_session):
        """Should handle exception when collecting OpenRouter metrics."""
        mock_create_session.return_value = Mock(spec=requests.Session)

        env_vars = {"OPENROUTER_API_KEY": "or-test-key"}
        with patch.dict(os.environ, env_vars):
            monitor = LLMGatewayMonitor()

            with patch.object(monitor, '_collect_openrouter_metrics', side_effect=Exception('API error')):
                with patch('monitors.llm_gateway_monitor.logger') as mock_logger:
                    metrics = monitor.get_gateway_metrics()

                    error_calls = [c for c in mock_logger.error.call_args_list
                                 if 'OpenRouter' in str(c)]
                    assert len(error_calls) > 0

    @patch('monitors.llm_gateway_monitor.LLMGatewayMonitor._create_session')
    def test_get_gateway_metrics_custom_gateway_exception(self, mock_create_session):
        """Should handle exception when collecting custom gateway metrics."""
        mock_create_session.return_value = Mock(spec=requests.Session)

        env_vars = {
            "CUSTOM_GATEWAY_ENDPOINTS": "http://custom:8000",
            "CUSTOM_GATEWAY_API_KEYS": "custom-key",
        }
        with patch.dict(os.environ, env_vars):
            monitor = LLMGatewayMonitor()

            with patch.object(monitor, '_collect_custom_gateway_metrics', side_effect=Exception('Gateway error')):
                with patch('monitors.llm_gateway_monitor.logger') as mock_logger:
                    metrics = monitor.get_gateway_metrics()

                    error_calls = [c for c in mock_logger.error.call_args_list
                                 if 'custom gateway' in str(c)]
                    assert len(error_calls) > 0

    @patch('monitors.llm_gateway_monitor.LLMGatewayMonitor._create_session')
    def test_get_provider_health_litellm_exception(self, mock_create_session):
        """Should handle exception when checking LiteLLM provider health."""
        mock_create_session.return_value = Mock(spec=requests.Session)

        env_vars = {
            "LITELLM_ENDPOINT": "http://localhost:8000",
            "LITELLM_API_KEY": "test-key",
        }
        with patch.dict(os.environ, env_vars):
            monitor = LLMGatewayMonitor()

            with patch.object(monitor, '_check_litellm_provider_health', side_effect=Exception('Health check failed')):
                with patch('monitors.llm_gateway_monitor.logger') as mock_logger:
                    health = monitor.get_provider_health()

                    error_calls = [c for c in mock_logger.error.call_args_list
                                 if 'LiteLLM' in str(c)]
                    assert len(error_calls) > 0

    @patch('monitors.llm_gateway_monitor.LLMGatewayMonitor._create_session')
    def test_get_provider_health_openrouter_exception(self, mock_create_session):
        """Should handle exception when checking OpenRouter provider health."""
        mock_create_session.return_value = Mock(spec=requests.Session)

        env_vars = {"OPENROUTER_API_KEY": "or-test-key"}
        with patch.dict(os.environ, env_vars):
            monitor = LLMGatewayMonitor()

            with patch.object(monitor, '_check_openrouter_provider_health', side_effect=Exception('Health check failed')):
                with patch('monitors.llm_gateway_monitor.logger') as mock_logger:
                    health = monitor.get_provider_health()

                    error_calls = [c for c in mock_logger.error.call_args_list
                                 if 'OpenRouter' in str(c)]
                    assert len(error_calls) > 0

    @patch('monitors.llm_gateway_monitor.LLMGatewayMonitor._create_session')
    def test_get_provider_health_custom_gateway_exception(self, mock_create_session):
        """Should handle exception when checking custom gateway provider health."""
        mock_create_session.return_value = Mock(spec=requests.Session)

        env_vars = {
            "CUSTOM_GATEWAY_ENDPOINTS": "http://custom:8000",
            "CUSTOM_GATEWAY_API_KEYS": "custom-key",
        }
        with patch.dict(os.environ, env_vars):
            monitor = LLMGatewayMonitor()

            with patch.object(monitor, '_check_custom_gateway_provider_health', side_effect=Exception('Gateway health check failed')):
                with patch('monitors.llm_gateway_monitor.logger') as mock_logger:
                    health = monitor.get_provider_health()

                    error_calls = [c for c in mock_logger.error.call_args_list
                                 if 'custom gateway' in str(c)]
                    assert len(error_calls) > 0

    @patch('monitors.llm_gateway_monitor.LLMGatewayMonitor._create_session')
    def test_create_session_returns_session(self, mock_create_session):
        """Should create session with retry strategy."""
        mock_create_session.return_value = Mock(spec=requests.Session)

        with patch.dict(os.environ, {}, clear=True):
            monitor = LLMGatewayMonitor()
            session = monitor._create_session()

            assert session is not None
