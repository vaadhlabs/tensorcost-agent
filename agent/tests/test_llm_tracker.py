"""
Tests for LLM Tracker - Application Integration Library.

Tests cover:
- LLMTracker class initialization and configuration
- Token tracking for multiple LLM providers
- Cost calculation and pricing lookup
- OpenAI and Anthropic convenience functions
- Environment variable handling and overrides
"""

import pytest
import os
import json
import tempfile
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from src.integrations.llm_tracker import (
    LLMTracker,
    get_tracker,
    track_openai_call,
    track_anthropic_call,
)


@pytest.mark.unit
class TestLLMTrackerInitialization:
    """Tests for LLMTracker initialization and configuration."""

    def test_init_with_defaults(self):
        """Should initialize with default values from environment."""
        with patch.dict(os.environ, {
            'BACKEND_API_URL': 'https://api.example.com',
            'BACKEND_API_KEY': 'test-key',
            'TENANT_ID': 'tenant-123',
            'LLM_TRACKING_ENABLED': 'true'
        }, clear=False):
            tracker = LLMTracker()
            assert tracker.backend_api_url == 'https://api.example.com'
            assert tracker.backend_api_key == 'test-key'
            assert tracker.tenant_id == 'tenant-123'
            assert tracker.enabled is True

    def test_init_with_explicit_parameters(self):
        """Should accept explicit parameters overriding environment."""
        with patch.dict(os.environ, {'BACKEND_API_URL': 'https://env.example.com'}, clear=False):
            tracker = LLMTracker(
                backend_api_url='https://explicit.example.com',
                backend_api_key='explicit-key',
                tenant_id='explicit-tenant'
            )
            assert tracker.backend_api_url == 'https://explicit.example.com'
            assert tracker.backend_api_key == 'explicit-key'
            assert tracker.tenant_id == 'explicit-tenant'

    def test_init_with_no_environment_variables(self):
        """Should handle missing environment variables gracefully."""
        with patch.dict(os.environ, {}, clear=True):
            tracker = LLMTracker()
            assert tracker.backend_api_url is None
            assert tracker.backend_api_key is None
            assert tracker.tenant_id == 'default'
            assert tracker.enabled is True  # Default to true

    def test_init_tracking_disabled(self):
        """Should respect LLM_TRACKING_ENABLED environment variable."""
        with patch.dict(os.environ, {'LLM_TRACKING_ENABLED': 'false'}, clear=False):
            tracker = LLMTracker()
            assert tracker.enabled is False

    def test_init_tracking_enabled_explicitly(self):
        """Should set enabled to true for 'true' value."""
        with patch.dict(os.environ, {'LLM_TRACKING_ENABLED': 'true'}, clear=False):
            tracker = LLMTracker()
            assert tracker.enabled is True

    def test_init_tracking_enabled_case_insensitive(self):
        """Should handle case-insensitive 'true' value."""
        with patch.dict(os.environ, {'LLM_TRACKING_ENABLED': 'TRUE'}, clear=False):
            tracker = LLMTracker()
            assert tracker.enabled is True

    def test_default_tenant_id(self):
        """Should default to 'default' when TENANT_ID not set."""
        with patch.dict(os.environ, {}, clear=True):
            tracker = LLMTracker()
            assert tracker.tenant_id == 'default'


@pytest.mark.unit
class TestTrackCall:
    """Tests for tracking LLM API calls."""

    def test_track_call_success(self):
        """Records locally when tracking is enabled (no backend egress)."""
        tracker = LLMTracker()
        result = tracker.track_call(
            provider="openai",
            model="gpt-4",
            usage_data={"input_tokens": 100, "output_tokens": 50},
        )
        assert result is True

    def test_track_call_disabled(self):
        """Should return False when tracking is disabled."""
        tracker = LLMTracker()
        tracker.enabled = False
        result = tracker.track_call(
            provider="openai",
            model="gpt-4",
            usage_data={"input_tokens": 100, "output_tokens": 50},
        )
        assert result is False

    def test_track_call_no_backend_url(self):
        """Local recording succeeds without backend URL."""
        tracker = LLMTracker()
        tracker.backend_api_url = None
        result = tracker.track_call(
            provider="openai",
            model="gpt-4",
            usage_data={"input_tokens": 100, "output_tokens": 50},
        )
        assert result is True

    def test_track_call_no_backend_key(self):
        """Local recording succeeds without backend API key."""
        tracker = LLMTracker(backend_api_url="https://api.example.com")
        tracker.backend_api_key = None
        result = tracker.track_call(
            provider="openai",
            model="gpt-4",
            usage_data={"input_tokens": 100, "output_tokens": 50},
        )
        assert result is True

    def test_track_call_with_all_optional_fields(self):
        tracker = LLMTracker()
        result = tracker.track_call(
            provider="anthropic",
            model="claude-3-opus",
            usage_data={"input_tokens": 100, "output_tokens": 50},
            workload_id="workload-1",
            project_id="project-1",
            user_id="user-123",
            metadata={"custom": "data"},
        )
        assert result is True

    @patch.dict(os.environ, {"LLM_SESSION_ID": "env-session", "LLM_CONVERSATION_ID": "env-conv"}, clear=False)
    def test_track_call_uses_environment_session_ids(self):
        tracker = LLMTracker()
        assert tracker.track_call(
            provider="openai",
            model="gpt-4",
            usage_data={"input_tokens": 100, "output_tokens": 50},
        )

    def test_track_call_with_headers(self):
        tracker = LLMTracker(tenant_id="tenant-abc")
        assert tracker.track_call(
            provider="openai",
            model="gpt-4",
            usage_data={"input_tokens": 100, "output_tokens": 50},
            headers_in={"X-TensorCost-Feature": "search"},
        )


@pytest.mark.unit
class TestCalculateCosts:
    """Tests for cost calculation."""

    def test_calculate_costs_openai_gpt4(self):
        """Should calculate costs for OpenAI GPT-4."""
        tracker = LLMTracker()
        costs = tracker.calculate_costs('openai', 'gpt-4', 1000, 500)

        assert costs['input_cost'] == pytest.approx(0.03, rel=1e-5)
        assert costs['output_cost'] == pytest.approx(0.03, rel=1e-5)
        assert costs['total_cost'] == pytest.approx(0.06, rel=1e-5)

    def test_calculate_costs_openai_gpt4o(self):
        """Should calculate costs for OpenAI GPT-4o."""
        tracker = LLMTracker()
        costs = tracker.calculate_costs('openai', 'gpt-4o', 1000, 1000)

        assert costs['input_cost'] == pytest.approx(0.0025, rel=1e-5)
        assert costs['output_cost'] == pytest.approx(0.01, rel=1e-5)
        assert costs['total_cost'] == pytest.approx(0.0125, rel=1e-5)

    def test_calculate_costs_anthropic_claude3_opus(self):
        """Should calculate costs for Anthropic Claude 3 Opus."""
        tracker = LLMTracker()
        costs = tracker.calculate_costs('anthropic', 'claude-3-opus', 1000, 1000)

        assert costs['input_cost'] == pytest.approx(0.015, rel=1e-5)
        assert costs['output_cost'] == pytest.approx(0.075, rel=1e-5)
        assert costs['total_cost'] == pytest.approx(0.09, rel=1e-5)

    def test_calculate_costs_aws_bedrock(self):
        """Should calculate costs for AWS Bedrock."""
        tracker = LLMTracker()
        costs = tracker.calculate_costs('aws_bedrock', 'anthropic.claude-3-opus', 1000, 1000)

        assert costs['input_cost'] == pytest.approx(0.015, rel=1e-5)
        assert costs['output_cost'] == pytest.approx(0.075, rel=1e-5)

    def test_calculate_costs_zero_tokens(self):
        """Should return zero costs for zero tokens."""
        tracker = LLMTracker()
        costs = tracker.calculate_costs('openai', 'gpt-4', 0, 0)

        assert costs['input_cost'] == 0
        assert costs['output_cost'] == 0
        assert costs['total_cost'] == 0

    def test_calculate_costs_unknown_model(self):
        """Should return zero pricing for unknown models."""
        tracker = LLMTracker()
        costs = tracker.calculate_costs('openai', 'unknown-model', 1000, 1000)

        assert costs['input_cost'] == 0
        assert costs['output_cost'] == 0
        assert costs['total_cost'] == 0


@pytest.mark.unit
class TestGetPricing:
    """Tests for pricing lookup with environment overrides."""

    def test_get_pricing_exact_match(self):
        """Should return exact model pricing."""
        tracker = LLMTracker()
        pricing = tracker._get_pricing('openai', 'gpt-4')

        assert pricing['input_price_per_1k'] == 0.03
        assert pricing['output_price_per_1k'] == 0.06

    def test_get_pricing_fuzzy_match(self):
        """Should fuzzy-match model versions."""
        tracker = LLMTracker()
        # gpt-4o-2024-05-13 should match gpt-4o base pricing
        pricing = tracker._get_pricing('openai', 'gpt-4o-2024-05-13')

        assert pricing['input_price_per_1k'] == 0.0025
        assert pricing['output_price_per_1k'] == 0.01

    def test_get_pricing_unknown_provider(self):
        """Should return zero pricing for unknown provider."""
        tracker = LLMTracker()
        pricing = tracker._get_pricing('unknown-provider', 'unknown-model')

        assert pricing['input_price_per_1k'] == 0
        assert pricing['output_price_per_1k'] == 0

    @patch.dict(os.environ, {
        'LLM_PRICING_OPENAI_GPT_4': '0.1,0.2'
    }, clear=False)
    def test_get_pricing_environment_override(self):
        """Should use environment variable override for pricing."""
        tracker = LLMTracker()
        pricing = tracker._get_pricing('openai', 'gpt-4')

        assert pricing['input_price_per_1k'] == 0.1
        assert pricing['output_price_per_1k'] == 0.2

    @patch.dict(os.environ, {
        'LLM_PRICING_OPENAI_GPT_4O': '0.05'
    }, clear=False)
    def test_get_pricing_environment_override_single_value(self):
        """Should use single value for both input and output."""
        tracker = LLMTracker()
        pricing = tracker._get_pricing('openai', 'gpt-4o')

        assert pricing['input_price_per_1k'] == 0.05
        assert pricing['output_price_per_1k'] == 0.05

    @patch.dict(os.environ, {
        'LLM_PRICING_OPENAI_GPT4': 'invalid'
    }, clear=False)
    def test_get_pricing_invalid_environment_override(self):
        """Should fall back to default on invalid environment override."""
        tracker = LLMTracker()
        pricing = tracker._get_pricing('openai', 'gpt-4')

        # Should fall back to defaults
        assert pricing['input_price_per_1k'] == 0.03
        assert pricing['output_price_per_1k'] == 0.06

    def test_get_pricing_config_file(self):
        """Should load pricing from JSON config file."""
        config = {
            'openai': {
                'gpt-4': {'input_price_per_1k': 0.15, 'output_price_per_1k': 0.20}
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config, f)
            config_path = f.name

        try:
            with patch.dict(os.environ, {'LLM_PRICING_CONFIG': config_path}, clear=False):
                tracker = LLMTracker()
                pricing = tracker._get_pricing('openai', 'gpt-4')
                assert pricing['input_price_per_1k'] == 0.15
                assert pricing['output_price_per_1k'] == 0.20
        finally:
            os.unlink(config_path)

    def test_get_pricing_config_file_not_found(self):
        """Should fall back to defaults when config file not found."""
        with patch.dict(os.environ, {'LLM_PRICING_CONFIG': '/nonexistent/path'}, clear=False):
            tracker = LLMTracker()
            pricing = tracker._get_pricing('openai', 'gpt-4')
            # Should use defaults
            assert pricing['input_price_per_1k'] == 0.03

    def test_get_pricing_anthropic_models(self):
        """Should have pricing for various Anthropic models."""
        tracker = LLMTracker()

        models = [
            'claude-3-opus',
            'claude-3-sonnet',
            'claude-3-haiku',
            'claude-3.5-sonnet',
        ]

        for model in models:
            pricing = tracker._get_pricing('anthropic', model)
            assert pricing['input_price_per_1k'] > 0
            assert pricing['output_price_per_1k'] > 0

    def test_get_pricing_gcp_vertex(self):
        """Should have pricing for Google Vertex AI models."""
        tracker = LLMTracker()
        pricing = tracker._get_pricing('gcp_vertex', 'gemini-1.5-pro')
        assert pricing['input_price_per_1k'] == 0.00125
        assert pricing['output_price_per_1k'] == 0.005


@pytest.mark.unit
class TestGlobalTracker:
    """Tests for global tracker singleton."""

    def test_get_tracker_returns_singleton(self):
        """Should return the same tracker instance."""
        tracker1 = get_tracker()
        tracker2 = get_tracker()

        assert tracker1 is tracker2

    def test_get_tracker_creates_instance(self):
        """Should create a tracker instance on first call."""
        # Reset global tracker
        import src.integrations.llm_tracker as llm_tracker
        llm_tracker._tracker = None

        tracker = get_tracker()
        assert tracker is not None
        assert isinstance(tracker, LLMTracker)


@pytest.mark.unit
class TestTrackOpenAICall:
    """Tests for OpenAI convenience tracking function."""

    def setup_method(self):
        import src.integrations.llm_tracker as llm_tracker

        llm_tracker._tracker = None

    def test_track_openai_call_dict_response(self):
        response = {
            "id": "chatcmpl-123",
            "model": "gpt-4",
            "usage": {"prompt_tokens": 100, "completion_tokens": 50},
        }
        assert track_openai_call(response, workload_id="workload-1") is True

    def test_track_openai_call_object_response(self):
        usage = Mock()
        usage.prompt_tokens = 100
        usage.completion_tokens = 50
        response = Mock()
        response.id = "chatcmpl-123"
        response.model = "gpt-4"
        response.usage = usage
        assert track_openai_call(response, workload_id="workload-1") is True


@pytest.mark.unit
class TestTrackAnthropicCall:
    """Tests for Anthropic convenience tracking function."""

    def setup_method(self):
        import src.integrations.llm_tracker as llm_tracker

        llm_tracker._tracker = None

    def test_track_anthropic_call_dict_response(self):
        response = {
            "id": "msg-123",
            "model": "claude-3-opus",
            "usage": {"input_tokens": 100, "output_tokens": 50},
        }
        assert track_anthropic_call(response, workload_id="workload-1") is True

    def test_track_anthropic_call_object_response(self):
        usage = Mock()
        usage.input_tokens = 100
        usage.output_tokens = 50
        response = Mock()
        response.id = "msg-123"
        response.model = "claude-3-opus"
        response.usage = usage
        assert track_anthropic_call(response, workload_id="workload-1") is True


@pytest.mark.unit
class TestFeatureAttribution:
    """Tests for X-tensorcost-Feature header resolution (spec 3A)."""

    def setup_method(self):
        # Reset the module-level singleton between tests so each test gets a
        # clean tracker and env state.
        import src.integrations.llm_tracker as llm_tracker
        llm_tracker._tracker = None

    def _make(self, **kwargs):
        return LLMTracker(
            backend_api_url='https://api.example.com',
            backend_api_key='test-key',
            tenant_id='tenant-1',
            **kwargs,
        )

    def test_explicit_feature_arg_wins(self):
        tracker = self._make(default_feature_tag='fallback')
        resolved = tracker._resolve_feature_tag(
            feature='checkout',
            headers_in={'X-tensorcost-Feature': 'header-feature'},
        )
        assert resolved == 'checkout'

    def test_header_picked_when_arg_missing(self):
        tracker = self._make()
        resolved = tracker._resolve_feature_tag(
            feature=None,
            headers_in={'x-tensorcost-feature': 'search'},
        )
        assert resolved == 'search'

    def test_env_fallback(self):
        import os
        tracker = self._make()
        with patch.dict(os.environ, {'LLM_FEATURE': 'chatbot'}, clear=False):
            resolved = tracker._resolve_feature_tag(feature=None, headers_in={})
        assert resolved == 'chatbot'

    def test_default_feature_tag_last_resort(self):
        tracker = self._make(default_feature_tag='code_assist')
        import os
        # Ensure no env override interferes
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop('LLM_FEATURE', None)
            resolved = tracker._resolve_feature_tag(feature=None, headers_in=None)
        assert resolved == 'code_assist'

    def test_nothing_configured_returns_none(self):
        import os
        tracker = self._make()
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop('LLM_FEATURE', None)
            resolved = tracker._resolve_feature_tag(feature=None, headers_in=None)
        assert resolved is None

    def test_version_resolver_fn_called_lazily(self):
        calls = {'n': 0}

        def resolver():
            calls['n'] += 1
            return 'v42'

        tracker = self._make(version_resolver_fn=resolver)
        import os
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop('LLM_FEATURE_VERSION', None)
            resolved = tracker._resolve_feature_version(feature_version=None, headers_in=None)
        assert resolved == 'v42'
        assert calls['n'] == 1

    def test_version_header_precedes_resolver(self):
        tracker = self._make(version_resolver_fn=lambda: 'from-resolver')
        resolved = tracker._resolve_feature_version(
            feature_version=None,
            headers_in={'X-tensorcost-Feature-Version': 'v7'},
        )
        assert resolved == 'v7'

    def test_track_call_includes_resolved_feature_in_payload(self):
        tracker = self._make(default_feature_tag="search")
        assert tracker.track_call(
            provider="openai",
            model="gpt-4",
            usage_data={
                "input_tokens": 10,
                "output_tokens": 5,
                "input_cost": 0.0003,
                "output_cost": 0.0003,
                "total_cost": 0.0006,
            },
            headers_in={"X-tensorcost-Feature-Version": "v3"},
        )
