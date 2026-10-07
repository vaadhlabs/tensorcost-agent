"""
Tests for AISpendMonitor class
Tests AI spend tracking from various LLM providers (OpenAI, Anthropic, AWS Bedrock, Azure)
"""

import pytest
import sys
import os
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import json

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from monitors.ai_spend_monitor import AISpendMonitor


class TestAISpendMonitorInitialization:
    """Test AISpendMonitor initialization with various configurations"""

    def test_init_with_no_env_vars(self):
        """Test initialization with no provider environment variables"""
        with patch.dict(os.environ, {}, clear=True):
            monitor = AISpendMonitor()
            assert monitor.providers == {}

    def test_init_with_openai_enabled(self):
        """Test initialization with OpenAI API key"""
        env_vars = {
            'OPENAI_API_KEY': 'sk-test-key-123',
            'OPENAI_ENABLED': 'true'
        }
        with patch.dict(os.environ, env_vars):
            monitor = AISpendMonitor()
            assert 'openai' in monitor.providers
            assert monitor.providers['openai']['api_key'] == 'sk-test-key-123'
            assert monitor.providers['openai']['enabled'] is True

    def test_init_with_openai_disabled(self):
        """Test initialization with OpenAI disabled"""
        env_vars = {
            'OPENAI_API_KEY': 'sk-test-key-123',
            'OPENAI_ENABLED': 'false'
        }
        with patch.dict(os.environ, env_vars):
            monitor = AISpendMonitor()
            assert 'openai' in monitor.providers
            assert monitor.providers['openai']['enabled'] is False

    def test_init_with_anthropic_enabled(self):
        """Test initialization with Anthropic API key"""
        env_vars = {
            'ANTHROPIC_API_KEY': 'sk-ant-test-key-123',
            'ANTHROPIC_ENABLED': 'true'
        }
        with patch.dict(os.environ, env_vars):
            monitor = AISpendMonitor()
            assert 'anthropic' in monitor.providers
            assert monitor.providers['anthropic']['api_key'] == 'sk-ant-test-key-123'
            assert monitor.providers['anthropic']['enabled'] is True

    def test_init_with_bedrock_enabled(self):
        """Test initialization with AWS Bedrock"""
        env_vars = {
            'AWS_ACCESS_KEY_ID': 'AKIAIOSFODNN7EXAMPLE',
            'AWS_SECRET_ACCESS_KEY': 'wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY',
            'BEDROCK_ENABLED': 'true',
            'AWS_REGION': 'us-west-2'
        }
        with patch.dict(os.environ, env_vars):
            monitor = AISpendMonitor()
            assert 'aws_bedrock' in monitor.providers
            assert monitor.providers['aws_bedrock']['enabled'] is True
            assert monitor.providers['aws_bedrock']['region'] == 'us-west-2'

    def test_init_with_bedrock_disabled_by_default(self):
        """Test that Bedrock is disabled by default even with AWS keys"""
        env_vars = {
            'AWS_ACCESS_KEY_ID': 'AKIAIOSFODNN7EXAMPLE',
            'AWS_SECRET_ACCESS_KEY': 'wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY',
        }
        with patch.dict(os.environ, env_vars):
            monitor = AISpendMonitor()
            assert 'aws_bedrock' in monitor.providers
            assert monitor.providers['aws_bedrock']['enabled'] is False

    def test_init_with_azure_openai_enabled(self):
        """Test initialization with Azure OpenAI"""
        env_vars = {
            'AZURE_OPENAI_API_KEY': 'azure-key-123',
            'AZURE_OPENAI_ENDPOINT': 'https://myresource.openai.azure.com/',
            'AZURE_OPENAI_ENABLED': 'true'
        }
        with patch.dict(os.environ, env_vars):
            monitor = AISpendMonitor()
            assert 'azure_openai' in monitor.providers
            assert monitor.providers['azure_openai']['api_key'] == 'azure-key-123'
            assert monitor.providers['azure_openai']['endpoint'] == 'https://myresource.openai.azure.com/'
            assert monitor.providers['azure_openai']['enabled'] is True

    def test_init_with_all_providers(self):
        """Test initialization with all providers enabled"""
        env_vars = {
            'OPENAI_API_KEY': 'sk-test-key-123',
            'OPENAI_ENABLED': 'true',
            'ANTHROPIC_API_KEY': 'sk-ant-test-key-123',
            'ANTHROPIC_ENABLED': 'true',
            'AWS_ACCESS_KEY_ID': 'AKIAIOSFODNN7EXAMPLE',
            'AWS_SECRET_ACCESS_KEY': 'wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY',
            'BEDROCK_ENABLED': 'true',
            'AZURE_OPENAI_API_KEY': 'azure-key-123',
            'AZURE_OPENAI_ENDPOINT': 'https://myresource.openai.azure.com/',
            'AZURE_OPENAI_ENABLED': 'true'
        }
        with patch.dict(os.environ, env_vars):
            monitor = AISpendMonitor()
            assert len(monitor.providers) == 4
            assert 'openai' in monitor.providers
            assert 'anthropic' in monitor.providers
            assert 'aws_bedrock' in monitor.providers
            assert 'azure_openai' in monitor.providers


class TestGetAISpend:
    """Test get_ai_spend method that dispatches to enabled providers"""

    def test_get_ai_spend_no_providers(self):
        """Test get_ai_spend with no configured providers"""
        with patch.dict(os.environ, {}, clear=True):
            monitor = AISpendMonitor()
            result = monitor.get_ai_spend(days=1)
            assert result == []

    def test_get_ai_spend_dispatches_to_enabled_providers(self):
        """Test that get_ai_spend calls methods for enabled providers"""
        env_vars = {
            'OPENAI_API_KEY': 'sk-test-key-123',
            'OPENAI_ENABLED': 'true',
            'ANTHROPIC_API_KEY': 'sk-ant-test-key-123',
            'ANTHROPIC_ENABLED': 'false'
        }
        with patch.dict(os.environ, env_vars):
            monitor = AISpendMonitor()

            with patch.object(monitor, '_get_openai_usage', return_value=[{'data': 'openai'}]) as mock_openai, \
                 patch.object(monitor, '_get_anthropic_usage', return_value=[{'data': 'anthropic'}]) as mock_anthropic:

                result = monitor.get_ai_spend(days=1)

                # OpenAI should be called (enabled)
                mock_openai.assert_called_once_with(1)
                # Anthropic should NOT be called (disabled)
                mock_anthropic.assert_not_called()
                # Result should include openai data
                assert len(result) == 1
                assert result[0]['data'] == 'openai'

    def test_get_ai_spend_aggregates_multiple_providers(self):
        """Test that get_ai_spend aggregates data from multiple providers"""
        env_vars = {
            'OPENAI_API_KEY': 'sk-test-key-123',
            'OPENAI_ENABLED': 'true',
            'ANTHROPIC_API_KEY': 'sk-ant-test-key-123',
            'ANTHROPIC_ENABLED': 'true'
        }
        with patch.dict(os.environ, env_vars):
            monitor = AISpendMonitor()

            openai_data = [{'provider': 'openai', 'cost': 1.50}]
            anthropic_data = [{'provider': 'anthropic', 'cost': 2.50}]

            with patch.object(monitor, '_get_openai_usage', return_value=openai_data), \
                 patch.object(monitor, '_get_anthropic_usage', return_value=anthropic_data):

                result = monitor.get_ai_spend(days=1)

                assert len(result) == 2
                assert result[0]['provider'] == 'openai'
                assert result[1]['provider'] == 'anthropic'


class TestOpenAIUsage:
    """Test _get_openai_usage method"""

    def test_get_openai_usage_no_api_key(self):
        """Test _get_openai_usage returns empty when no API key"""
        with patch.dict(os.environ, {}, clear=True):
            monitor = AISpendMonitor()
            result = monitor._get_openai_usage(days=1)
            assert result == []

    def test_get_openai_usage_successful_api_call(self):
        """Test successful OpenAI usage API call"""
        env_vars = {
            'OPENAI_API_KEY': 'sk-test-key-123',
            'OPENAI_ENABLED': 'true'
        }

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'data': [
                {
                    'start_time': int((datetime.utcnow() - timedelta(days=1)).timestamp()),
                    'results': [
                        {
                            'model': 'gpt-4',
                            'input_tokens': 1000,
                            'output_tokens': 500
                        }
                    ]
                }
            ]
        }

        with patch.dict(os.environ, env_vars):
            monitor = AISpendMonitor()
            with patch('requests.get', return_value=mock_response):
                result = monitor._get_openai_usage(days=1)

                assert len(result) == 1
                assert result[0]['provider'] == 'openai'
                assert result[0]['model_name'] == 'gpt-4'
                assert result[0]['input_tokens'] == 1000
                assert result[0]['output_tokens'] == 500
                assert result[0]['total_tokens'] == 1500
                assert result[0]['spend_type'] == 'llm_inference'

    def test_get_openai_usage_api_403_forbidden(self):
        """Test OpenAI usage API returns 403 Forbidden"""
        env_vars = {
            'OPENAI_API_KEY': 'sk-test-key-123',
            'OPENAI_ENABLED': 'true'
        }

        mock_response = Mock()
        mock_response.status_code = 403

        with patch.dict(os.environ, env_vars):
            monitor = AISpendMonitor()
            with patch('requests.get', return_value=mock_response):
                result = monitor._get_openai_usage(days=1)

                assert result == []

    def test_get_openai_usage_api_error_500(self):
        """Test OpenAI usage API returns 500 error"""
        env_vars = {
            'OPENAI_API_KEY': 'sk-test-key-123',
            'OPENAI_ENABLED': 'true'
        }

        mock_response = Mock()
        mock_response.status_code = 500

        with patch.dict(os.environ, env_vars):
            monitor = AISpendMonitor()
            with patch('requests.get', return_value=mock_response):
                result = monitor._get_openai_usage(days=1)

                assert result == []

    def test_get_openai_usage_exception_handling(self):
        """Test _get_openai_usage handles exceptions gracefully"""
        env_vars = {
            'OPENAI_API_KEY': 'sk-test-key-123',
            'OPENAI_ENABLED': 'true'
        }

        with patch.dict(os.environ, env_vars):
            monitor = AISpendMonitor()
            with patch('requests.get', side_effect=Exception('Network error')):
                result = monitor._get_openai_usage(days=1)

                assert result == []

    def test_get_openai_usage_cost_calculation(self):
        """Test cost calculation for OpenAI usage"""
        env_vars = {
            'OPENAI_API_KEY': 'sk-test-key-123',
            'OPENAI_ENABLED': 'true'
        }

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'data': [
                {
                    'start_time': int((datetime.utcnow() - timedelta(days=1)).timestamp()),
                    'results': [
                        {
                            'model': 'gpt-4',
                            'input_tokens': 1000,
                            'output_tokens': 1000
                        }
                    ]
                }
            ]
        }

        with patch.dict(os.environ, env_vars):
            monitor = AISpendMonitor()
            with patch('requests.get', return_value=mock_response):
                result = monitor._get_openai_usage(days=1)

                # gpt-4: input=$0.03, output=$0.06 per 1k tokens
                # 1000 input tokens = $0.03
                # 1000 output tokens = $0.06
                # total = $0.09
                assert abs(result[0]['input_cost'] - 0.03) < 0.0001
                assert abs(result[0]['output_cost'] - 0.06) < 0.0001
                assert abs(result[0]['total_cost'] - 0.09) < 0.0001


class TestAnthropicUsage:
    """Test _get_anthropic_usage method"""

    def test_get_anthropic_usage_no_api_key(self):
        """Test _get_anthropic_usage returns empty when no API key"""
        with patch.dict(os.environ, {}, clear=True):
            monitor = AISpendMonitor()
            result = monitor._get_anthropic_usage(days=1)
            assert result == []

    def test_get_anthropic_usage_successful_api_call(self):
        """Test successful Anthropic usage API call"""
        env_vars = {
            'ANTHROPIC_API_KEY': 'sk-ant-test-key-123',
            'ANTHROPIC_ENABLED': 'true'
        }

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'data': [
                {
                    'model': 'claude-3-opus',
                    'input_tokens': 2000,
                    'output_tokens': 800,
                    'date': datetime.utcnow().isoformat()
                }
            ]
        }

        with patch.dict(os.environ, env_vars):
            monitor = AISpendMonitor()
            with patch('requests.get', return_value=mock_response):
                result = monitor._get_anthropic_usage(days=1)

                assert len(result) == 1
                assert result[0]['provider'] == 'anthropic'
                assert result[0]['model_name'] == 'claude-3-opus'
                assert result[0]['input_tokens'] == 2000
                assert result[0]['output_tokens'] == 800
                assert result[0]['total_tokens'] == 2800

    def test_get_anthropic_usage_api_403_forbidden(self):
        """Test Anthropic usage API returns 403 Forbidden"""
        env_vars = {
            'ANTHROPIC_API_KEY': 'sk-ant-test-key-123',
            'ANTHROPIC_ENABLED': 'true'
        }

        mock_response = Mock()
        mock_response.status_code = 403

        with patch.dict(os.environ, env_vars):
            monitor = AISpendMonitor()
            with patch('requests.get', return_value=mock_response):
                result = monitor._get_anthropic_usage(days=1)

                assert result == []

    def test_get_anthropic_usage_api_error_500(self):
        """Test Anthropic usage API returns 500 error"""
        env_vars = {
            'ANTHROPIC_API_KEY': 'sk-ant-test-key-123',
            'ANTHROPIC_ENABLED': 'true'
        }

        mock_response = Mock()
        mock_response.status_code = 500

        with patch.dict(os.environ, env_vars):
            monitor = AISpendMonitor()
            with patch('requests.get', return_value=mock_response):
                result = monitor._get_anthropic_usage(days=1)

                assert result == []

    def test_get_anthropic_usage_exception_handling(self):
        """Test _get_anthropic_usage handles exceptions gracefully"""
        env_vars = {
            'ANTHROPIC_API_KEY': 'sk-ant-test-key-123',
            'ANTHROPIC_ENABLED': 'true'
        }

        with patch.dict(os.environ, env_vars):
            monitor = AISpendMonitor()
            with patch('requests.get', side_effect=Exception('Connection timeout')):
                result = monitor._get_anthropic_usage(days=1)

                assert result == []

    def test_get_anthropic_usage_cost_calculation(self):
        """Test cost calculation for Anthropic usage"""
        env_vars = {
            'ANTHROPIC_API_KEY': 'sk-ant-test-key-123',
            'ANTHROPIC_ENABLED': 'true'
        }

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'data': [
                {
                    'model': 'claude-3-sonnet',
                    'input_tokens': 1000,
                    'output_tokens': 1000,
                    'date': datetime.utcnow().isoformat()
                }
            ]
        }

        with patch.dict(os.environ, env_vars):
            monitor = AISpendMonitor()
            with patch('requests.get', return_value=mock_response):
                result = monitor._get_anthropic_usage(days=1)

                # claude-3-sonnet: input=$0.003, output=$0.015 per 1k tokens
                # 1000 input tokens = $0.003
                # 1000 output tokens = $0.015
                # total = $0.018
                assert abs(result[0]['input_cost'] - 0.003) < 0.0001
                assert abs(result[0]['output_cost'] - 0.015) < 0.0001
                assert abs(result[0]['total_cost'] - 0.018) < 0.0001


class TestBedrockUsage:
    """Test _get_bedrock_usage method"""

    def test_get_bedrock_usage_disabled(self):
        """Test _get_bedrock_usage returns empty when disabled"""
        env_vars = {
            'AWS_ACCESS_KEY_ID': 'AKIAIOSFODNN7EXAMPLE',
            'AWS_SECRET_ACCESS_KEY': 'test-secret',
            'BEDROCK_ENABLED': 'false'
        }
        with patch.dict(os.environ, env_vars):
            monitor = AISpendMonitor()
            result = monitor._get_bedrock_usage(days=1)
            assert result == []

    def test_get_bedrock_usage_successful_cost_explorer_call(self):
        """Test successful AWS Cost Explorer API call for Bedrock"""
        env_vars = {
            'AWS_ACCESS_KEY_ID': 'AKIAIOSFODNN7EXAMPLE',
            'AWS_SECRET_ACCESS_KEY': 'test-secret',
            'BEDROCK_ENABLED': 'true',
            'AWS_REGION': 'us-east-1'
        }

        mock_client = MagicMock()
        mock_client.get_cost_and_usage.return_value = {
            'ResultsByTime': [
                {
                    'TimePeriod': {'Start': '2024-01-01', 'End': '2024-01-02'},
                    'Groups': [
                        {
                            'Keys': ['USE2-Anthropic-Claude-3-Sonnet'],
                            'Metrics': {'BlendedCost': {'Amount': '1.50'}}
                        }
                    ]
                }
            ]
        }

        with patch.dict(os.environ, env_vars):
            monitor = AISpendMonitor()
            with patch('boto3.client', return_value=mock_client):
                result = monitor._get_bedrock_usage(days=1)

                assert len(result) == 1
                assert result[0]['provider'] == 'aws_bedrock'
                assert result[0]['spend_type'] == 'llm_inference'
                assert result[0]['total_cost'] == 1.50

    def test_get_bedrock_usage_zero_cost_filtered(self):
        """Test that zero-cost entries are filtered out"""
        env_vars = {
            'AWS_ACCESS_KEY_ID': 'AKIAIOSFODNN7EXAMPLE',
            'AWS_SECRET_ACCESS_KEY': 'test-secret',
            'BEDROCK_ENABLED': 'true'
        }

        mock_client = MagicMock()
        mock_client.get_cost_and_usage.return_value = {
            'ResultsByTime': [
                {
                    'TimePeriod': {'Start': '2024-01-01', 'End': '2024-01-02'},
                    'Groups': [
                        {
                            'Keys': ['USE2-Anthropic-Claude-3-Sonnet'],
                            'Metrics': {'BlendedCost': {'Amount': '0.00'}}
                        }
                    ]
                }
            ]
        }

        with patch.dict(os.environ, env_vars):
            monitor = AISpendMonitor()
            with patch('boto3.client', return_value=mock_client):
                result = monitor._get_bedrock_usage(days=1)

                assert result == []

    def test_get_bedrock_usage_exception_handling(self):
        """Test _get_bedrock_usage handles exceptions gracefully"""
        env_vars = {
            'AWS_ACCESS_KEY_ID': 'AKIAIOSFODNN7EXAMPLE',
            'AWS_SECRET_ACCESS_KEY': 'test-secret',
            'BEDROCK_ENABLED': 'true'
        }

        with patch.dict(os.environ, env_vars):
            monitor = AISpendMonitor()
            with patch('boto3.client', side_effect=Exception('AWS credentials invalid')):
                result = monitor._get_bedrock_usage(days=1)

                assert result == []


class TestAzureOpenAIUsage:
    """Test _get_azure_openai_usage method"""

    def test_get_azure_openai_usage_disabled(self):
        """Test _get_azure_openai_usage returns empty when disabled"""
        env_vars = {
            'AZURE_OPENAI_API_KEY': 'azure-key-123',
            'AZURE_OPENAI_ENDPOINT': 'https://myresource.openai.azure.com/',
            'AZURE_OPENAI_ENABLED': 'false'
        }
        with patch.dict(os.environ, env_vars):
            monitor = AISpendMonitor()
            result = monitor._get_azure_openai_usage(days=1)
            assert result == []

    def test_get_azure_openai_usage_returns_empty_list(self):
        """Test _get_azure_openai_usage returns empty list (no direct API support)"""
        env_vars = {
            'AZURE_OPENAI_API_KEY': 'azure-key-123',
            'AZURE_OPENAI_ENDPOINT': 'https://myresource.openai.azure.com/',
            'AZURE_OPENAI_ENABLED': 'true'
        }
        with patch.dict(os.environ, env_vars):
            monitor = AISpendMonitor()
            result = monitor._get_azure_openai_usage(days=1)

            # Azure doesn't have a direct usage API, so this returns empty
            assert result == []

    def test_get_azure_openai_usage_exception_handling(self):
        """Test _get_azure_openai_usage handles exceptions gracefully"""
        env_vars = {
            'AZURE_OPENAI_API_KEY': 'azure-key-123',
            'AZURE_OPENAI_ENDPOINT': 'https://myresource.openai.azure.com/',
            'AZURE_OPENAI_ENABLED': 'true'
        }
        with patch.dict(os.environ, env_vars):
            monitor = AISpendMonitor()
            result = monitor._get_azure_openai_usage(days=1)

            # Should still return empty list even if there are issues
            assert result == []


class TestParseLLMResponse:
    """Test parse_llm_response method"""

    def test_parse_llm_response_openai_gpt4(self):
        """Test parsing OpenAI GPT-4 response"""
        with patch.dict(os.environ, {}, clear=True):
            monitor = AISpendMonitor()

            response_data = {
                'id': 'chatcmpl-123',
                'usage': {
                    'prompt_tokens': 100,
                    'completion_tokens': 50,
                    'total_tokens': 150
                }
            }

            result = monitor.parse_llm_response('openai', 'gpt-4', response_data)

            assert result['provider'] == 'openai'
            assert result['model_name'] == 'gpt-4'
            assert result['input_tokens'] == 100
            assert result['output_tokens'] == 50
            assert result['total_tokens'] == 150
            assert result['spend_type'] == 'llm_inference'
            assert result['currency'] == 'USD'
            assert result['request_id'] == 'chatcmpl-123'

    def test_parse_llm_response_cost_calculation_gpt4(self):
        """Test cost calculation for GPT-4 response"""
        with patch.dict(os.environ, {}, clear=True):
            monitor = AISpendMonitor()

            response_data = {
                'id': 'chatcmpl-123',
                'usage': {
                    'prompt_tokens': 1000,
                    'completion_tokens': 1000,
                    'total_tokens': 2000
                }
            }

            result = monitor.parse_llm_response('openai', 'gpt-4', response_data)

            # gpt-4: input=$0.03, output=$0.06 per 1k
            # 1000 input = $0.03, 1000 output = $0.06, total = $0.09
            assert abs(result['input_cost'] - 0.03) < 0.0001
            assert abs(result['output_cost'] - 0.06) < 0.0001
            assert abs(result['total_cost'] - 0.09) < 0.0001

    def test_parse_llm_response_anthropic_claude(self):
        """Test parsing Anthropic Claude response"""
        with patch.dict(os.environ, {}, clear=True):
            monitor = AISpendMonitor()

            response_data = {
                'id': 'msg-123',
                'usage': {
                    'prompt_tokens': 500,
                    'completion_tokens': 200,
                    'total_tokens': 700
                }
            }

            result = monitor.parse_llm_response('anthropic', 'claude-3-sonnet', response_data)

            assert result['provider'] == 'anthropic'
            assert result['model_name'] == 'claude-3-sonnet'
            assert result['input_tokens'] == 500
            assert result['output_tokens'] == 200
            assert result['total_tokens'] == 700

    def test_parse_llm_response_unknown_model_zero_cost(self):
        """Test unknown model returns zero cost"""
        with patch.dict(os.environ, {}, clear=True):
            monitor = AISpendMonitor()

            response_data = {
                'id': 'test-123',
                'usage': {
                    'prompt_tokens': 1000,
                    'completion_tokens': 500,
                    'total_tokens': 1500
                }
            }

            result = monitor.parse_llm_response('openai', 'unknown-model-xyz', response_data)

            assert result['input_cost'] == 0.0
            assert result['output_cost'] == 0.0
            assert result['total_cost'] == 0.0

    def test_parse_llm_response_missing_usage_data(self):
        """Test handling missing usage data"""
        with patch.dict(os.environ, {}, clear=True):
            monitor = AISpendMonitor()

            response_data = {
                'id': 'test-123'
                # no 'usage' key
            }

            result = monitor.parse_llm_response('openai', 'gpt-4', response_data)

            assert result['input_tokens'] == 0
            assert result['output_tokens'] == 0
            assert result['total_tokens'] == 0
            assert result['total_cost'] == 0.0

    def test_parse_llm_response_metadata(self):
        """Test response includes metadata"""
        with patch.dict(os.environ, {}, clear=True):
            monitor = AISpendMonitor()

            response_data = {
                'id': 'chatcmpl-123',
                'usage': {
                    'prompt_tokens': 100,
                    'completion_tokens': 50,
                    'total_tokens': 150
                }
            }

            result = monitor.parse_llm_response('openai', 'gpt-4', response_data)

            assert 'metadata' in result
            assert result['metadata']['response_id'] == 'chatcmpl-123'
            assert result['metadata']['model_version'] == 'gpt-4'


class TestGetPricing:
    """Test _get_pricing method"""

    def test_get_pricing_openai_models(self):
        """Test pricing retrieval for OpenAI models"""
        with patch.dict(os.environ, {}, clear=True):
            monitor = AISpendMonitor()

            # Test multiple OpenAI models
            gpt4_pricing = monitor._get_pricing('openai', 'gpt-4')
            assert gpt4_pricing['input_price_per_1k'] == 0.03
            assert gpt4_pricing['output_price_per_1k'] == 0.06

            gpt4turbo_pricing = monitor._get_pricing('openai', 'gpt-4-turbo')
            assert gpt4turbo_pricing['input_price_per_1k'] == 0.01
            assert gpt4turbo_pricing['output_price_per_1k'] == 0.03

            gpt35_pricing = monitor._get_pricing('openai', 'gpt-3.5-turbo')
            assert gpt35_pricing['input_price_per_1k'] == 0.0015
            assert gpt35_pricing['output_price_per_1k'] == 0.002

    def test_get_pricing_anthropic_models(self):
        """Test pricing retrieval for Anthropic models"""
        with patch.dict(os.environ, {}, clear=True):
            monitor = AISpendMonitor()

            # Test multiple Anthropic models
            opus_pricing = monitor._get_pricing('anthropic', 'claude-3-opus')
            assert opus_pricing['input_price_per_1k'] == 0.015
            assert opus_pricing['output_price_per_1k'] == 0.075

            sonnet_pricing = monitor._get_pricing('anthropic', 'claude-3-sonnet')
            assert sonnet_pricing['input_price_per_1k'] == 0.003
            assert sonnet_pricing['output_price_per_1k'] == 0.015

            haiku_pricing = monitor._get_pricing('anthropic', 'claude-3-haiku')
            assert haiku_pricing['input_price_per_1k'] == 0.00025
            assert haiku_pricing['output_price_per_1k'] == 0.00125

    def test_get_pricing_aws_bedrock_models(self):
        """Test pricing retrieval for AWS Bedrock models"""
        with patch.dict(os.environ, {}, clear=True):
            monitor = AISpendMonitor()

            bedrock_pricing = monitor._get_pricing('aws_bedrock', 'anthropic.claude-3-opus')
            assert bedrock_pricing['input_price_per_1k'] == 0.015
            assert bedrock_pricing['output_price_per_1k'] == 0.075

    def test_get_pricing_azure_models(self):
        """Test pricing retrieval for Azure OpenAI models"""
        with patch.dict(os.environ, {}, clear=True):
            monitor = AISpendMonitor()

            azure_pricing = monitor._get_pricing('azure_openai', 'gpt-4')
            assert azure_pricing['input_price_per_1k'] == 0.03
            assert azure_pricing['output_price_per_1k'] == 0.06

    def test_get_pricing_unknown_provider(self):
        """Test pricing retrieval for unknown provider returns zero"""
        with patch.dict(os.environ, {}, clear=True):
            monitor = AISpendMonitor()

            pricing = monitor._get_pricing('unknown_provider', 'some-model')
            assert pricing['input_price_per_1k'] == 0
            assert pricing['output_price_per_1k'] == 0

    def test_get_pricing_unknown_model(self):
        """Test pricing retrieval for unknown model returns zero"""
        with patch.dict(os.environ, {}, clear=True):
            monitor = AISpendMonitor()

            pricing = monitor._get_pricing('openai', 'unknown-model-xyz')
            assert pricing['input_price_per_1k'] == 0
            assert pricing['output_price_per_1k'] == 0


class TestParseTrainingRun:
    """Test parse_training_run method"""

    def test_parse_training_run_basic(self):
        """Test basic training run parsing"""
        with patch.dict(os.environ, {}, clear=True):
            monitor = AISpendMonitor()

            training_data = {
                'run_id': 'train-123',
                'duration_hours': 5,
                'instance_type': 'ml.p3.2xlarge',
                'project_id': 'proj-456',
                'completed_at': '2024-01-01T12:00:00Z',
                'config': {'batch_size': 32}
            }

            result = monitor.parse_training_run('aws', 'gpt-custom-01', training_data)

            assert result['provider'] == 'aws'
            assert result['model_name'] == 'gpt-custom-01'
            assert result['spend_type'] == 'llm_training'
            assert result['training_run_id'] == 'train-123'
            assert result['training_duration_hours'] == 5
            assert result['project_id'] == 'proj-456'
            assert 'metadata' in result
            assert result['metadata']['instance_type'] == 'ml.p3.2xlarge'

    def test_parse_training_run_cost_calculation(self):
        """Test cost calculation for training runs"""
        with patch.dict(os.environ, {}, clear=True):
            monitor = AISpendMonitor()

            training_data = {
                'run_id': 'train-123',
                'duration_hours': 10,
                'instance_type': 'ml.p3.2xlarge',  # $3.06/hour
                'project_id': 'proj-456',
                'completed_at': '2024-01-01T12:00:00Z',
                'config': {}
            }

            result = monitor.parse_training_run('aws', 'gpt-custom-01', training_data)

            # 10 hours * $3.06/hour = $30.60
            expected_cost = 10 * 3.06
            assert abs(result['training_cost'] - expected_cost) < 0.01
            assert abs(result['total_cost'] - expected_cost) < 0.01

    def test_parse_training_run_unknown_instance_type(self):
        """Test training run with unknown instance type returns zero cost"""
        with patch.dict(os.environ, {}, clear=True):
            monitor = AISpendMonitor()

            training_data = {
                'run_id': 'train-123',
                'duration_hours': 10,
                'instance_type': 'unknown-instance-type',
                'project_id': 'proj-456',
                'completed_at': '2024-01-01T12:00:00Z',
                'config': {}
            }

            result = monitor.parse_training_run('aws', 'gpt-custom-01', training_data)

            assert result['training_cost'] == 0.0
            assert result['total_cost'] == 0.0

    def test_parse_training_run_zero_duration(self):
        """Test training run with zero duration"""
        with patch.dict(os.environ, {}, clear=True):
            monitor = AISpendMonitor()

            training_data = {
                'run_id': 'train-123',
                'duration_hours': 0,
                'instance_type': 'ml.p3.2xlarge',
                'project_id': 'proj-456',
                'completed_at': '2024-01-01T12:00:00Z',
                'config': {}
            }

            result = monitor.parse_training_run('aws', 'gpt-custom-01', training_data)

            assert result['training_cost'] == 0.0
            assert result['total_cost'] == 0.0


class TestGetTrainingInstanceCost:
    """Test _get_training_instance_cost method"""

    def test_get_training_instance_cost_aws_instances(self):
        """Test AWS instance pricing"""
        with patch.dict(os.environ, {}, clear=True):
            monitor = AISpendMonitor()

            # Test various AWS instance types
            cost1 = monitor._get_training_instance_cost('aws', 'ml.p3.2xlarge')
            assert cost1 == 3.06

            cost2 = monitor._get_training_instance_cost('aws', 'ml.p3.8xlarge')
            assert cost2 == 12.24

            cost3 = monitor._get_training_instance_cost('aws', 'ml.p4d.24xlarge')
            assert cost3 == 32.77

    def test_get_training_instance_cost_gcp_instances(self):
        """Test GCP instance pricing"""
        with patch.dict(os.environ, {}, clear=True):
            monitor = AISpendMonitor()

            cost1 = monitor._get_training_instance_cost('gcp', 'a2-highgpu-1g')
            assert cost1 == 2.25

            cost2 = monitor._get_training_instance_cost('gcp', 'a2-highgpu-2g')
            assert cost2 == 4.50

    def test_get_training_instance_cost_unknown_provider(self):
        """Test unknown provider returns zero cost"""
        with patch.dict(os.environ, {}, clear=True):
            monitor = AISpendMonitor()

            cost = monitor._get_training_instance_cost('unknown_provider', 'some-instance')
            assert cost == 0.0

    def test_get_training_instance_cost_unknown_instance(self):
        """Test unknown instance type returns zero cost"""
        with patch.dict(os.environ, {}, clear=True):
            monitor = AISpendMonitor()

            cost = monitor._get_training_instance_cost('aws', 'unknown-instance-xyz')
            assert cost == 0.0

    def test_get_training_instance_cost_returns_float(self):
        """Test that costs are returned as floats"""
        with patch.dict(os.environ, {}, clear=True):
            monitor = AISpendMonitor()

            cost = monitor._get_training_instance_cost('aws', 'ml.p3.2xlarge')
            assert isinstance(cost, float)
