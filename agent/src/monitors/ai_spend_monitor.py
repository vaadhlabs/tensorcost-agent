"""
AI Spend Monitor
Collects LLM token costs and training run costs from various providers
"""

import os
import logging
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import requests

logger = logging.getLogger(__name__)

class AISpendMonitor:
    """Monitor AI spend from LLM providers"""
    
    def __init__(self):
        """Initialize AI spend monitor"""
        self.providers = {}
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Initialize provider configurations"""
        # OpenAI
        if os.getenv('OPENAI_API_KEY'):
            self.providers['openai'] = {
                'api_key': os.getenv('OPENAI_API_KEY'),
                'enabled': os.getenv('OPENAI_ENABLED', 'true').lower() == 'true'
            }
        
        # Anthropic
        if os.getenv('ANTHROPIC_API_KEY'):
            self.providers['anthropic'] = {
                'api_key': os.getenv('ANTHROPIC_API_KEY'),
                'enabled': os.getenv('ANTHROPIC_ENABLED', 'true').lower() == 'true'
            }
        
        # AWS Bedrock
        if os.getenv('AWS_ACCESS_KEY_ID'):
            self.providers['aws_bedrock'] = {
                'enabled': os.getenv('BEDROCK_ENABLED', 'false').lower() == 'true',
                'region': os.getenv('AWS_REGION', 'us-east-1')
            }
        
        # Azure OpenAI
        if os.getenv('AZURE_OPENAI_API_KEY'):
            self.providers['azure_openai'] = {
                'api_key': os.getenv('AZURE_OPENAI_API_KEY'),
                'endpoint': os.getenv('AZURE_OPENAI_ENDPOINT'),
                'enabled': os.getenv('AZURE_OPENAI_ENABLED', 'true').lower() == 'true'
            }
    
    def get_ai_spend(self, days: int = 1) -> List[Dict[str, Any]]:
        """
        Collect AI spend data from configured providers
        
        Note: This is a framework. In production, you would:
        1. Integrate with actual LLM provider APIs to get usage data
        2. Parse billing/usage APIs from providers
        3. Calculate costs based on token usage and pricing
        
        For now, this returns example structure that matches the backend schema.
        """
        ai_spend_data = []
        
        # OpenAI usage (if enabled)
        if self.providers.get('openai', {}).get('enabled'):
            openai_spend = self._get_openai_usage(days)
            ai_spend_data.extend(openai_spend)
        
        # Anthropic usage (if enabled)
        if self.providers.get('anthropic', {}).get('enabled'):
            anthropic_spend = self._get_anthropic_usage(days)
            ai_spend_data.extend(anthropic_spend)
        
        # AWS Bedrock usage (if enabled)
        if self.providers.get('aws_bedrock', {}).get('enabled'):
            bedrock_spend = self._get_bedrock_usage(days)
            ai_spend_data.extend(bedrock_spend)
        
        # Azure OpenAI usage (if enabled)
        if self.providers.get('azure_openai', {}).get('enabled'):
            azure_spend = self._get_azure_openai_usage(days)
            ai_spend_data.extend(azure_spend)
        
        return ai_spend_data
    
    def _get_openai_usage(self, days: int) -> List[Dict[str, Any]]:
        """Get OpenAI usage data from the /usage endpoint."""
        api_key = self.providers.get('openai', {}).get('api_key')
        if not api_key:
            return []

        spend_data = []
        try:
            # OpenAI provides a /v1/usage endpoint for organization usage
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=days)

            headers = {
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json',
            }

            # Try the organization usage API
            response = requests.get(
                'https://api.openai.com/v1/organization/usage',
                headers=headers,
                params={
                    'start_time': int(start_date.timestamp()),
                    'end_time': int(end_date.timestamp()),
                },
                timeout=30,
            )

            if response.status_code == 200:
                data = response.json()
                for bucket in data.get('data', []):
                    for result in bucket.get('results', []):
                        input_tokens = result.get('input_tokens', 0)
                        output_tokens = result.get('output_tokens', 0)
                        model = result.get('model', 'unknown')

                        pricing = self._get_pricing('openai', model)
                        input_cost = (input_tokens / 1000) * pricing.get('input_price_per_1k', 0)
                        output_cost = (output_tokens / 1000) * pricing.get('output_price_per_1k', 0)

                        spend_data.append({
                            'provider': 'openai',
                            'model_name': model,
                            'spend_type': 'llm_inference',
                            'input_tokens': input_tokens,
                            'output_tokens': output_tokens,
                            'total_tokens': input_tokens + output_tokens,
                            'input_cost': round(input_cost, 6),
                            'output_cost': round(output_cost, 6),
                            'total_cost': round(input_cost + output_cost, 6),
                            'currency': 'USD',
                            'timestamp': datetime.utcfromtimestamp(bucket.get('start_time', 0)).isoformat(),
                        })
            elif response.status_code == 403:
                logger.warning("OpenAI usage API returned 403 - organization-level access may be required")
            else:
                logger.warning(f"OpenAI usage API returned {response.status_code}")

        except Exception as e:
            logger.error(f"Error fetching OpenAI usage: {e}")

        return spend_data
    
    def _get_anthropic_usage(self, days: int) -> List[Dict[str, Any]]:
        """Get Anthropic usage data from the /usage endpoint."""
        api_key = self.providers.get('anthropic', {}).get('api_key')
        if not api_key:
            return []

        spend_data = []
        try:
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=days)

            headers = {
                'x-api-key': api_key,
                'anthropic-version': '2023-06-01',
                'Content-Type': 'application/json',
            }

            response = requests.get(
                'https://api.anthropic.com/v1/organizations/usage',
                headers=headers,
                params={
                    'start_date': start_date.strftime('%Y-%m-%d'),
                    'end_date': end_date.strftime('%Y-%m-%d'),
                },
                timeout=30,
            )

            if response.status_code == 200:
                data = response.json()
                for usage in data.get('data', []):
                    input_tokens = usage.get('input_tokens', 0)
                    output_tokens = usage.get('output_tokens', 0)
                    model = usage.get('model', 'unknown')

                    pricing = self._get_pricing('anthropic', model)
                    input_cost = (input_tokens / 1000) * pricing.get('input_price_per_1k', 0)
                    output_cost = (output_tokens / 1000) * pricing.get('output_price_per_1k', 0)

                    spend_data.append({
                        'provider': 'anthropic',
                        'model_name': model,
                        'spend_type': 'llm_inference',
                        'input_tokens': input_tokens,
                        'output_tokens': output_tokens,
                        'total_tokens': input_tokens + output_tokens,
                        'input_cost': round(input_cost, 6),
                        'output_cost': round(output_cost, 6),
                        'total_cost': round(input_cost + output_cost, 6),
                        'currency': 'USD',
                        'timestamp': usage.get('date', datetime.utcnow().isoformat()),
                    })
            elif response.status_code == 403:
                logger.warning("Anthropic usage API returned 403 - admin access may be required")
            else:
                logger.warning(f"Anthropic usage API returned {response.status_code}")

        except Exception as e:
            logger.error(f"Error fetching Anthropic usage: {e}")

        return spend_data
    
    def _get_bedrock_usage(self, days: int) -> List[Dict[str, Any]]:
        """Get AWS Bedrock usage via CloudWatch metrics and Cost Explorer."""
        if not self.providers.get('aws_bedrock', {}).get('enabled'):
            return []

        spend_data = []
        try:
            import boto3
            region = self.providers['aws_bedrock'].get('region', 'us-east-1')
            ce_client = boto3.client('ce', region_name='us-east-1')

            end_date = datetime.utcnow().strftime('%Y-%m-%d')
            start_date = (datetime.utcnow() - timedelta(days=days)).strftime('%Y-%m-%d')

            response = ce_client.get_cost_and_usage(
                TimePeriod={'Start': start_date, 'End': end_date},
                Granularity='DAILY',
                Metrics=['BlendedCost'],
                Filter={
                    'Dimensions': {
                        'Key': 'SERVICE',
                        'Values': ['Amazon Bedrock'],
                    }
                },
                GroupBy=[{'Type': 'DIMENSION', 'Key': 'USAGE_TYPE'}],
            )

            for result in response.get('ResultsByTime', []):
                for group in result.get('Groups', []):
                    usage_type = group['Keys'][0]
                    cost = float(group['Metrics']['BlendedCost']['Amount'])
                    if cost > 0:
                        # Parse model from usage type (e.g., "USE2-Anthropic-Claude-3-Sonnet")
                        model_name = usage_type.split('-', 1)[-1] if '-' in usage_type else usage_type
                        spend_data.append({
                            'provider': 'aws_bedrock',
                            'model_name': model_name,
                            'spend_type': 'llm_inference',
                            'total_cost': round(cost, 6),
                            'currency': 'USD',
                            'timestamp': result['TimePeriod']['Start'],
                        })

        except Exception as e:
            logger.error(f"Error fetching Bedrock usage: {e}")

        return spend_data
    
    def _get_azure_openai_usage(self, days: int) -> List[Dict[str, Any]]:
        """Get Azure OpenAI usage via Azure Cost Management."""
        config = self.providers.get('azure_openai', {})
        if not config.get('enabled'):
            return []

        spend_data = []
        try:
            # Use Azure Cost Management API
            endpoint = config.get('endpoint', '')
            api_key = config.get('api_key', '')

            # Azure OpenAI doesn't have a direct usage API like OpenAI
            # Fall back to tracking via the middleware interceptor
            logger.info("Azure OpenAI usage tracked via middleware interceptor")

        except Exception as e:
            logger.error(f"Error fetching Azure OpenAI usage: {e}")

        return spend_data
    
    def parse_llm_response(self, provider: str, model: str, response_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse LLM API response to extract token usage and calculate costs
        
        This should be called from your application when making LLM API calls
        """
        # Extract token usage from response
        input_tokens = response_data.get('usage', {}).get('prompt_tokens', 0)
        output_tokens = response_data.get('usage', {}).get('completion_tokens', 0)
        total_tokens = response_data.get('usage', {}).get('total_tokens', input_tokens + output_tokens)
        
        # Calculate costs based on provider pricing
        pricing = self._get_pricing(provider, model)
        input_cost = (input_tokens / 1000) * pricing.get('input_price_per_1k', 0)
        output_cost = (output_tokens / 1000) * pricing.get('output_price_per_1k', 0)
        total_cost = input_cost + output_cost
        
        return {
            'provider': provider,
            'model_name': model,
            'spend_type': 'llm_inference',
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'total_tokens': total_tokens,
            'input_cost': round(input_cost, 6),
            'output_cost': round(output_cost, 6),
            'total_cost': round(total_cost, 6),
            'currency': 'USD',
            'timestamp': datetime.utcnow().isoformat(),
            'request_id': response_data.get('id'),
            'metadata': {
                'response_id': response_data.get('id'),
                'model_version': model
            }
        }
    
    def _get_pricing(self, provider: str, model: str) -> Dict[str, float]:
        """
        Get pricing for provider/model combination

        Pricing as of 2024 (update as needed)
        """
        pricing_map = {
            'openai': {
                'gpt-4': {'input_price_per_1k': 0.03, 'output_price_per_1k': 0.06},
                'gpt-4-turbo': {'input_price_per_1k': 0.01, 'output_price_per_1k': 0.03},
                'gpt-4o': {'input_price_per_1k': 0.005, 'output_price_per_1k': 0.015},
                'gpt-4o-mini': {'input_price_per_1k': 0.00015, 'output_price_per_1k': 0.0006},
                'gpt-3.5-turbo': {'input_price_per_1k': 0.0015, 'output_price_per_1k': 0.002},
                'o1': {'input_price_per_1k': 0.015, 'output_price_per_1k': 0.06},
                'o3-mini': {'input_price_per_1k': 0.0011, 'output_price_per_1k': 0.0044},
            },
            'anthropic': {
                'claude-3-opus': {'input_price_per_1k': 0.015, 'output_price_per_1k': 0.075},
                'claude-3-sonnet': {'input_price_per_1k': 0.003, 'output_price_per_1k': 0.015},
                'claude-3-haiku': {'input_price_per_1k': 0.00025, 'output_price_per_1k': 0.00125},
                'claude-3.5-sonnet': {'input_price_per_1k': 0.003, 'output_price_per_1k': 0.015},
                'claude-3.5-haiku': {'input_price_per_1k': 0.0008, 'output_price_per_1k': 0.004},
                'claude-4-sonnet': {'input_price_per_1k': 0.003, 'output_price_per_1k': 0.015},
                'claude-4-opus': {'input_price_per_1k': 0.015, 'output_price_per_1k': 0.075},
            },
            'aws_bedrock': {
                'anthropic.claude-3-opus': {'input_price_per_1k': 0.015, 'output_price_per_1k': 0.075},
                'anthropic.claude-3-sonnet': {'input_price_per_1k': 0.003, 'output_price_per_1k': 0.015},
            },
            'azure_openai': {
                'gpt-4': {'input_price_per_1k': 0.03, 'output_price_per_1k': 0.06},
                'gpt-35-turbo': {'input_price_per_1k': 0.0015, 'output_price_per_1k': 0.002},
            }
        }

        return pricing_map.get(provider, {}).get(model, {'input_price_per_1k': 0, 'output_price_per_1k': 0})
    
    def parse_training_run(self, provider: str, model: str, training_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse training run data to calculate costs
        """
        duration_hours = training_data.get('duration_hours', 0)
        instance_type = training_data.get('instance_type', '')
        
        # Calculate training cost based on instance type and duration
        instance_hourly_cost = self._get_training_instance_cost(provider, instance_type)
        training_cost = duration_hours * instance_hourly_cost
        
        return {
            'provider': provider,
            'model_name': model,
            'spend_type': 'llm_training',
            'training_run_id': training_data.get('run_id'),
            'training_duration_hours': duration_hours,
            'training_cost': round(training_cost, 6),
            'cost_per_training_run': round(training_cost, 6),
            'total_cost': round(training_cost, 6),
            'currency': 'USD',
            'timestamp': training_data.get('completed_at', datetime.utcnow().isoformat()),
            'project_id': training_data.get('project_id'),
            'metadata': {
                'instance_type': instance_type,
                'training_config': training_data.get('config', {})
            }
        }
    
    def _get_training_instance_cost(self, provider: str, instance_type: str) -> float:
        """Get hourly cost for training instance"""
        # Example pricing - update with actual costs
        pricing = {
            'aws': {
                'ml.p3.2xlarge': 3.06,
                'ml.p3.8xlarge': 12.24,
                'ml.p4d.24xlarge': 32.77,
            },
            'gcp': {
                'a2-highgpu-1g': 2.25,
                'a2-highgpu-2g': 4.50,
            }
        }
        
        return pricing.get(provider, {}).get(instance_type, 0.0)
