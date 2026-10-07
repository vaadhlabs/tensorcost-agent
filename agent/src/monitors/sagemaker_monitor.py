"""
SageMaker Monitor for training jobs, endpoints, and processing jobs
"""

import os
import logging
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import psutil
import pynvml

import boto3
from botocore.exceptions import ClientError
import sagemaker
from sagemaker import get_execution_role

logger = logging.getLogger(__name__)

class SageMakerMonitor:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize SageMaker monitoring client"""
        config = config or {}
        self.region = config.get('region') or os.getenv('AWS_REGION', 'us-east-1')
        
        # Initialize AWS clients
        self.sagemaker_client = boto3.client('sagemaker', region_name=self.region)
        self.cloudwatch_client = boto3.client('cloudwatch', region_name=self.region)
        self.cost_explorer_client = boto3.client('ce', region_name=self.region)
        self.ec2_client = boto3.client('ec2', region_name=self.region)
        
        # Initialize SageMaker session
        self.sagemaker_session = sagemaker.Session(boto3.Session(region_name=self.region))
        
        # Initialize NVML for GPU monitoring
        try:
            pynvml.nvmlInit()
            self.gpu_available = True
            logger.info("NVML initialized successfully")
        except Exception as e:
            self.gpu_available = False
            logger.warning(f"NVML not available: {e}")
    
    def get_training_jobs(self) -> List[Dict[str, Any]]:
        """Get all SageMaker training jobs"""
        training_jobs = []
        
        try:
            # Get training jobs
            response = self.sagemaker_client.list_training_jobs(
                StatusEquals='InProgress',
                MaxResults=100
            )
            
            for job in response['TrainingJobSummaries']:
                # Get detailed job information
                job_details = self.sagemaker_client.describe_training_job(
                    TrainingJobName=job['TrainingJobName']
                )
                
                # Check if job uses GPU instances
                instance_type = job_details['ResourceConfig']['InstanceType']
                if 'gpu' in instance_type.lower() or 'p' in instance_type.lower():
                    training_job_data = {
                        'job_name': job['TrainingJobName'],
                        'job_arn': job['TrainingJobArn'],
                        'status': job['TrainingJobStatus'],
                        'instance_type': instance_type,
                        'instance_count': job_details['ResourceConfig']['InstanceCount'],
                        'created_at': job['CreationTime'].isoformat(),
                        'started_at': job_details.get('TrainingStartTime', {}).isoformat() if job_details.get('TrainingStartTime') else None,
                        'cloud_provider': 'aws',
                        'region': self.region,
                        'experiment_name': job_details.get('ExperimentConfig', {}).get('ExperimentName'),
                        'trial_name': job_details.get('ExperimentConfig', {}).get('TrialName'),
                        'tags': {tag['Key']: tag['Value'] for tag in job_details.get('Tags', [])}
                    }
                    training_jobs.append(training_job_data)
            
            logger.info(f"Found {len(training_jobs)} GPU training jobs")
            
        except ClientError as e:
            logger.error(f"Error getting SageMaker training jobs: {e}")
        except Exception as e:
            logger.error(f"Error getting SageMaker training jobs: {e}")
        
        return training_jobs
    
    def get_endpoints(self) -> List[Dict[str, Any]]:
        """Get all SageMaker endpoints"""
        endpoints = []
        
        try:
            # Get endpoints
            response = self.sagemaker_client.list_endpoints(
                StatusEquals='InService',
                MaxResults=100
            )
            
            for endpoint in response['Endpoints']:
                # Get detailed endpoint information
                endpoint_details = self.sagemaker_client.describe_endpoint(
                    EndpointName=endpoint['EndpointName']
                )
                
                # Get endpoint configuration
                config_details = self.sagemaker_client.describe_endpoint_config(
                    EndpointConfigName=endpoint_details['EndpointConfigName']
                )
                
                # Check if endpoint uses GPU instances
                has_gpu = False
                for variant in config_details['ProductionVariants']:
                    instance_type = variant['InstanceType']
                    if 'gpu' in instance_type.lower() or 'p' in instance_type.lower():
                        has_gpu = True
                        break
                
                if has_gpu:
                    endpoint_data = {
                        'endpoint_name': endpoint['EndpointName'],
                        'endpoint_arn': endpoint['EndpointArn'],
                        'status': endpoint['EndpointStatus'],
                        'endpoint_config_name': endpoint_details['EndpointConfigName'],
                        'created_at': endpoint['CreationTime'].isoformat(),
                        'last_modified': endpoint['LastModifiedTime'].isoformat(),
                        'cloud_provider': 'aws',
                        'region': self.region,
                        'tags': {tag['Key']: tag['Value'] for tag in endpoint_details.get('Tags', [])}
                    }
                    endpoints.append(endpoint_data)
            
            logger.info(f"Found {len(endpoints)} GPU endpoints")
            
        except ClientError as e:
            logger.error(f"Error getting SageMaker endpoints: {e}")
        except Exception as e:
            logger.error(f"Error getting SageMaker endpoints: {e}")
        
        return endpoints
    
    def get_processing_jobs(self) -> List[Dict[str, Any]]:
        """Get all SageMaker processing jobs"""
        processing_jobs = []
        
        try:
            # Get processing jobs
            response = self.sagemaker_client.list_processing_jobs(
                StatusEquals='InProgress',
                MaxResults=100
            )
            
            for job in response['ProcessingJobSummaries']:
                # Get detailed job information
                job_details = self.sagemaker_client.describe_processing_job(
                    ProcessingJobName=job['ProcessingJobName']
                )
                
                # Check if job uses GPU instances
                instance_type = job_details['AppSpecification']['ImageUri']
                if 'gpu' in instance_type.lower() or 'p' in instance_type.lower():
                    processing_job_data = {
                        'job_name': job['ProcessingJobName'],
                        'job_arn': job['ProcessingJobArn'],
                        'status': job['ProcessingJobStatus'],
                        'instance_type': job_details['ProcessingResources']['ClusterConfig']['InstanceType'],
                        'instance_count': job_details['ProcessingResources']['ClusterConfig']['InstanceCount'],
                        'created_at': job['CreationTime'].isoformat(),
                        'started_at': job_details.get('ProcessingStartTime', {}).isoformat() if job_details.get('ProcessingStartTime') else None,
                        'cloud_provider': 'aws',
                        'region': self.region,
                        'experiment_name': job_details.get('ExperimentConfig', {}).get('ExperimentName'),
                        'trial_name': job_details.get('ExperimentConfig', {}).get('TrialName'),
                        'tags': {tag['Key']: tag['Value'] for tag in job_details.get('Tags', [])}
                    }
                    processing_jobs.append(processing_job_data)
            
            logger.info(f"Found {len(processing_jobs)} GPU processing jobs")
            
        except ClientError as e:
            logger.error(f"Error getting SageMaker processing jobs: {e}")
        except Exception as e:
            logger.error(f"Error getting SageMaker processing jobs: {e}")
        
        return processing_jobs

    def get_gpu_instances(self) -> List[Dict[str, Any]]:
        """Synthetic inventory from in-flight GPU training jobs and endpoints."""
        instances: List[Dict[str, Any]] = []
        seen = set()

        for job in self.get_training_jobs():
            iid = job.get('job_name')
            if not iid or iid in seen:
                continue
            seen.add(iid)
            instances.append({
                'instance_id': iid,
                'cloud_id': iid,
                'instance_type': job.get('instance_type', 'unknown'),
                'state': (job.get('status') or 'running').lower(),
                'cloud_provider': 'sagemaker',
                'region': job.get('region', self.region),
                'tags': job.get('tags') or {},
                'gpu_count': job.get('instance_count', 1),
                'gpu_type': None,
            })

        for ep in self.get_endpoints():
            iid = ep.get('endpoint_name')
            if not iid or iid in seen:
                continue
            seen.add(iid)
            instances.append({
                'instance_id': iid,
                'cloud_id': iid,
                'instance_type': ep.get('instance_type', 'unknown'),
                'state': (ep.get('status') or 'inservice').lower(),
                'cloud_provider': 'sagemaker',
                'region': ep.get('region', self.region),
                'tags': ep.get('tags') or {},
                'gpu_count': ep.get('instance_count', 1),
                'gpu_type': None,
                'endpoint_name': iid,
            })

        return instances
    
    def get_gpu_utilization(self, instances: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Get GPU utilization metrics for SageMaker endpoints (local NVML when on-host)."""
        endpoint_like = [
            ep for ep in instances
            if ep.get('endpoint_name') or ep.get('instance_id')
        ]
        if not endpoint_like:
            return []
        # Normalize to the shape expected by the legacy NVML matcher below.
        normalized = []
        for ep in endpoint_like:
            normalized.append({
                'endpoint_name': ep.get('endpoint_name') or ep.get('instance_id'),
            })
        metrics = self._collect_endpoint_gpu_metrics(normalized)
        for m in metrics:
            m.setdefault('instance_id', m.get('endpoint_name'))
            m['cloud_provider'] = 'sagemaker'
        return metrics

    def _collect_endpoint_gpu_metrics(self, endpoints: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Get GPU utilization metrics for SageMaker endpoints"""
        metrics = []
        
        if not self.gpu_available:
            logger.warning("GPU monitoring not available - NVML not initialized")
            return metrics
        
        try:
            device_count = pynvml.nvmlDeviceGetCount()
            
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                
                # Get GPU utilization
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_util = utilization.gpu
                memory_util = utilization.memory
                
                # Get memory info
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                memory_used = memory_info.used
                memory_total = memory_info.total
                
                # Get temperature
                try:
                    temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                except:
                    temperature = None
                
                # Get power usage
                try:
                    power_usage = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts
                except:
                    power_usage = None
                
                # Find matching endpoint
                endpoint = None
                for ep in endpoints:
                    if ep['endpoint_name'] in os.uname().nodename:  # Simple matching
                        endpoint = ep
                        break
                
                if endpoint:
                    metric_data = {
                        'endpoint_name': endpoint['endpoint_name'],
                        'gpu_index': i,
                        'gpu_utilization': round(gpu_util, 2),
                        'memory_utilization': round(memory_util, 2),
                        'memory_used_mb': round(memory_used / 1024 / 1024, 2),
                        'memory_total_mb': round(memory_total / 1024 / 1024, 2),
                        'temperature_c': temperature,
                        'power_usage_w': power_usage,
                        'is_idle': gpu_util < 10 and memory_util < 10,
                        'timestamp': datetime.utcnow().isoformat(),
                        'cloud_provider': 'aws'
                    }
                    metrics.append(metric_data)
            
            logger.info(f"Collected GPU metrics for {len(metrics)} devices")
            
        except Exception as e:
            logger.error(f"Error getting GPU utilization: {e}")
        
        return metrics
    
    def get_cost_data(self, training_jobs: List[Dict[str, Any]], endpoints: List[Dict[str, Any]], processing_jobs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Get cost data for SageMaker resources"""
        cost_data = []
        
        try:
            # Get cost data for the last 30 days
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=30)
            
            # Query cost data
            response = self.cost_explorer_client.get_cost_and_usage(
                TimePeriod={
                    'Start': start_date.strftime('%Y-%m-%d'),
                    'End': end_date.strftime('%Y-%m-%d')
                },
                Granularity='DAILY',
                Metrics=['BlendedCost'],
                GroupBy=[
                    {
                        'Type': 'DIMENSION',
                        'Key': 'SERVICE'
                    }
                ]
            )
            
            # Process cost data
            for result in response['ResultsByTime']:
                date = result['TimePeriod']['Start']
                
                for group in result['Groups']:
                    service = group['Keys'][0]
                    cost = float(group['Metrics']['BlendedCost']['Amount'])
                    
                    if service == 'Amazon SageMaker' and cost > 0:
                        cost_entry = {
                            'service': service,
                            'cost': cost,
                            'currency': 'USD',
                            'date': date,
                            'cloud_provider': 'aws',
                            'region': self.region
                        }
                        cost_data.append(cost_entry)
            
            # Add estimated costs for individual resources
            for job in training_jobs:
                estimated_cost = self._estimate_training_job_cost(job)
                cost_entry = {
                    'resource_name': job['job_name'],
                    'resource_type': 'training_job',
                    'service': 'Amazon SageMaker',
                    'cost': estimated_cost,
                    'currency': 'USD',
                    'date': datetime.utcnow().isoformat(),
                    'cloud_provider': 'aws',
                    'region': self.region
                }
                cost_data.append(cost_entry)
            
            for endpoint in endpoints:
                estimated_cost = self._estimate_endpoint_cost(endpoint)
                cost_entry = {
                    'resource_name': endpoint['endpoint_name'],
                    'resource_type': 'endpoint',
                    'service': 'Amazon SageMaker',
                    'cost': estimated_cost,
                    'currency': 'USD',
                    'date': datetime.utcnow().isoformat(),
                    'cloud_provider': 'aws',
                    'region': self.region
                }
                cost_data.append(cost_entry)
            
            logger.info(f"Collected cost data for {len(cost_data)} SageMaker resources")
            
        except ClientError as e:
            logger.error(f"Error getting SageMaker cost data: {e}")
        except Exception as e:
            logger.error(f"Error getting SageMaker cost data: {e}")
        
        return cost_data
    
    def _estimate_training_job_cost(self, job: Dict[str, Any]) -> float:
        """Estimate cost for a SageMaker training job"""
        # This is a simplified cost estimation
        # In practice, you'd use the actual billing data
        
        instance_costs = {
            'ml.p3.2xlarge': 3.06,
            'ml.p3.8xlarge': 12.24,
            'ml.p3.16xlarge': 24.48,
            'ml.p3dn.24xlarge': 31.22,
            'ml.g4dn.xlarge': 0.526,
            'ml.g4dn.2xlarge': 0.752,
            'ml.g4dn.4xlarge': 1.204,
            'ml.g4dn.8xlarge': 2.176,
            'ml.g4dn.12xlarge': 3.912,
            'ml.g4dn.16xlarge': 5.216,
            'ml.g5.xlarge': 1.006,
            'ml.g5.2xlarge': 1.212,
            'ml.g5.4xlarge': 1.624,
            'ml.g5.8xlarge': 2.448,
            'ml.g5.12xlarge': 3.672,
            'ml.g5.16xlarge': 4.896,
            'ml.g5.24xlarge': 7.344,
            'ml.g5.48xlarge': 14.688
        }
        
        instance_type = job['instance_type']
        instance_count = job['instance_count']
        hourly_cost = instance_costs.get(instance_type, 1.0)
        
        # Estimate runtime (simplified)
        runtime_hours = 1.0  # This would be calculated from actual start/end times
        
        return round(hourly_cost * instance_count * runtime_hours, 2)
    
    def _estimate_endpoint_cost(self, endpoint: Dict[str, Any]) -> float:
        """Estimate cost for a SageMaker endpoint"""
        # This is a simplified cost estimation
        # In practice, you'd use the actual billing data
        
        instance_costs = {
            'ml.m5.large': 0.115,
            'ml.m5.xlarge': 0.23,
            'ml.m5.2xlarge': 0.46,
            'ml.m5.4xlarge': 0.922,
            'ml.m5.12xlarge': 2.765,
            'ml.m5.24xlarge': 5.53,
            'ml.c5.large': 0.102,
            'ml.c5.xlarge': 0.204,
            'ml.c5.2xlarge': 0.408,
            'ml.c5.4xlarge': 0.816,
            'ml.c5.9xlarge': 1.836,
            'ml.c5.18xlarge': 3.672,
            'ml.p3.2xlarge': 3.06,
            'ml.p3.8xlarge': 12.24,
            'ml.p3.16xlarge': 24.48,
            'ml.g4dn.xlarge': 0.526,
            'ml.g4dn.2xlarge': 0.752,
            'ml.g4dn.4xlarge': 1.204,
            'ml.g4dn.8xlarge': 2.176,
            'ml.g4dn.12xlarge': 3.912,
            'ml.g4dn.16xlarge': 5.216
        }
        
        # Get endpoint configuration details
        try:
            config_details = self.sagemaker_client.describe_endpoint_config(
                EndpointConfigName=endpoint['endpoint_config_name']
            )
            
            total_cost = 0
            for variant in config_details['ProductionVariants']:
                instance_type = variant['InstanceType']
                instance_count = variant['InitialInstanceCount']
                hourly_cost = instance_costs.get(instance_type, 1.0)
                total_cost += hourly_cost * instance_count
            
            return round(total_cost, 2)
            
        except Exception as e:
            logger.error(f"Error estimating endpoint cost: {e}")
            return 0.0
    
    def get_endpoint_metrics(self, endpoint_name: str) -> Dict[str, Any]:
        """Get detailed metrics for a SageMaker endpoint"""
        try:
            # Get CloudWatch metrics for the endpoint
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=1)
            
            # Get invocation metrics
            response = self.cloudwatch_client.get_metric_statistics(
                Namespace='AWS/SageMaker',
                MetricName='Invocations',
                Dimensions=[
                    {
                        'Name': 'EndpointName',
                        'Value': endpoint_name
                    }
                ],
                StartTime=start_time,
                EndTime=end_time,
                Period=300,
                Statistics=['Sum', 'Average']
            )
            
            metrics_data = {
                'endpoint_name': endpoint_name,
                'invocations': [],
                'timestamp': datetime.utcnow().isoformat()
            }
            
            # Process metrics
            for datapoint in response['Datapoints']:
                metrics_data['invocations'].append({
                    'timestamp': datapoint['Timestamp'].isoformat(),
                    'sum': datapoint['Sum'],
                    'average': datapoint['Average']
                })
            
            return metrics_data
            
        except Exception as e:
            logger.error(f"Error getting metrics for endpoint {endpoint_name}: {e}")
            return {}
    
    def get_experiment_costs(self, experiment_name: str) -> Dict[str, Any]:
        """Get cost breakdown for a SageMaker experiment"""
        try:
            # Get all trials in the experiment
            response = self.sagemaker_client.list_trials(
                ExperimentName=experiment_name
            )

            total_cost = 0
            trial_costs = []

            for trial in response['TrialSummaries']:
                # Get trial details
                trial_details = self.sagemaker_client.describe_trial(
                    TrialName=trial['TrialName']
                )

                # Get training jobs for this trial
                training_jobs = self.sagemaker_client.list_training_jobs(
                    MaxResults=100
                )

                trial_cost = 0
                for job in training_jobs['TrainingJobSummaries']:
                    if job.get('ExperimentConfig', {}).get('TrialName') == trial['TrialName']:
                        # Estimate cost for this training job
                        job_cost = self._estimate_training_job_cost({
                            'instance_type': 'ml.p3.2xlarge',  # Simplified
                            'instance_count': 1
                        })
                        trial_cost += job_cost

                trial_costs.append({
                    'trial_name': trial['TrialName'],
                    'cost': trial_cost
                })
                total_cost += trial_cost

            return {
                'experiment_name': experiment_name,
                'total_cost': total_cost,
                'trial_costs': trial_costs,
                'timestamp': datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Error getting experiment costs for {experiment_name}: {e}")
            return {}

    def get_recommendations(self) -> List[Dict[str, Any]]:
        """Get SageMaker optimization recommendations (right-sizing + spot training)."""
        recommendations = []

        # Right-sizing recommendations for endpoints
        try:
            endpoints = self.get_endpoints()
            for ep in endpoints:
                rec = self._check_endpoint_rightsizing(ep)
                if rec:
                    recommendations.append(rec)
        except Exception as e:
            logger.warning(f"Error getting endpoint rightsizing recommendations: {e}")

        # Managed spot training recommendations
        try:
            recs = self._get_spot_training_recommendations()
            recommendations.extend(recs)
        except Exception as e:
            logger.warning(f"Error getting spot training recommendations: {e}")

        logger.info(f"Found {len(recommendations)} SageMaker recommendations")
        return recommendations

    def _check_endpoint_rightsizing(self, endpoint: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Check if a SageMaker endpoint is over-provisioned based on invocation metrics."""
        try:
            metrics = self.get_endpoint_metrics(endpoint['endpoint_name'])
            invocations = metrics.get('invocations', [])

            if not invocations:
                return None

            # Calculate average invocations per 5-min period
            avg_invocations = sum(d.get('sum', 0) for d in invocations) / max(len(invocations), 1)

            # Get endpoint config to check instance type
            config = self.sagemaker_client.describe_endpoint_config(
                EndpointConfigName=endpoint['endpoint_config_name']
            )

            for variant in config.get('ProductionVariants', []):
                instance_type = variant.get('InstanceType', '')
                instance_count = variant.get('InitialInstanceCount', 1)

                # If low invocations and using large instance, suggest downsizing
                if avg_invocations < 10 and instance_count > 1:
                    current_cost = self._estimate_endpoint_cost(endpoint)
                    smaller_cost = current_cost / instance_count  # rough estimate for 1 instance
                    return {
                        'instance_type': instance_type,
                        'recommendation_type': 'right_size',
                        'resource_name': endpoint['endpoint_name'],
                        'resource_type': 'sagemaker_endpoint',
                        'current_monthly_cost': round(current_cost * 730, 2),
                        'recommended_monthly_cost': round(smaller_cost * 730, 2),
                        'potential_monthly_savings': round((current_cost - smaller_cost) * 730, 2),
                        'potential_savings_pct': round((1 - smaller_cost / current_cost) * 100, 1) if current_cost > 0 else 0,
                        'cloud_provider': 'aws',
                        'source_data': {
                            'reason': f"Low invocation rate ({avg_invocations:.1f}/5min) with {instance_count} instances",
                            'avg_invocations_per_5min': avg_invocations,
                            'current_instance_count': instance_count,
                            'recommended_instance_count': 1,
                        }
                    }

        except Exception as e:
            logger.debug(f"Error checking endpoint rightsizing for {endpoint.get('endpoint_name')}: {e}")
        return None

    def _get_spot_training_recommendations(self) -> List[Dict[str, Any]]:
        """Recommend managed spot training for recent non-spot training jobs."""
        recommendations = []
        try:
            # Get recently completed training jobs (last 7 days)
            response = self.sagemaker_client.list_training_jobs(
                StatusEquals='Completed',
                MaxResults=50,
                SortBy='CreationTime',
                SortOrder='Descending',
            )

            for job_summary in response.get('TrainingJobSummaries', []):
                try:
                    job = self.sagemaker_client.describe_training_job(
                        TrainingJobName=job_summary['TrainingJobName']
                    )

                    # Skip if already using managed spot
                    if job.get('EnableManagedSpotTraining', False):
                        continue

                    instance_type = job['ResourceConfig']['InstanceType']
                    instance_count = job['ResourceConfig']['InstanceCount']

                    # Only recommend for GPU instances
                    if not any(prefix in instance_type for prefix in ['ml.p', 'ml.g']):
                        continue

                    hourly_cost = self._estimate_training_job_cost({
                        'instance_type': instance_type,
                        'instance_count': instance_count,
                    })

                    # Managed spot typically saves 60-90%
                    spot_savings_pct = 70  # conservative estimate
                    spot_cost = hourly_cost * (1 - spot_savings_pct / 100)

                    recommendations.append({
                        'instance_type': instance_type,
                        'recommendation_type': 'spot',
                        'resource_name': job_summary['TrainingJobName'],
                        'resource_type': 'sagemaker_training',
                        'current_monthly_cost': round(hourly_cost * 730, 2),
                        'recommended_monthly_cost': round(spot_cost * 730, 2),
                        'potential_monthly_savings': round((hourly_cost - spot_cost) * 730, 2),
                        'potential_savings_pct': spot_savings_pct,
                        'cloud_provider': 'aws',
                        'source_data': {
                            'reason': f"Training job {job_summary['TrainingJobName']} used {instance_count}x {instance_type} without managed spot",
                            'instance_count': instance_count,
                            'managed_spot_enabled': False,
                        }
                    })
                except Exception as e:
                    logger.debug(f"Error checking training job {job_summary.get('TrainingJobName')}: {e}")

        except Exception as e:
            logger.warning(f"Error getting spot training recommendations: {e}")

        return recommendations
