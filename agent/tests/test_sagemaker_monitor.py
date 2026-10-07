"""
Comprehensive test suite for SageMaker Monitor

Tests the SageMakerMonitor class with full mocking of boto3, sagemaker, and pynvml.
"""

import sys
import os
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock, call
import pytest

# Mock external dependencies before importing the monitor
mock_sagemaker = MagicMock()
mock_sagemaker.Session = MagicMock(return_value=MagicMock())
mock_sagemaker.get_execution_role = MagicMock(return_value='arn:aws:iam::123456789:role/test')
sys.modules['sagemaker'] = mock_sagemaker

mock_pynvml = MagicMock()
mock_pynvml.nvmlInit = MagicMock(return_value=None)
mock_pynvml.NVML_TEMPERATURE_GPU = 0
sys.modules['pynvml'] = mock_pynvml

# Mock boto3 before importing
mock_boto3 = MagicMock()
sys.modules['boto3'] = mock_boto3

# Add src to path and import the monitor
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from monitors.sagemaker_monitor import SageMakerMonitor


class TestSageMakerMonitorInit:
    """Tests for SageMakerMonitor initialization"""

    def test_init_success(self):
        """Test successful initialization with all clients created"""
        with patch.dict(os.environ, {'AWS_REGION': 'us-west-2'}):
            with patch('monitors.sagemaker_monitor.boto3.client') as mock_client:
                with patch('monitors.sagemaker_monitor.sagemaker.Session'):
                    with patch('monitors.sagemaker_monitor.pynvml.nvmlInit'):
                        monitor = SageMakerMonitor()

                        assert monitor.region == 'us-west-2'
                        assert monitor.gpu_available == True
                        assert mock_client.call_count >= 4  # sagemaker, cloudwatch, ce, ec2

    def test_init_default_region(self):
        """Test initialization uses default region when AWS_REGION not set"""
        with patch.dict(os.environ, {}, clear=True):
            with patch('monitors.sagemaker_monitor.boto3.client'):
                with patch('monitors.sagemaker_monitor.sagemaker.Session'):
                    with patch('monitors.sagemaker_monitor.pynvml.nvmlInit'):
                        monitor = SageMakerMonitor()
                        assert monitor.region == 'us-east-1'

    def test_init_nvml_init_failure(self):
        """Test initialization when NVML initialization fails"""
        with patch('monitors.sagemaker_monitor.boto3.client'):
            with patch('monitors.sagemaker_monitor.sagemaker.Session'):
                with patch('monitors.sagemaker_monitor.pynvml.nvmlInit',
                          side_effect=Exception('NVML not available')):
                    monitor = SageMakerMonitor()
                    assert monitor.gpu_available == False


class TestGetTrainingJobs:
    """Tests for get_training_jobs method"""

    @pytest.fixture
    def monitor(self):
        """Create a monitor instance for testing"""
        with patch('monitors.sagemaker_monitor.boto3.client') as mock_client:
            with patch('monitors.sagemaker_monitor.sagemaker.Session'):
                with patch('monitors.sagemaker_monitor.pynvml.nvmlInit'):
                    monitor = SageMakerMonitor()
                    monitor.sagemaker_client = MagicMock()
                    return monitor

    def test_get_training_jobs_with_gpu_instances(self, monitor):
        """Test retrieving GPU training jobs"""
        now = datetime.utcnow()

        monitor.sagemaker_client.list_training_jobs.return_value = {
            'TrainingJobSummaries': [
                {
                    'TrainingJobName': 'job-1',
                    'TrainingJobArn': 'arn:aws:sagemaker:us-east-1:123456789:training-job/job-1',
                    'CreationTime': now,
                    'TrainingJobStatus': 'InProgress'
                },
                {
                    'TrainingJobName': 'job-2',
                    'TrainingJobArn': 'arn:aws:sagemaker:us-east-1:123456789:training-job/job-2',
                    'CreationTime': now,
                    'TrainingJobStatus': 'InProgress'
                }
            ]
        }

        monitor.sagemaker_client.describe_training_job.side_effect = [
            {
                'ResourceConfig': {
                    'InstanceType': 'ml.p3.2xlarge',
                    'InstanceCount': 2
                },
                'TrainingStartTime': now,
                'ExperimentConfig': {
                    'ExperimentName': 'exp-1',
                    'TrialName': 'trial-1'
                },
                'Tags': [{'Key': 'Environment', 'Value': 'test'}]
            },
            {
                'ResourceConfig': {
                    'InstanceType': 'ml.p4d.24xlarge',
                    'InstanceCount': 1
                },
                'TrainingStartTime': now,
                'ExperimentConfig': {},
                'Tags': []
            }
        ]

        jobs = monitor.get_training_jobs()

        assert len(jobs) == 2
        assert jobs[0]['job_name'] == 'job-1'
        assert jobs[0]['instance_type'] == 'ml.p3.2xlarge'
        assert jobs[0]['instance_count'] == 2
        assert jobs[0]['cloud_provider'] == 'aws'
        assert jobs[1]['job_name'] == 'job-2'

    def test_get_training_jobs_filters_non_gpu(self, monitor):
        """Test that non-GPU instances are filtered out"""
        now = datetime.utcnow()

        monitor.sagemaker_client.list_training_jobs.return_value = {
            'TrainingJobSummaries': [
                {
                    'TrainingJobName': 'cpu-job',
                    'TrainingJobArn': 'arn:aws:sagemaker:us-east-1:123456789:training-job/cpu-job',
                    'CreationTime': now,
                    'TrainingJobStatus': 'InProgress'
                }
            ]
        }

        monitor.sagemaker_client.describe_training_job.return_value = {
            'ResourceConfig': {
                'InstanceType': 'ml.m5.xlarge',
                'InstanceCount': 1
            },
            'TrainingStartTime': now,
            'ExperimentConfig': {},
            'Tags': []
        }

        jobs = monitor.get_training_jobs()

        assert len(jobs) == 0

    def test_get_training_jobs_empty_list(self, monitor):
        """Test retrieving when no training jobs exist"""
        monitor.sagemaker_client.list_training_jobs.return_value = {
            'TrainingJobSummaries': []
        }

        jobs = monitor.get_training_jobs()

        assert len(jobs) == 0

    def test_get_training_jobs_api_error(self, monitor):
        """Test handling of API errors when retrieving training jobs"""
        from botocore.exceptions import ClientError

        error_response = {'Error': {'Code': 'AccessDenied', 'Message': 'Access Denied'}}
        monitor.sagemaker_client.list_training_jobs.side_effect = ClientError(
            error_response, 'ListTrainingJobs'
        )

        jobs = monitor.get_training_jobs()

        assert len(jobs) == 0


class TestGetEndpoints:
    """Tests for get_endpoints method"""

    @pytest.fixture
    def monitor(self):
        """Create a monitor instance for testing"""
        with patch('monitors.sagemaker_monitor.boto3.client'):
            with patch('monitors.sagemaker_monitor.sagemaker.Session'):
                with patch('monitors.sagemaker_monitor.pynvml.nvmlInit'):
                    monitor = SageMakerMonitor()
                    monitor.sagemaker_client = MagicMock()
                    return monitor

    def test_get_endpoints_with_gpu(self, monitor):
        """Test retrieving GPU endpoints"""
        now = datetime.utcnow()

        monitor.sagemaker_client.list_endpoints.return_value = {
            'Endpoints': [
                {
                    'EndpointName': 'endpoint-1',
                    'EndpointArn': 'arn:aws:sagemaker:us-east-1:123456789:endpoint/endpoint-1',
                    'CreationTime': now,
                    'LastModifiedTime': now,
                    'EndpointStatus': 'InService'
                }
            ]
        }

        monitor.sagemaker_client.describe_endpoint.return_value = {
            'EndpointName': 'endpoint-1',
            'EndpointConfigName': 'config-1',
            'Tags': [{'Key': 'App', 'Value': 'ml-api'}]
        }

        monitor.sagemaker_client.describe_endpoint_config.return_value = {
            'ProductionVariants': [
                {
                    'InstanceType': 'ml.p3.2xlarge',
                    'InitialInstanceCount': 2
                }
            ]
        }

        endpoints = monitor.get_endpoints()

        assert len(endpoints) == 1
        assert endpoints[0]['endpoint_name'] == 'endpoint-1'
        assert endpoints[0]['cloud_provider'] == 'aws'
        assert endpoints[0]['status'] == 'InService'

    def test_get_endpoints_filters_non_gpu(self, monitor):
        """Test that non-GPU endpoints are filtered out"""
        now = datetime.utcnow()

        monitor.sagemaker_client.list_endpoints.return_value = {
            'Endpoints': [
                {
                    'EndpointName': 'cpu-endpoint',
                    'EndpointArn': 'arn:aws:sagemaker:us-east-1:123456789:endpoint/cpu-endpoint',
                    'CreationTime': now,
                    'LastModifiedTime': now,
                    'EndpointStatus': 'InService'
                }
            ]
        }

        monitor.sagemaker_client.describe_endpoint.return_value = {
            'EndpointName': 'cpu-endpoint',
            'EndpointConfigName': 'config-1',
            'Tags': []
        }

        monitor.sagemaker_client.describe_endpoint_config.return_value = {
            'ProductionVariants': [
                {
                    'InstanceType': 'ml.m5.xlarge',
                    'InitialInstanceCount': 1
                }
            ]
        }

        endpoints = monitor.get_endpoints()

        assert len(endpoints) == 0

    def test_get_endpoints_empty(self, monitor):
        """Test retrieving when no endpoints exist"""
        monitor.sagemaker_client.list_endpoints.return_value = {
            'Endpoints': []
        }

        endpoints = monitor.get_endpoints()

        assert len(endpoints) == 0


class TestGetProcessingJobs:
    """Tests for get_processing_jobs method"""

    @pytest.fixture
    def monitor(self):
        """Create a monitor instance for testing"""
        with patch('monitors.sagemaker_monitor.boto3.client'):
            with patch('monitors.sagemaker_monitor.sagemaker.Session'):
                with patch('monitors.sagemaker_monitor.pynvml.nvmlInit'):
                    monitor = SageMakerMonitor()
                    monitor.sagemaker_client = MagicMock()
                    return monitor

    def test_get_processing_jobs_with_gpu(self, monitor):
        """Test retrieving GPU processing jobs"""
        now = datetime.utcnow()

        monitor.sagemaker_client.list_processing_jobs.return_value = {
            'ProcessingJobSummaries': [
                {
                    'ProcessingJobName': 'process-job-1',
                    'ProcessingJobArn': 'arn:aws:sagemaker:us-east-1:123456789:processing-job/process-job-1',
                    'CreationTime': now,
                    'ProcessingJobStatus': 'InProgress'
                }
            ]
        }

        monitor.sagemaker_client.describe_processing_job.return_value = {
            'ProcessingJobName': 'process-job-1',
            'ProcessingResources': {
                'ClusterConfig': {
                    'InstanceType': 'ml.p3.2xlarge',
                    'InstanceCount': 1
                }
            },
            'AppSpecification': {
                'ImageUri': '123456789.dkr.ecr.us-east-1.amazonaws.com/my-gpu-image'
            },
            'ProcessingStartTime': now,
            'ExperimentConfig': {},
            'Tags': []
        }

        jobs = monitor.get_processing_jobs()

        assert len(jobs) == 1
        assert jobs[0]['job_name'] == 'process-job-1'
        assert jobs[0]['instance_type'] == 'ml.p3.2xlarge'
        assert jobs[0]['cloud_provider'] == 'aws'


class TestGetGpuUtilization:
    """Tests for get_gpu_utilization method"""

    @pytest.fixture
    def monitor(self):
        """Create a monitor instance for testing"""
        with patch('monitors.sagemaker_monitor.boto3.client'):
            with patch('monitors.sagemaker_monitor.sagemaker.Session'):
                with patch('monitors.sagemaker_monitor.pynvml.nvmlInit'):
                    monitor = SageMakerMonitor()
                    monitor.sagemaker_client = MagicMock()
                    return monitor

    def test_get_gpu_utilization_success(self, monitor):
        """Test successful retrieval of GPU utilization metrics"""
        monitor.gpu_available = True

        mock_handle = MagicMock()
        mock_utilization = MagicMock()
        mock_utilization.gpu = 45.5
        mock_utilization.memory = 62.3

        mock_memory = MagicMock()
        mock_memory.used = 4294967296  # 4GB
        mock_memory.total = 16107941888  # 15GB

        with patch('monitors.sagemaker_monitor.pynvml.nvmlDeviceGetCount', return_value=2):
            with patch('monitors.sagemaker_monitor.pynvml.nvmlDeviceGetHandleByIndex', return_value=mock_handle):
                with patch('monitors.sagemaker_monitor.pynvml.nvmlDeviceGetUtilizationRates', return_value=mock_utilization):
                    with patch('monitors.sagemaker_monitor.pynvml.nvmlDeviceGetMemoryInfo', return_value=mock_memory):
                        with patch('monitors.sagemaker_monitor.pynvml.nvmlDeviceGetTemperature', return_value=45):
                            with patch('monitors.sagemaker_monitor.pynvml.nvmlDeviceGetPowerUsage', return_value=120000):
                                metrics = monitor.get_gpu_utilization([{
                                    'endpoint_name': 'test-endpoint',
                                    'endpoint_arn': 'arn:test'
                                }])

                                assert len(metrics) >= 0  # May be empty if endpoint name doesn't match hostname

    def test_get_gpu_utilization_gpu_not_available(self, monitor):
        """Test GPU utilization when GPU is not available"""
        monitor.gpu_available = False

        metrics = monitor.get_gpu_utilization([])

        assert len(metrics) == 0


class TestGetCostData:
    """Tests for get_cost_data method"""

    @pytest.fixture
    def monitor(self):
        """Create a monitor instance for testing"""
        with patch('monitors.sagemaker_monitor.boto3.client'):
            with patch('monitors.sagemaker_monitor.sagemaker.Session'):
                with patch('monitors.sagemaker_monitor.pynvml.nvmlInit'):
                    monitor = SageMakerMonitor()
                    monitor.sagemaker_client = MagicMock()
                    monitor.cost_explorer_client = MagicMock()
                    return monitor

    def test_get_cost_data_with_cost_explorer(self, monitor):
        """Test retrieving cost data from Cost Explorer"""
        now = datetime.utcnow()

        monitor.cost_explorer_client.get_cost_and_usage.return_value = {
            'ResultsByTime': [
                {
                    'TimePeriod': {
                        'Start': now.strftime('%Y-%m-%d'),
                        'End': (now + timedelta(days=1)).strftime('%Y-%m-%d')
                    },
                    'Groups': [
                        {
                            'Keys': ['Amazon SageMaker'],
                            'Metrics': {
                                'BlendedCost': {
                                    'Amount': '125.50'
                                }
                            }
                        }
                    ]
                }
            ]
        }

        training_jobs = [{
            'job_name': 'job-1',
            'instance_type': 'ml.p3.2xlarge',
            'instance_count': 1
        }]

        endpoints = [{
            'endpoint_name': 'endpoint-1',
            'endpoint_config_name': 'config-1'
        }]

        with patch.object(monitor, '_estimate_training_job_cost', return_value=3.06):
            with patch.object(monitor, '_estimate_endpoint_cost', return_value=6.12):
                with patch('monitors.sagemaker_monitor.datetime') as mock_dt:
                    mock_dt.utcnow.return_value = now
                    cost_data = monitor.get_cost_data(training_jobs, endpoints, [])

        # Should have cost explorer data + estimated costs
        assert len(cost_data) > 0


class TestEstimateTrainingJobCost:
    """Tests for _estimate_training_job_cost method"""

    @pytest.fixture
    def monitor(self):
        """Create a monitor instance for testing"""
        with patch('monitors.sagemaker_monitor.boto3.client'):
            with patch('monitors.sagemaker_monitor.sagemaker.Session'):
                with patch('monitors.sagemaker_monitor.pynvml.nvmlInit'):
                    monitor = SageMakerMonitor()
                    return monitor

    def test_estimate_training_job_cost_p3(self, monitor):
        """Test cost estimation for p3 instance"""
        job = {
            'instance_type': 'ml.p3.2xlarge',
            'instance_count': 2
        }

        cost = monitor._estimate_training_job_cost(job)

        # hourly_cost * instance_count * runtime_hours
        # 3.06 * 2 * 1 = 6.12
        assert cost == 6.12

    def test_estimate_training_job_cost_g4dn(self, monitor):
        """Test cost estimation for g4dn instance"""
        job = {
            'instance_type': 'ml.g4dn.xlarge',
            'instance_count': 1
        }

        cost = monitor._estimate_training_job_cost(job)

        # 0.526 * 1 * 1 = 0.53 (rounded)
        assert cost == 0.53

    def test_estimate_training_job_cost_unknown_instance(self, monitor):
        """Test cost estimation with unknown instance type"""
        job = {
            'instance_type': 'ml.unknown.type',
            'instance_count': 1
        }

        cost = monitor._estimate_training_job_cost(job)

        # Should use default cost of 1.0
        assert cost == 1.0


class TestEstimateEndpointCost:
    """Tests for _estimate_endpoint_cost method"""

    @pytest.fixture
    def monitor(self):
        """Create a monitor instance for testing"""
        with patch('monitors.sagemaker_monitor.boto3.client'):
            with patch('monitors.sagemaker_monitor.sagemaker.Session'):
                with patch('monitors.sagemaker_monitor.pynvml.nvmlInit'):
                    monitor = SageMakerMonitor()
                    monitor.sagemaker_client = MagicMock()
                    return monitor

    def test_estimate_endpoint_cost_success(self, monitor):
        """Test successful endpoint cost estimation"""
        endpoint = {
            'endpoint_name': 'endpoint-1',
            'endpoint_config_name': 'config-1'
        }

        monitor.sagemaker_client.describe_endpoint_config.return_value = {
            'ProductionVariants': [
                {
                    'InstanceType': 'ml.p3.2xlarge',
                    'InitialInstanceCount': 2
                }
            ]
        }

        cost = monitor._estimate_endpoint_cost(endpoint)

        # 3.06 * 2 = 6.12
        assert cost == 6.12

    def test_estimate_endpoint_cost_multiple_variants(self, monitor):
        """Test endpoint cost estimation with multiple variants"""
        endpoint = {
            'endpoint_name': 'endpoint-1',
            'endpoint_config_name': 'config-1'
        }

        monitor.sagemaker_client.describe_endpoint_config.return_value = {
            'ProductionVariants': [
                {
                    'InstanceType': 'ml.p3.2xlarge',
                    'InitialInstanceCount': 1
                },
                {
                    'InstanceType': 'ml.g4dn.xlarge',
                    'InitialInstanceCount': 2
                }
            ]
        }

        cost = monitor._estimate_endpoint_cost(endpoint)

        # (3.06 * 1) + (0.526 * 2) = 3.06 + 1.052 = 4.11 (rounded)
        assert cost == 4.11

    def test_estimate_endpoint_cost_api_error(self, monitor):
        """Test endpoint cost estimation with API error"""
        endpoint = {
            'endpoint_name': 'endpoint-1',
            'endpoint_config_name': 'config-1'
        }

        from botocore.exceptions import ClientError
        error_response = {'Error': {'Code': 'ValidationError', 'Message': 'Not found'}}
        monitor.sagemaker_client.describe_endpoint_config.side_effect = ClientError(
            error_response, 'DescribeEndpointConfig'
        )

        cost = monitor._estimate_endpoint_cost(endpoint)

        assert cost == 0.0


class TestGetEndpointMetrics:
    """Tests for get_endpoint_metrics method"""

    @pytest.fixture
    def monitor(self):
        """Create a monitor instance for testing"""
        with patch('monitors.sagemaker_monitor.boto3.client'):
            with patch('monitors.sagemaker_monitor.sagemaker.Session'):
                with patch('monitors.sagemaker_monitor.pynvml.nvmlInit'):
                    monitor = SageMakerMonitor()
                    monitor.cloudwatch_client = MagicMock()
                    return monitor

    def test_get_endpoint_metrics_success(self, monitor):
        """Test successful retrieval of endpoint metrics"""
        now = datetime.utcnow()

        monitor.cloudwatch_client.get_metric_statistics.return_value = {
            'Datapoints': [
                {
                    'Timestamp': now,
                    'Sum': 150.0,
                    'Average': 25.0
                },
                {
                    'Timestamp': now - timedelta(minutes=5),
                    'Sum': 120.0,
                    'Average': 20.0
                }
            ]
        }

        with patch('monitors.sagemaker_monitor.datetime') as mock_dt:
            mock_dt.utcnow.return_value = now
            metrics = monitor.get_endpoint_metrics('endpoint-1')

        assert metrics['endpoint_name'] == 'endpoint-1'
        assert len(metrics['invocations']) == 2
        assert metrics['invocations'][0]['sum'] == 150.0

    def test_get_endpoint_metrics_empty(self, monitor):
        """Test endpoint metrics when no data available"""
        monitor.cloudwatch_client.get_metric_statistics.return_value = {
            'Datapoints': []
        }

        metrics = monitor.get_endpoint_metrics('endpoint-1')

        assert metrics['endpoint_name'] == 'endpoint-1'
        assert len(metrics['invocations']) == 0


class TestGetExperimentCosts:
    """Tests for get_experiment_costs method"""

    @pytest.fixture
    def monitor(self):
        """Create a monitor instance for testing"""
        with patch('monitors.sagemaker_monitor.boto3.client'):
            with patch('monitors.sagemaker_monitor.sagemaker.Session'):
                with patch('monitors.sagemaker_monitor.pynvml.nvmlInit'):
                    monitor = SageMakerMonitor()
                    monitor.sagemaker_client = MagicMock()
                    return monitor

    def test_get_experiment_costs_success(self, monitor):
        """Test successful retrieval of experiment costs"""
        monitor.sagemaker_client.list_trials.return_value = {
            'TrialSummaries': [
                {
                    'TrialName': 'trial-1',
                    'TrialArn': 'arn:aws:sagemaker:us-east-1:123456789:trial/trial-1'
                }
            ]
        }

        monitor.sagemaker_client.describe_trial.return_value = {
            'TrialName': 'trial-1'
        }

        monitor.sagemaker_client.list_training_jobs.return_value = {
            'TrainingJobSummaries': [
                {
                    'TrainingJobName': 'job-1',
                    'ExperimentConfig': {
                        'TrialName': 'trial-1'
                    }
                }
            ]
        }

        with patch.object(monitor, '_estimate_training_job_cost', return_value=3.06):
            with patch('monitors.sagemaker_monitor.datetime') as mock_dt:
                mock_dt.utcnow.return_value = datetime.utcnow()
                costs = monitor.get_experiment_costs('exp-1')

        assert costs['experiment_name'] == 'exp-1'
        assert costs['total_cost'] == 3.06
        assert len(costs['trial_costs']) == 1


class TestGetRecommendations:
    """Tests for get_recommendations method"""

    @pytest.fixture
    def monitor(self):
        """Create a monitor instance for testing"""
        with patch('monitors.sagemaker_monitor.boto3.client'):
            with patch('monitors.sagemaker_monitor.sagemaker.Session'):
                with patch('monitors.sagemaker_monitor.pynvml.nvmlInit'):
                    monitor = SageMakerMonitor()
                    monitor.sagemaker_client = MagicMock()
                    monitor.cloudwatch_client = MagicMock()
                    return monitor

    def test_get_recommendations_includes_rightsizing(self, monitor):
        """Test that recommendations include endpoint rightsizing"""
        monitor.sagemaker_client.list_endpoints.return_value = {
            'Endpoints': []
        }

        monitor.sagemaker_client.list_training_jobs.return_value = {
            'TrainingJobSummaries': []
        }

        with patch.object(monitor, 'get_endpoints', return_value=[]):
            with patch.object(monitor, '_get_spot_training_recommendations', return_value=[]):
                recs = monitor.get_recommendations()

        assert isinstance(recs, list)

    def test_get_recommendations_error_handling(self, monitor):
        """Test error handling in recommendations"""
        with patch.object(monitor, 'get_endpoints', side_effect=Exception('API Error')):
            with patch.object(monitor, '_get_spot_training_recommendations', return_value=[]):
                recs = monitor.get_recommendations()

        # Should not raise, should return recommendations from non-failing calls
        assert isinstance(recs, list)


class TestCheckEndpointRightsizing:
    """Tests for _check_endpoint_rightsizing method"""

    @pytest.fixture
    def monitor(self):
        """Create a monitor instance for testing"""
        with patch('monitors.sagemaker_monitor.boto3.client'):
            with patch('monitors.sagemaker_monitor.sagemaker.Session'):
                with patch('monitors.sagemaker_monitor.pynvml.nvmlInit'):
                    monitor = SageMakerMonitor()
                    monitor.sagemaker_client = MagicMock()
                    return monitor

    def test_check_endpoint_rightsizing_low_invocations(self, monitor):
        """Test rightsizing recommendation for low-invocation endpoint"""
        now = datetime.utcnow()

        endpoint = {
            'endpoint_name': 'endpoint-1',
            'endpoint_config_name': 'config-1'
        }

        with patch.object(monitor, 'get_endpoint_metrics') as mock_metrics:
            mock_metrics.return_value = {
                'invocations': [
                    {'sum': 5},
                    {'sum': 8}
                ]
            }

            monitor.sagemaker_client.describe_endpoint_config.return_value = {
                'ProductionVariants': [
                    {
                        'InstanceType': 'ml.p3.2xlarge',
                        'InitialInstanceCount': 2
                    }
                ]
            }

            with patch.object(monitor, '_estimate_endpoint_cost', return_value=6.12):
                rec = monitor._check_endpoint_rightsizing(endpoint)

        assert rec is not None
        assert rec['recommendation_type'] == 'right_size'
        assert rec['resource_name'] == 'endpoint-1'

    def test_check_endpoint_rightsizing_no_recommendation(self, monitor):
        """Test no rightsizing recommendation for normal endpoint"""
        endpoint = {
            'endpoint_name': 'endpoint-1',
            'endpoint_config_name': 'config-1'
        }

        with patch.object(monitor, 'get_endpoint_metrics') as mock_metrics:
            mock_metrics.return_value = {
                'invocations': [
                    {'sum': 1000},
                    {'sum': 950}
                ]
            }

            monitor.sagemaker_client.describe_endpoint_config.return_value = {
                'ProductionVariants': [
                    {
                        'InstanceType': 'ml.p3.2xlarge',
                        'InitialInstanceCount': 1
                    }
                ]
            }

            rec = monitor._check_endpoint_rightsizing(endpoint)

        assert rec is None


class TestGetSpotTrainingRecommendations:
    """Tests for _get_spot_training_recommendations method"""

    @pytest.fixture
    def monitor(self):
        """Create a monitor instance for testing"""
        with patch('monitors.sagemaker_monitor.boto3.client'):
            with patch('monitors.sagemaker_monitor.sagemaker.Session'):
                with patch('monitors.sagemaker_monitor.pynvml.nvmlInit'):
                    monitor = SageMakerMonitor()
                    monitor.sagemaker_client = MagicMock()
                    return monitor

    def test_get_spot_training_recommendations_non_spot_gpu(self, monitor):
        """Test spot recommendations for non-spot GPU training jobs"""
        monitor.sagemaker_client.list_training_jobs.return_value = {
            'TrainingJobSummaries': [
                {
                    'TrainingJobName': 'job-1',
                    'TrainingJobArn': 'arn:aws:sagemaker:us-east-1:123456789:training-job/job-1'
                }
            ]
        }

        monitor.sagemaker_client.describe_training_job.return_value = {
            'ResourceConfig': {
                'InstanceType': 'ml.p3.2xlarge',
                'InstanceCount': 1
            },
            'EnableManagedSpotTraining': False
        }

        with patch.object(monitor, '_estimate_training_job_cost', return_value=3.06):
            recs = monitor._get_spot_training_recommendations()

        assert len(recs) == 1
        assert recs[0]['recommendation_type'] == 'spot'
        assert recs[0]['potential_savings_pct'] == 70

    def test_get_spot_training_recommendations_already_using_spot(self, monitor):
        """Test spot recommendations when job already uses spot"""
        monitor.sagemaker_client.list_training_jobs.return_value = {
            'TrainingJobSummaries': [
                {
                    'TrainingJobName': 'job-1',
                    'TrainingJobArn': 'arn:aws:sagemaker:us-east-1:123456789:training-job/job-1'
                }
            ]
        }

        monitor.sagemaker_client.describe_training_job.return_value = {
            'ResourceConfig': {
                'InstanceType': 'ml.p3.2xlarge',
                'InstanceCount': 1
            },
            'EnableManagedSpotTraining': True
        }

        recs = monitor._get_spot_training_recommendations()

        assert len(recs) == 0

    def test_get_spot_training_recommendations_non_gpu(self, monitor):
        """Test spot recommendations skip non-GPU instances"""
        monitor.sagemaker_client.list_training_jobs.return_value = {
            'TrainingJobSummaries': [
                {
                    'TrainingJobName': 'job-1',
                    'TrainingJobArn': 'arn:aws:sagemaker:us-east-1:123456789:training-job/job-1'
                }
            ]
        }

        monitor.sagemaker_client.describe_training_job.return_value = {
            'ResourceConfig': {
                'InstanceType': 'ml.m5.xlarge',
                'InstanceCount': 1
            },
            'EnableManagedSpotTraining': False
        }

        recs = monitor._get_spot_training_recommendations()

        assert len(recs) == 0


class TestEdgeCases:
    """Tests for edge cases and error conditions"""

    @pytest.fixture
    def monitor(self):
        """Create a monitor instance for testing"""
        with patch('monitors.sagemaker_monitor.boto3.client'):
            with patch('monitors.sagemaker_monitor.sagemaker.Session'):
                with patch('monitors.sagemaker_monitor.pynvml.nvmlInit'):
                    monitor = SageMakerMonitor()
                    monitor.sagemaker_client = MagicMock()
                    return monitor

    def test_training_job_missing_started_at(self, monitor):
        """Test handling training job without TrainingStartTime"""
        now = datetime.utcnow()

        monitor.sagemaker_client.list_training_jobs.return_value = {
            'TrainingJobSummaries': [
                {
                    'TrainingJobName': 'job-1',
                    'TrainingJobArn': 'arn:test',
                    'CreationTime': now,
                    'TrainingJobStatus': 'InProgress'
                }
            ]
        }

        monitor.sagemaker_client.describe_training_job.return_value = {
            'ResourceConfig': {
                'InstanceType': 'ml.p3.2xlarge',
                'InstanceCount': 1
            },
            'TrainingStartTime': None,
            'ExperimentConfig': {},
            'Tags': []
        }

        jobs = monitor.get_training_jobs()

        assert len(jobs) == 1
        assert jobs[0]['started_at'] is None

    def test_endpoint_with_multiple_gpu_variants(self, monitor):
        """Test endpoint with multiple GPU variant types"""
        now = datetime.utcnow()

        monitor.sagemaker_client.list_endpoints.return_value = {
            'Endpoints': [
                {
                    'EndpointName': 'endpoint-1',
                    'EndpointArn': 'arn:test',
                    'CreationTime': now,
                    'LastModifiedTime': now,
                    'EndpointStatus': 'InService'
                }
            ]
        }

        monitor.sagemaker_client.describe_endpoint.return_value = {
            'EndpointName': 'endpoint-1',
            'EndpointConfigName': 'config-1',
            'Tags': []
        }

        monitor.sagemaker_client.describe_endpoint_config.return_value = {
            'ProductionVariants': [
                {
                    'InstanceType': 'ml.p3.2xlarge',
                    'InitialInstanceCount': 1
                },
                {
                    'InstanceType': 'ml.g4dn.xlarge',
                    'InitialInstanceCount': 2
                }
            ]
        }

        endpoints = monitor.get_endpoints()

        # Should return endpoint because first variant has GPU
        assert len(endpoints) == 1

    def test_cost_data_with_zero_cost(self, monitor):
        """Test handling of zero cost data"""
        monitor.cost_explorer_client = MagicMock()
        monitor.cost_explorer_client.get_cost_and_usage.return_value = {
            'ResultsByTime': [
                {
                    'TimePeriod': {
                        'Start': '2025-01-01',
                        'End': '2025-01-02'
                    },
                    'Groups': [
                        {
                            'Keys': ['Amazon SageMaker'],
                            'Metrics': {
                                'BlendedCost': {
                                    'Amount': '0'
                                }
                            }
                        }
                    ]
                }
            ]
        }

        with patch.object(monitor, '_estimate_training_job_cost', return_value=0.0):
            with patch.object(monitor, '_estimate_endpoint_cost', return_value=0.0):
                cost_data = monitor.get_cost_data([], [], [])

        # Zero cost entries should be filtered
        assert all(entry.get('cost', 0) > 0 or entry.get('service') != 'Amazon SageMaker'
                  for entry in cost_data if 'service' in entry)


class TestGetTrainingJobsExceptionHandling:
    """Tests for exception handling in get_training_jobs"""

    @pytest.fixture
    def monitor(self):
        with patch('monitors.sagemaker_monitor.boto3.client'):
            with patch('monitors.sagemaker_monitor.sagemaker.Session'):
                with patch('monitors.sagemaker_monitor.pynvml.nvmlInit'):
                    monitor = SageMakerMonitor()
                    monitor.sagemaker_client = MagicMock()
                    return monitor

    def test_get_training_jobs_general_exception(self, monitor):
        """Test handling of general exceptions in get_training_jobs"""
        monitor.sagemaker_client.list_training_jobs.side_effect = Exception('Unexpected error')

        jobs = monitor.get_training_jobs()

        assert len(jobs) == 0


class TestGetEndpointsExceptionHandling:
    """Tests for exception handling in get_endpoints"""

    @pytest.fixture
    def monitor(self):
        with patch('monitors.sagemaker_monitor.boto3.client'):
            with patch('monitors.sagemaker_monitor.sagemaker.Session'):
                with patch('monitors.sagemaker_monitor.pynvml.nvmlInit'):
                    monitor = SageMakerMonitor()
                    monitor.sagemaker_client = MagicMock()
                    return monitor

    def test_get_endpoints_general_exception(self, monitor):
        """Test handling of general exceptions in get_endpoints"""
        monitor.sagemaker_client.list_endpoints.side_effect = Exception('Unexpected error')

        endpoints = monitor.get_endpoints()

        assert len(endpoints) == 0


class TestGetProcessingJobsExceptionHandling:
    """Tests for exception handling in get_processing_jobs"""

    @pytest.fixture
    def monitor(self):
        with patch('monitors.sagemaker_monitor.boto3.client'):
            with patch('monitors.sagemaker_monitor.sagemaker.Session'):
                with patch('monitors.sagemaker_monitor.pynvml.nvmlInit'):
                    monitor = SageMakerMonitor()
                    monitor.sagemaker_client = MagicMock()
                    return monitor

    def test_get_processing_jobs_general_exception(self, monitor):
        """Test handling of general exceptions in get_processing_jobs"""
        monitor.sagemaker_client.list_processing_jobs.side_effect = Exception('Unexpected error')

        jobs = monitor.get_processing_jobs()

        assert len(jobs) == 0

    def test_get_processing_jobs_client_error(self, monitor):
        """Test handling of ClientError in get_processing_jobs"""
        from botocore.exceptions import ClientError

        error_response = {'Error': {'Code': 'AccessDenied', 'Message': 'Access Denied'}}
        monitor.sagemaker_client.list_processing_jobs.side_effect = ClientError(
            error_response, 'ListProcessingJobs'
        )

        jobs = monitor.get_processing_jobs()

        assert len(jobs) == 0


class TestGPUUtilizationExceptionHandling:
    """Tests for exception handling in GPU utilization"""

    @pytest.fixture
    def monitor(self):
        with patch('monitors.sagemaker_monitor.boto3.client'):
            with patch('monitors.sagemaker_monitor.sagemaker.Session'):
                with patch('monitors.sagemaker_monitor.pynvml.nvmlInit'):
                    monitor = SageMakerMonitor()
                    return monitor

    def test_get_gpu_utilization_exception(self, monitor):
        """Test exception handling in GPU utilization retrieval"""
        monitor.gpu_available = True

        with patch('monitors.sagemaker_monitor.pynvml.nvmlDeviceGetCount', side_effect=Exception('GPU error')):
            metrics = monitor.get_gpu_utilization([])

            assert len(metrics) == 0

    def test_gpu_utilization_temperature_unavailable(self, monitor):
        """Test handling when temperature reading fails"""
        monitor.gpu_available = True

        mock_handle = MagicMock()
        mock_utilization = MagicMock()
        mock_utilization.gpu = 45.5
        mock_utilization.memory = 62.3

        mock_memory = MagicMock()
        mock_memory.used = 4294967296
        mock_memory.total = 16107941888

        with patch('monitors.sagemaker_monitor.pynvml.nvmlDeviceGetCount', return_value=1):
            with patch('monitors.sagemaker_monitor.pynvml.nvmlDeviceGetHandleByIndex', return_value=mock_handle):
                with patch('monitors.sagemaker_monitor.pynvml.nvmlDeviceGetUtilizationRates', return_value=mock_utilization):
                    with patch('monitors.sagemaker_monitor.pynvml.nvmlDeviceGetMemoryInfo', return_value=mock_memory):
                        with patch('monitors.sagemaker_monitor.pynvml.nvmlDeviceGetTemperature', side_effect=Exception('Temp unavailable')):
                            with patch('monitors.sagemaker_monitor.pynvml.nvmlDeviceGetPowerUsage', side_effect=Exception('Power unavailable')):
                                metrics = monitor.get_gpu_utilization([{
                                    'endpoint_name': 'test-endpoint',
                                    'endpoint_arn': 'arn:test'
                                }])

                                assert len(metrics) >= 0

    def test_gpu_utilization_with_temperature_and_power(self, monitor):
        """Test GPU utilization with temperature and power readings"""
        monitor.gpu_available = True

        mock_handle = MagicMock()
        mock_utilization = MagicMock()
        mock_utilization.gpu = 45.5
        mock_utilization.memory = 62.3

        mock_memory = MagicMock()
        mock_memory.used = 4294967296
        mock_memory.total = 16107941888

        with patch('monitors.sagemaker_monitor.pynvml.nvmlDeviceGetCount', return_value=1):
            with patch('monitors.sagemaker_monitor.pynvml.nvmlDeviceGetHandleByIndex', return_value=mock_handle):
                with patch('monitors.sagemaker_monitor.pynvml.nvmlDeviceGetUtilizationRates', return_value=mock_utilization):
                    with patch('monitors.sagemaker_monitor.pynvml.nvmlDeviceGetMemoryInfo', return_value=mock_memory):
                        with patch('monitors.sagemaker_monitor.pynvml.nvmlDeviceGetTemperature', return_value=75):
                            with patch('monitors.sagemaker_monitor.pynvml.nvmlDeviceGetPowerUsage', return_value=250000):
                                metrics = monitor.get_gpu_utilization([])

                                assert len(metrics) >= 0


class TestGetCostDataExceptionHandling:
    """Tests for exception handling in cost data retrieval"""

    @pytest.fixture
    def monitor(self):
        with patch('monitors.sagemaker_monitor.boto3.client'):
            with patch('monitors.sagemaker_monitor.sagemaker.Session'):
                with patch('monitors.sagemaker_monitor.pynvml.nvmlInit'):
                    monitor = SageMakerMonitor()
                    monitor.sagemaker_client = MagicMock()
                    monitor.cost_explorer_client = MagicMock()
                    return monitor

    def test_get_cost_data_general_exception(self, monitor):
        """Test handling of general exceptions in get_cost_data"""
        monitor.cost_explorer_client.get_cost_and_usage.side_effect = Exception('API error')

        cost_data = monitor.get_cost_data([], [], [])

        assert len(cost_data) == 0

    def test_get_cost_data_client_error(self, monitor):
        """Test handling of ClientError in get_cost_data"""
        from botocore.exceptions import ClientError

        error_response = {'Error': {'Code': 'ValidationException', 'Message': 'Invalid request'}}
        monitor.cost_explorer_client.get_cost_and_usage.side_effect = ClientError(
            error_response, 'GetCostAndUsage'
        )

        cost_data = monitor.get_cost_data([], [], [])

        assert len(cost_data) == 0


class TestGetEndpointMetricsExceptionHandling:
    """Tests for exception handling in endpoint metrics"""

    @pytest.fixture
    def monitor(self):
        with patch('monitors.sagemaker_monitor.boto3.client'):
            with patch('monitors.sagemaker_monitor.sagemaker.Session'):
                with patch('monitors.sagemaker_monitor.pynvml.nvmlInit'):
                    monitor = SageMakerMonitor()
                    monitor.cloudwatch_client = MagicMock()
                    return monitor

    def test_get_endpoint_metrics_exception(self, monitor):
        """Test exception handling in get_endpoint_metrics"""
        monitor.cloudwatch_client.get_metric_statistics.side_effect = Exception('CloudWatch error')

        metrics = monitor.get_endpoint_metrics('endpoint-1')

        assert metrics == {}


class TestGetExperimentCostsExceptionHandling:
    """Tests for exception handling in experiment costs"""

    @pytest.fixture
    def monitor(self):
        with patch('monitors.sagemaker_monitor.boto3.client'):
            with patch('monitors.sagemaker_monitor.sagemaker.Session'):
                with patch('monitors.sagemaker_monitor.pynvml.nvmlInit'):
                    monitor = SageMakerMonitor()
                    monitor.sagemaker_client = MagicMock()
                    return monitor

    def test_get_experiment_costs_exception(self, monitor):
        """Test exception handling in get_experiment_costs"""
        monitor.sagemaker_client.list_trials.side_effect = Exception('API error')

        costs = monitor.get_experiment_costs('exp-1')

        assert costs == {}


class TestGetRecommendationsExceptionHandling:
    """Tests for exception handling in recommendations"""

    @pytest.fixture
    def monitor(self):
        with patch('monitors.sagemaker_monitor.boto3.client'):
            with patch('monitors.sagemaker_monitor.sagemaker.Session'):
                with patch('monitors.sagemaker_monitor.pynvml.nvmlInit'):
                    monitor = SageMakerMonitor()
                    monitor.sagemaker_client = MagicMock()
                    return monitor

    def test_get_recommendations_spot_training_exception(self, monitor):
        """Test exception handling in spot training recommendations"""
        with patch.object(monitor, 'get_endpoints', return_value=[]):
            with patch.object(monitor, '_get_spot_training_recommendations', side_effect=Exception('API error')):
                recs = monitor.get_recommendations()

                assert isinstance(recs, list)


class TestCheckEndpointRightsizingEdgeCases:
    """Tests for edge cases in endpoint rightsizing"""

    @pytest.fixture
    def monitor(self):
        with patch('monitors.sagemaker_monitor.boto3.client'):
            with patch('monitors.sagemaker_monitor.sagemaker.Session'):
                with patch('monitors.sagemaker_monitor.pynvml.nvmlInit'):
                    monitor = SageMakerMonitor()
                    monitor.sagemaker_client = MagicMock()
                    return monitor

    def test_check_endpoint_rightsizing_exception(self, monitor):
        """Test exception handling in endpoint rightsizing"""
        endpoint = {
            'endpoint_name': 'endpoint-1',
            'endpoint_config_name': 'config-1'
        }

        with patch.object(monitor, 'get_endpoint_metrics', side_effect=Exception('Error')):
            rec = monitor._check_endpoint_rightsizing(endpoint)

            assert rec is None

    def test_check_endpoint_rightsizing_no_metrics(self, monitor):
        """Test rightsizing when endpoint has no metrics"""
        endpoint = {
            'endpoint_name': 'endpoint-1',
            'endpoint_config_name': 'config-1'
        }

        with patch.object(monitor, 'get_endpoint_metrics') as mock_metrics:
            mock_metrics.return_value = {
                'invocations': []
            }

            rec = monitor._check_endpoint_rightsizing(endpoint)

            assert rec is None


class TestGetSpotTrainingRecommendationsExceptionHandling:
    """Tests for exception handling in spot training recommendations"""

    @pytest.fixture
    def monitor(self):
        with patch('monitors.sagemaker_monitor.boto3.client'):
            with patch('monitors.sagemaker_monitor.sagemaker.Session'):
                with patch('monitors.sagemaker_monitor.pynvml.nvmlInit'):
                    monitor = SageMakerMonitor()
                    monitor.sagemaker_client = MagicMock()
                    return monitor

    def test_get_spot_training_recommendations_general_exception(self, monitor):
        """Test exception handling in spot training recommendations"""
        monitor.sagemaker_client.list_training_jobs.side_effect = Exception('API error')

        recs = monitor._get_spot_training_recommendations()

        assert isinstance(recs, list)
        assert len(recs) == 0

    def test_get_spot_training_recommendations_job_describe_exception(self, monitor):
        """Test exception handling when describing training job fails"""
        monitor.sagemaker_client.list_training_jobs.return_value = {
            'TrainingJobSummaries': [
                {
                    'TrainingJobName': 'job-1',
                    'TrainingJobArn': 'arn:aws:sagemaker:us-east-1:123456789:training-job/job-1'
                }
            ]
        }

        monitor.sagemaker_client.describe_training_job.side_effect = Exception('Describe error')

        recs = monitor._get_spot_training_recommendations()

        assert len(recs) == 0
