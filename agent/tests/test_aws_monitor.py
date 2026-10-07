import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from monitors.aws_monitor import AWSMonitor


class TestAWSMonitor:
    """Test cases for AWS monitoring functionality."""

    def test_init(self, mock_aws_services):
        """Test AWSMonitor initialization."""
        monitor = AWSMonitor()

        # Check that clients are initialized
        assert monitor.ec2_client is not None
        assert monitor.cloudwatch is not None
        assert monitor.ce_client is not None
        # Check regions are set
        assert len(monitor._regions) > 0

    def test_get_gpu_instances(self, mock_aws_services, sample_gpu_instance):
        """Test getting GPU instances from AWS."""
        monitor = AWSMonitor()

        # Mock EC2 describe_instances response
        mock_response = {
            'Reservations': [
                {
                    'Instances': [
                        {
                            'InstanceId': sample_gpu_instance['instance_id'],
                            'InstanceType': sample_gpu_instance['instance_type'],
                            'State': {'Name': 'running'},
                            'Placement': {'AvailabilityZone': 'us-east-1a'},
                            'LaunchTime': datetime.now(),
                            'Tags': [
                                {'Key': 'Environment', 'Value': 'test'},
                                {'Key': 'experiment_id', 'Value': sample_gpu_instance['experiment_id']}
                            ]
                        }
                    ]
                }
            ]
        }

        with patch.object(monitor, '_ec2_client_for_region', return_value=mock_aws_services['ec2']):
            with patch.object(mock_aws_services['ec2'], 'describe_instances', return_value=mock_response):
                instances = monitor.get_gpu_instances()

                assert len(instances) == 1
                assert instances[0]['instance_id'] == sample_gpu_instance['instance_id']
                assert instances[0]['instance_type'] == sample_gpu_instance['instance_type']
                assert instances[0]['state'] == 'running'
                assert instances[0]['region'] == 'us-east-1'

    def test_get_gpu_instances_empty(self, mock_aws_services):
        """Test getting GPU instances when none exist."""
        monitor = AWSMonitor()

        # Mock empty response
        mock_response = {'Reservations': []}

        with patch.object(monitor, '_ec2_client_for_region', return_value=mock_aws_services['ec2']):
            with patch.object(mock_aws_services['ec2'], 'describe_instances', return_value=mock_response):
                instances = monitor.get_gpu_instances()

                assert len(instances) == 0

    def test_get_gpu_instances_error(self, mock_aws_services):
        """Test error handling when getting GPU instances."""
        monitor = AWSMonitor()

        with patch.object(monitor, '_ec2_client_for_region', return_value=mock_aws_services['ec2']):
            with patch.object(mock_aws_services['ec2'], 'describe_instances', side_effect=Exception('AWS Error')):
                instances = monitor.get_gpu_instances()

                assert len(instances) == 0

    def test_get_gpu_utilization_skips_cloudwatch(self, mock_aws_services, sample_gpu_instance):
        """AWS monitor does not read CloudWatch GPUUtilization — NVML/DCGM on-host only."""
        monitor = AWSMonitor()
        instance = {
            'instance_id': sample_gpu_instance['instance_id'],
            'instance_type': sample_gpu_instance['instance_type'],
            'state': 'running',
            'tags': {'experiment_id': 'exp-123'}
        }
        with patch.object(monitor.cloudwatch, 'get_metric_statistics') as mock_cw:
            metrics = monitor.get_gpu_utilization([instance])
            assert metrics == []
            mock_cw.assert_not_called()

    def test_get_cost_data(self, mock_aws_services):
        """Test getting cost data from Cost Explorer."""
        monitor = AWSMonitor()

        # Mock Cost Explorer get_cost_and_usage response
        # The service key must contain 'EC2' to be included (line 217 of source checks: if 'EC2' in group['Keys'][0])
        mock_response = {
            'ResultsByTime': [
                {
                    'TimePeriod': {'Start': '2024-01-01', 'End': '2024-01-02'},
                    'Groups': [
                        {
                            'Keys': ['EC2-Instances', 'g4dn.xlarge'],
                            'Metrics': {
                                'BlendedCost': {'Amount': '12.624', 'Unit': 'USD'}
                            }
                        }
                    ]
                }
            ]
        }

        with patch.object(monitor.ce_client, 'get_cost_and_usage', return_value=mock_response):
            cost_data = monitor.get_cost_data()

            assert len(cost_data) == 1
            assert cost_data[0]['date'] == '2024-01-01'
            assert float(cost_data[0]['cost']) == pytest.approx(12.624, rel=1e-3)
            assert cost_data[0]['currency'] == 'USD'
            assert cost_data[0]['instance_type'] == 'g4dn.xlarge'

    def test_get_cost_data_error(self, mock_aws_services):
        """Test error handling when getting cost data."""
        monitor = AWSMonitor()

        with patch.object(monitor.ce_client, 'get_cost_and_usage', side_effect=Exception('Cost Explorer Error')):
            cost_data = monitor.get_cost_data()

            assert len(cost_data) == 0

    def test_get_gpu_count(self, mock_aws_services):
        """Test getting GPU count for instance type."""
        monitor = AWSMonitor()

        assert monitor._get_gpu_count('g4dn.xlarge') == 1
        assert monitor._get_gpu_count('p3.8xlarge') == 4
        assert monitor._get_gpu_count('p3.16xlarge') == 8
        assert monitor._get_gpu_count('p5.48xlarge') == 8
        assert monitor._get_gpu_count('p5e.48xlarge') == 8
        assert monitor._get_gpu_count('p6-b200.48xlarge') == 8
        assert monitor._get_gpu_count('u-p6e-gb200x72') == 72
        assert monitor._get_gpu_count('unknown.type') == 0

    def test_get_gpu_type(self, mock_aws_services):
        """Test getting GPU type for instance type."""
        monitor = AWSMonitor()

        assert monitor._get_gpu_type('g4dn.xlarge') == 'T4'
        assert monitor._get_gpu_type('p3.2xlarge') == 'V100'
        assert monitor._get_gpu_type('p5.48xlarge') == 'H100'
        assert monitor._get_gpu_type('p5e.48xlarge') == 'H200'
        assert monitor._get_gpu_type('p5en.48xlarge') == 'H200'
        assert monitor._get_gpu_type('p6-b200.48xlarge') == 'B200'
        assert monitor._get_gpu_type('u-p6e-gb200x72') == 'GB200'
        assert monitor._get_gpu_type('g6e.xlarge') == 'L40S'
        assert monitor._get_gpu_type('unknown.type') == 'Unknown'

    def test_get_spot_pricing(self, mock_aws_services):
        """Test getting spot pricing data."""
        monitor = AWSMonitor()

        # Mock EC2 describe_spot_price_history response
        mock_response = {
            'SpotPriceHistory': [
                {
                    'InstanceType': 'g4dn.xlarge',
                    'SpotPrice': '0.3',
                    'AvailabilityZone': 'us-east-1a',
                    'Timestamp': datetime.now()
                }
            ]
        }

        with patch.object(monitor.ec2_client, 'describe_spot_price_history', return_value=mock_response):
            spot_data = monitor.get_spot_pricing()

            assert len(spot_data) >= 1
            assert spot_data[0]['instance_type'] == 'g4dn.xlarge'
            assert float(spot_data[0]['spot_price_hourly']) == 0.3

    def test_get_spot_candidates(self, mock_aws_services):
        """Test identifying spot instance candidates."""
        monitor = AWSMonitor()

        instances = [
            {
                'instance_id': 'i-123',
                'instance_type': 'g4dn.xlarge',
                'state': 'running',
                'tags': {'fault_tolerant': 'true'}
            },
            {
                'instance_id': 'i-456',
                'instance_type': 'p3.2xlarge',
                'state': 'running',
                'tags': {'workload_type': 'training'}
            },
            {
                'instance_id': 'i-789',
                'instance_type': 'g4dn.xlarge',
                'state': 'stopped',
                'tags': {'fault_tolerant': 'true'}
            }
        ]

        spot_pricing = [
            {
                'instance_type': 'g4dn.xlarge',
                'spot_price_hourly': 0.3,
                'on_demand_price_hourly': 0.526,
                'availability_zone': 'us-east-1a',
                'spot_monthly': 219.0,
                'on_demand_monthly': 384.0,
                'savings_pct': 43.0
            },
            {
                'instance_type': 'p3.2xlarge',
                'spot_price_hourly': 1.5,
                'on_demand_price_hourly': 3.06,
                'availability_zone': 'us-east-1a',
                'spot_monthly': 1095.0,
                'on_demand_monthly': 2234.0,
                'savings_pct': 51.0
            }
        ]

        candidates = monitor.get_spot_candidates(instances, spot_pricing)

        # Should find 2 candidates (the two running instances tagged as fault_tolerant or training)
        assert len(candidates) == 2
        assert all(c['recommendation_type'] == 'spot' for c in candidates)

    def test_stop_instance(self, mock_aws_services):
        """Test stopping an instance."""
        monitor = AWSMonitor()

        mock_response = {
            'StoppingInstances': [
                {'CurrentState': {'Name': 'stopping'}}
            ]
        }

        with patch.object(monitor.ec2_client, 'stop_instances', return_value=mock_response):
            result = monitor.stop_instance('i-123')

            assert result['instance_id'] == 'i-123'
            assert result['state'] == 'stopping'

    def test_start_instance(self, mock_aws_services):
        """Test starting an instance."""
        monitor = AWSMonitor()

        mock_response = {
            'StartingInstances': [
                {'CurrentState': {'Name': 'running'}}
            ]
        }

        with patch.object(monitor.ec2_client, 'start_instances', return_value=mock_response):
            result = monitor.start_instance('i-123')

            assert result['instance_id'] == 'i-123'
            assert result['state'] == 'running'

    def test_init_no_credentials(self):
        """Test initialization fails when no credentials found."""
        with patch('monitors.aws_monitor.boto3.Session') as mock_session_class:
            mock_session = Mock()
            mock_session.get_credentials.return_value = None
            mock_session_class.return_value = mock_session

            with pytest.raises(Exception, match="No AWS credentials found"):
                AWSMonitor()

    def test_init_credential_error(self):
        """Test initialization fails when credential error occurs."""
        with patch('monitors.aws_monitor.boto3.Session') as mock_session_class:
            mock_session = Mock()
            mock_session.get_credentials.side_effect = Exception("AWS credential error")
            mock_session_class.return_value = mock_session

            with pytest.raises(Exception, match="AWS credential error"):
                AWSMonitor()

    def test_get_regions_from_env_var(self):
        """Test region configuration from AWS_REGIONS env var."""
        monitor = AWSMonitor()

        with patch.dict(os.environ, {'AWS_REGIONS': 'us-west-1, eu-west-1, ap-southeast-1'}):
            regions = monitor._get_regions()
            assert 'us-west-1' in regions
            assert 'eu-west-1' in regions
            assert 'ap-southeast-1' in regions
            assert len(regions) == 3

    def test_get_regions_fallback_to_single_region(self):
        """Test fallback to single region when AWS_REGIONS not set."""
        monitor = AWSMonitor()

        with patch.dict(os.environ, {'AWS_REGION': 'eu-central-1'}, clear=True):
            regions = monitor._get_regions()
            assert regions == ['eu-central-1']

    def test_ec2_client_for_region_caching(self, mock_aws_services):
        """Test that EC2 clients are cached per region."""
        monitor = AWSMonitor()

        with patch.object(monitor.session, 'client', return_value=mock_aws_services['ec2']) as mock_client:
            # First call
            client1 = monitor._ec2_client_for_region('us-west-2')
            assert mock_client.call_count == 1

            # Second call should use cached value
            client2 = monitor._ec2_client_for_region('us-west-2')
            assert mock_client.call_count == 1  # No additional call
            assert client1 is client2

    def test_get_gpu_instances_with_multiple_regions(self, mock_aws_services):
        """Test getting GPU instances from multiple regions."""
        monitor = AWSMonitor(['us-east-1', 'us-west-2'])

        mock_response = {
            'Reservations': [
                {
                    'Instances': [
                        {
                            'InstanceId': 'i-region1',
                            'InstanceType': 'p3.2xlarge',
                            'State': {'Name': 'running'},
                            'Placement': {'AvailabilityZone': 'us-east-1a'},
                            'LaunchTime': datetime.now(),
                            'Tags': []
                        }
                    ]
                }
            ]
        }

        with patch.object(monitor, '_ec2_client_for_region', return_value=mock_aws_services['ec2']):
            with patch.object(mock_aws_services['ec2'], 'describe_instances', return_value=mock_response):
                instances = monitor.get_gpu_instances()
                assert len(instances) >= 1

    def test_get_gpu_utilization_idle_instance(self, mock_aws_services, sample_gpu_instance):
        """Idle detection is delegated to on-host NVML — aws_monitor returns no metrics."""
        monitor = AWSMonitor()
        instance = {
            'instance_id': 'i-idle',
            'instance_type': 'g4dn.xlarge',
            'state': 'running',
            'tags': {}
        }
        assert monitor.get_gpu_utilization([instance]) == []

    def test_get_gpu_utilization_stopped_instance(self, mock_aws_services):
        """Test that stopped instances are skipped in utilization metrics."""
        monitor = AWSMonitor()

        stopped_instance = {
            'instance_id': 'i-stopped',
            'instance_type': 'g4dn.xlarge',
            'state': 'stopped',
            'tags': {}
        }

        metrics = monitor.get_gpu_utilization([stopped_instance])
        assert len(metrics) == 0

    def test_get_gpu_utilization_error_handling(self, mock_aws_services):
        """No CloudWatch calls — errors from CW cannot occur in get_gpu_utilization."""
        monitor = AWSMonitor()
        instance = {
            'instance_id': 'i-error',
            'instance_type': 'g4dn.xlarge',
            'state': 'running',
            'tags': {}
        }
        assert monitor.get_gpu_utilization([instance]) == []

    def test_get_cost_data_no_ec2_services(self, mock_aws_services):
        """Test cost data filtering excludes non-EC2 services."""
        monitor = AWSMonitor()

        mock_response = {
            'ResultsByTime': [
                {
                    'TimePeriod': {'Start': '2024-01-01', 'End': '2024-01-02'},
                    'Groups': [
                        {
                            'Keys': ['RDS-Service', 'db.t3.medium'],
                            'Metrics': {
                                'BlendedCost': {'Amount': '5.0', 'Unit': 'USD'}
                            }
                        }
                    ]
                }
            ]
        }

        with patch.object(monitor.ce_client, 'get_cost_and_usage', return_value=mock_response):
            cost_data = monitor.get_cost_data()
            assert len(cost_data) == 0  # RDS is not EC2

    def test_get_gpu_type_all_types(self, mock_aws_services):
        """Test GPU type detection for all supported instance types."""
        monitor = AWSMonitor()

        test_cases = [
            ('p2.xlarge', 'K80'),
            ('p3.2xlarge', 'V100'),
            ('p4.24xlarge', 'A100'),
            ('p4de.24xlarge', 'A100-80GB'),
            ('p5.48xlarge', 'H100'),
            ('p5e.48xlarge', 'H200'),
            ('p5en.48xlarge', 'H200'),
            ('p6-b200.48xlarge', 'B200'),
            ('u-p6e-gb200x36', 'GB200'),
            ('g4dn.2xlarge', 'T4'),
            ('g5.8xlarge', 'A10G'),
            ('g6.4xlarge', 'L4'),
            ('g6e.2xlarge', 'L40S'),
            ('trn1.32xlarge', 'Trainium'),
            ('inf2.24xlarge', 'Inferentia2'),
        ]

        for instance_type, expected_gpu in test_cases:
            gpu_type = monitor._get_gpu_type(instance_type)
            assert gpu_type == expected_gpu, f"Failed for {instance_type}"

    def test_get_sp_recommendations_empty(self, mock_aws_services):
        """Test SP recommendations when none available."""
        monitor = AWSMonitor()

        mock_response = {
            'SavingsPlansPurchaseRecommendation': {
                'SavingsPlansPurchaseRecommendationDetails': []
            }
        }

        with patch.object(monitor.ce_client, 'get_savings_plans_purchase_recommendation', return_value=mock_response):
            recs = monitor._get_sp_recommendations()
            assert len(recs) == 0

    def test_get_sp_recommendations_success(self, mock_aws_services):
        """Test Savings Plan recommendations."""
        monitor = AWSMonitor()

        mock_response = {
            'SavingsPlansPurchaseRecommendation': {
                'SavingsPlansPurchaseRecommendationDetails': [
                    {
                        'CurrentAverageHourlyOnDemandSpend': 100,
                        'EstimatedAverageUtilization': 85,
                        'HourlyCommitmentToPurchase': 80,
                        'EstimatedMonthlySavingsAmount': 1500,
                        'EstimatedSavingsPercentage': 15
                    }
                ]
            }
        }

        with patch.object(monitor.ce_client, 'get_savings_plans_purchase_recommendation', return_value=mock_response):
            recs = monitor._get_sp_recommendations()

            assert len(recs) > 0
            assert recs[0]['recommendation_type'] == 'savings_plan'

    def test_get_spot_pricing_with_history(self, mock_aws_services):
        """Test spot pricing with multiple price points."""
        monitor = AWSMonitor()

        now = datetime.now()
        mock_response = {
            'SpotPriceHistory': [
                {
                    'InstanceType': 'g4dn.xlarge',
                    'SpotPrice': '0.3',
                    'AvailabilityZone': 'us-east-1a',
                    'Timestamp': now
                },
                {
                    'InstanceType': 'g4dn.xlarge',
                    'SpotPrice': '0.29',
                    'AvailabilityZone': 'us-east-1a',
                    'Timestamp': now - timedelta(hours=1)
                }
            ]
        }

        with patch.object(monitor.ec2_client, 'describe_spot_price_history', return_value=mock_response):
            spot_data = monitor.get_spot_pricing()
            # Should get latest price only
            assert len(spot_data) >= 1

    def test_get_spot_pricing_error(self, mock_aws_services):
        """Test spot pricing error handling."""
        monitor = AWSMonitor()

        with patch.object(monitor.ec2_client, 'describe_spot_price_history', side_effect=Exception("Spot pricing error")):
            spot_data = monitor.get_spot_pricing()
            assert len(spot_data) == 0

    def test_get_spot_candidates_by_fault_tolerant_tag(self, mock_aws_services):
        """Test spot candidates identified by fault_tolerant tag."""
        monitor = AWSMonitor()

        instances = [
            {
                'instance_id': 'i-ft-true',
                'instance_type': 'p3.2xlarge',
                'state': 'running',
                'tags': {'fault_tolerant': 'true'}
            }
        ]

        spot_pricing = [
            {
                'instance_type': 'p3.2xlarge',
                'spot_price_hourly': 1.5,
                'on_demand_price_hourly': 3.06,
                'availability_zone': 'us-east-1a',
                'spot_monthly': 1095.0,
                'on_demand_monthly': 2234.0,
                'savings_pct': 51.0
            }
        ]

        candidates = monitor.get_spot_candidates(instances, spot_pricing)
        assert len(candidates) == 1
        assert candidates[0]['source_data']['reason'] == 'Tagged as fault-tolerant'

    def test_get_spot_candidates_by_workload_type(self, mock_aws_services):
        """Test spot candidates identified by workload_type tag."""
        monitor = AWSMonitor()

        instances = [
            {
                'instance_id': 'i-batch',
                'instance_type': 'g4dn.xlarge',
                'state': 'running',
                'tags': {'workload_type': 'batch'}
            }
        ]

        spot_pricing = [
            {
                'instance_type': 'g4dn.xlarge',
                'spot_price_hourly': 0.3,
                'on_demand_price_hourly': 0.526,
                'availability_zone': 'us-east-1a',
                'spot_monthly': 219.0,
                'on_demand_monthly': 384.0,
                'savings_pct': 43.0
            }
        ]

        candidates = monitor.get_spot_candidates(instances, spot_pricing)
        assert len(candidates) == 1
        assert 'batch' in candidates[0]['source_data']['reason']

    def test_get_spot_candidates_by_tag_value(self, mock_aws_services):
        """Test spot candidates identified by tag values suggesting non-production."""
        monitor = AWSMonitor()

        instances = [
            {
                'instance_id': 'i-batch-job',
                'instance_type': 'g4dn.xlarge',
                'state': 'running',
                'tags': {'purpose': 'batch', 'team': 'ml'}
            }
        ]

        spot_pricing = [
            {
                'instance_type': 'g4dn.xlarge',
                'spot_price_hourly': 0.3,
                'on_demand_price_hourly': 0.526,
                'availability_zone': 'us-east-1a',
                'spot_monthly': 219.0,
                'on_demand_monthly': 384.0,
                'savings_pct': 43.0
            }
        ]

        candidates = monitor.get_spot_candidates(instances, spot_pricing)
        # Should find a candidate based on tag value 'batch' which is in spot_tags
        assert len(candidates) == 1
        assert candidates[0]['instance_id'] == 'i-batch-job'

    def test_get_spot_candidates_no_pricing_data(self, mock_aws_services):
        """Test spot candidates when no pricing data available."""
        monitor = AWSMonitor()

        instances = [
            {
                'instance_id': 'i-no-price',
                'instance_type': 'unknown.type',
                'state': 'running',
                'tags': {'fault_tolerant': 'true'}
            }
        ]

        spot_pricing = []

        candidates = monitor.get_spot_candidates(instances, spot_pricing)
        assert len(candidates) == 0

    def test_resize_instance(self, mock_aws_services):
        """Test resizing an instance."""
        monitor = AWSMonitor()

        mock_waiter = Mock()

        with patch.object(monitor.ec2_client, 'stop_instances'):
            with patch.object(monitor.ec2_client, 'get_waiter', return_value=mock_waiter):
                with patch.object(monitor.ec2_client, 'modify_instance_attribute'):
                    with patch.object(monitor.ec2_client, 'start_instances'):
                        result = monitor.resize_instance('i-123', 'p3.8xlarge')

                        assert result['instance_id'] == 'i-123'
                        assert result['new_type'] == 'p3.8xlarge'

    def test_restart_instance(self, mock_aws_services):
        """Test restarting an instance."""
        monitor = AWSMonitor()

        with patch.object(monitor.ec2_client, 'reboot_instances') as mock_reboot:
            result = monitor.restart_instance('i-123')

            mock_reboot.assert_called_once()
            assert result['instance_id'] == 'i-123'
            assert result['action'] == 'reboot'

    def test_terminate_instance(self, mock_aws_services):
        """Test terminating an instance."""
        monitor = AWSMonitor()

        with patch.object(monitor.ec2_client, 'terminate_instances') as mock_terminate:
            result = monitor.terminate_instance('i-123')

            mock_terminate.assert_called_once()
            assert result['instance_id'] == 'i-123'
            assert result['action'] == 'terminate'

    def test_stop_instance_with_region(self, mock_aws_services):
        """Test stopping an instance with specific region."""
        monitor = AWSMonitor()

        mock_response = {
            'StoppingInstances': [
                {'CurrentState': {'Name': 'stopping'}}
            ]
        }

        with patch.object(monitor, '_ec2_client_for_region', return_value=mock_aws_services['ec2']):
            with patch.object(mock_aws_services['ec2'], 'stop_instances', return_value=mock_response):
                result = monitor.stop_instance('i-123', region='us-west-2')

                assert result['instance_id'] == 'i-123'
                assert result['state'] == 'stopping'
