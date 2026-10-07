import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Mock GCP SDK imports before importing the monitor
_mock_compute_v1 = MagicMock()
sys.modules.setdefault('google', Mock())
sys.modules.setdefault('google.cloud', Mock())
sys.modules.setdefault('google.cloud.compute_v1', _mock_compute_v1)
sys.modules.setdefault('google.cloud.monitoring_v3', Mock())
sys.modules.setdefault('google.cloud.billing_v1', Mock())
sys.modules.setdefault('google.cloud.resourcemanager_v3', Mock())
sys.modules.setdefault('google.auth', Mock())
sys.modules.setdefault('google.oauth2', Mock())
sys.modules.setdefault('google.oauth2.service_account', Mock())


def _make_monitor(compute_client=None, monitoring_client=None,
                  billing_client=None, resource_client=None,
                  gpu_available=False):
    """Create a GCPMonitor with mocked dependencies — bypasses __init__."""
    from monitors.gcp_monitor import GCPMonitor
    monitor = object.__new__(GCPMonitor)
    monitor.project_id = 'test-project'
    monitor.region = 'us-central1'
    monitor.zone = 'us-central1-a'
    monitor.credentials = Mock()
    monitor.compute_client = compute_client or Mock()
    monitor.monitoring_client = monitoring_client or Mock()
    monitor.billing_client = billing_client or Mock()
    monitor.resource_client = resource_client or Mock()
    monitor.bq_client = None
    monitor.billing_dataset = None
    monitor.gpu_available = gpu_available
    monitor._pynvml = Mock() if gpu_available else None
    monitor.discover_labeled = False
    monitor.discover_label_key = "instance-type"
    monitor.discover_label_value = "gpu-monitoring"
    return monitor


def _make_gce_instance(name='gpu-vm-1', machine_type='zones/us-central1-a/machineTypes/n1-standard-8',
                       status='RUNNING', zone='zones/us-central1-a',
                       gpu_type='nvidia-tesla-v100', gpu_count=1,
                       labels=None, preemptible=False):
    """Create a mock GCE instance."""
    accelerator = Mock()
    accelerator.accelerator_type = gpu_type
    accelerator.accelerator_count = gpu_count

    scheduling = Mock()
    scheduling.preemptible = preemptible

    instance = Mock()
    instance.name = name
    instance.machine_type = machine_type
    instance.status = status
    instance.guest_accelerators = [accelerator]
    instance.labels = labels or {}
    instance.creation_timestamp = '2024-01-01T00:00:00Z'
    instance.scheduling = scheduling
    return instance, zone


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

class TestGCPMonitorInit:
    @patch('monitors.gcp_monitor.resourcemanager_v3.ProjectsClient')
    @patch('monitors.gcp_monitor.billing_v1.CloudBillingClient')
    @patch('monitors.gcp_monitor.monitoring_v3.MetricServiceClient')
    @patch('monitors.gcp_monitor.compute_v1.InstancesClient')
    @patch('monitors.gcp_monitor.service_account.Credentials.from_service_account_file')
    def test_init_with_service_account(self, mock_sa, mock_compute, mock_monitoring,
                                       mock_billing, mock_resource):
        """Should use service account key when path exists."""
        with patch.dict(os.environ, {
            'GCP_PROJECT_ID': 'my-project',
            'GCP_SERVICE_ACCOUNT_KEY_PATH': '/tmp/fake-key.json',
        }), patch('os.path.exists', return_value=True):
            from monitors.gcp_monitor import GCPMonitor
            monitor = GCPMonitor()
            mock_sa.assert_called_once_with('/tmp/fake-key.json')

    @patch('monitors.gcp_monitor.resourcemanager_v3.ProjectsClient')
    @patch('monitors.gcp_monitor.billing_v1.CloudBillingClient')
    @patch('monitors.gcp_monitor.monitoring_v3.MetricServiceClient')
    @patch('monitors.gcp_monitor.compute_v1.InstancesClient')
    @patch('monitors.gcp_monitor.default')
    def test_init_with_adc(self, mock_default, mock_compute, mock_monitoring,
                           mock_billing, mock_resource):
        """Should fall back to ADC when no service account key."""
        mock_default.return_value = (Mock(), 'project-id')
        with patch.dict(os.environ, {'GCP_PROJECT_ID': 'my-project'}, clear=False):
            # Remove service account path
            os.environ.pop('GCP_SERVICE_ACCOUNT_KEY_PATH', None)
            from monitors.gcp_monitor import GCPMonitor
            monitor = GCPMonitor()
            mock_default.assert_called_once()


# ---------------------------------------------------------------------------
# get_gpu_instances
# ---------------------------------------------------------------------------

class TestGetGpuInstances:
    def test_finds_gpu_instances_across_zones(self):
        """Should find GPU instances from multiple zones."""
        instance, zone = _make_gce_instance()
        zone_scope = Mock()
        zone_scope.instances = [instance]

        compute = Mock()
        compute.aggregated_list.return_value = [(zone, zone_scope)]
        monitor = _make_monitor(compute_client=compute)

        with patch.dict(os.environ, {'GCP_GPU_TYPES': 'nvidia-tesla-v100', 'GCP_MACHINE_TYPES': ''}):
            instances = monitor.get_gpu_instances()

        assert len(instances) == 1
        assert instances[0]['instance_id'] == 'us-central1-a/gpu-vm-1'
        assert instances[0]['gpu_count'] == 1
        assert instances[0]['gpu_type'] == 'nvidia-tesla-v100'
        assert instances[0]['cloud_provider'] == 'gcp'

    def test_filters_non_matching_gpu_types(self):
        """Should skip instances whose GPU type doesn't match config."""
        instance, zone = _make_gce_instance(gpu_type='nvidia-tesla-k80')
        zone_scope = Mock()
        zone_scope.instances = [instance]

        compute = Mock()
        compute.aggregated_list.return_value = [(zone, zone_scope)]
        monitor = _make_monitor(compute_client=compute)

        with patch.dict(os.environ, {'GCP_GPU_TYPES': 'nvidia-tesla-v100', 'GCP_MACHINE_TYPES': ''}):
            instances = monitor.get_gpu_instances()

        assert len(instances) == 0

    def test_instances_without_accelerators(self):
        """Should skip instances with no guest_accelerators."""
        instance = Mock()
        instance.name = 'cpu-vm'
        instance.guest_accelerators = []
        instance.labels = {}

        zone_scope = Mock()
        zone_scope.instances = [instance]

        compute = Mock()
        compute.aggregated_list.return_value = [('zones/us-central1-a', zone_scope)]
        monitor = _make_monitor(compute_client=compute)

        with patch.dict(os.environ, {'GCP_GPU_TYPES': 'nvidia-tesla-v100', 'GCP_MACHINE_TYPES': ''}):
            instances = monitor.get_gpu_instances()

        assert instances == []

    def test_empty_gpu_types_accepts_any_accelerator(self):
        """Unset GCP_GPU_TYPES means any guest accelerator qualifies."""
        instance, zone = _make_gce_instance(gpu_type='nvidia-tesla-t4')
        zone_scope = Mock()
        zone_scope.instances = [instance]
        compute = Mock()
        compute.aggregated_list.return_value = [(zone, zone_scope)]
        monitor = _make_monitor(compute_client=compute)

        with patch.dict(os.environ, {'GCP_GPU_TYPES': ''}, clear=False):
            # Clear any inherited filter from the environment.
            os.environ.pop('GCP_GPU_TYPES', None)
            instances = monitor.get_gpu_instances()

        assert len(instances) == 1
        assert instances[0]['gpu_type'] == 'nvidia-tesla-t4'

    def test_discover_labeled_without_accelerator(self):
        """MIG demo VMs labeled instance-type=gpu-monitoring are discovered."""
        instance = Mock()
        instance.name = 'test-gpu-instance-abc'
        instance.machine_type = 'zones/us-central1-a/machineTypes/n1-standard-4'
        instance.status = 'RUNNING'
        instance.guest_accelerators = []
        instance.labels = {'instance-type': 'gpu-monitoring'}
        instance.creation_timestamp = '2024-01-01T00:00:00Z'
        instance.scheduling = Mock(preemptible=True)

        zone = 'zones/us-central1-a'
        zone_scope = Mock()
        zone_scope.instances = [instance]
        compute = Mock()
        compute.aggregated_list.return_value = [(zone, zone_scope)]
        monitor = _make_monitor(compute_client=compute)
        monitor.discover_labeled = True

        with patch.dict(os.environ, {'GCP_GPU_TYPES': 'nvidia-tesla-v100'}):
            instances = monitor.get_gpu_instances()

        assert len(instances) == 1
        assert instances[0]['instance_id'] == 'us-central1-a/test-gpu-instance-abc'
        assert instances[0]['gpu_count'] == 0

    def test_handles_empty_zones(self):
        """Should handle zones with no instances."""
        zone_scope = Mock()
        zone_scope.instances = None

        compute = Mock()
        compute.aggregated_list.return_value = [('zones/us-central1-a', zone_scope)]
        monitor = _make_monitor(compute_client=compute)

        with patch.dict(os.environ, {'GCP_GPU_TYPES': 'nvidia-tesla-v100', 'GCP_MACHINE_TYPES': ''}):
            instances = monitor.get_gpu_instances()

        assert instances == []

    def test_handles_missing_labels(self):
        """Should default to empty dict when labels are None."""
        instance, zone = _make_gce_instance(labels=None)
        instance.labels = None

        zone_scope = Mock()
        zone_scope.instances = [instance]

        compute = Mock()
        compute.aggregated_list.return_value = [(zone, zone_scope)]
        monitor = _make_monitor(compute_client=compute)

        with patch.dict(os.environ, {'GCP_GPU_TYPES': 'nvidia-tesla-v100', 'GCP_MACHINE_TYPES': ''}):
            instances = monitor.get_gpu_instances()

        assert instances[0]['tags'] == {}

    def test_preemptible_detection(self):
        """Should detect preemptible instances."""
        instance, zone = _make_gce_instance(preemptible=True)
        zone_scope = Mock()
        zone_scope.instances = [instance]

        compute = Mock()
        compute.aggregated_list.return_value = [(zone, zone_scope)]
        monitor = _make_monitor(compute_client=compute)

        with patch.dict(os.environ, {'GCP_GPU_TYPES': 'nvidia-tesla-v100', 'GCP_MACHINE_TYPES': ''}):
            instances = monitor.get_gpu_instances()

        assert instances[0]['preemptible'] is True

    def test_api_error_returns_empty(self):
        """Should return empty list on API error."""
        compute = Mock()
        compute.aggregated_list.side_effect = Exception('Permission denied')
        monitor = _make_monitor(compute_client=compute)

        with patch.dict(os.environ, {'GCP_GPU_TYPES': 'nvidia-tesla-v100', 'GCP_MACHINE_TYPES': ''}):
            assert monitor.get_gpu_instances() == []

    def test_zone_parsing(self):
        """Should correctly extract region from zone string."""
        instance, _ = _make_gce_instance()
        zone_scope = Mock()
        zone_scope.instances = [instance]

        compute = Mock()
        compute.aggregated_list.return_value = [('zones/europe-west1-b', zone_scope)]
        monitor = _make_monitor(compute_client=compute)

        with patch.dict(os.environ, {'GCP_GPU_TYPES': 'nvidia-tesla-v100', 'GCP_MACHINE_TYPES': ''}):
            instances = monitor.get_gpu_instances()

        assert instances[0]['region'] == 'europe-west1'
        assert instances[0]['availability_zone'] == 'europe-west1-b'


# ---------------------------------------------------------------------------
# get_gpu_utilization
# ---------------------------------------------------------------------------

class TestGetGpuUtilization:
    def test_returns_empty_when_nvml_unavailable(self):
        monitor = _make_monitor(gpu_available=False)
        assert monitor.get_gpu_utilization([]) == []

    def test_collects_metrics(self):
        """Should read GPU metrics via NVML and match to instance."""
        monitor = _make_monitor(gpu_available=True)
        pynvml = monitor._pynvml
        pynvml.nvmlDeviceGetCount.return_value = 1
        pynvml.nvmlDeviceGetHandleByIndex.return_value = Mock()
        pynvml.nvmlDeviceGetUtilizationRates.return_value = Mock(gpu=80.0, memory=55.0)
        pynvml.nvmlDeviceGetMemoryInfo.return_value = Mock(
            used=8 * 1024 * 1024 * 1024,
            total=16 * 1024 * 1024 * 1024
        )
        pynvml.nvmlDeviceGetTemperature.return_value = 72
        pynvml.nvmlDeviceGetPowerUsage.return_value = 250000
        pynvml.NVML_TEMPERATURE_GPU = 0

        hostname = os.uname().nodename
        instances = [{'instance_id': hostname}]

        metrics = monitor.get_gpu_utilization(instances)

        assert len(metrics) == 1
        assert metrics[0]['gpu_utilization'] == 80.0
        assert metrics[0]['is_idle'] is False
        assert metrics[0]['cloud_provider'] == 'gcp'

    def test_idle_detection(self):
        """Should flag GPU as idle when both gpu and memory util are low."""
        monitor = _make_monitor(gpu_available=True)
        pynvml = monitor._pynvml
        pynvml.nvmlDeviceGetCount.return_value = 1
        pynvml.nvmlDeviceGetHandleByIndex.return_value = Mock()
        pynvml.nvmlDeviceGetUtilizationRates.return_value = Mock(gpu=5.0, memory=3.0)
        pynvml.nvmlDeviceGetMemoryInfo.return_value = Mock(used=100, total=1000)
        pynvml.nvmlDeviceGetTemperature.side_effect = Exception('N/A')
        pynvml.nvmlDeviceGetPowerUsage.side_effect = Exception('N/A')
        pynvml.NVML_TEMPERATURE_GPU = 0

        hostname = os.uname().nodename
        instances = [{'instance_id': hostname}]
        metrics = monitor.get_gpu_utilization(instances)

        assert metrics[0]['is_idle'] is True
        assert metrics[0]['temperature_c'] is None
        assert metrics[0]['power_usage_w'] is None

    def test_per_device_error_isolation(self):
        """Should skip failed GPUs and continue with others."""
        monitor = _make_monitor(gpu_available=True)
        pynvml = monitor._pynvml
        pynvml.nvmlDeviceGetCount.return_value = 2
        good_handle = Mock()
        pynvml.nvmlDeviceGetHandleByIndex.side_effect = [
            Exception('GPU 0 error'),
            good_handle
        ]
        pynvml.nvmlDeviceGetUtilizationRates.return_value = Mock(gpu=50, memory=30)
        pynvml.nvmlDeviceGetMemoryInfo.return_value = Mock(used=100, total=1000)
        pynvml.nvmlDeviceGetTemperature.return_value = 60
        pynvml.nvmlDeviceGetPowerUsage.return_value = 100000
        pynvml.NVML_TEMPERATURE_GPU = 0

        hostname = os.uname().nodename
        instances = [{'instance_id': hostname}]
        metrics = monitor.get_gpu_utilization(instances)
        assert len(metrics) == 1


# ---------------------------------------------------------------------------
# get_cost_data
# ---------------------------------------------------------------------------

class TestGetCostData:
    def test_missing_billing_account(self):
        """Should return empty list when billing account is not configured."""
        monitor = _make_monitor()
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop('GCP_BILLING_ACCOUNT_ID', None)
            assert monitor.get_cost_data() == []

    def test_estimates_cost_for_gpu_instances(self):
        """Should estimate costs for each GPU instance."""
        instance, zone = _make_gce_instance()
        zone_scope = Mock()
        zone_scope.instances = [instance]

        compute = Mock()
        compute.aggregated_list.return_value = [(zone, zone_scope)]
        monitor = _make_monitor(compute_client=compute)

        with patch.dict(os.environ, {
            'GCP_BILLING_ACCOUNT_ID': 'billing-123',
            'GCP_GPU_TYPES': 'nvidia-tesla-v100',
            'GCP_MACHINE_TYPES': '',
        }):
            costs = monitor.get_cost_data()

        assert len(costs) == 1
        assert costs[0]['service'] == 'Compute Engine'
        assert costs[0]['cloud_provider'] == 'gcp'
        assert costs[0]['cost'] > 0

    def test_api_error_returns_empty(self):
        """Should return empty list when get_gpu_instances fails."""
        compute = Mock()
        compute.aggregated_list.side_effect = Exception('API error')
        monitor = _make_monitor(compute_client=compute)

        with patch.dict(os.environ, {
            'GCP_BILLING_ACCOUNT_ID': 'billing-123',
            'GCP_GPU_TYPES': 'nvidia-tesla-v100',
            'GCP_MACHINE_TYPES': '',
        }):
            assert monitor.get_cost_data() == []


# ---------------------------------------------------------------------------
# _estimate_instance_cost
# ---------------------------------------------------------------------------

class TestEstimateInstanceCost:
    def test_known_instance_and_gpu(self):
        monitor = _make_monitor()
        instance = {
            'instance_type': 'n1-standard-8',
            'gpu_type': 'nvidia-tesla-v100',
            'gpu_count': 2,
            'preemptible': False,
        }
        cost = monitor._estimate_instance_cost(instance)
        expected = 0.38 + 2.48 * 2  # base + gpu*count
        assert cost == round(expected, 2)

    def test_unknown_instance_type_uses_default(self):
        monitor = _make_monitor()
        instance = {
            'instance_type': 'a2-highgpu-1g',
            'gpu_type': 'nvidia-tesla-a100',
            'gpu_count': 1,
            'preemptible': False,
        }
        cost = monitor._estimate_instance_cost(instance)
        expected = 0.19 + 2.93  # default base + known gpu
        assert cost == round(expected, 2)

    def test_preemptible_discount(self):
        """Should apply 80% discount for preemptible instances."""
        monitor = _make_monitor()
        instance = {
            'instance_type': 'n1-standard-8',
            'gpu_type': 'nvidia-tesla-v100',
            'gpu_count': 1,
            'preemptible': True,
        }
        cost = monitor._estimate_instance_cost(instance)
        expected = 0.38 * 0.2 + 2.48 * 0.2
        assert cost == round(expected, 2)

    def test_zero_gpu_count(self):
        monitor = _make_monitor()
        instance = {
            'instance_type': 'n1-standard-8',
            'gpu_type': '',
            'gpu_count': 0,
            'preemptible': False,
        }
        cost = monitor._estimate_instance_cost(instance)
        assert cost == 0.38  # base only


# ---------------------------------------------------------------------------
# Action methods
# ---------------------------------------------------------------------------

class TestActionMethods:
    def test_stop_instance(self):
        """Should call compute_v1 stop with correct request."""
        operation = Mock()
        compute = Mock()
        compute.stop.return_value = operation
        monitor = _make_monitor(compute_client=compute)

        result = monitor.stop_instance('us-central1-a/my-vm')

        compute.stop.assert_called_once()
        operation.result.assert_called_once()
        assert result == {'instance_id': 'us-central1-a/my-vm', 'action': 'stop'}

    def test_start_instance(self):
        operation = Mock()
        compute = Mock()
        compute.start.return_value = operation
        monitor = _make_monitor(compute_client=compute)

        result = monitor.start_instance('us-central1-a/my-vm')
        compute.start.assert_called_once()
        assert result['action'] == 'start'

    def test_resize_instance(self):
        """Should stop -> set_machine_type -> start."""
        compute = Mock()
        for method in ['stop', 'set_machine_type', 'start']:
            getattr(compute, method).return_value = Mock()

        monitor = _make_monitor(compute_client=compute)
        result = monitor.resize_instance('us-central1-a/my-vm', 'n1-standard-16')

        compute.stop.assert_called_once()
        compute.set_machine_type.assert_called_once()
        compute.start.assert_called_once()
        assert result == {'instance_id': 'us-central1-a/my-vm', 'new_type': 'n1-standard-16'}

    def test_restart_instance(self):
        operation = Mock()
        compute = Mock()
        compute.reset.return_value = operation
        monitor = _make_monitor(compute_client=compute)

        result = monitor.restart_instance('us-central1-a/my-vm')
        compute.reset.assert_called_once()
        assert result['action'] == 'reset'

    def test_terminate_instance(self):
        operation = Mock()
        compute = Mock()
        compute.delete.return_value = operation
        monitor = _make_monitor(compute_client=compute)

        result = monitor.terminate_instance('us-central1-a/my-vm')
        compute.delete.assert_called_once()
        assert result['action'] == 'delete'

    def test_invalid_instance_id_format(self):
        """Should raise ValueError for malformed instance IDs."""
        monitor = _make_monitor()

        with pytest.raises(ValueError, match="Invalid GCP instance ID"):
            monitor.stop_instance('just-a-name')

        with pytest.raises(ValueError, match="Invalid GCP instance ID"):
            monitor.start_instance('too/many/parts/here')

        with pytest.raises(ValueError, match="Invalid GCP instance ID"):
            monitor.terminate_instance('/missing-zone')


# ---------------------------------------------------------------------------
# get_instance_metrics
# ---------------------------------------------------------------------------

class TestGetInstanceMetrics:
    def test_returns_cpu_and_memory(self):
        """Should parse Cloud Monitoring time series response."""
        point = Mock()
        point.value.double_value = 0.75
        point.interval.end_time.isoformat.return_value = '2024-01-01T12:00:00Z'

        series = Mock()
        series.points = [point]

        monitoring = Mock()
        monitoring.list_time_series.side_effect = [[series], []]
        monitor = _make_monitor(monitoring_client=monitoring)

        result = monitor.get_instance_metrics('vm-1', 'us-central1-a')

        assert result['instance_id'] == 'vm-1'
        assert len(result['cpu_metrics']) == 1
        assert result['cpu_metrics'][0]['value'] == 75.0  # converted to percentage

    def test_returns_empty_on_error(self):
        monitoring = Mock()
        monitoring.list_time_series.side_effect = Exception('Permission denied')
        monitor = _make_monitor(monitoring_client=monitoring)
        assert monitor.get_instance_metrics('vm-1', 'zone-a') == {}

    def test_skips_null_datapoints(self):
        """Should skip points where double_value is None."""
        null_point = Mock()
        null_point.value.double_value = None

        good_point = Mock()
        good_point.value.double_value = 0.5
        good_point.interval.end_time.isoformat.return_value = '2024-01-01T12:05:00Z'

        series = Mock()
        series.points = [null_point, good_point]

        monitoring = Mock()
        monitoring.list_time_series.side_effect = [[series], []]
        monitor = _make_monitor(monitoring_client=monitoring)

        result = monitor.get_instance_metrics('vm-1', 'zone-a')
        assert len(result['cpu_metrics']) == 1


# ---------------------------------------------------------------------------
# get_resource_labels
# ---------------------------------------------------------------------------

class TestGetResourceLabels:
    def test_returns_labels(self):
        instance = Mock()
        instance.labels = {'env': 'prod', 'team': 'ml'}
        compute = Mock()
        compute.get.return_value = instance
        monitor = _make_monitor(compute_client=compute)

        labels = monitor.get_resource_labels('vm-1', 'us-central1-a')
        assert labels == {'env': 'prod', 'team': 'ml'}

    def test_returns_empty_when_no_labels(self):
        instance = Mock()
        instance.labels = None
        compute = Mock()
        compute.get.return_value = instance
        monitor = _make_monitor(compute_client=compute)
        assert monitor.get_resource_labels('vm-1', 'zone-a') == {}

    def test_returns_empty_on_error(self):
        compute = Mock()
        compute.get.side_effect = Exception('Not found')
        monitor = _make_monitor(compute_client=compute)
        assert monitor.get_resource_labels('vm-1', 'zone-a') == {}
