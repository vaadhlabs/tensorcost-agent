import pytest
from unittest.mock import Mock, patch, MagicMock, PropertyMock
from datetime import datetime, timedelta

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Mock Azure SDK imports before importing the monitor
sys.modules.setdefault('azure', Mock())
sys.modules.setdefault('azure.identity', Mock())
sys.modules.setdefault('azure.mgmt', Mock())
sys.modules.setdefault('azure.mgmt.compute', Mock())
sys.modules.setdefault('azure.mgmt.monitor', Mock())
sys.modules.setdefault('azure.mgmt.consumption', Mock())
sys.modules.setdefault('azure.mgmt.resource', Mock())
sys.modules.setdefault('azure.mgmt.costmanagement', Mock())


def _make_monitor(compute_client=None, monitor_client=None,
                  consumption_client=None, resource_client=None,
                  cost_client=None, gpu_available=False):
    """Create an AzureMonitor with mocked dependencies — bypasses __init__."""
    from monitors.azure_monitor import AzureMonitor
    monitor = object.__new__(AzureMonitor)
    monitor.subscription_id = 'test-sub-123'
    monitor.resource_group = 'test-rg'
    monitor.credential = Mock()
    monitor.compute_client = compute_client or Mock()
    monitor.monitor_client = monitor_client or Mock()
    monitor.consumption_client = consumption_client or Mock()
    monitor.resource_client = resource_client or Mock()
    monitor.cost_client = cost_client or Mock()
    monitor.gpu_available = gpu_available
    monitor._pynvml = Mock() if gpu_available else None
    return monitor


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

class TestAzureMonitorInit:
    @patch('monitors.azure_monitor.CostManagementClient')
    @patch('monitors.azure_monitor.ResourceManagementClient')
    @patch('monitors.azure_monitor.ConsumptionManagementClient')
    @patch('monitors.azure_monitor.MonitorManagementClient')
    @patch('monitors.azure_monitor.ComputeManagementClient')
    @patch('monitors.azure_monitor.DefaultAzureCredential')
    def test_init_uses_default_azure_credential(
        self, mock_dac, mock_compute, mock_monitor, mock_consumption, mock_resource, mock_cost
    ):
        """Should use DefaultAzureCredential (includes managed identity on Azure)."""
        os.environ['AZURE_SUBSCRIPTION_ID'] = 'sub-123'
        os.environ['AZURE_RESOURCE_GROUP'] = 'rg-test'
        from monitors.azure_monitor import AzureMonitor
        monitor = AzureMonitor()
        mock_dac.assert_called_once()
        assert monitor.subscription_id == 'sub-123'
        assert monitor.resource_group == 'rg-test'

    @patch('monitors.azure_monitor.CostManagementClient')
    @patch('monitors.azure_monitor.ResourceManagementClient')
    @patch('monitors.azure_monitor.ConsumptionManagementClient')
    @patch('monitors.azure_monitor.MonitorManagementClient')
    @patch('monitors.azure_monitor.ComputeManagementClient')
    @patch('monitors.azure_monitor.DefaultAzureCredential')
    def test_init_accepts_config_dict(
        self, mock_dac, mock_compute, mock_monitor, mock_consumption, mock_resource, mock_cost
    ):
        """Should read subscription_id and resource_group from config dict."""
        from monitors.azure_monitor import AzureMonitor
        monitor = AzureMonitor(
            {
                'subscription_id': 'cfg-sub',
                'resource_group': 'cfg-rg',
            }
        )
        assert monitor.subscription_id == 'cfg-sub'
        assert monitor.resource_group == 'cfg-rg'


# ---------------------------------------------------------------------------
# get_gpu_instances
# ---------------------------------------------------------------------------

class TestGetGpuInstances:
    def test_instances_with_resource_group(self):
        """Should list VMs in the specified resource group."""
        vm = Mock()
        vm.name = 'gpu-vm-1'
        vm.hardware_profile.vm_size = 'Standard_NC6s_v3'
        vm.provisioning_state = 'Succeeded'
        vm.location = 'eastus'
        vm.id = '/subscriptions/sub/resourceGroups/test-rg/providers/...'
        vm.tags = {'team': 'ml'}
        vm.time_created = datetime(2024, 1, 1)
        vm.zones = None

        compute = Mock()
        compute.virtual_machines.list.return_value = [vm]

        monitor = _make_monitor(compute_client=compute)

        with patch.dict(os.environ, {'AZURE_VM_SIZES': 'Standard_NC'}):
            instances = monitor.get_gpu_instances()

        assert len(instances) == 1
        assert instances[0]['instance_id'] == 'test-rg/gpu-vm-1'
        assert instances[0]['instance_type'] == 'Standard_NC6s_v3'
        assert instances[0]['cloud_provider'] == 'azure'
        assert instances[0]['tags'] == {'team': 'ml'}

    def test_instances_list_all_when_no_rg(self):
        """Should call list_all() when resource_group is not set."""
        compute = Mock()
        compute.virtual_machines.list_all.return_value = []

        monitor = _make_monitor(compute_client=compute)
        monitor.resource_group = None

        with patch.dict(os.environ, {'AZURE_VM_SIZES': 'Standard_NC'}):
            instances = monitor.get_gpu_instances()

        compute.virtual_machines.list_all.assert_called_once()
        assert instances == []

    def test_filters_non_gpu_vms(self):
        """Should exclude VMs whose size doesn't match GPU patterns."""
        gpu_vm = Mock()
        gpu_vm.name = 'gpu-vm'
        gpu_vm.hardware_profile.vm_size = 'Standard_NC6s_v3'
        gpu_vm.provisioning_state = 'Succeeded'
        gpu_vm.location = 'eastus'
        gpu_vm.id = '/subscriptions/s/resourceGroups/rg/providers/x'
        gpu_vm.tags = {}
        gpu_vm.time_created = None
        gpu_vm.zones = None

        cpu_vm = Mock()
        cpu_vm.name = 'cpu-vm'
        cpu_vm.hardware_profile.vm_size = 'Standard_D4s_v3'

        compute = Mock()
        compute.virtual_machines.list.return_value = [gpu_vm, cpu_vm]
        monitor = _make_monitor(compute_client=compute)

        with patch.dict(os.environ, {'AZURE_VM_SIZES': 'Standard_NC,Standard_ND'}):
            instances = monitor.get_gpu_instances()

        assert len(instances) == 1
        assert instances[0]['instance_id'] == 'rg/gpu-vm'

    def test_handles_missing_time_created(self):
        """Should handle VMs with no time_created."""
        vm = Mock()
        vm.name = 'vm-1'
        vm.hardware_profile.vm_size = 'Standard_NC6'
        vm.provisioning_state = 'Succeeded'
        vm.location = 'westus'
        vm.id = '/subscriptions/s/resourceGroups/rg/providers/x'
        vm.tags = None
        vm.time_created = None
        vm.zones = None

        compute = Mock()
        compute.virtual_machines.list.return_value = [vm]
        monitor = _make_monitor(compute_client=compute)

        with patch.dict(os.environ, {'AZURE_VM_SIZES': 'Standard_NC'}):
            instances = monitor.get_gpu_instances()

        assert instances[0]['created_at'] is None
        assert instances[0]['tags'] == {}

    def test_api_error_returns_empty(self):
        """Should return empty list on Azure API errors."""
        compute = Mock()
        compute.virtual_machines.list.side_effect = Exception('Azure error')
        monitor = _make_monitor(compute_client=compute)

        with patch.dict(os.environ, {'AZURE_VM_SIZES': 'Standard_NC'}):
            instances = monitor.get_gpu_instances()

        assert instances == []


# ---------------------------------------------------------------------------
# get_gpu_utilization
# ---------------------------------------------------------------------------

class TestGetGpuUtilization:
    def test_returns_empty_when_nvml_unavailable(self):
        """Should return empty metrics when NVML is not initialized."""
        monitor = _make_monitor(gpu_available=False)
        assert monitor.get_gpu_utilization([]) == []

    def test_collects_metrics_for_matching_instance(self):
        """Should collect GPU metrics and match to instance by hostname."""
        monitor = _make_monitor(gpu_available=True)
        pynvml = monitor._pynvml
        pynvml.nvmlDeviceGetCount.return_value = 1
        pynvml.nvmlDeviceGetHandleByIndex.return_value = Mock()
        pynvml.nvmlDeviceGetUtilizationRates.return_value = Mock(gpu=75.5, memory=40.2)
        pynvml.nvmlDeviceGetMemoryInfo.return_value = Mock(
            used=4 * 1024 * 1024 * 1024,
            total=16 * 1024 * 1024 * 1024
        )
        pynvml.nvmlDeviceGetTemperature.return_value = 65
        pynvml.nvmlDeviceGetPowerUsage.return_value = 150000  # milliwatts
        pynvml.NVML_TEMPERATURE_GPU = 0

        hostname = os.uname().nodename
        instances = [{'instance_id': hostname, 'cloud_provider': 'azure'}]

        metrics = monitor.get_gpu_utilization(instances)

        assert len(metrics) == 1
        assert metrics[0]['gpu_utilization'] == 75.5
        assert metrics[0]['memory_utilization'] == 40.2
        assert metrics[0]['temperature_c'] == 65
        assert metrics[0]['power_usage_w'] == 150.0
        assert metrics[0]['is_idle'] is False

    def test_idle_detection(self):
        """Should flag GPU as idle when util < 10% and memory < 10%."""
        monitor = _make_monitor(gpu_available=True)
        pynvml = monitor._pynvml
        pynvml.nvmlDeviceGetCount.return_value = 1
        pynvml.nvmlDeviceGetHandleByIndex.return_value = Mock()
        pynvml.nvmlDeviceGetUtilizationRates.return_value = Mock(gpu=3.0, memory=5.0)
        pynvml.nvmlDeviceGetMemoryInfo.return_value = Mock(used=100, total=1000)
        pynvml.nvmlDeviceGetTemperature.side_effect = Exception('no temp')
        pynvml.nvmlDeviceGetPowerUsage.side_effect = Exception('no power')
        pynvml.NVML_TEMPERATURE_GPU = 0

        hostname = os.uname().nodename
        instances = [{'instance_id': hostname}]

        metrics = monitor.get_gpu_utilization(instances)

        assert len(metrics) == 1
        assert metrics[0]['is_idle'] is True
        assert metrics[0]['temperature_c'] is None
        assert metrics[0]['power_usage_w'] is None

    def test_no_matching_instance_skips(self):
        """Should not emit metrics when hostname doesn't match and multiple instances exist."""
        monitor = _make_monitor(gpu_available=True)
        pynvml = monitor._pynvml
        pynvml.nvmlDeviceGetCount.return_value = 1
        pynvml.nvmlDeviceGetHandleByIndex.return_value = Mock()
        pynvml.nvmlDeviceGetUtilizationRates.return_value = Mock(gpu=50, memory=30)
        pynvml.nvmlDeviceGetMemoryInfo.return_value = Mock(used=100, total=1000)
        pynvml.NVML_TEMPERATURE_GPU = 0

        # More than one row disables single-VM fallback (see azure_monitor.get_gpu_utilization).
        instances = [
            {'instance_id': 'some-other-host'},
            {'instance_id': 'another-vm'},
        ]
        metrics = monitor.get_gpu_utilization(instances)
        assert metrics == []

    def test_single_instance_fallback_when_hostname_unknown(self):
        """With exactly one Azure GPU row, attribute NVML to it even if hostname differs."""
        monitor = _make_monitor(gpu_available=True)
        pynvml = monitor._pynvml
        pynvml.nvmlDeviceGetCount.return_value = 1
        pynvml.nvmlDeviceGetHandleByIndex.return_value = Mock()
        pynvml.nvmlDeviceGetUtilizationRates.return_value = Mock(gpu=50, memory=30)
        pynvml.nvmlDeviceGetMemoryInfo.return_value = Mock(used=100, total=1000)
        pynvml.NVML_TEMPERATURE_GPU = 0

        metrics = monitor.get_gpu_utilization([{'instance_id': 'some-other-host'}])
        assert len(metrics) == 1
        assert metrics[0]['instance_id'] == 'some-other-host'
        assert metrics[0]['source'] == 'nvml'

    def test_per_device_error_isolation(self):
        """Should continue sampling other GPUs if one device fails."""
        monitor = _make_monitor(gpu_available=True)
        pynvml = monitor._pynvml
        pynvml.nvmlDeviceGetCount.return_value = 2
        handle_ok = Mock()
        pynvml.nvmlDeviceGetHandleByIndex.side_effect = [
            Exception('GPU 0 fell off bus'),
            handle_ok
        ]
        pynvml.nvmlDeviceGetUtilizationRates.return_value = Mock(gpu=50, memory=30)
        pynvml.nvmlDeviceGetMemoryInfo.return_value = Mock(used=100, total=1000)
        pynvml.nvmlDeviceGetTemperature.return_value = 60
        pynvml.nvmlDeviceGetPowerUsage.return_value = 100000
        pynvml.NVML_TEMPERATURE_GPU = 0

        hostname = os.uname().nodename
        instances = [{'instance_id': hostname}]
        metrics = monitor.get_gpu_utilization(instances)
        # GPU 0 failed, GPU 1 succeeded
        assert len(metrics) == 1


# ---------------------------------------------------------------------------
# get_cost_data
# ---------------------------------------------------------------------------

class TestGetCostData:
    def test_cost_data_filters_gpu_vms(self):
        """Should filter usage records for Standard_N GPU VMs."""
        usage1 = Mock()
        usage1.instance_name = 'Standard_NC6_vm1'
        usage1.product = 'Virtual Machines NC Series'
        usage1.cost = 25.50
        usage1.billing_currency = 'USD'
        usage1.date = datetime(2024, 1, 15)
        usage1.resource_group = 'ml-rg'
        usage1.location = 'eastus'

        usage2 = Mock()
        usage2.instance_name = 'Standard_D4_vm2'  # Not a GPU VM

        consumption = Mock()
        consumption.usage_details.list.return_value = [usage1, usage2]
        monitor = _make_monitor(consumption_client=consumption)

        costs = monitor.get_cost_data()

        assert len(costs) == 1
        assert costs[0]['cost'] == 25.50
        assert costs[0]['cloud_provider'] == 'azure'

    def test_cost_data_empty(self):
        """Should return empty list when no usage records match."""
        consumption = Mock()
        consumption.usage_details.list.return_value = []
        monitor = _make_monitor(consumption_client=consumption)
        assert monitor.get_cost_data() == []

    def test_cost_data_api_error(self):
        """Should return empty list on API error."""
        consumption = Mock()
        consumption.usage_details.list.side_effect = Exception('Auth expired')
        monitor = _make_monitor(consumption_client=consumption)
        assert monitor.get_cost_data() == []


# ---------------------------------------------------------------------------
# get_instance_metrics
# ---------------------------------------------------------------------------

class TestGetInstanceMetrics:
    def test_returns_cpu_and_memory_metrics(self):
        """Should parse Azure Monitor response into structured metrics."""
        cpu_data = Mock()
        cpu_data.average = 45.5
        cpu_data.time_stamp = datetime(2024, 1, 1, 12, 0)

        mem_data = Mock()
        mem_data.average = 2 * 1024 * 1024 * 1024  # 2GB in bytes
        mem_data.time_stamp = datetime(2024, 1, 1, 12, 0)

        cpu_ts = Mock()
        cpu_ts.data = [cpu_data]
        cpu_metric = Mock()
        cpu_metric.timeseries = [cpu_ts]

        mem_ts = Mock()
        mem_ts.data = [mem_data]
        mem_metric = Mock()
        mem_metric.timeseries = [mem_ts]

        cpu_response = Mock()
        cpu_response.value = [cpu_metric]
        mem_response = Mock()
        mem_response.value = [mem_metric]

        monitor_client = Mock()
        monitor_client.metrics.list.side_effect = [cpu_response, mem_response]
        monitor = _make_monitor(monitor_client=monitor_client)

        result = monitor.get_instance_metrics('vm-1', 'rg-1')

        assert result['instance_id'] == 'vm-1'
        assert len(result['cpu_metrics']) == 1
        assert result['cpu_metrics'][0]['value'] == 45.5
        assert len(result['memory_metrics']) == 1

    def test_returns_empty_on_error(self):
        """Should return empty dict on API error."""
        monitor_client = Mock()
        monitor_client.metrics.list.side_effect = Exception('Not found')
        monitor = _make_monitor(monitor_client=monitor_client)
        assert monitor.get_instance_metrics('vm-1', 'rg-1') == {}

    def test_skips_null_datapoints(self):
        """Should skip datapoints where average is None."""
        data_null = Mock()
        data_null.average = None

        data_ok = Mock()
        data_ok.average = 30.0
        data_ok.time_stamp = datetime(2024, 1, 1, 12, 5)

        ts = Mock()
        ts.data = [data_null, data_ok]
        metric = Mock()
        metric.timeseries = [ts]
        response = Mock()
        response.value = [metric]

        empty_response = Mock()
        empty_response.value = []

        monitor_client = Mock()
        monitor_client.metrics.list.side_effect = [response, empty_response]
        monitor = _make_monitor(monitor_client=monitor_client)

        result = monitor.get_instance_metrics('vm-1', 'rg-1')
        assert len(result['cpu_metrics']) == 1
        assert result['cpu_metrics'][0]['value'] == 30.0


# ---------------------------------------------------------------------------
# Action methods
# ---------------------------------------------------------------------------

class TestActionMethods:
    def test_stop_instance(self):
        """Should deallocate VM with timeout."""
        poller = Mock()
        compute = Mock()
        compute.virtual_machines.begin_deallocate.return_value = poller
        monitor = _make_monitor(compute_client=compute)

        result = monitor.stop_instance('my-rg/my-vm')

        compute.virtual_machines.begin_deallocate.assert_called_once_with('my-rg', 'my-vm')
        poller.result.assert_called_once_with(timeout=600)
        assert result == {'instance_id': 'my-rg/my-vm', 'action': 'deallocate'}

    def test_start_instance(self):
        poller = Mock()
        compute = Mock()
        compute.virtual_machines.begin_start.return_value = poller
        monitor = _make_monitor(compute_client=compute)

        result = monitor.start_instance('rg/vm')
        poller.result.assert_called_once_with(timeout=600)
        assert result['action'] == 'start'

    def test_resize_instance(self):
        """Should deallocate -> update -> start with correct target type."""
        compute = Mock()
        for method_name in ['begin_deallocate', 'begin_update', 'begin_start']:
            getattr(compute.virtual_machines, method_name).return_value = Mock()

        monitor = _make_monitor(compute_client=compute)
        result = monitor.resize_instance('rg/vm', 'Standard_NC12')

        compute.virtual_machines.begin_deallocate.assert_called_once_with('rg', 'vm')
        compute.virtual_machines.begin_update.assert_called_once_with(
            'rg', 'vm', {'hardware_profile': {'vm_size': 'Standard_NC12'}}
        )
        compute.virtual_machines.begin_start.assert_called_once_with('rg', 'vm')
        assert result == {'instance_id': 'rg/vm', 'new_type': 'Standard_NC12'}

    def test_restart_instance(self):
        poller = Mock()
        compute = Mock()
        compute.virtual_machines.begin_restart.return_value = poller
        monitor = _make_monitor(compute_client=compute)

        result = monitor.restart_instance('rg/vm')
        poller.result.assert_called_once_with(timeout=600)
        assert result['action'] == 'restart'

    def test_terminate_instance(self):
        poller = Mock()
        compute = Mock()
        compute.virtual_machines.begin_delete.return_value = poller
        monitor = _make_monitor(compute_client=compute)

        result = monitor.terminate_instance('rg/vm')
        poller.result.assert_called_once_with(timeout=600)
        assert result['action'] == 'delete'

    def test_invalid_instance_id_format(self):
        """Should raise ValueError for malformed instance IDs."""
        monitor = _make_monitor()

        with pytest.raises(ValueError, match="Invalid Azure instance ID"):
            monitor.stop_instance('just-a-name')

        with pytest.raises(ValueError, match="Invalid Azure instance ID"):
            monitor.start_instance('too/many/parts')

        with pytest.raises(ValueError, match="Invalid Azure instance ID"):
            monitor.restart_instance('/missing-rg')


# ---------------------------------------------------------------------------
# get_resource_tags
# ---------------------------------------------------------------------------

class TestGetResourceTags:
    def test_returns_tags(self):
        resource = Mock()
        resource.tags = {'env': 'prod', 'team': 'ml'}
        resource_client = Mock()
        resource_client.resources.get_by_id.return_value = resource
        monitor = _make_monitor(resource_client=resource_client)

        tags = monitor.get_resource_tags('/subscriptions/sub/resource/id')
        assert tags == {'env': 'prod', 'team': 'ml'}

    def test_returns_empty_on_error(self):
        resource_client = Mock()
        resource_client.resources.get_by_id.side_effect = Exception('Not found')
        monitor = _make_monitor(resource_client=resource_client)
        assert monitor.get_resource_tags('bad-id') == {}

    def test_returns_empty_when_no_tags(self):
        resource = Mock()
        resource.tags = None
        resource_client = Mock()
        resource_client.resources.get_by_id.return_value = resource
        monitor = _make_monitor(resource_client=resource_client)
        assert monitor.get_resource_tags('some-id') == {}
