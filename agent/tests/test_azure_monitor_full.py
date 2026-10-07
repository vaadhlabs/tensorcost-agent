"""
Comprehensive tests for Azure Monitor — covers cost-by-tags, spot pricing,
the Azure Retail Prices API integration, and price caching.
"""

import pytest
import os
import sys
import time
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

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
    """Create an AzureMonitor with mocked dependencies -- bypasses __init__."""
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
    # Reset class-level price cache for test isolation
    AzureMonitor._azure_price_cache = {}
    AzureMonitor._azure_price_cache_ts = 0
    return monitor


def _stub_vm_power_state(compute_client, power_state='running'):
    """Mock virtual_machines.get(..., expand=instanceView) for Azure power state."""

    def _get(rg, name, expand=None):
        st = Mock()
        st.code = f'PowerState/{power_state}'
        detail = Mock()
        detail.instance_view = Mock()
        detail.instance_view.statuses = [st]
        return detail

    compute_client.virtual_machines.get.side_effect = _get


# ---------------------------------------------------------------------------
# get_gpu_instances
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestGetGpuInstancesFull:
    def test_returns_instances_with_all_fields(self):
        """Should populate all expected fields for a GPU VM."""
        vm = Mock()
        vm.name = 'gpu-vm-1'
        vm.hardware_profile.vm_size = 'Standard_NC6s_v3'
        vm.provisioning_state = 'Succeeded'
        vm.location = 'eastus'
        vm.id = '/subscriptions/sub/resourceGroups/test-rg/providers/Microsoft.Compute/virtualMachines/gpu-vm-1'
        vm.tags = {'team': 'ml', 'environment': 'dev'}
        vm.time_created = datetime(2024, 6, 15, 10, 30)
        vm.zones = ['1']

        compute = Mock()
        compute.virtual_machines.list.return_value = [vm]
        _stub_vm_power_state(compute, 'running')
        monitor = _make_monitor(compute_client=compute)

        with patch.dict(os.environ, {'AZURE_VM_SIZES': 'Standard_NC'}):
            instances = monitor.get_gpu_instances()

        assert len(instances) == 1
        inst = instances[0]
        assert inst['instance_id'] == 'test-rg/gpu-vm-1'
        assert inst['instance_type'] == 'Standard_NC6s_v3'
        assert inst['state'] == 'running'
        assert inst['gpu_count'] == 1
        assert inst['gpu_type'] == 'NVIDIA V100'
        assert inst['availability_zone'] == '1'
        assert inst['launch_time'] is not None
        assert inst['location'] == 'eastus'
        assert inst['resource_group'] == 'test-rg'
        assert inst['cloud_provider'] == 'azure'
        assert inst['tags'] == {'team': 'ml', 'environment': 'dev'}
        assert inst['created_at'] is not None

    def test_multiple_gpu_sizes_filter(self):
        """Should match VMs against multiple comma-separated GPU size prefixes."""
        vm_nc = Mock()
        vm_nc.name = 'nc-vm'
        vm_nc.hardware_profile.vm_size = 'Standard_NC6'
        vm_nc.provisioning_state = 'Succeeded'
        vm_nc.location = 'eastus'
        vm_nc.id = '/subscriptions/s/resourceGroups/rg/providers/x'
        vm_nc.tags = {}
        vm_nc.time_created = None
        vm_nc.zones = None

        vm_nd = Mock()
        vm_nd.name = 'nd-vm'
        vm_nd.hardware_profile.vm_size = 'Standard_ND6s'
        vm_nd.provisioning_state = 'Succeeded'
        vm_nd.location = 'eastus'
        vm_nd.id = '/subscriptions/s/resourceGroups/rg/providers/x'
        vm_nd.tags = {}
        vm_nd.time_created = None
        vm_nd.zones = None

        vm_cpu = Mock()
        vm_cpu.name = 'cpu-vm'
        vm_cpu.hardware_profile.vm_size = 'Standard_D4s_v3'

        compute = Mock()
        compute.virtual_machines.list.return_value = [vm_nc, vm_nd, vm_cpu]
        _stub_vm_power_state(compute, 'running')
        monitor = _make_monitor(compute_client=compute)

        with patch.dict(os.environ, {'AZURE_VM_SIZES': 'Standard_NC,Standard_ND'}):
            instances = monitor.get_gpu_instances()

        assert len(instances) == 2
        names = {i['instance_id'] for i in instances}
        assert names == {'rg/nc-vm', 'rg/nd-vm'}

    def test_empty_vm_sizes_env_uses_default_gpu_prefixes(self):
        """When AZURE_VM_SIZES is unset/empty, use built-in NC/NV/ND/NG substring filters."""
        vm = Mock()
        vm.name = 'some-vm'
        vm.hardware_profile.vm_size = 'Standard_NC6'
        vm.provisioning_state = 'Succeeded'
        vm.location = 'eastus'
        vm.id = '/subscriptions/s/resourceGroups/test-rg/providers/x'
        vm.tags = {}
        vm.time_created = None
        vm.zones = None

        compute = Mock()
        compute.virtual_machines.list.return_value = [vm]
        _stub_vm_power_state(compute, 'running')
        monitor = _make_monitor(compute_client=compute)

        with patch.dict(os.environ, {'AZURE_VM_SIZES': ''}, clear=False):
            instances = monitor.get_gpu_instances()

        assert len(instances) == 1
        assert instances[0]['instance_id'] == 'test-rg/some-vm'

    def test_list_all_when_no_resource_group(self):
        """Should call list_all() when resource_group is None."""
        compute = Mock()
        compute.virtual_machines.list_all.return_value = []
        monitor = _make_monitor(compute_client=compute)
        monitor.resource_group = None

        with patch.dict(os.environ, {'AZURE_VM_SIZES': 'Standard_NC'}):
            monitor.get_gpu_instances()

        compute.virtual_machines.list_all.assert_called_once()

    def test_api_exception_returns_empty_list(self):
        """Should catch exceptions and return empty list."""
        compute = Mock()
        compute.virtual_machines.list.side_effect = Exception('Azure API down')
        monitor = _make_monitor(compute_client=compute)

        with patch.dict(os.environ, {'AZURE_VM_SIZES': 'Standard_NC'}):
            result = monitor.get_gpu_instances()

        assert result == []


# ---------------------------------------------------------------------------
# get_gpu_utilization
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestGetGpuUtilizationFull:
    def test_returns_empty_when_nvml_not_available(self):
        """Should return empty list when GPU monitoring is not available."""
        monitor = _make_monitor(gpu_available=False)
        assert monitor.get_gpu_utilization([]) == []

    def test_collects_metrics_with_matching_hostname(self):
        """Should collect GPU metrics and match to instance by hostname."""
        monitor = _make_monitor(gpu_available=True)
        pynvml = monitor._pynvml
        pynvml.nvmlDeviceGetCount.return_value = 1
        pynvml.nvmlDeviceGetHandleByIndex.return_value = Mock()
        pynvml.nvmlDeviceGetUtilizationRates.return_value = Mock(gpu=85.3, memory=42.1)
        pynvml.nvmlDeviceGetMemoryInfo.return_value = Mock(
            used=4 * 1024 * 1024 * 1024,
            total=16 * 1024 * 1024 * 1024
        )
        pynvml.nvmlDeviceGetTemperature.return_value = 70
        pynvml.nvmlDeviceGetPowerUsage.return_value = 200000
        pynvml.NVML_TEMPERATURE_GPU = 0

        hostname = os.uname().nodename
        instances = [{'instance_id': hostname, 'cloud_provider': 'azure'}]
        metrics = monitor.get_gpu_utilization(instances)

        assert len(metrics) == 1
        m = metrics[0]
        assert m['gpu_utilization'] == 85.3
        assert m['memory_utilization'] == 42.1
        assert m['memory_used_mb'] == pytest.approx(4096.0, rel=1e-1)
        assert m['temperature_c'] == 70
        assert m['power_usage_w'] == 200.0
        assert m['is_idle'] is False
        assert m['cloud_provider'] == 'azure'
        assert m['source'] == 'nvml'
        assert 'cpu_utilization' in m

    def test_marks_idle_correctly(self):
        """Should flag GPU as idle when gpu < 10% and memory < 10%."""
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

    def test_multiple_gpus_reported(self):
        """Should report metrics for multiple GPUs when hostname matches."""
        monitor = _make_monitor(gpu_available=True)
        pynvml = monitor._pynvml
        pynvml.nvmlDeviceGetCount.return_value = 2
        pynvml.nvmlDeviceGetHandleByIndex.return_value = Mock()
        pynvml.nvmlDeviceGetUtilizationRates.return_value = Mock(gpu=50, memory=30)
        pynvml.nvmlDeviceGetMemoryInfo.return_value = Mock(used=1024**3, total=8*1024**3)
        pynvml.nvmlDeviceGetTemperature.return_value = 55
        pynvml.nvmlDeviceGetPowerUsage.return_value = 100000
        pynvml.NVML_TEMPERATURE_GPU = 0

        hostname = os.uname().nodename
        instances = [{'instance_id': hostname}]
        metrics = monitor.get_gpu_utilization(instances)

        assert len(metrics) == 2

    def test_no_matching_hostname_returns_empty(self):
        """Should return empty when hostname doesn't match and multiple instances exist."""
        monitor = _make_monitor(gpu_available=True)
        pynvml = monitor._pynvml
        pynvml.nvmlDeviceGetCount.return_value = 1
        pynvml.nvmlDeviceGetHandleByIndex.return_value = Mock()
        pynvml.nvmlDeviceGetUtilizationRates.return_value = Mock(gpu=50, memory=30)
        pynvml.nvmlDeviceGetMemoryInfo.return_value = Mock(used=100, total=1000)
        pynvml.NVML_TEMPERATURE_GPU = 0

        instances = [
            {'instance_id': 'some-other-host'},
            {'instance_id': 'other-vm'},
        ]
        metrics = monitor.get_gpu_utilization(instances)
        assert metrics == []

    def test_device_error_isolation(self):
        """Should continue sampling other GPUs if one device fails."""
        monitor = _make_monitor(gpu_available=True)
        pynvml = monitor._pynvml
        pynvml.nvmlDeviceGetCount.return_value = 2
        good_handle = Mock()
        pynvml.nvmlDeviceGetHandleByIndex.side_effect = [
            Exception('GPU 0 error'), good_handle
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

    def test_nvml_device_count_error_returns_empty(self):
        """Should return empty list if nvmlDeviceGetCount raises."""
        monitor = _make_monitor(gpu_available=True)
        monitor._pynvml.nvmlDeviceGetCount.side_effect = Exception('NVML driver error')
        metrics = monitor.get_gpu_utilization([{'instance_id': 'x'}])
        assert metrics == []


# ---------------------------------------------------------------------------
# get_cost_data
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestGetCostDataFull:
    def test_filters_gpu_vms_by_standard_n(self):
        """Should only include usage records with 'Standard_N' in instance name."""
        gpu_usage = Mock()
        gpu_usage.instance_name = 'Standard_NC6_vm1'
        gpu_usage.product = 'NC Series'
        gpu_usage.cost = 10.50
        gpu_usage.billing_currency = 'USD'
        gpu_usage.date = datetime(2024, 1, 15)
        gpu_usage.resource_group = 'ml-rg'
        gpu_usage.location = 'eastus'

        cpu_usage = Mock()
        cpu_usage.instance_name = 'Standard_D4_vm2'

        consumption = Mock()
        consumption.usage_details.list.return_value = [gpu_usage, cpu_usage]
        monitor = _make_monitor(consumption_client=consumption)

        costs = monitor.get_cost_data()

        assert len(costs) == 1
        assert costs[0]['cost'] == 10.50
        assert costs[0]['cloud_provider'] == 'azure'
        assert costs[0]['instance_id'] == 'Standard_NC6_vm1'

    def test_empty_usage_returns_empty_list(self):
        consumption = Mock()
        consumption.usage_details.list.return_value = []
        monitor = _make_monitor(consumption_client=consumption)
        assert monitor.get_cost_data() == []

    def test_api_error_returns_empty_list(self):
        consumption = Mock()
        consumption.usage_details.list.side_effect = Exception('Auth expired')
        monitor = _make_monitor(consumption_client=consumption)
        assert monitor.get_cost_data() == []

    def test_null_instance_name_skipped(self):
        """Should skip usage records where instance_name is None."""
        usage = Mock()
        usage.instance_name = None

        consumption = Mock()
        consumption.usage_details.list.return_value = [usage]
        monitor = _make_monitor(consumption_client=consumption)
        assert monitor.get_cost_data() == []


# ---------------------------------------------------------------------------
# get_cost_by_tags
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestGetCostByTags:
    def test_groups_costs_by_tag_key(self):
        """Should group GPU VM costs by specified tag key."""
        usage1 = Mock()
        usage1.instance_name = 'Standard_NC6_vm1'
        usage1.cost = 10.0
        usage1.tags = {'team': 'ml'}

        usage2 = Mock()
        usage2.instance_name = 'Standard_NC12_vm2'
        usage2.cost = 20.0
        usage2.tags = {'team': 'ml'}

        usage3 = Mock()
        usage3.instance_name = 'Standard_ND6_vm3'
        usage3.cost = 15.0
        usage3.tags = {'team': 'infra'}

        consumption = Mock()
        consumption.usage_details.list.return_value = [usage1, usage2, usage3]
        monitor = _make_monitor(consumption_client=consumption)

        result = monitor.get_cost_by_tags(tag_key='team')

        assert len(result) == 2
        by_tag = {r['tag_value']: r for r in result}
        assert by_tag['ml']['total_cost'] == 30.0
        assert by_tag['ml']['record_count'] == 2
        assert by_tag['infra']['total_cost'] == 15.0
        assert by_tag['infra']['record_count'] == 1
        assert all(r['tag_key'] == 'team' for r in result)

    def test_untagged_resources_grouped(self):
        """Should group resources without the tag under 'untagged'."""
        usage = Mock()
        usage.instance_name = 'Standard_NC6_vm1'
        usage.cost = 5.0
        usage.tags = {}

        consumption = Mock()
        consumption.usage_details.list.return_value = [usage]
        monitor = _make_monitor(consumption_client=consumption)

        result = monitor.get_cost_by_tags(tag_key='team')

        assert len(result) == 1
        assert result[0]['tag_value'] == 'untagged'
        assert result[0]['total_cost'] == 5.0

    def test_no_tag_key_groups_all(self):
        """Should group all costs under 'all' when tag_key is None."""
        usage = Mock()
        usage.instance_name = 'Standard_NC6_vm1'
        usage.cost = 7.5
        usage.tags = {'team': 'ml'}

        consumption = Mock()
        consumption.usage_details.list.return_value = [usage]
        monitor = _make_monitor(consumption_client=consumption)

        result = monitor.get_cost_by_tags(tag_key=None)

        assert len(result) == 1
        assert result[0]['tag_value'] == 'all'
        assert result[0]['tag_key'] == 'all'

    def test_filters_non_gpu_usage(self):
        """Should skip usage records for non-GPU VMs."""
        gpu_usage = Mock()
        gpu_usage.instance_name = 'Standard_NC6_vm1'
        gpu_usage.cost = 10.0
        gpu_usage.tags = {'team': 'ml'}

        cpu_usage = Mock()
        cpu_usage.instance_name = 'Standard_D4_vm2'
        cpu_usage.cost = 5.0
        cpu_usage.tags = {'team': 'ml'}

        consumption = Mock()
        consumption.usage_details.list.return_value = [gpu_usage, cpu_usage]
        monitor = _make_monitor(consumption_client=consumption)

        result = monitor.get_cost_by_tags(tag_key='team')

        assert len(result) == 1
        assert result[0]['total_cost'] == 10.0

    def test_api_error_returns_empty(self):
        consumption = Mock()
        consumption.usage_details.list.side_effect = Exception('Cost API error')
        monitor = _make_monitor(consumption_client=consumption)
        assert monitor.get_cost_by_tags(tag_key='team') == []

    def test_tags_attribute_missing_uses_empty_dict(self):
        """Should treat missing tags attribute as empty dict."""
        usage = Mock(spec=[])
        usage.instance_name = 'Standard_NC6_vm1'
        usage.cost = 8.0
        # getattr(usage, 'tags', {}) will return {} since spec=[] has no tags

        consumption = Mock()
        consumption.usage_details.list.return_value = [usage]
        monitor = _make_monitor(consumption_client=consumption)

        result = monitor.get_cost_by_tags(tag_key='team')
        assert len(result) == 1
        assert result[0]['tag_value'] == 'untagged'


# ---------------------------------------------------------------------------
# _refresh_azure_prices
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestRefreshAzurePrices:
    def test_fetches_prices_from_api(self):
        """Should parse Azure Retail Prices API response and populate cache."""
        from monitors.azure_monitor import AzureMonitor

        api_response = {
            'Items': [
                {
                    'armSkuName': 'Standard_NC6',
                    'unitPrice': 0.90,
                    'skuName': 'NC6',
                    'meterName': 'NC6',
                },
                {
                    'armSkuName': 'Standard_NC6',
                    'unitPrice': 0.27,
                    'skuName': 'NC6 Spot',
                    'meterName': 'NC6 Spot',
                },
            ],
            'NextPageLink': None,
        }

        mock_response = Mock()
        mock_response.json.return_value = api_response
        mock_response.raise_for_status = Mock()

        monitor = _make_monitor()

        with patch('requests.get', return_value=mock_response) as mock_get, \
             patch.dict(os.environ, {'AZURE_LOCATION': 'eastus'}):
            monitor._refresh_azure_prices()

        assert 'Standard_NC6' in AzureMonitor._azure_price_cache
        assert AzureMonitor._azure_price_cache['Standard_NC6']['on_demand'] == 0.90
        assert AzureMonitor._azure_price_cache['Standard_NC6']['spot'] == 0.27

    def test_cache_prevents_repeated_api_calls(self):
        """Should not call API again within 6-hour cache window."""
        from monitors.azure_monitor import AzureMonitor

        monitor = _make_monitor()
        AzureMonitor._azure_price_cache = {'Standard_NC6': {'on_demand': 0.90, 'spot': 0.27}}
        AzureMonitor._azure_price_cache_ts = time.time()  # fresh

        with patch('requests.get') as mock_get:
            monitor._refresh_azure_prices()

        mock_get.assert_not_called()

    def test_stale_cache_triggers_refresh(self):
        """Should call API when cache is older than 6 hours."""
        from monitors.azure_monitor import AzureMonitor

        AzureMonitor._azure_price_cache = {'old': {'on_demand': 1.0, 'spot': 0.3}}
        AzureMonitor._azure_price_cache_ts = time.time() - 25000  # > 21600s

        api_response = {'Items': [], 'NextPageLink': None}
        mock_response = Mock()
        mock_response.json.return_value = api_response
        mock_response.raise_for_status = Mock()

        monitor = _make_monitor()

        with patch('requests.get', return_value=mock_response) as mock_get, \
             patch.dict(os.environ, {'AZURE_LOCATION': 'eastus'}):
            monitor._refresh_azure_prices()

        mock_get.assert_called_once()

    def test_api_failure_keeps_existing_cache(self):
        """Should log warning and keep cache on API failure."""
        from monitors.azure_monitor import AzureMonitor

        # Empty cache, stale timestamp to force refresh
        AzureMonitor._azure_price_cache = {}
        AzureMonitor._azure_price_cache_ts = 0

        monitor = _make_monitor()

        with patch('requests.get', side_effect=Exception('Network error')), \
             patch.dict(os.environ, {'AZURE_LOCATION': 'eastus'}):
            monitor._refresh_azure_prices()

        # Cache should remain empty (no crash)
        assert AzureMonitor._azure_price_cache == {}

    def test_pagination_follows_next_page_link(self):
        """Should follow NextPageLink for paginated results."""
        from monitors.azure_monitor import AzureMonitor

        page1_response = Mock()
        page1_response.json.return_value = {
            'Items': [{
                'armSkuName': 'Standard_NC6',
                'unitPrice': 0.90,
                'skuName': 'NC6',
                'meterName': 'NC6',
            }],
            'NextPageLink': 'https://prices.azure.com/api/retail/prices?page=2',
        }
        page1_response.raise_for_status = Mock()

        page2_response = Mock()
        page2_response.json.return_value = {
            'Items': [{
                'armSkuName': 'Standard_NC12',
                'unitPrice': 1.80,
                'skuName': 'NC12',
                'meterName': 'NC12',
            }],
            'NextPageLink': None,
        }
        page2_response.raise_for_status = Mock()

        monitor = _make_monitor()

        with patch('requests.get', side_effect=[page1_response, page2_response]), \
             patch.dict(os.environ, {'AZURE_LOCATION': 'eastus'}):
            monitor._refresh_azure_prices()

        assert 'Standard_NC6' in AzureMonitor._azure_price_cache
        assert 'Standard_NC12' in AzureMonitor._azure_price_cache

    def test_skips_low_priority_for_on_demand(self):
        """Should not use 'Low Priority' SKU as on-demand price."""
        from monitors.azure_monitor import AzureMonitor

        api_response = {
            'Items': [
                {
                    'armSkuName': 'Standard_NC6',
                    'unitPrice': 0.27,
                    'skuName': 'NC6 Low Priority',
                    'meterName': 'NC6',
                },
                {
                    'armSkuName': 'Standard_NC6',
                    'unitPrice': 0.90,
                    'skuName': 'NC6',
                    'meterName': 'NC6',
                },
            ],
            'NextPageLink': None,
        }

        mock_response = Mock()
        mock_response.json.return_value = api_response
        mock_response.raise_for_status = Mock()

        monitor = _make_monitor()

        with patch('requests.get', return_value=mock_response), \
             patch.dict(os.environ, {'AZURE_LOCATION': 'eastus'}):
            monitor._refresh_azure_prices()

        assert AzureMonitor._azure_price_cache['Standard_NC6']['on_demand'] == 0.90


# ---------------------------------------------------------------------------
# _get_azure_on_demand_price
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestGetAzureOnDemandPrice:
    def test_cache_hit_returns_api_price(self):
        """Should return API-fetched price when available in cache."""
        from monitors.azure_monitor import AzureMonitor

        monitor = _make_monitor()
        AzureMonitor._azure_price_cache = {
            'Standard_NC6': {'on_demand': 0.95, 'spot': 0.28}
        }
        AzureMonitor._azure_price_cache_ts = time.time()

        price = monitor._get_azure_on_demand_price('Standard_NC6')
        assert price == 0.95

    def test_cache_miss_returns_fallback(self):
        """Should return fallback price when VM size not in cache."""
        from monitors.azure_monitor import AzureMonitor

        AzureMonitor._azure_price_cache = {}
        AzureMonitor._azure_price_cache_ts = time.time()

        monitor = _make_monitor()

        with patch('requests.get', side_effect=Exception('skip')):
            price = monitor._get_azure_on_demand_price('Standard_NC6')

        assert price == 0.90  # fallback value

    def test_unknown_vm_size_returns_zero(self):
        """Should return 0 for unknown VM sizes not in fallback."""
        from monitors.azure_monitor import AzureMonitor

        AzureMonitor._azure_price_cache = {}
        AzureMonitor._azure_price_cache_ts = time.time()

        monitor = _make_monitor()

        with patch('requests.get', side_effect=Exception('skip')):
            price = monitor._get_azure_on_demand_price('Standard_UNKNOWN_SIZE')

        assert price == 0


# ---------------------------------------------------------------------------
# _get_azure_spot_price
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestGetAzureSpotPrice:
    def test_returns_real_spot_price_from_cache(self):
        """Should return real spot price when available in API cache."""
        from monitors.azure_monitor import AzureMonitor

        AzureMonitor._azure_price_cache = {
            'Standard_NC6': {'on_demand': 0.90, 'spot': 0.27}
        }
        AzureMonitor._azure_price_cache_ts = time.time()

        monitor = _make_monitor()
        price = monitor._get_azure_spot_price('Standard_NC6')
        assert price == 0.27

    def test_fallback_estimate_when_no_spot_price(self):
        """Should estimate spot at 30% of on-demand when spot price missing."""
        from monitors.azure_monitor import AzureMonitor

        # Cache entry with on_demand but no spot
        AzureMonitor._azure_price_cache = {
            'Standard_NC6': {'on_demand': 0.90, 'spot': 0}
        }
        AzureMonitor._azure_price_cache_ts = time.time()

        monitor = _make_monitor()
        price = monitor._get_azure_spot_price('Standard_NC6')
        assert price == pytest.approx(0.27, rel=1e-2)

    def test_fallback_for_unknown_uses_on_demand_fallback(self):
        """Should use fallback on-demand * 0.3 for unknown VM sizes."""
        from monitors.azure_monitor import AzureMonitor

        AzureMonitor._azure_price_cache = {}
        AzureMonitor._azure_price_cache_ts = time.time()

        monitor = _make_monitor()

        with patch('requests.get', side_effect=Exception('skip')):
            price = monitor._get_azure_spot_price('Standard_NC6')

        expected = round(0.90 * 0.3, 4)
        assert price == expected

    def test_returns_zero_for_completely_unknown_size(self):
        """Should return 0 when on-demand is also 0."""
        from monitors.azure_monitor import AzureMonitor

        AzureMonitor._azure_price_cache = {}
        AzureMonitor._azure_price_cache_ts = time.time()

        monitor = _make_monitor()

        with patch('requests.get', side_effect=Exception('skip')):
            price = monitor._get_azure_spot_price('Standard_UNKNOWN')

        assert price == 0


# ---------------------------------------------------------------------------
# get_spot_pricing
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestGetSpotPricing:
    def test_returns_spot_data_for_configured_vm_sizes(self):
        """Should return spot pricing info for each configured VM size."""
        from monitors.azure_monitor import AzureMonitor

        monitor = _make_monitor()
        AzureMonitor._azure_price_cache = {
            'Standard_NC6': {'on_demand': 0.90, 'spot': 0.27}
        }
        AzureMonitor._azure_price_cache_ts = time.time()

        with patch.dict(os.environ, {'AZURE_VM_SIZES': 'Standard_NC6'}):
            spot_data = monitor.get_spot_pricing()

        assert len(spot_data) == 1
        entry = spot_data[0]
        assert entry['instance_type'] == 'Standard_NC6'
        assert entry['spot_price_hourly'] == 0.27
        assert entry['on_demand_price_hourly'] == 0.90
        assert entry['savings_pct'] == 70.0
        assert entry['cloud_provider'] == 'azure'
        assert entry['source'] == 'Azure Retail Prices API'

    def test_empty_vm_sizes_returns_empty(self):
        """Should return empty list when no VM sizes configured."""
        monitor = _make_monitor()

        with patch.dict(os.environ, {'AZURE_VM_SIZES': ''}):
            assert monitor.get_spot_pricing() == []

    def test_uses_fallback_source_when_no_spot_in_cache(self):
        """Should indicate 'estimated' source when using fallback pricing."""
        from monitors.azure_monitor import AzureMonitor

        AzureMonitor._azure_price_cache = {
            'Standard_NC6': {'on_demand': 0.90, 'spot': 0}
        }
        AzureMonitor._azure_price_cache_ts = time.time()

        monitor = _make_monitor()

        with patch.dict(os.environ, {'AZURE_VM_SIZES': 'Standard_NC6'}):
            spot_data = monitor.get_spot_pricing()

        assert len(spot_data) == 1
        assert spot_data[0]['source'] == 'estimated'

    def test_multiple_vm_sizes(self):
        """Should return pricing for multiple VM sizes."""
        from monitors.azure_monitor import AzureMonitor

        AzureMonitor._azure_price_cache = {
            'Standard_NC6': {'on_demand': 0.90, 'spot': 0.27},
            'Standard_NC12': {'on_demand': 1.80, 'spot': 0.54},
        }
        AzureMonitor._azure_price_cache_ts = time.time()

        monitor = _make_monitor()

        with patch.dict(os.environ, {'AZURE_VM_SIZES': 'Standard_NC6,Standard_NC12'}):
            spot_data = monitor.get_spot_pricing()

        assert len(spot_data) == 2

    def test_monthly_cost_calculations(self):
        """Should correctly calculate monthly costs (hourly * 730)."""
        from monitors.azure_monitor import AzureMonitor

        monitor = _make_monitor()
        AzureMonitor._azure_price_cache = {
            'Standard_NC6': {'on_demand': 1.0, 'spot': 0.3}
        }
        AzureMonitor._azure_price_cache_ts = time.time()

        with patch.dict(os.environ, {'AZURE_VM_SIZES': 'Standard_NC6'}):
            spot_data = monitor.get_spot_pricing()

        assert spot_data[0]['spot_monthly'] == 219.0
        assert spot_data[0]['on_demand_monthly'] == 730.0


# ---------------------------------------------------------------------------
# get_spot_candidates
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestGetSpotCandidates:
    def _sample_spot_pricing(self):
        return [{
            'instance_type': 'Standard_NC6',
            'spot_price_hourly': 0.27,
            'on_demand_price_hourly': 0.90,
            'spot_monthly': 197.10,
            'on_demand_monthly': 657.0,
            'savings_pct': 70.0,
        }]

    def test_fault_tolerant_tag_is_candidate(self):
        """Should select instances tagged as fault-tolerant."""
        instances = [{
            'instance_id': 'gpu-vm-1',
            'instance_type': 'Standard_NC6',
            'state': 'Succeeded',
            'tags': {'fault_tolerant': 'true'},
        }]
        monitor = _make_monitor()
        candidates = monitor.get_spot_candidates(instances, self._sample_spot_pricing())

        assert len(candidates) == 1
        assert candidates[0]['instance_id'] == 'gpu-vm-1'
        assert candidates[0]['recommendation_type'] == 'spot'
        assert candidates[0]['source_data']['reason'] == 'Tagged as fault-tolerant'

    def test_dev_environment_tag_is_candidate(self):
        """Should select instances with dev environment tag."""
        instances = [{
            'instance_id': 'dev-vm',
            'instance_type': 'Standard_NC6',
            'state': 'Running',
            'tags': {'environment': 'dev'},
        }]
        monitor = _make_monitor()
        candidates = monitor.get_spot_candidates(instances, self._sample_spot_pricing())

        assert len(candidates) == 1
        assert 'Environment: dev' in candidates[0]['source_data']['reason']

    def test_training_tag_value_is_candidate(self):
        """Should select instances with tag values matching spot workloads."""
        instances = [{
            'instance_id': 'train-vm',
            'instance_type': 'Standard_NC6',
            'state': 'Succeeded',
            'tags': {'workload': 'training'},
        }]
        monitor = _make_monitor()
        candidates = monitor.get_spot_candidates(instances, self._sample_spot_pricing())

        assert len(candidates) == 1
        assert 'non-production' in candidates[0]['source_data']['reason']

    def test_non_running_instances_excluded(self):
        """Should skip instances that are not in Running or Succeeded state."""
        instances = [{
            'instance_id': 'stopped-vm',
            'instance_type': 'Standard_NC6',
            'state': 'Stopped',
            'tags': {'fault_tolerant': 'true'},
        }]
        monitor = _make_monitor()
        candidates = monitor.get_spot_candidates(instances, self._sample_spot_pricing())
        assert candidates == []

    def test_no_matching_pricing_excluded(self):
        """Should skip instances without matching spot pricing data."""
        instances = [{
            'instance_id': 'big-vm',
            'instance_type': 'Standard_NC24s_v3',
            'state': 'Succeeded',
            'tags': {'fault_tolerant': 'true'},
        }]
        monitor = _make_monitor()
        candidates = monitor.get_spot_candidates(instances, self._sample_spot_pricing())
        assert candidates == []

    def test_no_matching_tags_excluded(self):
        """Should skip instances without matching spot-related tags."""
        instances = [{
            'instance_id': 'prod-vm',
            'instance_type': 'Standard_NC6',
            'state': 'Succeeded',
            'tags': {'environment': 'production', 'team': 'ml'},
        }]
        monitor = _make_monitor()
        candidates = monitor.get_spot_candidates(instances, self._sample_spot_pricing())
        assert candidates == []

    def test_savings_calculations_correct(self):
        """Should correctly calculate savings fields."""
        instances = [{
            'instance_id': 'gpu-vm',
            'instance_type': 'Standard_NC6',
            'state': 'Succeeded',
            'tags': {'environment': 'test'},
        }]
        monitor = _make_monitor()
        candidates = monitor.get_spot_candidates(instances, self._sample_spot_pricing())

        assert len(candidates) == 1
        c = candidates[0]
        assert c['current_monthly_cost'] == 657.0
        assert c['recommended_monthly_cost'] == 197.10
        assert c['potential_monthly_savings'] == pytest.approx(459.90, rel=1e-2)
        assert c['potential_savings_pct'] == 70.0

    def test_best_spot_price_selected(self):
        """Should use the best (lowest) spot price when duplicates exist."""
        spot_pricing = [
            {
                'instance_type': 'Standard_NC6',
                'spot_price_hourly': 0.30,
                'on_demand_price_hourly': 0.90,
                'spot_monthly': 219.0,
                'on_demand_monthly': 657.0,
                'savings_pct': 66.7,
            },
            {
                'instance_type': 'Standard_NC6',
                'spot_price_hourly': 0.25,
                'on_demand_price_hourly': 0.90,
                'spot_monthly': 182.50,
                'on_demand_monthly': 657.0,
                'savings_pct': 72.2,
            },
        ]
        instances = [{
            'instance_id': 'vm1',
            'instance_type': 'Standard_NC6',
            'state': 'Succeeded',
            'tags': {'environment': 'dev'},
        }]
        monitor = _make_monitor()
        candidates = monitor.get_spot_candidates(instances, spot_pricing)

        assert len(candidates) == 1
        assert candidates[0]['recommended_monthly_cost'] == 182.50


# ---------------------------------------------------------------------------
# get_resource_tags
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestGetResourceTagsFull:
    def test_returns_tags_dict(self):
        resource = Mock()
        resource.tags = {'env': 'prod', 'team': 'ml'}
        resource_client = Mock()
        resource_client.resources.get_by_id.return_value = resource
        monitor = _make_monitor(resource_client=resource_client)

        tags = monitor.get_resource_tags('/subscriptions/sub/resource/id')
        assert tags == {'env': 'prod', 'team': 'ml'}

    def test_returns_empty_on_api_error(self):
        resource_client = Mock()
        resource_client.resources.get_by_id.side_effect = Exception('Not found')
        monitor = _make_monitor(resource_client=resource_client)
        assert monitor.get_resource_tags('bad-id') == {}

    def test_returns_empty_when_tags_none(self):
        resource = Mock()
        resource.tags = None
        resource_client = Mock()
        resource_client.resources.get_by_id.return_value = resource
        monitor = _make_monitor(resource_client=resource_client)
        assert monitor.get_resource_tags('some-id') == {}

    def test_calls_api_with_correct_version(self):
        """Should pass the correct API version to get_by_id."""
        resource = Mock()
        resource.tags = {}
        resource_client = Mock()
        resource_client.resources.get_by_id.return_value = resource
        monitor = _make_monitor(resource_client=resource_client)

        monitor.get_resource_tags('/subscriptions/sub/resource/id')
        resource_client.resources.get_by_id.assert_called_once_with(
            '/subscriptions/sub/resource/id', '2021-04-01'
        )
