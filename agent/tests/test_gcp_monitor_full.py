"""
Comprehensive tests for GCP Monitor — covers cost-by-labels, spot pricing,
the Cloud Billing Catalog API integration, GPU price caching, and spot candidates.
"""

import pytest
import os
import sys
import time
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

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
_mock_bigquery = MagicMock()
sys.modules.setdefault('google.cloud.bigquery', _mock_bigquery)


def _make_monitor(compute_client=None, monitoring_client=None,
                  billing_client=None, resource_client=None,
                  gpu_available=False):
    """Create a GCPMonitor with mocked dependencies -- bypasses __init__."""
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
    monitor.gpu_available = gpu_available
    monitor._pynvml = Mock() if gpu_available else None
    monitor.bq_client = None
    monitor.billing_dataset = None
    monitor.discover_labeled = False
    monitor.discover_label_key = "instance-type"
    monitor.discover_label_value = "gpu-monitoring"
    # Reset class-level price cache for test isolation
    GCPMonitor._gcp_gpu_price_cache = {}
    GCPMonitor._gcp_gpu_price_cache_ts = 0
    return monitor


def _make_gce_instance(name='gpu-vm-1',
                       machine_type='zones/us-central1-a/machineTypes/n1-standard-8',
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
# get_gpu_instances
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestGetGpuInstancesFull:
    def test_returns_all_fields(self):
        """Should populate all expected fields for a GPU instance."""
        instance, zone = _make_gce_instance(
            labels={'team': 'ml', 'environment': 'dev'},
            gpu_count=2,
        )
        zone_scope = Mock()
        zone_scope.instances = [instance]

        compute = Mock()
        compute.aggregated_list.return_value = [(zone, zone_scope)]
        monitor = _make_monitor(compute_client=compute)

        with patch.dict(os.environ, {'GCP_GPU_TYPES': 'nvidia-tesla-v100', 'GCP_MACHINE_TYPES': ''}):
            instances = monitor.get_gpu_instances()

        assert len(instances) == 1
        inst = instances[0]
        assert inst['instance_id'] == 'us-central1-a/gpu-vm-1'
        assert inst['instance_type'] == 'n1-standard-8'
        assert inst['state'] == 'RUNNING'
        assert inst['cloud_provider'] == 'gcp'
        assert inst['gpu_count'] == 2
        assert inst['gpu_type'] == 'nvidia-tesla-v100'
        assert inst['tags'] == {'team': 'ml', 'environment': 'dev'}
        assert inst['project_id'] == 'test-project'

    def test_multiple_zones(self):
        """Should aggregate instances from multiple zones."""
        inst1, zone1 = _make_gce_instance(name='vm-1')
        inst2, _ = _make_gce_instance(name='vm-2')
        zone2 = 'zones/europe-west1-b'

        scope1 = Mock()
        scope1.instances = [inst1]
        scope2 = Mock()
        scope2.instances = [inst2]

        compute = Mock()
        compute.aggregated_list.return_value = [(zone1, scope1), (zone2, scope2)]
        monitor = _make_monitor(compute_client=compute)

        with patch.dict(os.environ, {'GCP_GPU_TYPES': 'nvidia-tesla-v100', 'GCP_MACHINE_TYPES': ''}):
            instances = monitor.get_gpu_instances()

        assert len(instances) == 2

    def test_no_accelerators_skipped(self):
        """Should skip instances with no guest_accelerators."""
        instance = Mock()
        instance.name = 'cpu-vm'
        instance.guest_accelerators = []

        zone_scope = Mock()
        zone_scope.instances = [instance]

        compute = Mock()
        compute.aggregated_list.return_value = [('zones/us-central1-a', zone_scope)]
        monitor = _make_monitor(compute_client=compute)

        with patch.dict(os.environ, {'GCP_GPU_TYPES': 'nvidia-tesla-v100', 'GCP_MACHINE_TYPES': ''}):
            instances = monitor.get_gpu_instances()

        assert instances == []

    def test_api_error_returns_empty(self):
        compute = Mock()
        compute.aggregated_list.side_effect = Exception('Permission denied')
        monitor = _make_monitor(compute_client=compute)

        with patch.dict(os.environ, {'GCP_GPU_TYPES': 'nvidia-tesla-v100', 'GCP_MACHINE_TYPES': ''}):
            assert monitor.get_gpu_instances() == []

    def test_empty_zones_handled(self):
        """Should handle zones where instances is None."""
        scope = Mock()
        scope.instances = None

        compute = Mock()
        compute.aggregated_list.return_value = [('zones/us-central1-a', scope)]
        monitor = _make_monitor(compute_client=compute)

        with patch.dict(os.environ, {'GCP_GPU_TYPES': 'nvidia-tesla-v100', 'GCP_MACHINE_TYPES': ''}):
            assert monitor.get_gpu_instances() == []


# ---------------------------------------------------------------------------
# get_cost_data
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestGetCostDataFull:
    def test_missing_billing_account_returns_empty(self):
        monitor = _make_monitor()
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop('GCP_BILLING_ACCOUNT_ID', None)
            assert monitor.get_cost_data() == []

    def test_returns_cost_entries_for_gpu_instances(self):
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
        assert costs[0]['instance_id'] == 'us-central1-a/gpu-vm-1'

    def test_error_returns_empty(self):
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
# get_cost_by_labels
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestGetCostByLabels:
    def _setup_instances(self, monitor, instances_data):
        """Helper to mock get_gpu_instances for cost_by_labels."""
        with patch.object(monitor, 'get_gpu_instances', return_value=instances_data), \
             patch.object(monitor, '_estimate_instance_cost', side_effect=lambda inst: inst.get('_cost', 1.0)):
            return monitor.get_cost_by_labels

    def test_groups_by_label_key(self):
        """Should group costs by specified label key."""
        monitor = _make_monitor()
        instances = [
            {'instance_id': 'vm1', 'labels': {'team': 'ml'}, '_cost': 10.0,
             'instance_type': 'n1-standard-8', 'gpu_type': 'nvidia-tesla-v100',
             'gpu_count': 1, 'preemptible': False},
            {'instance_id': 'vm2', 'labels': {'team': 'ml'}, '_cost': 15.0,
             'instance_type': 'n1-standard-8', 'gpu_type': 'nvidia-tesla-v100',
             'gpu_count': 1, 'preemptible': False},
            {'instance_id': 'vm3', 'labels': {'team': 'infra'}, '_cost': 5.0,
             'instance_type': 'n1-standard-8', 'gpu_type': 'nvidia-tesla-v100',
             'gpu_count': 1, 'preemptible': False},
        ]

        with patch.object(monitor, 'get_gpu_instances', return_value=instances), \
             patch.object(monitor, '_estimate_instance_cost', side_effect=lambda inst: inst['_cost']):
            result = monitor.get_cost_by_labels(label_key='team')

        assert len(result) == 2
        by_label = {r['label_value']: r for r in result}
        assert by_label['ml']['total_cost'] == 25.0
        assert by_label['ml']['record_count'] == 2
        assert by_label['infra']['total_cost'] == 5.0

    def test_unlabeled_resources_grouped(self):
        """Should group resources without the label under 'unlabeled'."""
        monitor = _make_monitor()
        instances = [
            {'instance_id': 'vm1', 'labels': {},
             'instance_type': 'n1-standard-8', 'gpu_type': 'nvidia-tesla-v100',
             'gpu_count': 1, 'preemptible': False},
        ]

        with patch.object(monitor, 'get_gpu_instances', return_value=instances), \
             patch.object(monitor, '_estimate_instance_cost', return_value=3.0):
            result = monitor.get_cost_by_labels(label_key='team')

        assert len(result) == 1
        assert result[0]['label_value'] == 'unlabeled'

    def test_no_label_key_groups_all(self):
        """Should group all under 'all' when label_key is None."""
        monitor = _make_monitor()
        instances = [
            {'instance_id': 'vm1', 'labels': {'team': 'ml'},
             'instance_type': 'n1-standard-8', 'gpu_type': 'nvidia-tesla-v100',
             'gpu_count': 1, 'preemptible': False},
        ]

        with patch.object(monitor, 'get_gpu_instances', return_value=instances), \
             patch.object(monitor, '_estimate_instance_cost', return_value=5.0):
            result = monitor.get_cost_by_labels(label_key=None)

        assert len(result) == 1
        assert result[0]['label_value'] == 'all'
        assert result[0]['label_key'] == 'all'

    def test_empty_instances_returns_empty(self):
        monitor = _make_monitor()
        with patch.object(monitor, 'get_gpu_instances', return_value=[]):
            result = monitor.get_cost_by_labels(label_key='team')
        assert result == []

    def test_error_returns_empty(self):
        monitor = _make_monitor()
        with patch.object(monitor, 'get_gpu_instances', side_effect=Exception('fail')):
            result = monitor.get_cost_by_labels(label_key='team')
        assert result == []


# ---------------------------------------------------------------------------
# _refresh_gcp_gpu_prices
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestRefreshGcpGpuPrices:
    def setup_method(self):
        """Reset class-level cache before each test."""
        from monitors.gcp_monitor import GCPMonitor
        GCPMonitor._gcp_gpu_price_cache = {}
        GCPMonitor._gcp_gpu_price_cache_ts = 0

    def test_fetches_prices_from_billing_catalog(self):
        """Should parse Cloud Billing Catalog SKUs and populate cache."""
        from monitors.gcp_monitor import GCPMonitor

        # Create mock SKU for T4 on-demand
        sku_t4 = Mock()
        sku_t4.description = 'Nvidia Tesla T4 GPU running in Americas'
        sku_t4.service_regions = ['us-central1']

        rate = Mock()
        rate.unit_price = Mock()
        rate.unit_price.units = 0
        rate.unit_price.nanos = 350000000  # $0.35

        pricing_expr = Mock()
        pricing_expr.tiered_rates = [rate]

        pricing_info = Mock()
        pricing_info.pricing_expression = pricing_expr

        sku_t4.pricing_info = [pricing_info]

        # Create mock SKU for T4 spot/preemptible
        sku_t4_spot = Mock()
        sku_t4_spot.description = 'Nvidia Tesla T4 GPU running in Americas (Preemptible)'
        sku_t4_spot.service_regions = ['us-central1']

        rate_spot = Mock()
        rate_spot.unit_price = Mock()
        rate_spot.unit_price.units = 0
        rate_spot.unit_price.nanos = 110000000  # $0.11

        pricing_expr_spot = Mock()
        pricing_expr_spot.tiered_rates = [rate_spot]

        pricing_info_spot = Mock()
        pricing_info_spot.pricing_expression = pricing_expr_spot

        sku_t4_spot.pricing_info = [pricing_info_spot]

        mock_catalog_client = Mock()
        mock_catalog_client.list_skus.return_value = [sku_t4, sku_t4_spot]

        monitor = _make_monitor()

        with patch('monitors.gcp_monitor.billing_v1.CloudCatalogClient', return_value=mock_catalog_client):
            monitor._refresh_gcp_gpu_prices()

        assert 'nvidia-tesla-t4' in GCPMonitor._gcp_gpu_price_cache
        assert GCPMonitor._gcp_gpu_price_cache['nvidia-tesla-t4']['on_demand'] == 0.35
        assert GCPMonitor._gcp_gpu_price_cache['nvidia-tesla-t4']['spot'] == 0.11

    def test_cache_prevents_repeated_api_calls(self):
        """Should not call API again within 6-hour cache window."""
        from monitors.gcp_monitor import GCPMonitor

        monitor = _make_monitor()
        GCPMonitor._gcp_gpu_price_cache = {'nvidia-tesla-t4': {'on_demand': 0.35, 'spot': 0.11}}
        GCPMonitor._gcp_gpu_price_cache_ts = time.time()

        with patch('monitors.gcp_monitor.billing_v1.CloudCatalogClient') as mock_client:
            monitor._refresh_gcp_gpu_prices()

        mock_client.assert_not_called()

    def test_stale_cache_triggers_refresh(self):
        """Should refresh when cache is older than 6 hours."""
        from monitors.gcp_monitor import GCPMonitor

        GCPMonitor._gcp_gpu_price_cache = {'old': {'on_demand': 1.0, 'spot': 0.3}}
        GCPMonitor._gcp_gpu_price_cache_ts = time.time() - 25000  # > 21600s

        mock_catalog_client = Mock()
        mock_catalog_client.list_skus.return_value = []

        monitor = _make_monitor()

        with patch('monitors.gcp_monitor.billing_v1.CloudCatalogClient', return_value=mock_catalog_client):
            monitor._refresh_gcp_gpu_prices()

        mock_catalog_client.list_skus.assert_called_once()

    def test_api_failure_keeps_empty_cache(self):
        """Should not crash on API failure."""
        from monitors.gcp_monitor import GCPMonitor

        GCPMonitor._gcp_gpu_price_cache = {}
        GCPMonitor._gcp_gpu_price_cache_ts = 0

        monitor = _make_monitor()

        with patch('google.cloud.billing_v1.CloudCatalogClient', side_effect=Exception('Auth error')):
            monitor._refresh_gcp_gpu_prices()

        assert GCPMonitor._gcp_gpu_price_cache == {}

    def test_skips_non_gpu_skus(self):
        """Should ignore SKUs that don't match GPU keywords."""
        from monitors.gcp_monitor import GCPMonitor

        sku_cpu = Mock()
        sku_cpu.description = 'N1 Predefined Instance Core running in Americas'
        sku_cpu.service_regions = ['us-central1']
        sku_cpu.pricing_info = []

        mock_catalog_client = Mock()
        mock_catalog_client.list_skus.return_value = [sku_cpu]

        monitor = _make_monitor()

        with patch('monitors.gcp_monitor.billing_v1.CloudCatalogClient', return_value=mock_catalog_client):
            monitor._refresh_gcp_gpu_prices()

        assert GCPMonitor._gcp_gpu_price_cache == {}

    def test_skips_wrong_region_skus(self):
        """Should skip SKUs not applicable to the configured region."""
        from monitors.gcp_monitor import GCPMonitor

        sku = Mock()
        sku.description = 'Nvidia Tesla T4 GPU running in Europe'
        sku.service_regions = ['europe-west1']

        rate = Mock()
        rate.unit_price = Mock()
        rate.unit_price.units = 0
        rate.unit_price.nanos = 350000000

        pricing_expr = Mock()
        pricing_expr.tiered_rates = [rate]

        pricing_info = Mock()
        pricing_info.pricing_expression = pricing_expr
        sku.pricing_info = [pricing_info]

        mock_catalog_client = Mock()
        mock_catalog_client.list_skus.return_value = [sku]

        monitor = _make_monitor()

        with patch('monitors.gcp_monitor.billing_v1.CloudCatalogClient', return_value=mock_catalog_client):
            monitor._refresh_gcp_gpu_prices()

        # Should not cache since region doesn't match
        assert GCPMonitor._gcp_gpu_price_cache == {}


# ---------------------------------------------------------------------------
# _GCP_GPU_PRICES property
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestGcpGpuPricesProperty:
    def setup_method(self):
        from monitors.gcp_monitor import GCPMonitor
        GCPMonitor._gcp_gpu_price_cache = {}
        GCPMonitor._gcp_gpu_price_cache_ts = 0

    def test_returns_fallback_when_cache_empty(self):
        """Should return fallback prices when API cache is empty."""
        from monitors.gcp_monitor import GCPMonitor

        GCPMonitor._gcp_gpu_price_cache = {}
        GCPMonitor._gcp_gpu_price_cache_ts = time.time()

        monitor = _make_monitor()
        prices = monitor._GCP_GPU_PRICES

        assert 'nvidia-tesla-t4' in prices
        assert prices['nvidia-tesla-t4'] == {'on_demand': 0.35, 'spot': 0.11}

    def test_merges_cache_over_fallback(self):
        """Should merge API data over fallback, with API data winning."""
        from monitors.gcp_monitor import GCPMonitor

        monitor = _make_monitor()
        GCPMonitor._gcp_gpu_price_cache = {
            'nvidia-tesla-t4': {'on_demand': 0.40, 'spot': 0.12},
        }
        GCPMonitor._gcp_gpu_price_cache_ts = time.time()

        with patch.object(monitor, '_refresh_gcp_gpu_prices'):
            prices = monitor._GCP_GPU_PRICES

        # API data should override fallback for T4
        assert prices['nvidia-tesla-t4']['on_demand'] == 0.40
        assert prices['nvidia-tesla-t4']['spot'] == 0.12

        # Fallback should still be present for other types
        assert 'nvidia-tesla-v100' in prices
        assert prices['nvidia-tesla-v100'] == {'on_demand': 2.48, 'spot': 0.74}

    def test_cache_fills_gaps_in_fallback(self):
        """Should add new GPU types from API not present in fallback."""
        from monitors.gcp_monitor import GCPMonitor

        monitor = _make_monitor()
        GCPMonitor._gcp_gpu_price_cache = {
            'nvidia-h100': {'on_demand': 10.0, 'spot': 3.0},
        }
        GCPMonitor._gcp_gpu_price_cache_ts = time.time()

        prices = monitor._GCP_GPU_PRICES

        assert 'nvidia-h100' in prices
        assert prices['nvidia-h100'] == {'on_demand': 10.0, 'spot': 3.0}


# ---------------------------------------------------------------------------
# _estimate_instance_cost
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestEstimateInstanceCostFull:
    def setup_method(self):
        from monitors.gcp_monitor import GCPMonitor
        GCPMonitor._gcp_gpu_price_cache = {}
        GCPMonitor._gcp_gpu_price_cache_ts = 0
    def test_known_instance_and_gpu(self):
        monitor = _make_monitor()
        instance = {
            'instance_type': 'n1-standard-8',
            'gpu_type': 'nvidia-tesla-v100',
            'gpu_count': 2,
            'preemptible': False,
        }
        cost = monitor._estimate_instance_cost(instance)
        expected = 0.38 + 2.48 * 2
        assert cost == round(expected, 2)

    def test_preemptible_applies_80pct_discount(self):
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

    def test_unknown_instance_type_uses_default(self):
        monitor = _make_monitor()
        instance = {
            'instance_type': 'a2-highgpu-1g',
            'gpu_type': 'nvidia-tesla-a100',
            'gpu_count': 1,
            'preemptible': False,
        }
        cost = monitor._estimate_instance_cost(instance)
        expected = 0.19 + 2.934  # default base + known gpu
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
        assert cost == 0.38

    def test_uses_api_prices_when_available(self):
        """Should use real API prices from _GCP_GPU_PRICES property when cached."""
        from monitors.gcp_monitor import GCPMonitor

        monitor = _make_monitor()
        GCPMonitor._gcp_gpu_price_cache = {
            'nvidia-tesla-v100': {'on_demand': 3.00, 'spot': 0.90},
        }
        GCPMonitor._gcp_gpu_price_cache_ts = time.time()

        instance = {
            'instance_type': 'n1-standard-8',
            'gpu_type': 'nvidia-tesla-v100',
            'gpu_count': 1,
            'preemptible': False,
        }
        cost = monitor._estimate_instance_cost(instance)
        expected = 0.38 + 3.00  # API price instead of fallback 2.48
        assert cost == round(expected, 2)


# ---------------------------------------------------------------------------
# get_spot_pricing
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestGetSpotPricingFull:
    def setup_method(self):
        from monitors.gcp_monitor import GCPMonitor
        GCPMonitor._gcp_gpu_price_cache = {}
        GCPMonitor._gcp_gpu_price_cache_ts = 0

    def test_returns_all_gpu_types(self):
        """Should return spot pricing for all known GPU types."""
        monitor = _make_monitor()
        spot_data = monitor.get_spot_pricing()

        assert len(spot_data) >= 6  # at least fallback types
        types = {d['instance_type'] for d in spot_data}
        assert 'nvidia-tesla-t4' in types
        assert 'nvidia-tesla-v100' in types

    def test_entry_has_correct_fields(self):
        """Should include all expected fields in spot data entries."""
        monitor = _make_monitor()
        spot_data = monitor.get_spot_pricing()

        for entry in spot_data:
            assert 'instance_type' in entry
            assert 'spot_price_hourly' in entry
            assert 'on_demand_price_hourly' in entry
            assert 'spot_monthly' in entry
            assert 'on_demand_monthly' in entry
            assert 'savings_pct' in entry
            assert entry['cloud_provider'] == 'gcp'

    def test_monthly_calculation(self):
        """Should calculate monthly as hourly * 730."""
        monitor = _make_monitor()
        spot_data = monitor.get_spot_pricing()

        for entry in spot_data:
            assert entry['spot_monthly'] == round(entry['spot_price_hourly'] * 730, 2)
            assert entry['on_demand_monthly'] == round(entry['on_demand_price_hourly'] * 730, 2)

    def test_savings_percentage(self):
        """Should correctly calculate savings percentage."""
        monitor = _make_monitor()
        spot_data = monitor.get_spot_pricing()

        for entry in spot_data:
            expected = round(
                (1 - entry['spot_price_hourly'] / entry['on_demand_price_hourly']) * 100, 1
            )
            assert entry['savings_pct'] == expected

    def test_uses_api_prices_when_cached(self):
        """Should use API-fetched prices when available."""
        from monitors.gcp_monitor import GCPMonitor

        monitor = _make_monitor()
        GCPMonitor._gcp_gpu_price_cache = {
            'nvidia-tesla-t4': {'on_demand': 0.40, 'spot': 0.12},
        }
        GCPMonitor._gcp_gpu_price_cache_ts = time.time()

        spot_data = monitor.get_spot_pricing()

        t4_entry = next(d for d in spot_data if d['instance_type'] == 'nvidia-tesla-t4')
        assert t4_entry['on_demand_price_hourly'] == 0.40
        assert t4_entry['spot_price_hourly'] == 0.12


# ---------------------------------------------------------------------------
# get_spot_candidates
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestGetSpotCandidatesFull:
    def _sample_spot_pricing(self):
        return [{
            'instance_type': 'nvidia-tesla-v100',
            'spot_price_hourly': 0.74,
            'on_demand_price_hourly': 2.48,
            'spot_monthly': 540.20,
            'on_demand_monthly': 1810.40,
            'savings_pct': 70.2,
        }]

    def test_fault_tolerant_label_is_candidate(self):
        instances = [{
            'instance_id': 'gpu-vm-1',
            'instance_type': 'n1-standard-8',
            'state': 'RUNNING',
            'gpu_type': 'nvidia-tesla-v100',
            'gpu_count': 1,
            'preemptible': False,
            'tags': {'fault_tolerant': 'true'},
        }]
        monitor = _make_monitor()
        candidates = monitor.get_spot_candidates(instances, self._sample_spot_pricing())

        assert len(candidates) == 1
        assert candidates[0]['recommendation_type'] == 'spot'
        assert 'fault-tolerant' in candidates[0]['source_data']['reason']

    def test_dev_environment_label_is_candidate(self):
        instances = [{
            'instance_id': 'dev-vm',
            'instance_type': 'n1-standard-8',
            'state': 'RUNNING',
            'gpu_type': 'nvidia-tesla-v100',
            'gpu_count': 1,
            'preemptible': False,
            'tags': {'environment': 'dev'},
        }]
        monitor = _make_monitor()
        candidates = monitor.get_spot_candidates(instances, self._sample_spot_pricing())

        assert len(candidates) == 1
        assert 'Environment: dev' in candidates[0]['source_data']['reason']

    def test_training_label_is_candidate(self):
        instances = [{
            'instance_id': 'train-vm',
            'instance_type': 'n1-standard-8',
            'state': 'RUNNING',
            'gpu_type': 'nvidia-tesla-v100',
            'gpu_count': 1,
            'preemptible': False,
            'tags': {'workload': 'training'},
        }]
        monitor = _make_monitor()
        candidates = monitor.get_spot_candidates(instances, self._sample_spot_pricing())

        assert len(candidates) == 1
        assert 'non-production' in candidates[0]['source_data']['reason']

    def test_preemptible_instances_excluded(self):
        """Should skip instances already running as preemptible."""
        instances = [{
            'instance_id': 'preempt-vm',
            'instance_type': 'n1-standard-8',
            'state': 'RUNNING',
            'gpu_type': 'nvidia-tesla-v100',
            'gpu_count': 1,
            'preemptible': True,
            'tags': {'environment': 'dev'},
        }]
        monitor = _make_monitor()
        candidates = monitor.get_spot_candidates(instances, self._sample_spot_pricing())
        assert candidates == []

    def test_non_running_instances_excluded(self):
        instances = [{
            'instance_id': 'stopped-vm',
            'instance_type': 'n1-standard-8',
            'state': 'TERMINATED',
            'gpu_type': 'nvidia-tesla-v100',
            'gpu_count': 1,
            'preemptible': False,
            'tags': {'environment': 'dev'},
        }]
        monitor = _make_monitor()
        candidates = monitor.get_spot_candidates(instances, self._sample_spot_pricing())
        assert candidates == []

    def test_no_matching_gpu_type_excluded(self):
        """Should skip instances whose gpu_type has no pricing data."""
        instances = [{
            'instance_id': 'vm',
            'instance_type': 'n1-standard-8',
            'state': 'RUNNING',
            'gpu_type': 'nvidia-tesla-h100',
            'gpu_count': 1,
            'preemptible': False,
            'tags': {'environment': 'dev'},
        }]
        monitor = _make_monitor()
        candidates = monitor.get_spot_candidates(instances, self._sample_spot_pricing())
        assert candidates == []

    def test_no_matching_labels_excluded(self):
        instances = [{
            'instance_id': 'prod-vm',
            'instance_type': 'n1-standard-8',
            'state': 'RUNNING',
            'gpu_type': 'nvidia-tesla-v100',
            'gpu_count': 1,
            'preemptible': False,
            'tags': {'environment': 'production', 'team': 'ml'},
        }]
        monitor = _make_monitor()
        candidates = monitor.get_spot_candidates(instances, self._sample_spot_pricing())
        assert candidates == []

    def test_savings_scale_with_gpu_count(self):
        """Should multiply pricing by GPU count for multi-GPU instances."""
        instances = [{
            'instance_id': 'multi-gpu',
            'instance_type': 'n1-standard-16',
            'state': 'RUNNING',
            'gpu_type': 'nvidia-tesla-v100',
            'gpu_count': 4,
            'preemptible': False,
            'tags': {'environment': 'dev'},
        }]
        monitor = _make_monitor()
        candidates = monitor.get_spot_candidates(instances, self._sample_spot_pricing())

        assert len(candidates) == 1
        c = candidates[0]
        assert c['current_monthly_cost'] == round(1810.40 * 4, 2)
        assert c['recommended_monthly_cost'] == round(540.20 * 4, 2)

    def test_instance_type_includes_gpu_info(self):
        """Should include GPU type and count in the candidate instance_type field."""
        instances = [{
            'instance_id': 'vm1',
            'instance_type': 'n1-standard-8',
            'state': 'RUNNING',
            'gpu_type': 'nvidia-tesla-v100',
            'gpu_count': 2,
            'preemptible': False,
            'tags': {'environment': 'test'},
        }]
        monitor = _make_monitor()
        candidates = monitor.get_spot_candidates(instances, self._sample_spot_pricing())

        assert '2x nvidia-tesla-v100' in candidates[0]['instance_type']
        assert 'n1-standard-8' in candidates[0]['instance_type']


# ---------------------------------------------------------------------------
# BigQuery billing integration
# ---------------------------------------------------------------------------

def _make_monitor_with_bq(bq_client=None, billing_dataset='billing_dataset.gcp_billing_export'):
    """Create a GCPMonitor with BigQuery attributes pre-set."""
    monitor = _make_monitor()
    monitor.bq_client = bq_client or Mock()
    monitor.billing_dataset = billing_dataset
    return monitor


def _make_bq_row(service_desc='Compute Engine', sku_desc='Nvidia Tesla V100 GPU',
                 usage_start_time=None, cost=12.50, currency='USD',
                 labels=None, project_id='test-project', region='us-central1'):
    """Create a mock BigQuery result row."""
    row = Mock()
    row.service_description = service_desc
    row.sku_description = sku_desc
    row.usage_start_time = usage_start_time or datetime(2026, 3, 1, 10, 0, 0)
    row.cost = cost
    row.currency = currency
    row.labels = labels  # list of dicts like [{'key': 'team', 'value': 'ml'}]
    row.project_id = project_id
    row.region = region
    return row


@pytest.mark.unit
class TestGetCostDataFromBigQuery:
    """Tests for get_cost_data_from_bigquery()."""

    def test_returns_cost_entries_from_bq(self):
        """Should return correctly shaped cost entries from BigQuery rows."""
        rows = [
            _make_bq_row(cost=10.0, labels=[{'key': 'team', 'value': 'ml'}]),
            _make_bq_row(cost=5.25, labels=None),
        ]

        bq_client = Mock()
        bq_client.query.return_value.result.return_value = rows
        monitor = _make_monitor_with_bq(bq_client=bq_client)

        result = monitor.get_cost_data_from_bigquery()

        assert len(result) == 2
        assert result[0]['cost'] == 10.0
        assert result[0]['service'] == 'Compute Engine'
        assert result[0]['cloud_provider'] == 'gcp'
        assert result[0]['labels'] == {'team': 'ml'}
        assert result[1]['labels'] == {}

    def test_uses_parameterized_query(self):
        """Should pass project_id as a query parameter."""
        bq_client = Mock()
        bq_client.query.return_value.result.return_value = []
        monitor = _make_monitor_with_bq(bq_client=bq_client)
        monitor.project_id = 'my-project'

        monitor.get_cost_data_from_bigquery()

        call_args = bq_client.query.call_args
        query_str = call_args[0][0]
        assert 'billing_dataset.gcp_billing_export' in query_str
        assert '%GPU%' in query_str
        # Check that the query uses parameterized format (@project_id)
        assert '@project_id' in query_str
        # Check job_config was passed
        assert 'job_config' in call_args[1]

    def test_raises_when_no_bq_client(self):
        """Should raise RuntimeError when BigQuery is not configured."""
        monitor = _make_monitor()
        monitor.bq_client = None
        monitor.billing_dataset = None

        with pytest.raises(RuntimeError):
            monitor.get_cost_data_from_bigquery()

    def test_empty_results(self):
        """Should return empty list when no billing rows match."""
        bq_client = Mock()
        bq_client.query.return_value.result.return_value = []
        monitor = _make_monitor_with_bq(bq_client=bq_client)

        result = monitor.get_cost_data_from_bigquery()
        assert result == []

    def test_date_is_iso_formatted(self):
        """Should convert usage_start_time to ISO format string."""
        ts = datetime(2026, 3, 15, 14, 30, 0)
        rows = [_make_bq_row(usage_start_time=ts)]
        bq_client = Mock()
        bq_client.query.return_value.result.return_value = rows
        monitor = _make_monitor_with_bq(bq_client=bq_client)

        result = monitor.get_cost_data_from_bigquery()
        assert result[0]['date'] == '2026-03-15T14:30:00'


# ---------------------------------------------------------------------------
# get_cost_data fallback logic
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestGetCostDataFallback:
    """Tests for BigQuery-first / estimation-fallback in get_cost_data()."""

    def test_uses_bigquery_when_configured(self):
        """Should call BigQuery path when bq_client and billing_dataset are set."""
        bq_client = Mock()
        bq_client.query.return_value.result.return_value = [
            _make_bq_row(cost=7.0, labels=None),
        ]
        monitor = _make_monitor_with_bq(bq_client=bq_client)

        result = monitor.get_cost_data()

        assert len(result) == 1
        assert result[0]['cost'] == 7.0
        bq_client.query.assert_called_once()

    def test_falls_back_on_bigquery_error(self):
        """Should fall back to estimation when BigQuery query raises."""
        bq_client = Mock()
        bq_client.query.side_effect = Exception('BQ quota exceeded')

        instance, zone = _make_gce_instance()
        zone_scope = Mock()
        zone_scope.instances = [instance]
        compute = Mock()
        compute.aggregated_list.return_value = [(zone, zone_scope)]

        monitor = _make_monitor_with_bq(bq_client=bq_client)
        monitor.compute_client = compute

        with patch.dict(os.environ, {
            'GCP_BILLING_ACCOUNT_ID': 'billing-123',
            'GCP_GPU_TYPES': 'nvidia-tesla-v100',
            'GCP_MACHINE_TYPES': '',
        }):
            result = monitor.get_cost_data()

        # Should have fallen back to estimation
        assert len(result) == 1
        assert result[0]['instance_id'] == 'us-central1-a/gpu-vm-1'
        assert result[0]['service'] == 'Compute Engine'

    def test_uses_estimation_when_bq_not_configured(self):
        """Should use estimation when bq_client is None."""
        instance, zone = _make_gce_instance()
        zone_scope = Mock()
        zone_scope.instances = [instance]
        compute = Mock()
        compute.aggregated_list.return_value = [(zone, zone_scope)]

        monitor = _make_monitor(compute_client=compute)
        monitor.bq_client = None
        monitor.billing_dataset = None

        with patch.dict(os.environ, {
            'GCP_BILLING_ACCOUNT_ID': 'billing-123',
            'GCP_GPU_TYPES': 'nvidia-tesla-v100',
            'GCP_MACHINE_TYPES': '',
        }):
            result = monitor.get_cost_data()

        assert len(result) == 1
        assert result[0]['service'] == 'Compute Engine'


# ---------------------------------------------------------------------------
# get_cost_by_labels_from_bigquery
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestGetCostByLabelsFromBigQuery:
    """Tests for get_cost_by_labels_from_bigquery()."""

    def test_returns_grouped_label_costs(self):
        """Should return label-grouped cost data from BigQuery."""
        row_ml = Mock()
        row_ml.label_value = 'ml'
        row_ml.total_cost = 120.50
        row_ml.record_count = 15

        row_infra = Mock()
        row_infra.label_value = 'infra'
        row_infra.total_cost = 45.00
        row_infra.record_count = 5

        bq_client = Mock()
        bq_client.query.return_value.result.return_value = [row_ml, row_infra]
        monitor = _make_monitor_with_bq(bq_client=bq_client)

        result = monitor.get_cost_by_labels_from_bigquery(label_key='team')

        assert len(result) == 2
        by_label = {r['label_value']: r for r in result}
        assert by_label['ml']['total_cost'] == 120.50
        assert by_label['ml']['record_count'] == 15
        assert by_label['ml']['label_key'] == 'team'
        assert by_label['infra']['total_cost'] == 45.00

    def test_passes_label_key_as_parameter(self):
        """Should pass label_key as a BigQuery query parameter."""
        bq_client = Mock()
        bq_client.query.return_value.result.return_value = []
        monitor = _make_monitor_with_bq(bq_client=bq_client)

        monitor.get_cost_by_labels_from_bigquery(label_key='environment')

        call_args = bq_client.query.call_args
        query_str = call_args[0][0]
        # Check that the query uses parameterized format (@label_key and @project_id)
        assert '@label_key' in query_str
        assert '@project_id' in query_str
        # Check job_config was passed
        assert 'job_config' in call_args[1]

    def test_raises_when_not_configured(self):
        """Should raise when BigQuery is not available."""
        monitor = _make_monitor()
        monitor.bq_client = None
        monitor.billing_dataset = None

        with pytest.raises(RuntimeError):
            monitor.get_cost_by_labels_from_bigquery(label_key='team')

    def test_empty_results(self):
        bq_client = Mock()
        bq_client.query.return_value.result.return_value = []
        monitor = _make_monitor_with_bq(bq_client=bq_client)

        result = monitor.get_cost_by_labels_from_bigquery(label_key='team')
        assert result == []

    def test_get_cost_by_labels_uses_bq_when_available(self):
        """get_cost_by_labels() should delegate to BigQuery when configured."""
        row = Mock()
        row.label_value = 'ml'
        row.total_cost = 50.0
        row.record_count = 3

        bq_client = Mock()
        bq_client.query.return_value.result.return_value = [row]
        monitor = _make_monitor_with_bq(bq_client=bq_client)

        result = monitor.get_cost_by_labels(label_key='team')

        assert len(result) == 1
        assert result[0]['label_value'] == 'ml'
        bq_client.query.assert_called_once()

    def test_get_cost_by_labels_falls_back_on_bq_error(self):
        """get_cost_by_labels() should fall back to estimation on BQ error."""
        bq_client = Mock()
        bq_client.query.side_effect = Exception('BQ error')

        monitor = _make_monitor_with_bq(bq_client=bq_client)

        with patch.object(monitor, 'get_gpu_instances', return_value=[]), \
             patch.object(monitor, '_estimate_instance_cost', return_value=1.0):
            result = monitor.get_cost_by_labels(label_key='team')

        assert result == []

    def test_get_cost_by_labels_no_label_key_skips_bq(self):
        """get_cost_by_labels(label_key=None) should skip BigQuery path."""
        bq_client = Mock()
        monitor = _make_monitor_with_bq(bq_client=bq_client)

        with patch.object(monitor, 'get_gpu_instances', return_value=[]):
            monitor.get_cost_by_labels(label_key=None)

        bq_client.query.assert_not_called()
