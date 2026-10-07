"""
Azure Monitor for GPU instances and cost tracking
"""

import os
import logging
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import psutil
import requests

from azure.identity import DefaultAzureCredential
from azure.mgmt.compute import ComputeManagementClient
from azure.mgmt.monitor import MonitorManagementClient
from azure.mgmt.consumption import ConsumptionManagementClient
from azure.mgmt.resource import ResourceManagementClient
from azure.mgmt.costmanagement import CostManagementClient

logger = logging.getLogger(__name__)

class AzureMonitor:
    def __init__(self, config=None):
        """Initialize Azure monitoring client.

        Args:
            config: Optional dict from YAML/env-backed config (monitoring.azure), e.g.
                subscription_id, resource_group. Falls back to environment variables.
        """
        config = config or {}
        self.subscription_id = config.get('subscription_id') or os.getenv('AZURE_SUBSCRIPTION_ID')
        self.resource_group = (config.get('resource_group') or os.getenv('AZURE_RESOURCE_GROUP') or '').strip() or None

        # On Azure VMs, DefaultAzureCredential picks up managed identity automatically.
        self.credential = DefaultAzureCredential(exclude_interactive_browser_credential=True)
        logger.info("Using DefaultAzureCredential for Azure authentication")
        
        # Initialize Azure clients
        self.compute_client = ComputeManagementClient(
            self.credential, 
            self.subscription_id
        )
        self.monitor_client = MonitorManagementClient(
            self.credential, 
            self.subscription_id
        )
        self.consumption_client = ConsumptionManagementClient(
            self.credential, 
            self.subscription_id
        )
        self.resource_client = ResourceManagementClient(
            self.credential, 
            self.subscription_id
        )
        self.cost_client = CostManagementClient(
            self.credential, 
            self.subscription_id
        )
        
        # Initialize NVML for GPU monitoring
        self.gpu_available = False
        try:
            import pynvml
            pynvml.nvmlInit()
            self.gpu_available = True
            self._pynvml = pynvml
            logger.info("NVML initialized successfully")
        except Exception as e:
            self._pynvml = None
            logger.warning(f"NVML not available: {e}")

    # Azure GPU VM size -> (gpu_count, marketing gpu_type). Fallback: (1, "NVIDIA GPU").
    _AZURE_GPU_SKU_INFO = {
        'Standard_NC6': (1, 'NVIDIA K80'),
        'Standard_NC12': (2, 'NVIDIA K80'),
        'Standard_NC24': (4, 'NVIDIA K80'),
        'Standard_NC6s_v2': (1, 'NVIDIA P100'),
        'Standard_NC12s_v2': (2, 'NVIDIA P100'),
        'Standard_NC24s_v2': (4, 'NVIDIA P100'),
        'Standard_NC6s_v3': (1, 'NVIDIA V100'),
        'Standard_NC12s_v3': (2, 'NVIDIA V100'),
        'Standard_NC24s_v3': (4, 'NVIDIA V100'),
        'Standard_NC4as_T4_v3': (1, 'NVIDIA T4'),
        'Standard_NC8as_T4_v3': (1, 'NVIDIA T4'),
        'Standard_NC16as_T4_v3': (1, 'NVIDIA T4'),
        'Standard_NC64as_T4_v3': (4, 'NVIDIA T4'),
        'Standard_NV6': (1, 'NVIDIA M60'),
        'Standard_NV12': (2, 'NVIDIA M60'),
        'Standard_NV24': (4, 'NVIDIA M60'),
        'Standard_NV6s_v2': (1, 'NVIDIA P40'),
        'Standard_NV12s_v2': (2, 'NVIDIA P40'),
        'Standard_NV24s_v2': (4, 'NVIDIA P40'),
        'Standard_NV12s_v3': (2, 'NVIDIA V100'),
        'Standard_NV24s_v3': (4, 'NVIDIA V100'),
        'Standard_NV48s_v3': (8, 'NVIDIA V100'),
        'Standard_NV6ads_A10_v5': (1, 'NVIDIA A10'),
        'Standard_NV12ads_A10_v5': (1, 'NVIDIA A10'),
        'Standard_NV18ads_A10_v5': (1, 'NVIDIA A10'),
        'Standard_NV36ads_A10_v5': (1, 'NVIDIA A10'),
        'Standard_NV36adms_A10_v5': (1, 'NVIDIA A10'),
        'Standard_NV72ads_A10_v5': (2, 'NVIDIA A10'),
        # ND A100 v4
        'Standard_ND96asr_v4': (8, 'NVIDIA A100'),
        'Standard_ND96amsr_A100_v4': (8, 'NVIDIA A100-80GB'),
        # NC A100 v4
        'Standard_NC24ads_A100_v4': (1, 'NVIDIA A100'),
        'Standard_NC48ads_A100_v4': (2, 'NVIDIA A100'),
        'Standard_NC96ads_A100_v4': (4, 'NVIDIA A100'),
        # ND H100 v5
        'Standard_ND96isr_H100_v5': (8, 'NVIDIA H100'),
        # ND MI300X v5 (AMD)
        'Standard_ND96isr_MI300X_v5': (8, 'AMD MI300X'),
    }

    @staticmethod
    def _gpu_size_filter_tokens() -> List[str]:
        raw = os.getenv('AZURE_VM_SIZES', '')
        tokens = [s.strip() for s in raw.split(',') if s.strip()]
        if tokens:
            return tokens
        return ['Standard_NC', 'Standard_NV', 'Standard_ND', 'Standard_NG']

    def _vm_size_matches_gpu_filter(self, vm_size: str) -> bool:
        for t in self._gpu_size_filter_tokens():
            if t == vm_size or t in vm_size:
                return True
        return False

    def _azure_gpu_info_for_size(self, vm_size: str) -> Tuple[int, str]:
        if vm_size in self._AZURE_GPU_SKU_INFO:
            return self._AZURE_GPU_SKU_INFO[vm_size]
        return (1, 'NVIDIA GPU')

    @staticmethod
    def _imds_vm_name() -> Optional[str]:
        try:
            r = requests.get(
                'http://169.254.169.254/metadata/instance/compute/name?api-version=2021-02-01&format=text',
                headers={'Metadata': 'true'},
                timeout=2,
            )
            if r.status_code == 200 and r.text.strip():
                return r.text.strip()
        except Exception:
            pass
        return None

    def _read_power_state(self, resource_group: str, vm_name: str) -> str:
        try:
            detail = self.compute_client.virtual_machines.get(
                resource_group, vm_name, expand='instanceView'
            )
            for st in detail.instance_view.statuses or []:
                if st.code and st.code.startswith('PowerState/'):
                    return st.code.split('/')[-1].lower()
        except Exception as e:
            logger.debug(f"Power state for {vm_name}: {e}")
        return 'unknown'

    def get_gpu_instances(self) -> List[Dict[str, Any]]:
        """Get all Azure VM instances with GPU capabilities"""
        instances = []

        try:
            if self.resource_group:
                vms = self.compute_client.virtual_machines.list(self.resource_group)
            else:
                vms = self.compute_client.virtual_machines.list_all()

            for vm in vms:
                vm_size = vm.hardware_profile.vm_size
                if not self._vm_size_matches_gpu_filter(vm_size):
                    continue

                rg = vm.id.split('/')[4]
                gpu_count, gpu_type = self._azure_gpu_info_for_size(vm_size)
                power = self._read_power_state(rg, vm.name) if rg else 'unknown'
                zones = vm.zones or []
                az = zones[0] if zones else None

                instance_data = {
                    'instance_id': f"{rg}/{vm.name}",
                    'cloud_id': vm.name,
                    'instance_type': vm_size,
                    'state': power,
                    'launch_time': vm.time_created.isoformat() if vm.time_created else None,
                    'location': vm.location,
                    'resource_group': rg,
                    'tags': vm.tags or {},
                    'created_at': vm.time_created.isoformat() if vm.time_created else None,
                    'cloud_provider': 'azure',
                    'region': vm.location,
                    'availability_zone': az,
                    'gpu_count': gpu_count,
                    'gpu_type': gpu_type,
                }
                instances.append(instance_data)

            logger.info(f"Found {len(instances)} GPU-enabled Azure VMs")

        except Exception as e:
            logger.error(f"Error getting Azure GPU instances: {e}")

        return instances
    
    def get_gpu_utilization(self, instances: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Get GPU utilization metrics for Azure instances"""
        metrics = []

        if not self.gpu_available or not self._pynvml:
            logger.warning("GPU monitoring not available - NVML not initialized")
            return metrics

        pynvml = self._pynvml
        try:
            device_count = pynvml.nvmlDeviceGetCount()
            cpu_util = float(psutil.cpu_percent(interval=0.08))

            for i in range(device_count):
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)

                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_util = utilization.gpu
                    memory_util = utilization.memory

                    memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

                    temperature = None
                    try:
                        temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    except Exception:
                        pass

                    power_usage = None
                    try:
                        power_usage = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                    except Exception:
                        pass

                    # Match this host to an Azure VM row (hostname / IMDS name)
                    instance = None
                    hostname = os.uname().nodename
                    imds_name = self._imds_vm_name()
                    for inst in instances:
                        iid = str(inst.get('instance_id', ''))
                        if iid == hostname or iid in hostname or (imds_name and iid == imds_name):
                            instance = inst
                            break
                    if instance is None and len(instances) == 1:
                        instance = instances[0]

                    if instance:
                        metric_data = {
                            'instance_id': instance['instance_id'],
                            'gpu_index': i,
                            'gpu_utilization': round(gpu_util, 2),
                            'cpu_utilization': round(cpu_util, 2),
                            'memory_utilization': round(memory_util, 2),
                            'memory_used_mb': round(memory_info.used / 1024 / 1024, 2),
                            'memory_total_mb': round(memory_info.total / 1024 / 1024, 2),
                            'temperature_c': temperature,
                            'power_usage_w': power_usage,
                            'is_idle': gpu_util < 10 and memory_util < 10,
                            'timestamp': datetime.utcnow().isoformat(),
                            'cloud_provider': 'azure',
                            'source': 'nvml',
                        }
                        metrics.append(metric_data)
                except Exception as e:
                    logger.debug(f"Failed to read GPU {i}: {e}")

            logger.info(f"Collected GPU metrics for {len(metrics)} devices")

        except Exception as e:
            logger.error(f"Error getting GPU utilization: {e}")

        return metrics
    
    def get_cost_data(self) -> List[Dict[str, Any]]:
        """Get cost data from Azure Cost Management"""
        cost_data = []
        
        try:
            # Get cost data for the last 30 days
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=30)
            
            # Query cost data
            scope = f"/subscriptions/{self.subscription_id}"
            
            # Get usage details
            usage_details = self.consumption_client.usage_details.list(
                scope=scope,
                filter=f"properties/usageStart ge '{start_date.isoformat()}' and properties/usageEnd le '{end_date.isoformat()}'"
            )
            
            for usage in usage_details:
                # Filter for compute resources
                if usage.instance_name and 'Standard_N' in usage.instance_name:
                    cost_entry = {
                        'instance_id': usage.instance_name,
                        'service': usage.product,
                        'cost': float(usage.cost),
                        'currency': usage.billing_currency,
                        'date': usage.date.isoformat(),
                        'resource_group': usage.resource_group,
                        'location': usage.location,
                        'cloud_provider': 'azure'
                    }
                    cost_data.append(cost_entry)
            
            logger.info(f"Collected cost data for {len(cost_data)} Azure resources")
            
        except Exception as e:
            logger.error(f"Error getting Azure cost data: {e}")
        
        return cost_data
    
    def get_instance_metrics(self, instance_id: str, resource_group: str) -> Dict[str, Any]:
        """Get detailed metrics for a specific Azure VM instance"""
        try:
            # Get VM metrics from Azure Monitor
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=1)
            
            # Get CPU percentage metric
            cpu_metric = self.monitor_client.metrics.list(
                resource_uri=f"/subscriptions/{self.subscription_id}/resourceGroups/{resource_group}/providers/Microsoft.Compute/virtualMachines/{instance_id}",
                timespan=f"{start_time.isoformat()}/{end_time.isoformat()}",
                interval="PT5M",
                metricnames="Percentage CPU"
            )
            
            # Get memory metrics
            memory_metric = self.monitor_client.metrics.list(
                resource_uri=f"/subscriptions/{self.subscription_id}/resourceGroups/{resource_group}/providers/Microsoft.Compute/virtualMachines/{instance_id}",
                timespan=f"{start_time.isoformat()}/{end_time.isoformat()}",
                interval="PT5M",
                metricnames="Available Memory Bytes"
            )
            
            metrics_data = {
                'instance_id': instance_id,
                'cpu_metrics': [],
                'memory_metrics': [],
                'timestamp': datetime.utcnow().isoformat()
            }
            
            # Process CPU metrics
            for metric in cpu_metric.value:
                for timeseries in metric.timeseries:
                    for data in timeseries.data:
                        if data.average is not None:
                            metrics_data['cpu_metrics'].append({
                                'timestamp': data.time_stamp.isoformat(),
                                'value': round(data.average, 2)
                            })
            
            # Process memory metrics
            for metric in memory_metric.value:
                for timeseries in metric.timeseries:
                    for data in timeseries.data:
                        if data.average is not None:
                            metrics_data['memory_metrics'].append({
                                'timestamp': data.time_stamp.isoformat(),
                                'value': round(data.average / 1024 / 1024, 2)  # Convert to MB
                            })
            
            return metrics_data
            
        except Exception as e:
            logger.error(f"Error getting metrics for instance {instance_id}: {e}")
            return {}
    
    def get_resource_tags(self, resource_id: str) -> Dict[str, str]:
        """Get tags for an Azure resource"""
        try:
            resource = self.resource_client.resources.get_by_id(resource_id, "2021-04-01")
            return resource.tags or {}
        except Exception as e:
            logger.error(f"Error getting tags for resource {resource_id}: {e}")
            return {}
    
    def get_cost_by_tags(self, tag_key: str = None) -> List[Dict[str, Any]]:
        """Get cost breakdown grouped by a specific tag key.
        Uses the Consumption API usage details which include resource tags.
        """
        try:
            scope = f"/subscriptions/{self.subscription_id}"
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=30)

            usage_details = self.consumption_client.usage_details.list(
                scope=scope,
                filter=f"properties/usageStart ge '{start_date.isoformat()}' and properties/usageEnd le '{end_date.isoformat()}'"
            )

            # Group costs by tag value
            tag_costs = {}
            for usage in usage_details:
                if not (usage.instance_name and 'Standard_N' in usage.instance_name):
                    continue
                resource_tags = getattr(usage, 'tags', {}) or {}
                tag_value = resource_tags.get(tag_key, 'untagged') if tag_key else 'all'
                if tag_value not in tag_costs:
                    tag_costs[tag_value] = {'total_cost': 0.0, 'record_count': 0}
                tag_costs[tag_value]['total_cost'] += float(usage.cost)
                tag_costs[tag_value]['record_count'] += 1

            result = [
                {'tag_key': tag_key or 'all', 'tag_value': tv, 'total_cost': round(d['total_cost'], 2), 'record_count': d['record_count']}
                for tv, d in tag_costs.items()
            ]
            logger.info(f"Cost by tag '{tag_key}': {len(result)} groups")
            return result
        except Exception as e:
            logger.error(f"Error getting cost by tags: {e}")
            return []

    # ── Spot Instance Intelligence ─────────────────────────────────────

    def get_spot_pricing(self) -> List[Dict[str, Any]]:
        """Get Azure Spot VM pricing for GPU VM sizes.
        Azure doesn't have a direct spot price history API like AWS.
        Uses the eviction rate and retail pricing API as approximation.
        """
        spot_data = []
        try:
            gpu_vm_sizes = os.getenv('AZURE_VM_SIZES', '').split(',')
            if not gpu_vm_sizes or gpu_vm_sizes == ['']:
                return spot_data

            for vm_size in gpu_vm_sizes:
                vm_size = vm_size.strip()
                if not vm_size:
                    continue
                try:
                    on_demand_hourly = self._get_azure_on_demand_price(vm_size)
                    spot_hourly = self._get_azure_spot_price(vm_size)
                    if on_demand_hourly > 0 and spot_hourly > 0:
                        savings_pct = round((1 - spot_hourly / on_demand_hourly) * 100, 1)
                        # Check if we used real API data or fallback
                        cached = self._azure_price_cache.get(vm_size)
                        source = 'Azure Retail Prices API' if (cached and cached['spot'] > 0) else 'estimated'
                        spot_data.append({
                            'instance_type': vm_size,
                            'availability_zone': self.resource_group or 'default',
                            'spot_price_hourly': round(spot_hourly, 4),
                            'on_demand_price_hourly': on_demand_hourly,
                            'spot_monthly': round(spot_hourly * 730, 2),
                            'on_demand_monthly': round(on_demand_hourly * 730, 2),
                            'savings_pct': savings_pct,
                            'cloud_provider': 'azure',
                            'source': source
                        })
                except Exception as e:
                    logger.debug(f"Spot pricing error for {vm_size}: {e}")

            logger.info(f"Collected Azure spot pricing for {len(spot_data)} VM sizes")
        except Exception as e:
            logger.warning(f"Error fetching Azure spot pricing: {e}")

        return spot_data

    # Fallback prices — used ONLY if the Retail Prices API call fails
    _AZURE_GPU_PRICES_FALLBACK = {
        'Standard_NC6': 0.90, 'Standard_NC12': 1.80, 'Standard_NC24': 3.60,
        'Standard_NC6s_v3': 3.06, 'Standard_NC12s_v3': 6.12, 'Standard_NC24s_v3': 12.24,
        'Standard_ND6s': 2.07, 'Standard_ND12s': 4.14, 'Standard_ND24s': 8.28,
        'Standard_NV6': 1.14, 'Standard_NV12': 2.28, 'Standard_NV24': 4.56,
        'Standard_NC4as_T4_v3': 0.526, 'Standard_NC8as_T4_v3': 0.752,
        'Standard_NC16as_T4_v3': 1.204, 'Standard_NC64as_T4_v3': 4.352,
    }

    # Cache for API-fetched prices: {vm_size: {'on_demand': float, 'spot': float}}
    _azure_price_cache = {}
    _azure_price_cache_ts = 0

    def _refresh_azure_prices(self):
        """Fetch real-time pricing from the Azure Retail Prices API (public, no auth).
        https://learn.microsoft.com/en-us/rest/api/cost-management/retail-prices/azure-retail-prices
        """
        import time
        # Cache for 6 hours
        if self._azure_price_cache and (time.time() - self._azure_price_cache_ts) < 21600:
            return

        try:
            import requests
            region = getattr(self, 'location', None) or os.getenv('AZURE_LOCATION', 'eastus')
            # Query for Virtual Machines, Consumption (pay-as-you-go), in the target region
            base_url = 'https://prices.azure.com/api/retail/prices'
            odata_filter = (
                f"serviceFamily eq 'Compute' and serviceName eq 'Virtual Machines' "
                f"and armRegionName eq '{region}' and priceType eq 'Consumption' "
                f"and contains(armSkuName, 'Standard_N')"
            )

            prices = {}
            next_url = f"{base_url}?$filter={odata_filter}"

            while next_url:
                resp = requests.get(next_url, timeout=15)
                resp.raise_for_status()
                data = resp.json()

                for item in data.get('Items', []):
                    sku = item.get('armSkuName', '')
                    unit_price = item.get('unitPrice', 0)
                    sku_name = item.get('skuName', '')
                    is_spot = 'Spot' in item.get('meterName', '') or 'Spot' in sku_name

                    if sku and unit_price > 0:
                        if sku not in prices:
                            prices[sku] = {'on_demand': 0, 'spot': 0}
                        if is_spot:
                            prices[sku]['spot'] = unit_price
                        else:
                            # Take the base (non-low-priority) price
                            if 'Low Priority' not in sku_name:
                                prices[sku]['on_demand'] = unit_price

                next_url = data.get('NextPageLink')
                # Safety: limit pages to avoid runaway
                if len(prices) > 200:
                    break

            if prices:
                AzureMonitor._azure_price_cache = prices
                AzureMonitor._azure_price_cache_ts = time.time()
                logger.info(f"Fetched {len(prices)} Azure VM prices from Retail Prices API ({region})")
            else:
                logger.warning("Azure Retail Prices API returned no GPU VM prices — using fallback")

        except Exception as e:
            logger.warning(f"Azure Retail Prices API call failed, using fallback: {e}")

    def _get_azure_on_demand_price(self, vm_size: str) -> float:
        """Get on-demand hourly price for a VM size. Uses real API data with fallback."""
        self._refresh_azure_prices()
        cached = self._azure_price_cache.get(vm_size)
        if cached and cached['on_demand'] > 0:
            return cached['on_demand']
        return self._AZURE_GPU_PRICES_FALLBACK.get(vm_size, 0)

    def _get_azure_spot_price(self, vm_size: str) -> float:
        """Get spot hourly price for a VM size. Uses real API data with estimate fallback."""
        self._refresh_azure_prices()
        cached = self._azure_price_cache.get(vm_size)
        if cached and cached['spot'] > 0:
            return cached['spot']
        # Fallback: estimate spot at 30% of on-demand
        on_demand = self._get_azure_on_demand_price(vm_size)
        return round(on_demand * 0.3, 4) if on_demand > 0 else 0

    def get_spot_candidates(self, instances: List[Dict[str, Any]], spot_pricing: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify Azure VMs that are good Spot VM candidates based on tags."""
        candidates = []
        best_spot = {}
        for sp in spot_pricing:
            itype = sp['instance_type']
            if itype not in best_spot or sp['spot_price_hourly'] < best_spot[itype]['spot_price_hourly']:
                best_spot[itype] = sp

        spot_tags = {'batch', 'training', 'dev', 'test', 'experiment', 'ci', 'staging'}

        for inst in instances:
            # get_gpu_instances uses Azure power state (e.g. running); older paths used provisioning state.
            state_norm = (inst.get('state') or '').lower()
            if state_norm not in ('running', 'succeeded'):
                continue
            itype = inst.get('instance_type', '')
            if itype not in best_spot:
                continue

            tags = inst.get('tags', {})
            tag_values = {v.lower() for v in tags.values()}

            is_candidate = False
            reason = ''
            if tags.get('fault_tolerant', '').lower() in ('true', 'yes'):
                is_candidate, reason = True, 'Tagged as fault-tolerant'
            elif tags.get('environment', '').lower() in ('dev', 'staging', 'test'):
                is_candidate, reason = True, f"Environment: {tags.get('environment')}"
            elif tag_values & spot_tags:
                is_candidate, reason = True, 'Tags suggest non-production workload'

            if is_candidate:
                sp = best_spot[itype]
                candidates.append({
                    'instance_id': inst['instance_id'],
                    'instance_type': itype,
                    'recommendation_type': 'spot',
                    'term': 'on_demand',
                    'payment_option': 'spot',
                    'current_monthly_cost': sp['on_demand_monthly'],
                    'recommended_monthly_cost': sp['spot_monthly'],
                    'potential_monthly_savings': round(sp['on_demand_monthly'] - sp['spot_monthly'], 2),
                    'potential_savings_pct': sp['savings_pct'],
                    'cloud_provider': 'azure',
                    'source_data': {'reason': reason, 'instance_id': inst['instance_id'], 'tags': tags}
                })

        logger.info(f"Found {len(candidates)} Azure spot candidates")
        return candidates

    # ── Action Methods ──────────────────────────────────────────────────
    # instance_id format: "resourceGroup/vmName"

    # Timeout for async operations (seconds). Prevents indefinite hangs.
    _ACTION_TIMEOUT = 600  # 10 minutes

    def _parse_instance_id(self, instance_id: str):
        """Parse 'resourceGroup/vmName' format and validate."""
        parts = instance_id.split('/')
        if len(parts) != 2 or not all(parts):
            raise ValueError(
                f"Invalid Azure instance ID: {instance_id}. Expected: resourceGroup/vmName"
            )
        return parts[0], parts[1]

    def stop_instance(self, instance_id):
        """Stop (deallocate) an Azure VM."""
        rg, vm_name = self._parse_instance_id(instance_id)
        poller = self.compute_client.virtual_machines.begin_deallocate(rg, vm_name)
        poller.result(timeout=self._ACTION_TIMEOUT)
        logger.info(f"Deallocated Azure VM {vm_name} in {rg}")
        return {'instance_id': instance_id, 'action': 'deallocate'}

    def start_instance(self, instance_id):
        """Start an Azure VM."""
        rg, vm_name = self._parse_instance_id(instance_id)
        poller = self.compute_client.virtual_machines.begin_start(rg, vm_name)
        poller.result(timeout=self._ACTION_TIMEOUT)
        logger.info(f"Started Azure VM {vm_name} in {rg}")
        return {'instance_id': instance_id, 'action': 'start'}

    def resize_instance(self, instance_id, target_type):
        """Resize an Azure VM (deallocate -> update -> start)."""
        rg, vm_name = self._parse_instance_id(instance_id)
        self.compute_client.virtual_machines.begin_deallocate(rg, vm_name).result(timeout=self._ACTION_TIMEOUT)
        self.compute_client.virtual_machines.begin_update(
            rg, vm_name, {'hardware_profile': {'vm_size': target_type}}
        ).result(timeout=self._ACTION_TIMEOUT)
        self.compute_client.virtual_machines.begin_start(rg, vm_name).result(timeout=self._ACTION_TIMEOUT)
        logger.info(f"Resized Azure VM {vm_name} to {target_type}")
        return {'instance_id': instance_id, 'new_type': target_type}

    def restart_instance(self, instance_id):
        """Restart an Azure VM."""
        rg, vm_name = self._parse_instance_id(instance_id)
        self.compute_client.virtual_machines.begin_restart(rg, vm_name).result(timeout=self._ACTION_TIMEOUT)
        logger.info(f"Restarted Azure VM {vm_name}")
        return {'instance_id': instance_id, 'action': 'restart'}

    def terminate_instance(self, instance_id):
        """Delete an Azure VM."""
        rg, vm_name = self._parse_instance_id(instance_id)
        self.compute_client.virtual_machines.begin_delete(rg, vm_name).result(timeout=self._ACTION_TIMEOUT)
        logger.info(f"Deleted Azure VM {vm_name}")
        return {'instance_id': instance_id, 'action': 'delete'}
