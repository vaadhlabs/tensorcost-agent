"""
Google Cloud Monitor for GPU instances and cost tracking
"""

import os
import logging
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any
import psutil

from google.cloud import compute_v1
from google.cloud import monitoring_v3
from google.cloud import billing_v1
from google.cloud import resourcemanager_v3
from google.auth import default
from google.oauth2 import service_account

logger = logging.getLogger(__name__)

class GCPMonitor:
    def __init__(self, config=None):
        """Initialize Google Cloud monitoring client.

        Args:
            config: Optional dict from YAML/env-backed config (monitoring.gcp),
                e.g. project_id. Falls back to environment variables.
        """
        config = config or {}
        self.project_id = (
            config.get("project_id")
            or os.getenv("GCP_PROJECT_ID")
            or ""
        ).strip() or None
        self.region = (
            config.get("region") or os.getenv("GCP_REGION", "us-central1")
        ).strip() or "us-central1"
        self.zone = (
            config.get("zone") or os.getenv("GCP_ZONE", "us-central1-a")
        ).strip() or "us-central1-a"
        # When true (or GCP_DISCOVER_LABELED=true), also surface MIG VMs that
        # carry instance-type=gpu-monitoring even without guest accelerators.
        labeled_flag = (
            str(config.get("discover_labeled", "")).lower()
            or os.getenv("GCP_DISCOVER_LABELED", "false").lower()
        )
        self.discover_labeled = labeled_flag in ("true", "1", "yes")
        self.discover_label_key = os.getenv(
            "GCP_DISCOVER_LABEL_KEY", "instance-type"
        )
        self.discover_label_value = os.getenv(
            "GCP_DISCOVER_LABEL_VALUE", "gpu-monitoring"
        )

        if not self.project_id:
            raise ValueError(
                "GCP_PROJECT_ID is required when GCP_ENABLED=true "
                "(or pass monitoring.gcp.project_id)"
            )

        # Initialize authentication
        self._init_auth()
        
        # Initialize Google Cloud clients
        self.compute_client = compute_v1.InstancesClient()
        self.monitoring_client = monitoring_v3.MetricServiceClient()
        self.billing_client = billing_v1.CloudBillingClient()
        self.resource_client = resourcemanager_v3.ProjectsClient()
        
        # Initialize BigQuery client for real billing data (optional)
        self.bq_client = None
        self.billing_dataset = os.getenv('GCP_BILLING_DATASET')
        if self.billing_dataset:
            try:
                from google.cloud import bigquery
                self.bq_client = bigquery.Client(project=self.project_id)
                logger.info(f"BigQuery client initialized for billing dataset: {self.billing_dataset}")
            except Exception as e:
                logger.warning(f"Failed to initialize BigQuery client: {e}")

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
    
    def _init_auth(self):
        """Initialize Google Cloud authentication"""
        try:
            # Try service account key first
            service_account_path = os.getenv('GCP_SERVICE_ACCOUNT_KEY_PATH')
            if service_account_path and os.path.exists(service_account_path):
                self.credentials = service_account.Credentials.from_service_account_file(
                    service_account_path
                )
                logger.info("Using service account key for authentication")
            else:
                # Use Application Default Credentials
                self.credentials, _ = default()
                logger.info("Using Application Default Credentials")
        except Exception as e:
            logger.error(f"Error initializing GCP authentication: {e}")
            raise
    
    def get_gpu_instances(self) -> List[Dict[str, Any]]:
        """Get all GCE instances with GPU accelerators (and optionally labeled MIG VMs)."""
        instances = []
        
        try:
            # Get all instances in the project
            request = compute_v1.AggregatedListInstancesRequest(project=self.project_id)
            response = self.compute_client.aggregated_list(request=request)
            
            # Filter for GPU-enabled instances. Empty GCP_GPU_TYPES means
            # "any accelerator" — the previous `in ['']` check matched nothing.
            gpu_types = [
                t.strip()
                for t in os.getenv("GCP_GPU_TYPES", "").split(",")
                if t.strip()
            ]
            
            for zone, zone_instances in response:
                if zone_instances.instances:
                    for instance in zone_instances.instances:
                        gcp_labels = dict(instance.labels) if instance.labels else {}
                        has_gpu = False
                        gpu_count = 0
                        gpu_type = None
                        
                        if instance.guest_accelerators:
                            for accelerator in instance.guest_accelerators:
                                # API returns full URL or short name depending
                                # on client; match either form.
                                acc_type = accelerator.accelerator_type or ""
                                short = acc_type.rsplit("/", 1)[-1]
                                if not gpu_types or short in gpu_types or acc_type in gpu_types:
                                    has_gpu = True
                                    gpu_count += accelerator.accelerator_count or 0
                                    gpu_type = short or acc_type

                        labeled_match = (
                            self.discover_labeled
                            and gcp_labels.get(self.discover_label_key)
                            == self.discover_label_value
                        )

                        if has_gpu or labeled_match:
                            zone_name = zone.split('/')[-1]
                            instance_data = {
                                'instance_id': f"{zone_name}/{instance.name}",
                                'cloud_id': instance.name,
                                'instance_type': instance.machine_type.split('/')[-1],
                                'state': instance.status,
                                'location': zone.split('/')[-1],
                                'project_id': self.project_id,
                                'tags': gcp_labels,
                                'created_at': instance.creation_timestamp,
                                'cloud_provider': 'gcp',
                                'region': zone.split('/')[-1].rsplit('-', 1)[0],
                                'availability_zone': zone.split('/')[-1],
                                'gpu_count': gpu_count,
                                'gpu_type': gpu_type,
                                'preemptible': instance.scheduling.preemptible if instance.scheduling else False
                            }
                            instances.append(instance_data)
            
            logger.info(f"Found {len(instances)} GPU-enabled GCE instances")
            
        except Exception as e:
            logger.error(f"Error getting GCP GPU instances: {e}")
        
        return instances
    
    def get_gpu_utilization(self, instances: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Get GPU utilization metrics for GCP instances"""
        metrics = []

        if not self.gpu_available or not self._pynvml:
            logger.warning("GPU monitoring not available - NVML not initialized")
            return metrics

        pynvml = self._pynvml
        try:
            device_count = pynvml.nvmlDeviceGetCount()

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

                    # Match GPU to an instance — works when agent runs on the GPU host
                    instance = None
                    hostname = os.uname().nodename
                    for inst in instances:
                        if inst['instance_id'] == hostname or inst['instance_id'] in hostname:
                            instance = inst
                            break

                    if instance:
                        metric_data = {
                            'instance_id': instance['instance_id'],
                            'gpu_index': i,
                            'gpu_utilization': round(gpu_util, 2),
                            'memory_utilization': round(memory_util, 2),
                            'memory_used_mb': round(memory_info.used / 1024 / 1024, 2),
                            'memory_total_mb': round(memory_info.total / 1024 / 1024, 2),
                            'temperature_c': temperature,
                            'power_usage_w': power_usage,
                            'is_idle': gpu_util < 10 and memory_util < 10,
                            'timestamp': datetime.utcnow().isoformat(),
                            'cloud_provider': 'gcp'
                        }
                        metrics.append(metric_data)
                except Exception as e:
                    logger.debug(f"Failed to read GPU {i}: {e}")

            logger.info(f"Collected GPU metrics for {len(metrics)} devices")

        except Exception as e:
            logger.error(f"Error getting GPU utilization: {e}")

        return metrics
    
    def get_cost_data_from_bigquery(self) -> List[Dict[str, Any]]:
        """Get real GPU cost data from BigQuery billing export.

        Requires ``GCP_BILLING_DATASET`` env var (e.g.
        ``billing_dataset.gcp_billing_export``) and a working BigQuery client.
        Returns data in the same shape as :meth:`get_cost_data` for drop-in
        compatibility, with an additional ``labels`` field from the export.
        """
        if not self.bq_client or not self.billing_dataset:
            raise RuntimeError("BigQuery client or billing dataset not configured")

        query = f"""
            SELECT
                service.description  AS service_description,
                sku.description      AS sku_description,
                usage_start_time,
                cost,
                currency,
                labels,
                project.id           AS project_id,
                location.region      AS region
            FROM `{self.billing_dataset}`
            WHERE project.id = @project_id
              AND usage_start_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)
              AND sku.description LIKE '%GPU%'
            ORDER BY usage_start_time DESC
        """

        from google.cloud import bigquery as bq

        job_config = bq.QueryJobConfig(
            query_parameters=[
                bq.ScalarQueryParameter("project_id", "STRING", self.project_id),
            ]
        )

        results = self.bq_client.query(query, job_config=job_config).result()

        cost_data = []
        for row in results:
            # Convert BigQuery label repeated field to a plain dict
            labels_dict = {}
            if row.labels:
                for label in row.labels:
                    labels_dict[label['key']] = label['value']

            cost_entry = {
                'instance_id': None,  # billing export doesn't carry instance name
                'service': row.service_description,
                'sku_description': row.sku_description,
                'cost': float(row.cost),
                'currency': row.currency,
                'date': row.usage_start_time.isoformat() if row.usage_start_time else None,
                'project_id': row.project_id,
                'location': row.region,
                'cloud_provider': 'gcp',
                'labels': labels_dict,
            }
            cost_data.append(cost_entry)

        logger.info(f"Collected {len(cost_data)} cost records from BigQuery")
        return cost_data

    def get_cost_data(self) -> List[Dict[str, Any]]:
        """Get cost data from Google Cloud Billing.

        Tries BigQuery billing export first (if ``GCP_BILLING_DATASET`` is
        set); falls back to the estimation method on failure or when BigQuery
        is not configured.
        """
        # --- Attempt BigQuery path first ---
        if self.bq_client and self.billing_dataset:
            try:
                logger.info("Fetching cost data from BigQuery billing export")
                return self.get_cost_data_from_bigquery()
            except Exception as e:
                logger.warning(f"BigQuery cost query failed, falling back to estimation: {e}")

        # --- Fallback: estimation based on instance list ---
        logger.info("Using estimation method for cost data")
        cost_data = []

        try:
            billing_account_id = os.getenv('GCP_BILLING_ACCOUNT_ID')
            if not billing_account_id:
                logger.warning("GCP_BILLING_ACCOUNT_ID not configured - skipping cost data collection")
                return cost_data

            for instance in self.get_gpu_instances():
                estimated_cost = self._estimate_instance_cost(instance)

                cost_entry = {
                    'instance_id': instance['instance_id'],
                    'service': 'Compute Engine',
                    'cost': estimated_cost,
                    'currency': 'USD',
                    'date': datetime.utcnow().isoformat(),
                    'project_id': instance['project_id'],
                    'location': instance['location'],
                    'cloud_provider': 'gcp'
                }
                cost_data.append(cost_entry)

            logger.info(f"Collected estimated cost data for {len(cost_data)} GCP resources")

        except Exception as e:
            logger.error(f"Error getting GCP cost data: {e}")

        return cost_data
    
    def _estimate_instance_cost(self, instance: Dict[str, Any]) -> float:
        """Estimate hourly cost for a GCP instance using real API pricing when available."""
        base_costs = {
            'n1-standard-4': 0.19,
            'n1-standard-8': 0.38,
            'n1-standard-16': 0.76,
            'n1-standard-32': 1.52,
            'n1-standard-64': 3.04,
            'n1-standard-96': 4.56
        }

        # GPU costs from real API (via _GCP_GPU_PRICES property) or fallback
        gpu_prices = self._GCP_GPU_PRICES
        gpu_costs = {k: v['on_demand'] for k, v in gpu_prices.items()}
        
        instance_type = instance['instance_type']
        gpu_type = instance.get('gpu_type', '')
        gpu_count = instance.get('gpu_count', 0)
        
        base_cost = base_costs.get(instance_type, 0.19)
        gpu_cost = gpu_costs.get(gpu_type, 0.45) * gpu_count
        
        # Apply preemptible discount
        if instance.get('preemptible', False):
            base_cost *= 0.2  # 80% discount for preemptible instances
            gpu_cost *= 0.2
        
        return round(base_cost + gpu_cost, 2)
    
    def get_instance_metrics(self, instance_id: str, zone: str) -> Dict[str, Any]:
        """Get detailed metrics for a specific GCE instance"""
        try:
            # Get instance metrics from Cloud Monitoring
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=1)
            
            # Get CPU utilization metric
            cpu_metric = self.monitoring_client.list_time_series(
                name=f"projects/{self.project_id}",
                filter=f'metric.type="compute.googleapis.com/instance/cpu/utilization" AND resource.labels.instance_name="{instance_id}" AND resource.labels.zone="{zone}"',
                interval={
                    'start_time': start_time,
                    'end_time': end_time
                }
            )
            
            # Get memory utilization metric
            memory_metric = self.monitoring_client.list_time_series(
                name=f"projects/{self.project_id}",
                filter=f'metric.type="compute.googleapis.com/instance/memory/utilization" AND resource.labels.instance_name="{instance_id}" AND resource.labels.zone="{zone}"',
                interval={
                    'start_time': start_time,
                    'end_time': end_time
                }
            )
            
            metrics_data = {
                'instance_id': instance_id,
                'cpu_metrics': [],
                'memory_metrics': [],
                'timestamp': datetime.utcnow().isoformat()
            }
            
            # Process CPU metrics
            for series in cpu_metric:
                for point in series.points:
                    if point.value.double_value is not None:
                        metrics_data['cpu_metrics'].append({
                            'timestamp': point.interval.end_time.isoformat(),
                            'value': round(point.value.double_value * 100, 2)  # Convert to percentage
                        })
            
            # Process memory metrics
            for series in memory_metric:
                for point in series.points:
                    if point.value.double_value is not None:
                        metrics_data['memory_metrics'].append({
                            'timestamp': point.interval.end_time.isoformat(),
                            'value': round(point.value.double_value * 100, 2)  # Convert to percentage
                        })
            
            return metrics_data
            
        except Exception as e:
            logger.error(f"Error getting metrics for instance {instance_id}: {e}")
            return {}
    
    def get_resource_labels(self, instance_id: str, zone: str) -> Dict[str, str]:
        """Get labels for a GCP resource"""
        try:
            request = compute_v1.GetInstanceRequest(
                project=self.project_id,
                zone=zone,
                instance=instance_id
            )
            instance = self.compute_client.get(request=request)
            return dict(instance.labels) if instance.labels else {}
        except Exception as e:
            logger.error(f"Error getting labels for instance {instance_id}: {e}")
            return {}
    
    def get_cost_by_labels_from_bigquery(self, label_key: str) -> List[Dict[str, Any]]:
        """Get cost breakdown grouped by a billing-export label key via BigQuery.

        Returns ``[{label_key, label_value, total_cost, record_count}]``.
        """
        if not self.bq_client or not self.billing_dataset:
            raise RuntimeError("BigQuery client or billing dataset not configured")

        query = f"""
            SELECT
                label.value AS label_value,
                ROUND(SUM(cost), 2) AS total_cost,
                COUNT(*) AS record_count
            FROM `{self.billing_dataset}`,
                 UNNEST(labels) AS label
            WHERE project.id = @project_id
              AND usage_start_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)
              AND sku.description LIKE '%GPU%'
              AND label.key = @label_key
            GROUP BY label.value
            ORDER BY total_cost DESC
        """

        from google.cloud import bigquery as bq

        job_config = bq.QueryJobConfig(
            query_parameters=[
                bq.ScalarQueryParameter("project_id", "STRING", self.project_id),
                bq.ScalarQueryParameter("label_key", "STRING", label_key),
            ]
        )

        results = self.bq_client.query(query, job_config=job_config).result()

        data = []
        for row in results:
            data.append({
                'label_key': label_key,
                'label_value': row.label_value,
                'total_cost': float(row.total_cost),
                'record_count': int(row.record_count),
            })

        logger.info(f"BigQuery cost by label '{label_key}': {len(data)} groups")
        return data

    def get_cost_by_labels(self, label_key: str = None) -> List[Dict[str, Any]]:
        """Get cost breakdown grouped by a specific label key.

        Tries BigQuery billing export first (when configured and a label_key
        is provided); falls back to instance-list estimation.
        """
        # --- Attempt BigQuery path ---
        if self.bq_client and self.billing_dataset and label_key:
            try:
                logger.info(f"Fetching cost by label '{label_key}' from BigQuery")
                return self.get_cost_by_labels_from_bigquery(label_key)
            except Exception as e:
                logger.warning(f"BigQuery cost-by-label query failed, falling back to estimation: {e}")

        # --- Fallback: estimation ---
        logger.info(f"Using estimation method for cost by label '{label_key}'")
        try:
            instances = self.get_gpu_instances()
            label_costs = {}

            for inst in instances:
                labels = inst.get('labels', {}) or {}
                label_value = labels.get(label_key, 'unlabeled') if label_key else 'all'
                estimated_cost = self._estimate_instance_cost(inst)

                if label_value not in label_costs:
                    label_costs[label_value] = {'total_cost': 0.0, 'record_count': 0}
                label_costs[label_value]['total_cost'] += estimated_cost
                label_costs[label_value]['record_count'] += 1

            result = [
                {'label_key': label_key or 'all', 'label_value': lv, 'total_cost': round(d['total_cost'], 2), 'record_count': d['record_count']}
                for lv, d in label_costs.items()
            ]
            logger.info(f"Cost by label '{label_key}': {len(result)} groups")
            return result
        except Exception as e:
            logger.error(f"Error getting cost by labels: {e}")
            return []

    # ── Spot / Preemptible Instance Intelligence ──────────────────────

    # Fallback prices — used ONLY if Cloud Billing Catalog API call fails
    _GCP_GPU_PRICES_FALLBACK = {
        'nvidia-tesla-t4': {'on_demand': 0.35, 'spot': 0.11},
        'nvidia-tesla-v100': {'on_demand': 2.48, 'spot': 0.74},
        'nvidia-tesla-a100': {'on_demand': 2.934, 'spot': 0.88},
        'nvidia-tesla-a100-80gb': {'on_demand': 3.67, 'spot': 1.10},
        'nvidia-tesla-k80': {'on_demand': 0.45, 'spot': 0.135},
        'nvidia-tesla-p100': {'on_demand': 1.46, 'spot': 0.43},
        'nvidia-l4': {'on_demand': 0.744, 'spot': 0.223},
        'nvidia-h100-80gb': {'on_demand': 10.82, 'spot': 3.25},
    }

    _gcp_gpu_price_cache = {}
    _gcp_gpu_price_cache_ts = 0

    def _refresh_gcp_gpu_prices(self):
        """Fetch real GPU pricing from the GCP Cloud Billing Catalog API.
        Queries the Compute Engine service SKUs for GPU accelerator pricing.
        """
        import time
        # Cache for 6 hours
        if self._gcp_gpu_price_cache and (time.time() - self._gcp_gpu_price_cache_ts) < 21600:
            return

        try:
            from google.cloud import billing_v1

            client = billing_v1.CloudCatalogClient()
            # List all Compute Engine SKUs
            # Service ID for Compute Engine: 6F81-5844-456A
            service_name = 'services/6F81-5844-456A'

            prices = {}
            gpu_keywords = ['GPU', 'Nvidia', 'Tesla', 'A100', 'V100', 'T4', 'K80', 'P100', 'L4']
            region = getattr(self, 'region', 'us-central1')

            for sku in client.list_skus(parent=service_name):
                description = sku.description or ''
                # Only look at GPU-related SKUs
                if not any(kw.lower() in description.lower() for kw in gpu_keywords):
                    continue
                # Check if this SKU applies to our region
                regions = [r for sr in sku.service_regions for r in [sr]]
                if region not in regions and 'global' not in regions:
                    continue

                # Extract pricing
                for tier in (sku.pricing_info or []):
                    expr = tier.pricing_expression
                    if not expr or not expr.tiered_rates:
                        continue
                    unit_price = 0
                    for rate in expr.tiered_rates:
                        if rate.unit_price:
                            unit_price = rate.unit_price.units + rate.unit_price.nanos / 1e9

                    if unit_price <= 0:
                        continue

                    # Map SKU description to gpu_type
                    desc_lower = description.lower()
                    is_spot = 'preemptible' in desc_lower or 'spot' in desc_lower

                    gpu_type = None
                    for gtype in ['nvidia-tesla-t4', 'nvidia-tesla-v100', 'nvidia-tesla-a100', 'nvidia-tesla-a100-80gb',
                                  'nvidia-tesla-k80', 'nvidia-tesla-p100', 'nvidia-l4', 'nvidia-h100-80gb']:
                        short = gtype.replace('nvidia-tesla-', '').replace('nvidia-', '')
                        if short in desc_lower:
                            gpu_type = gtype
                            break

                    if gpu_type:
                        if gpu_type not in prices:
                            prices[gpu_type] = {'on_demand': 0, 'spot': 0}
                        if is_spot:
                            prices[gpu_type]['spot'] = round(unit_price, 4)
                        else:
                            prices[gpu_type]['on_demand'] = round(unit_price, 4)

            if prices:
                GCPMonitor._gcp_gpu_price_cache = prices
                GCPMonitor._gcp_gpu_price_cache_ts = time.time()
                logger.info(f"Fetched {len(prices)} GCP GPU prices from Cloud Billing Catalog API ({region})")
            else:
                logger.warning("GCP Billing Catalog returned no GPU prices — using fallback")

        except Exception as e:
            logger.warning(f"GCP Billing Catalog API call failed, using fallback: {e}")

    @property
    def _GCP_GPU_PRICES(self):
        """Return real prices if available, fallback otherwise."""
        self._refresh_gcp_gpu_prices()
        if self._gcp_gpu_price_cache:
            # Merge: API data wins, fallback fills gaps
            merged = dict(self._GCP_GPU_PRICES_FALLBACK)
            merged.update(self._gcp_gpu_price_cache)
            return merged
        return self._GCP_GPU_PRICES_FALLBACK

    def get_spot_pricing(self) -> List[Dict[str, Any]]:
        """Get GCP Spot VM pricing for GPU accelerator types.
        GCP publishes spot pricing as a fixed discount per accelerator type.
        """
        spot_data = []
        try:
            for gpu_type, prices in self._GCP_GPU_PRICES.items():
                spot_data.append({
                    'instance_type': gpu_type,
                    'availability_zone': 'all',
                    'spot_price_hourly': prices['spot'],
                    'on_demand_price_hourly': prices['on_demand'],
                    'spot_monthly': round(prices['spot'] * 730, 2),
                    'on_demand_monthly': round(prices['on_demand'] * 730, 2),
                    'savings_pct': round((1 - prices['spot'] / prices['on_demand']) * 100, 1),
                    'cloud_provider': 'gcp',
                    'note': 'Per-GPU pricing. Total cost depends on GPU count and machine type.'
                })
            logger.info(f"Collected GCP spot pricing for {len(spot_data)} GPU types")
        except Exception as e:
            logger.warning(f"Error fetching GCP spot pricing: {e}")
        return spot_data

    def get_spot_candidates(self, instances: List[Dict[str, Any]], spot_pricing: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify GCE instances that could use Spot/Preemptible VMs.
        Instances already running as preemptible are excluded.
        """
        candidates = []
        # Build GPU type -> pricing lookup
        gpu_prices = {sp['instance_type']: sp for sp in spot_pricing}

        spot_labels = {'batch', 'training', 'dev', 'test', 'experiment', 'ci', 'staging'}

        for inst in instances:
            # Skip already-preemptible instances
            if inst.get('preemptible', False):
                continue
            if inst.get('state') != 'RUNNING':
                continue

            gpu_type = (inst.get('gpu_type') or '').lower()
            gpu_count = inst.get('gpu_count', 1)
            pricing = gpu_prices.get(gpu_type)
            if not pricing:
                continue

            tags = inst.get('tags', {})
            label_values = {v.lower() for v in tags.values()}

            is_candidate = False
            reason = ''
            if tags.get('fault_tolerant', '').lower() in ('true', 'yes'):
                is_candidate, reason = True, 'Labeled as fault-tolerant'
            elif tags.get('environment', '').lower() in ('dev', 'staging', 'test'):
                is_candidate, reason = True, f"Environment: {tags.get('environment')}"
            elif label_values & spot_labels:
                is_candidate, reason = True, 'Labels suggest non-production workload'

            if is_candidate:
                on_demand_monthly = pricing['on_demand_monthly'] * gpu_count
                spot_monthly = pricing['spot_monthly'] * gpu_count
                candidates.append({
                    'instance_id': inst['instance_id'],
                    'instance_type': f"{inst.get('instance_type', 'unknown')} ({gpu_count}x {gpu_type})",
                    'recommendation_type': 'spot',
                    'term': 'on_demand',
                    'payment_option': 'spot',
                    'current_monthly_cost': round(on_demand_monthly, 2),
                    'recommended_monthly_cost': round(spot_monthly, 2),
                    'potential_monthly_savings': round(on_demand_monthly - spot_monthly, 2),
                    'potential_savings_pct': pricing['savings_pct'],
                    'cloud_provider': 'gcp',
                    'source_data': {
                        'reason': reason,
                        'instance_id': inst['instance_id'],
                        'gpu_type': gpu_type,
                        'gpu_count': gpu_count,
                        'tags': tags
                    }
                })

        logger.info(f"Found {len(candidates)} GCP spot candidates")
        return candidates

    # ── Action Methods ──────────────────────────────────────────────────
    # instance_id format: "zone/instanceName"

    def _parse_instance_id(self, instance_id: str):
        """Parse 'zone/name' format and validate."""
        parts = instance_id.split('/')
        if len(parts) != 2 or not all(parts):
            raise ValueError(
                f"Invalid GCP instance ID: {instance_id}. Expected: zone/instanceName"
            )
        return parts[0], parts[1]

    def stop_instance(self, instance_id):
        """Stop a GCP instance."""
        zone, name = self._parse_instance_id(instance_id)
        request = compute_v1.StopInstanceRequest(
            project=self.project_id, zone=zone, instance=name
        )
        operation = self.compute_client.stop(request=request)
        operation.result()  # wait for completion
        logger.info(f"Stopped GCP instance {name} in {zone}")
        return {'instance_id': instance_id, 'action': 'stop'}

    def start_instance(self, instance_id):
        """Start a GCP instance."""
        zone, name = self._parse_instance_id(instance_id)
        request = compute_v1.StartInstanceRequest(
            project=self.project_id, zone=zone, instance=name
        )
        operation = self.compute_client.start(request=request)
        operation.result()
        logger.info(f"Started GCP instance {name} in {zone}")
        return {'instance_id': instance_id, 'action': 'start'}

    def resize_instance(self, instance_id, target_type):
        """Resize a GCP instance (stop -> set machine type -> start)."""
        zone, name = self._parse_instance_id(instance_id)

        # Stop the instance and wait
        stop_req = compute_v1.StopInstanceRequest(
            project=self.project_id, zone=zone, instance=name
        )
        self.compute_client.stop(request=stop_req).result()

        # Set new machine type
        machine_type_url = f"zones/{zone}/machineTypes/{target_type}"
        set_req = compute_v1.SetMachineTypeInstanceRequest(
            project=self.project_id,
            zone=zone,
            instance=name,
            instances_set_machine_type_request_resource=compute_v1.InstancesSetMachineTypeRequest(
                machine_type=machine_type_url
            ),
        )
        self.compute_client.set_machine_type(request=set_req).result()

        # Start instance
        start_req = compute_v1.StartInstanceRequest(
            project=self.project_id, zone=zone, instance=name
        )
        self.compute_client.start(request=start_req).result()

        logger.info(f"Resized GCP instance {name} to {target_type}")
        return {'instance_id': instance_id, 'new_type': target_type}

    def restart_instance(self, instance_id):
        """Reset (hard restart) a GCP instance."""
        zone, name = self._parse_instance_id(instance_id)
        request = compute_v1.ResetInstanceRequest(
            project=self.project_id, zone=zone, instance=name
        )
        operation = self.compute_client.reset(request=request)
        operation.result()
        logger.info(f"Reset GCP instance {name}")
        return {'instance_id': instance_id, 'action': 'reset'}

    def terminate_instance(self, instance_id):
        """Delete a GCP instance."""
        zone, name = self._parse_instance_id(instance_id)
        request = compute_v1.DeleteInstanceRequest(
            project=self.project_id, zone=zone, instance=name
        )
        operation = self.compute_client.delete(request=request)
        operation.result()
        logger.info(f"Deleted GCP instance {name}")
        return {'instance_id': instance_id, 'action': 'delete'}
