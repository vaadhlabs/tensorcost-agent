"""
Kubernetes Monitor for GPU nodes and pods
"""

import os
import logging
import time
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional
import psutil

from kubernetes import client, config
from kubernetes.client.rest import ApiException

logger = logging.getLogger(__name__)

# pynvml is optional — prefer shared NvmlSampler when available
try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False


class KubernetesMonitor:
    def __init__(self, nvml_sampler=None, namespaces: Optional[List[str]] = None):
        """Initialize Kubernetes monitoring client.

        Args:
            nvml_sampler: Optional shared NvmlSampler instance. When provided,
                GPU metrics come from the sampler's ring buffer (more accurate
                idle detection). Falls back to inline pynvml if not provided.
            namespaces: Optional list of namespaces to scrape. When omitted,
                reads K8S_NAMESPACES (comma-separated) or KUBERNETES_NAMESPACE.
                Use ``['all']`` or include ``all`` to list every namespace.
        """
        self.kubeconfig_path = os.getenv('KUBECONFIG_PATH', '~/.kube/config')
        self.context = os.getenv('KUBERNETES_CONTEXT')
        raw_ns = os.getenv('K8S_NAMESPACES', '').strip()
        if namespaces is not None:
            self.namespaces = [n.strip() for n in namespaces if n and n.strip()]
        elif raw_ns:
            self.namespaces = [n.strip() for n in raw_ns.split(',') if n.strip()]
        else:
            self.namespaces = [os.getenv('KUBERNETES_NAMESPACE', 'default')]
        if not self.namespaces:
            self.namespaces = ['default']
        # Back-compat: single namespace field used by older tests/callers.
        self.namespace = 'all' if 'all' in self.namespaces else self.namespaces[0]
        self.gpu_resource_name = os.getenv('K8S_GPU_RESOURCE_NAME', 'nvidia.com/gpu')
        self.nvml_sampler = nvml_sampler

        # Initialize Kubernetes client
        self._init_k8s_client()

        # Initialize inline NVML only if no shared sampler provided
        self.gpu_available = False
        if self.nvml_sampler and self.nvml_sampler.is_available():
            self.gpu_available = True
            logger.info("Using shared NvmlSampler for GPU monitoring")
        elif PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.gpu_available = True
                logger.info("NVML initialized (inline fallback)")
            except Exception as e:
                logger.warning(f"NVML not available: {e}")
        else:
            logger.warning("pynvml not installed — GPU monitoring unavailable")
    
    def _init_k8s_client(self):
        """Initialize Kubernetes client"""
        try:
            # Try to load kubeconfig
            if os.path.exists(os.path.expanduser(self.kubeconfig_path)):
                config.load_kube_config(config_file=os.path.expanduser(self.kubeconfig_path), context=self.context)
                logger.info(f"Loaded kubeconfig from {self.kubeconfig_path}")
            else:
                # Try to load from cluster (in-cluster config)
                config.load_incluster_config()
                logger.info("Loaded in-cluster Kubernetes config")
            
            # Initialize API clients
            self.v1 = client.CoreV1Api()
            self.apps_v1 = client.AppsV1Api()
            self.metrics_v1 = client.CustomObjectsApi()
            
            logger.info("Kubernetes client initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing Kubernetes client: {e}")
            raise

    def get_gpu_instances(self) -> List[Dict[str, Any]]:
        """Inventory hook used by the unified monitoring loop."""
        return self.get_gpu_nodes()
    
    def get_gpu_nodes(self) -> List[Dict[str, Any]]:
        """Get all Kubernetes nodes with GPU capabilities"""
        nodes = []
        
        try:
            # Get all nodes
            node_list = self.v1.list_node()
            
            for node in node_list.items:
                # Check if node has GPU resources
                gpu_count = 0
                gpu_type = None
                
                # Check node capacity for GPU resources
                if self.gpu_resource_name in node.status.capacity:
                    gpu_count = int(node.status.capacity[self.gpu_resource_name])
                
                # Check node labels for GPU type
                if node.metadata.labels:
                    for label, value in node.metadata.labels.items():
                        if 'accelerator' in label.lower() or 'gpu' in label.lower():
                            gpu_type = value
                            break
                
                if gpu_count > 0:
                    # Merge labels + annotations into tags (labels take priority)
                    k8s_tags = dict(node.metadata.annotations) if node.metadata.annotations else {}
                    if node.metadata.labels:
                        k8s_tags.update(dict(node.metadata.labels))
                    node_data = {
                        'instance_id': node.metadata.name,
                        'gpu_count': gpu_count,
                        'gpu_type': gpu_type,
                        'state': node.status.phase or 'Ready',
                        'tags': k8s_tags,
                        'launch_time': node.metadata.creation_timestamp.isoformat() if node.metadata.creation_timestamp else None,
                        'cloud_provider': 'kubernetes',
                        'availability_zone': node.metadata.labels.get('topology.kubernetes.io/zone', 'unknown') if node.metadata.labels else 'unknown',
                        'instance_type': node.metadata.labels.get('node.kubernetes.io/instance-type', 'unknown') if node.metadata.labels else 'unknown'
                    }
                    nodes.append(node_data)
            
            logger.info(f"Found {len(nodes)} GPU-enabled Kubernetes nodes")
            
        except ApiException as e:
            logger.error(f"Error getting Kubernetes nodes: {e}")
        except Exception as e:
            logger.error(f"Error getting Kubernetes nodes: {e}")
        
        return nodes
    
    def get_gpu_pods(self) -> List[Dict[str, Any]]:
        """Get all pods that are using GPU resources"""
        pods = []
        
        try:
            pod_items = []
            if 'all' in self.namespaces:
                pod_list = self.v1.list_pod_for_all_namespaces()
                pod_items = pod_list.items
            else:
                seen: set = set()
                for ns in self.namespaces:
                    pod_list = self.v1.list_namespaced_pod(namespace=ns)
                    for pod in pod_list.items:
                        key = (pod.metadata.namespace, pod.metadata.name)
                        if key in seen:
                            continue
                        seen.add(key)
                        pod_items.append(pod)
            
            for pod in pod_items:
                # Check if pod is using GPU resources
                gpu_requests = 0
                gpu_limits = 0
                
                for container in pod.spec.containers:
                    if container.resources and container.resources.requests:
                        if self.gpu_resource_name in container.resources.requests:
                            gpu_requests += int(container.resources.requests[self.gpu_resource_name])
                    
                    if container.resources and container.resources.limits:
                        if self.gpu_resource_name in container.resources.limits:
                            gpu_limits += int(container.resources.limits[self.gpu_resource_name])
                
                if gpu_requests > 0 or gpu_limits > 0:
                    pod_data = {
                        'pod_name': pod.metadata.name,
                        'namespace': pod.metadata.namespace,
                        'node_name': pod.spec.node_name,
                        'gpu_requests': gpu_requests,
                        'gpu_limits': gpu_limits,
                        'status': pod.status.phase,
                        'labels': dict(pod.metadata.labels) if pod.metadata.labels else {},
                        'annotations': dict(pod.metadata.annotations) if pod.metadata.annotations else {},
                        'created_at': pod.metadata.creation_timestamp.isoformat() if pod.metadata.creation_timestamp else None,
                        'cloud_provider': 'kubernetes',
                        'containers': [c.name for c in pod.spec.containers]
                    }
                    pods.append(pod_data)
            
            logger.info(f"Found {len(pods)} GPU-using pods")
            
        except ApiException as e:
            logger.error(f"Error getting Kubernetes pods: {e}")
        except Exception as e:
            logger.error(f"Error getting Kubernetes pods: {e}")
        
        return pods
    
    def get_gpu_utilization(self, pods: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Get GPU utilization metrics for Kubernetes pods.

        Uses shared NvmlSampler if available (ring-buffer-based idle detection).
        Falls back to inline pynvml reads otherwise.
        """
        metrics = []

        if not self.gpu_available:
            logger.warning("GPU monitoring not available - NVML not initialized")
            return metrics

        try:
            # Prefer shared sampler — gives time-windowed idle detection
            if self.nvml_sampler and self.nvml_sampler.is_available():
                gpu_samples = self.nvml_sampler.get_latest_metrics()
            else:
                gpu_samples = self._read_gpus_inline()

            # Match GPU samples to pods on this node
            hostname = os.uname().nodename
            for sample in gpu_samples:
                pod = None
                for p in pods:
                    if p['node_name'] in hostname:
                        pod = p
                        break

                if pod:
                    metric_data = {
                        'instance_id': pod['node_name'],
                        'pod_name': pod['pod_name'],
                        'namespace': pod['namespace'],
                        'node_name': pod['node_name'],
                        'gpu_index': sample['gpu_index'],
                        'gpu_utilization': sample['gpu_utilization'],
                        'memory_utilization': sample['memory_utilization'],
                        'memory_used_mb': sample['memory_used_mb'],
                        'memory_total_mb': sample['memory_total_mb'],
                        'temperature_c': sample.get('temperature_c'),
                        'power_usage_w': sample.get('power_usage_w'),
                        'is_idle': sample['is_idle'],
                        'timestamp': sample.get('timestamp', datetime.now(timezone.utc).isoformat()),
                        'cloud_provider': 'kubernetes',
                    }
                    metrics.append(metric_data)

            logger.info(f"Collected GPU metrics for {len(metrics)} devices")

        except Exception as e:
            logger.error(f"Error getting GPU utilization: {e}")

        return metrics

    def _read_gpus_inline(self) -> List[Dict]:
        """Inline pynvml reads — fallback when no shared sampler is available."""
        samples = []
        device_count = pynvml.nvmlDeviceGetCount()

        for i in range(device_count):
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
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

                gpu_util = round(utilization.gpu, 2)
                mem_util = round(utilization.memory, 2)
                samples.append({
                    'gpu_index': i,
                    'gpu_utilization': gpu_util,
                    'memory_utilization': mem_util,
                    'memory_used_mb': round(memory_info.used / 1024 / 1024, 2),
                    'memory_total_mb': round(memory_info.total / 1024 / 1024, 2),
                    'temperature_c': temperature,
                    'power_usage_w': power_usage,
                    'is_idle': gpu_util < 10 and mem_util < 10,
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                })
            except Exception as e:
                logger.debug(f"Failed to read GPU {i}: {e}")

        return samples
    
    def get_cost_data(self, pods: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Get cost data for Kubernetes pods"""
        cost_data = []
        
        try:
            # Estimate costs based on pod resources and node types
            for pod in pods:
                # Get node information
                node_info = self._get_node_info(pod['node_name'])
                
                # Calculate estimated cost
                estimated_cost = self._estimate_pod_cost(pod, node_info)
                
                cost_entry = {
                    'pod_name': pod['pod_name'],
                    'namespace': pod['namespace'],
                    'node_name': pod['node_name'],
                    'service': 'Kubernetes',
                    'cost': estimated_cost,
                    'currency': 'USD',
                    'date': datetime.utcnow().isoformat(),
                    'cloud_provider': 'kubernetes'
                }
                cost_data.append(cost_entry)
            
            logger.info(f"Collected cost data for {len(cost_data)} Kubernetes pods")
            
        except Exception as e:
            logger.error(f"Error getting Kubernetes cost data: {e}")
        
        return cost_data
    
    def _get_node_info(self, node_name: str) -> Dict[str, Any]:
        """Get information about a specific node"""
        try:
            node = self.v1.read_node(name=node_name)
            return {
                'instance_type': node.metadata.labels.get('node.kubernetes.io/instance-type', 'unknown'),
                'zone': node.metadata.labels.get('topology.kubernetes.io/zone', 'unknown'),
                'labels': dict(node.metadata.labels) if node.metadata.labels else {}
            }
        except Exception as e:
            logger.error(f"Error getting node info for {node_name}: {e}")
            return {}
    
    def _estimate_pod_cost(self, pod: Dict[str, Any], node_info: Dict[str, Any]) -> float:
        """Estimate cost for a Kubernetes pod"""
        # This is a simplified cost estimation
        # In practice, you'd use actual billing data or more sophisticated cost models
        
        base_costs = {
            't3.medium': 0.0416,
            't3.large': 0.0832,
            't3.xlarge': 0.1664,
            'm5.large': 0.096,
            'm5.xlarge': 0.192,
            'm5.2xlarge': 0.384,
            'c5.large': 0.085,
            'c5.xlarge': 0.17,
            'c5.2xlarge': 0.34
        }
        
        gpu_costs = {
            'nvidia-tesla-k80': 0.45,
            'nvidia-tesla-p4': 0.60,
            'nvidia-tesla-p100': 1.46,
            'nvidia-tesla-v100': 2.48,
            'nvidia-tesla-t4': 0.35,
            'nvidia-tesla-a100': 2.93
        }
        
        instance_type = node_info.get('instance_type', 't3.medium')
        gpu_requests = pod.get('gpu_requests', 0)
        
        base_cost = base_costs.get(instance_type, 0.0416)
        gpu_cost = 0
        
        # Estimate GPU cost based on node labels
        if gpu_requests > 0:
            node_labels = node_info.get('labels', {})
            for label, value in node_labels.items():
                if 'accelerator' in label.lower() or 'gpu' in label.lower():
                    gpu_cost = gpu_costs.get(value, 0.45) * gpu_requests
                    break
        
        return round(base_cost + gpu_cost, 2)

    def get_namespace_resource_usage(self, namespace: str) -> Dict[str, Any]:
        """Get resource usage summary for a namespace"""
        try:
            # Get all pods in namespace
            pod_list = self.v1.list_namespaced_pod(namespace=namespace)
            
            total_gpu_requests = 0
            total_gpu_limits = 0
            pod_count = 0
            
            for pod in pod_list.items:
                if pod.status.phase == 'Running':
                    pod_count += 1
                    
                    for container in pod.spec.containers:
                        if container.resources and container.resources.requests:
                            if self.gpu_resource_name in container.resources.requests:
                                total_gpu_requests += int(container.resources.requests[self.gpu_resource_name])
                        
                        if container.resources and container.resources.limits:
                            if self.gpu_resource_name in container.resources.limits:
                                total_gpu_limits += int(container.resources.limits[self.gpu_resource_name])
            
            return {
                'namespace': namespace,
                'pod_count': pod_count,
                'total_gpu_requests': total_gpu_requests,
                'total_gpu_limits': total_gpu_limits,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting namespace resource usage for {namespace}: {e}")
            return {}
