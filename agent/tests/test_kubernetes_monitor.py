import pytest
from unittest.mock import Mock, patch, MagicMock, call
from datetime import datetime, timezone, timedelta
import sys
import os

# Create a real ApiException class for mocking
class ApiException(Exception):
    """Mock Kubernetes ApiException."""
    def __init__(self, status, reason):
        self.status = status
        self.reason = reason
        super().__init__(f"({status}) {reason}")

# Mock kubernetes module before importing KubernetesMonitor
kubernetes_mock = MagicMock()
kubernetes_client_rest_mock = MagicMock()
kubernetes_client_rest_mock.ApiException = ApiException

sys.modules['kubernetes'] = kubernetes_mock
sys.modules['kubernetes.client'] = MagicMock()
sys.modules['kubernetes.client.rest'] = kubernetes_client_rest_mock

# Mock pynvml - it's optional and may not be available
sys.modules['pynvml'] = MagicMock()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from monitors.kubernetes_monitor import KubernetesMonitor


class TestKubernetesMonitorInit:
    """Test cases for KubernetesMonitor initialization."""

    @patch('monitors.kubernetes_monitor.config.load_kube_config')
    @patch('monitors.kubernetes_monitor.client.CoreV1Api')
    @patch('monitors.kubernetes_monitor.client.AppsV1Api')
    @patch('monitors.kubernetes_monitor.client.CustomObjectsApi')
    @patch('os.path.exists')
    def test_init_with_kubeconfig(self, mock_exists, mock_custom, mock_apps, mock_core, mock_load_config):
        """Test KubernetesMonitor initialization with kubeconfig file."""
        mock_exists.return_value = True

        with patch('monitors.kubernetes_monitor.PYNVML_AVAILABLE', False):
            monitor = KubernetesMonitor()

            # Verify kubeconfig was loaded
            mock_load_config.assert_called_once()
            assert monitor.v1 is not None
            assert monitor.apps_v1 is not None
            assert monitor.metrics_v1 is not None
            assert monitor.namespace == 'default'
            assert monitor.gpu_resource_name == 'nvidia.com/gpu'
            assert monitor.nvml_sampler is None
            assert monitor.gpu_available == False

    @patch('monitors.kubernetes_monitor.config.load_incluster_config')
    @patch('monitors.kubernetes_monitor.client.CoreV1Api')
    @patch('monitors.kubernetes_monitor.client.AppsV1Api')
    @patch('monitors.kubernetes_monitor.client.CustomObjectsApi')
    @patch('os.path.exists')
    def test_init_with_incluster_config(self, mock_exists, mock_custom, mock_apps, mock_core, mock_load_incluster):
        """Test KubernetesMonitor initialization with in-cluster config."""
        mock_exists.return_value = False

        monitor = KubernetesMonitor()

        # Verify in-cluster config was loaded
        mock_load_incluster.assert_called_once()
        assert monitor.v1 is not None

    @patch('monitors.kubernetes_monitor.config.load_kube_config', side_effect=Exception('Config error'))
    @patch('os.path.exists')
    def test_init_error_handling(self, mock_exists, mock_load_config):
        """Test KubernetesMonitor initialization error handling."""
        mock_exists.return_value = True

        with pytest.raises(Exception):
            KubernetesMonitor()

    @patch('monitors.kubernetes_monitor.config.load_kube_config')
    @patch('monitors.kubernetes_monitor.client.CoreV1Api')
    @patch('monitors.kubernetes_monitor.client.AppsV1Api')
    @patch('monitors.kubernetes_monitor.client.CustomObjectsApi')
    @patch('os.path.exists')
    @patch('monitors.kubernetes_monitor.pynvml')
    def test_init_with_pynvml_available(self, mock_pynvml, mock_exists, mock_custom, mock_apps, mock_core, mock_load_config):
        """Test KubernetesMonitor initialization with pynvml available."""
        mock_exists.return_value = True
        mock_pynvml.nvmlInit.return_value = None

        # Patch the module-level PYNVML_AVAILABLE variable
        with patch('monitors.kubernetes_monitor.PYNVML_AVAILABLE', True):
            monitor = KubernetesMonitor()

            assert monitor.gpu_available == True
            mock_pynvml.nvmlInit.assert_called_once()

    @patch('monitors.kubernetes_monitor.config.load_kube_config')
    @patch('monitors.kubernetes_monitor.client.CoreV1Api')
    @patch('monitors.kubernetes_monitor.client.AppsV1Api')
    @patch('monitors.kubernetes_monitor.client.CustomObjectsApi')
    @patch('os.path.exists')
    @patch('monitors.kubernetes_monitor.pynvml')
    def test_init_with_pynvml_init_error(self, mock_pynvml, mock_exists, mock_custom, mock_apps, mock_core, mock_load_config):
        """Test KubernetesMonitor initialization when pynvml init fails."""
        mock_exists.return_value = True
        mock_pynvml.nvmlInit.side_effect = Exception('NVML init failed')

        with patch('monitors.kubernetes_monitor.PYNVML_AVAILABLE', True):
            monitor = KubernetesMonitor()

            assert monitor.gpu_available == False

    @patch('monitors.kubernetes_monitor.config.load_kube_config')
    @patch('monitors.kubernetes_monitor.client.CoreV1Api')
    @patch('monitors.kubernetes_monitor.client.AppsV1Api')
    @patch('monitors.kubernetes_monitor.client.CustomObjectsApi')
    @patch('os.path.exists')
    def test_init_with_nvml_sampler(self, mock_exists, mock_custom, mock_apps, mock_core, mock_load_config):
        """Test KubernetesMonitor initialization with shared NvmlSampler."""
        mock_exists.return_value = True
        mock_sampler = Mock()
        mock_sampler.is_available.return_value = True

        monitor = KubernetesMonitor(nvml_sampler=mock_sampler)

        assert monitor.nvml_sampler is mock_sampler
        assert monitor.gpu_available == True

    @patch('monitors.kubernetes_monitor.config.load_kube_config')
    @patch('monitors.kubernetes_monitor.client.CoreV1Api')
    @patch('monitors.kubernetes_monitor.client.AppsV1Api')
    @patch('monitors.kubernetes_monitor.client.CustomObjectsApi')
    @patch('os.path.exists')
    def test_init_with_env_variables(self, mock_exists, mock_custom, mock_apps, mock_core, mock_load_config):
        """Test KubernetesMonitor initialization with environment variables."""
        mock_exists.return_value = True

        with patch.dict(os.environ, {
            'KUBECONFIG_PATH': '/custom/path',
            'KUBERNETES_CONTEXT': 'custom-context',
            'KUBERNETES_NAMESPACE': 'custom-ns',
            'K8S_GPU_RESOURCE_NAME': 'nvidia.com/custom-gpu'
        }):
            monitor = KubernetesMonitor()

            assert monitor.kubeconfig_path == '/custom/path'
            assert monitor.context == 'custom-context'
            assert monitor.namespace == 'custom-ns'
            assert monitor.gpu_resource_name == 'nvidia.com/custom-gpu'


class TestGetGPUNodes:
    """Test cases for get_gpu_nodes method."""

    @patch('monitors.kubernetes_monitor.config.load_kube_config')
    @patch('monitors.kubernetes_monitor.client.CoreV1Api')
    @patch('monitors.kubernetes_monitor.client.AppsV1Api')
    @patch('monitors.kubernetes_monitor.client.CustomObjectsApi')
    @patch('os.path.exists')
    def test_get_gpu_nodes_success(self, mock_exists, mock_custom, mock_apps, mock_core, mock_load_config):
        """Test getting GPU nodes successfully."""
        mock_exists.return_value = True

        monitor = KubernetesMonitor()

        # Mock node
        mock_node = Mock()
        mock_node.metadata.name = 'gpu-node-1'
        mock_node.metadata.creation_timestamp = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        mock_node.metadata.labels = {
            'topology.kubernetes.io/zone': 'us-east-1a',
            'node.kubernetes.io/instance-type': 'g4dn.xlarge',
            'accelerator': 'nvidia-tesla-t4'
        }
        mock_node.metadata.annotations = {'annotation-key': 'annotation-value'}
        mock_node.status.phase = 'Ready'
        mock_node.status.capacity = {'nvidia.com/gpu': '1', 'cpu': '4'}

        mock_node_list = Mock()
        mock_node_list.items = [mock_node]

        monitor.v1.list_node = Mock(return_value=mock_node_list)

        nodes = monitor.get_gpu_nodes()

        assert len(nodes) == 1
        assert nodes[0]['instance_id'] == 'gpu-node-1'
        assert nodes[0]['gpu_count'] == 1
        assert nodes[0]['gpu_type'] == 'nvidia-tesla-t4'
        assert nodes[0]['state'] == 'Ready'
        assert nodes[0]['cloud_provider'] == 'kubernetes'
        assert nodes[0]['availability_zone'] == 'us-east-1a'
        assert nodes[0]['instance_type'] == 'g4dn.xlarge'
        assert 'annotation-key' in nodes[0]['tags']
        assert 'accelerator' in nodes[0]['tags']

    @patch('monitors.kubernetes_monitor.config.load_kube_config')
    @patch('monitors.kubernetes_monitor.client.CoreV1Api')
    @patch('monitors.kubernetes_monitor.client.AppsV1Api')
    @patch('monitors.kubernetes_monitor.client.CustomObjectsApi')
    @patch('os.path.exists')
    def test_get_gpu_nodes_multiple(self, mock_exists, mock_custom, mock_apps, mock_core, mock_load_config):
        """Test getting multiple GPU nodes."""
        mock_exists.return_value = True

        monitor = KubernetesMonitor()

        # Create two GPU nodes
        mock_node1 = Mock()
        mock_node1.metadata.name = 'gpu-node-1'
        mock_node1.metadata.creation_timestamp = datetime(2024, 1, 1, tzinfo=timezone.utc)
        mock_node1.metadata.labels = {'topology.kubernetes.io/zone': 'us-east-1a'}
        mock_node1.metadata.annotations = None
        mock_node1.status.phase = 'Ready'
        mock_node1.status.capacity = {'nvidia.com/gpu': '4'}

        mock_node2 = Mock()
        mock_node2.metadata.name = 'gpu-node-2'
        mock_node2.metadata.creation_timestamp = datetime(2024, 1, 2, tzinfo=timezone.utc)
        mock_node2.metadata.labels = {'topology.kubernetes.io/zone': 'us-east-1b'}
        mock_node2.metadata.annotations = None
        mock_node2.status.phase = 'Ready'
        mock_node2.status.capacity = {'nvidia.com/gpu': '2'}

        # Also include a node without GPU
        mock_node_no_gpu = Mock()
        mock_node_no_gpu.metadata.name = 'cpu-node-1'
        mock_node_no_gpu.metadata.creation_timestamp = datetime(2024, 1, 1, tzinfo=timezone.utc)
        mock_node_no_gpu.metadata.labels = None
        mock_node_no_gpu.metadata.annotations = None
        mock_node_no_gpu.status.phase = 'Ready'
        mock_node_no_gpu.status.capacity = {}

        mock_node_list = Mock()
        mock_node_list.items = [mock_node1, mock_node2, mock_node_no_gpu]

        monitor.v1.list_node = Mock(return_value=mock_node_list)

        nodes = monitor.get_gpu_nodes()

        assert len(nodes) == 2
        assert nodes[0]['instance_id'] == 'gpu-node-1'
        assert nodes[0]['gpu_count'] == 4
        assert nodes[1]['instance_id'] == 'gpu-node-2'
        assert nodes[1]['gpu_count'] == 2

    @patch('monitors.kubernetes_monitor.config.load_kube_config')
    @patch('monitors.kubernetes_monitor.client.CoreV1Api')
    @patch('monitors.kubernetes_monitor.client.AppsV1Api')
    @patch('monitors.kubernetes_monitor.client.CustomObjectsApi')
    @patch('os.path.exists')
    def test_get_gpu_nodes_empty(self, mock_exists, mock_custom, mock_apps, mock_core, mock_load_config):
        """Test getting GPU nodes when none exist."""
        mock_exists.return_value = True

        monitor = KubernetesMonitor()

        mock_node_list = Mock()
        mock_node_list.items = []

        monitor.v1.list_node = Mock(return_value=mock_node_list)

        nodes = monitor.get_gpu_nodes()

        assert len(nodes) == 0

    @patch('monitors.kubernetes_monitor.config.load_kube_config')
    @patch('monitors.kubernetes_monitor.client.CoreV1Api')
    @patch('monitors.kubernetes_monitor.client.AppsV1Api')
    @patch('monitors.kubernetes_monitor.client.CustomObjectsApi')
    @patch('os.path.exists')
    def test_get_gpu_nodes_api_exception(self, mock_exists, mock_custom, mock_apps, mock_core, mock_load_config):
        """Test error handling for API exceptions."""
        mock_exists.return_value = True

        monitor = KubernetesMonitor()
        monitor.v1.list_node = Mock(side_effect=ApiException(500, 'API error'))

        nodes = monitor.get_gpu_nodes()

        assert len(nodes) == 0

    @patch('monitors.kubernetes_monitor.config.load_kube_config')
    @patch('monitors.kubernetes_monitor.client.CoreV1Api')
    @patch('monitors.kubernetes_monitor.client.AppsV1Api')
    @patch('monitors.kubernetes_monitor.client.CustomObjectsApi')
    @patch('os.path.exists')
    def test_get_gpu_nodes_generic_exception(self, mock_exists, mock_custom, mock_apps, mock_core, mock_load_config):
        """Test error handling for generic exceptions."""
        mock_exists.return_value = True

        monitor = KubernetesMonitor()
        monitor.v1.list_node = Mock(side_effect=Exception('Unexpected error'))

        nodes = monitor.get_gpu_nodes()

        assert len(nodes) == 0

    @patch('monitors.kubernetes_monitor.config.load_kube_config')
    @patch('monitors.kubernetes_monitor.client.CoreV1Api')
    @patch('monitors.kubernetes_monitor.client.AppsV1Api')
    @patch('monitors.kubernetes_monitor.client.CustomObjectsApi')
    @patch('os.path.exists')
    def test_get_gpu_nodes_no_labels(self, mock_exists, mock_custom, mock_apps, mock_core, mock_load_config):
        """Test getting GPU nodes without labels."""
        mock_exists.return_value = True

        monitor = KubernetesMonitor()

        mock_node = Mock()
        mock_node.metadata.name = 'gpu-node-1'
        mock_node.metadata.creation_timestamp = datetime(2024, 1, 1, tzinfo=timezone.utc)
        mock_node.metadata.labels = None
        mock_node.metadata.annotations = None
        mock_node.status.phase = 'Ready'
        mock_node.status.capacity = {'nvidia.com/gpu': '1'}

        mock_node_list = Mock()
        mock_node_list.items = [mock_node]

        monitor.v1.list_node = Mock(return_value=mock_node_list)

        nodes = monitor.get_gpu_nodes()

        assert len(nodes) == 1
        assert nodes[0]['availability_zone'] == 'unknown'
        assert nodes[0]['instance_type'] == 'unknown'


class TestGetGPUPods:
    """Test cases for get_gpu_pods method."""

    @patch('monitors.kubernetes_monitor.config.load_kube_config')
    @patch('monitors.kubernetes_monitor.client.CoreV1Api')
    @patch('monitors.kubernetes_monitor.client.AppsV1Api')
    @patch('monitors.kubernetes_monitor.client.CustomObjectsApi')
    @patch('os.path.exists')
    def test_get_gpu_pods_success(self, mock_exists, mock_custom, mock_apps, mock_core, mock_load_config):
        """Test getting GPU pods successfully."""
        mock_exists.return_value = True

        monitor = KubernetesMonitor()

        # Mock pod with GPU resources
        mock_container = Mock()
        mock_container.name = 'gpu-container'
        mock_container.resources.requests = {'nvidia.com/gpu': '1'}
        mock_container.resources.limits = {'nvidia.com/gpu': '1'}

        mock_pod = Mock()
        mock_pod.metadata.name = 'gpu-pod-1'
        mock_pod.metadata.namespace = 'default'
        mock_pod.metadata.creation_timestamp = datetime(2024, 1, 1, tzinfo=timezone.utc)
        mock_pod.metadata.labels = {'app': 'ml-training'}
        mock_pod.metadata.annotations = {'annotation': 'value'}
        mock_pod.spec.containers = [mock_container]
        mock_pod.spec.node_name = 'gpu-node-1'
        mock_pod.status.phase = 'Running'

        mock_pod_list = Mock()
        mock_pod_list.items = [mock_pod]

        monitor.v1.list_namespaced_pod = Mock(return_value=mock_pod_list)

        pods = monitor.get_gpu_pods()

        assert len(pods) == 1
        assert pods[0]['pod_name'] == 'gpu-pod-1'
        assert pods[0]['namespace'] == 'default'
        assert pods[0]['gpu_requests'] == 1
        assert pods[0]['gpu_limits'] == 1
        assert pods[0]['status'] == 'Running'
        assert pods[0]['node_name'] == 'gpu-node-1'
        assert pods[0]['cloud_provider'] == 'kubernetes'
        assert 'gpu-container' in pods[0]['containers']

    @patch('monitors.kubernetes_monitor.config.load_kube_config')
    @patch('monitors.kubernetes_monitor.client.CoreV1Api')
    @patch('monitors.kubernetes_monitor.client.AppsV1Api')
    @patch('monitors.kubernetes_monitor.client.CustomObjectsApi')
    @patch('os.path.exists')
    def test_get_gpu_pods_all_namespaces(self, mock_exists, mock_custom, mock_apps, mock_core, mock_load_config):
        """Test getting GPU pods from all namespaces."""
        mock_exists.return_value = True

        with patch.dict(os.environ, {'KUBERNETES_NAMESPACE': 'all'}):
            monitor = KubernetesMonitor()

            mock_container = Mock()
            mock_container.name = 'container'
            mock_container.resources.requests = {'nvidia.com/gpu': '2'}
            mock_container.resources.limits = None

            mock_pod = Mock()
            mock_pod.metadata.name = 'gpu-pod-1'
            mock_pod.metadata.namespace = 'kube-system'
            mock_pod.metadata.creation_timestamp = datetime(2024, 1, 1, tzinfo=timezone.utc)
            mock_pod.metadata.labels = {}
            mock_pod.metadata.annotations = None
            mock_pod.spec.containers = [mock_container]
            mock_pod.spec.node_name = 'gpu-node-1'
            mock_pod.status.phase = 'Running'

            mock_pod_list = Mock()
            mock_pod_list.items = [mock_pod]

            monitor.v1.list_pod_for_all_namespaces = Mock(return_value=mock_pod_list)

            pods = monitor.get_gpu_pods()

            monitor.v1.list_pod_for_all_namespaces.assert_called_once()
            assert len(pods) == 1
            assert pods[0]['gpu_requests'] == 2

    @patch('monitors.kubernetes_monitor.config.load_kube_config')
    @patch('monitors.kubernetes_monitor.client.CoreV1Api')
    @patch('monitors.kubernetes_monitor.client.AppsV1Api')
    @patch('monitors.kubernetes_monitor.client.CustomObjectsApi')
    @patch('os.path.exists')
    def test_get_gpu_pods_multiple_containers(self, mock_exists, mock_custom, mock_apps, mock_core, mock_load_config):
        """Test getting GPU pods with multiple containers."""
        mock_exists.return_value = True

        monitor = KubernetesMonitor()

        # Mock pod with multiple containers
        mock_container1 = Mock()
        mock_container1.name = 'container1'
        mock_container1.resources.requests = {'nvidia.com/gpu': '1'}
        mock_container1.resources.limits = {'nvidia.com/gpu': '1'}

        mock_container2 = Mock()
        mock_container2.name = 'container2'
        mock_container2.resources.requests = {'nvidia.com/gpu': '2'}
        mock_container2.resources.limits = {'nvidia.com/gpu': '2'}

        mock_pod = Mock()
        mock_pod.metadata.name = 'multi-gpu-pod'
        mock_pod.metadata.namespace = 'default'
        mock_pod.metadata.creation_timestamp = datetime(2024, 1, 1, tzinfo=timezone.utc)
        mock_pod.metadata.labels = None
        mock_pod.metadata.annotations = None
        mock_pod.spec.containers = [mock_container1, mock_container2]
        mock_pod.spec.node_name = 'gpu-node-1'
        mock_pod.status.phase = 'Running'

        mock_pod_list = Mock()
        mock_pod_list.items = [mock_pod]

        monitor.v1.list_namespaced_pod = Mock(return_value=mock_pod_list)

        pods = monitor.get_gpu_pods()

        assert len(pods) == 1
        assert pods[0]['gpu_requests'] == 3  # 1 + 2
        assert pods[0]['gpu_limits'] == 3
        assert len(pods[0]['containers']) == 2

    @patch('monitors.kubernetes_monitor.config.load_kube_config')
    @patch('monitors.kubernetes_monitor.client.CoreV1Api')
    @patch('monitors.kubernetes_monitor.client.AppsV1Api')
    @patch('monitors.kubernetes_monitor.client.CustomObjectsApi')
    @patch('os.path.exists')
    def test_get_gpu_pods_no_resources(self, mock_exists, mock_custom, mock_apps, mock_core, mock_load_config):
        """Test getting GPU pods that don't request GPU resources."""
        mock_exists.return_value = True

        monitor = KubernetesMonitor()

        # Mock pod without GPU resources
        mock_container = Mock()
        mock_container.name = 'container'
        mock_container.resources = None

        mock_pod = Mock()
        mock_pod.metadata.name = 'cpu-pod'
        mock_pod.metadata.namespace = 'default'
        mock_pod.metadata.creation_timestamp = datetime(2024, 1, 1, tzinfo=timezone.utc)
        mock_pod.metadata.labels = None
        mock_pod.metadata.annotations = None
        mock_pod.spec.containers = [mock_container]
        mock_pod.spec.node_name = 'cpu-node'
        mock_pod.status.phase = 'Running'

        mock_pod_list = Mock()
        mock_pod_list.items = [mock_pod]

        monitor.v1.list_namespaced_pod = Mock(return_value=mock_pod_list)

        pods = monitor.get_gpu_pods()

        assert len(pods) == 0

    @patch('monitors.kubernetes_monitor.config.load_kube_config')
    @patch('monitors.kubernetes_monitor.client.CoreV1Api')
    @patch('monitors.kubernetes_monitor.client.AppsV1Api')
    @patch('monitors.kubernetes_monitor.client.CustomObjectsApi')
    @patch('os.path.exists')
    def test_get_gpu_pods_empty(self, mock_exists, mock_custom, mock_apps, mock_core, mock_load_config):
        """Test getting GPU pods when none exist."""
        mock_exists.return_value = True

        monitor = KubernetesMonitor()

        mock_pod_list = Mock()
        mock_pod_list.items = []

        monitor.v1.list_namespaced_pod = Mock(return_value=mock_pod_list)

        pods = monitor.get_gpu_pods()

        assert len(pods) == 0

    @patch('monitors.kubernetes_monitor.config.load_kube_config')
    @patch('monitors.kubernetes_monitor.client.CoreV1Api')
    @patch('monitors.kubernetes_monitor.client.AppsV1Api')
    @patch('monitors.kubernetes_monitor.client.CustomObjectsApi')
    @patch('os.path.exists')
    def test_get_gpu_pods_error(self, mock_exists, mock_custom, mock_apps, mock_core, mock_load_config):
        """Test error handling for get_gpu_pods."""
        mock_exists.return_value = True

        monitor = KubernetesMonitor()
        monitor.v1.list_namespaced_pod = Mock(side_effect=ApiException(500, 'API error'))

        pods = monitor.get_gpu_pods()

        assert len(pods) == 0


class TestGetGPUUtilization:
    """Test cases for get_gpu_utilization method."""

    @patch('monitors.kubernetes_monitor.config.load_kube_config')
    @patch('monitors.kubernetes_monitor.client.CoreV1Api')
    @patch('monitors.kubernetes_monitor.client.AppsV1Api')
    @patch('monitors.kubernetes_monitor.client.CustomObjectsApi')
    @patch('os.path.exists')
    @patch('os.uname')
    def test_get_gpu_utilization_not_available(self, mock_uname, mock_exists, mock_custom, mock_apps, mock_core, mock_load_config):
        """Test get_gpu_utilization when GPU not available."""
        mock_exists.return_value = True

        monitor = KubernetesMonitor()
        monitor.gpu_available = False

        pods = [{'pod_name': 'pod-1', 'node_name': 'node-1'}]

        metrics = monitor.get_gpu_utilization(pods)

        assert len(metrics) == 0

    @patch('monitors.kubernetes_monitor.config.load_kube_config')
    @patch('monitors.kubernetes_monitor.client.CoreV1Api')
    @patch('monitors.kubernetes_monitor.client.AppsV1Api')
    @patch('monitors.kubernetes_monitor.client.CustomObjectsApi')
    @patch('os.path.exists')
    @patch('os.uname')
    def test_get_gpu_utilization_with_sampler(self, mock_uname, mock_exists, mock_custom, mock_apps, mock_core, mock_load_config):
        """Test get_gpu_utilization with shared NvmlSampler."""
        mock_exists.return_value = True
        mock_uname.return_value = Mock(nodename='gpu-node-1')

        monitor = KubernetesMonitor()

        # Setup mock sampler
        mock_sampler = Mock()
        mock_sampler.is_available.return_value = True
        mock_sampler.get_latest_metrics.return_value = [
            {
                'gpu_index': 0,
                'gpu_utilization': 85.5,
                'memory_utilization': 45.2,
                'memory_used_mb': 4096,
                'memory_total_mb': 8192,
                'temperature_c': 65,
                'power_usage_w': 120.5,
                'is_idle': False,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
        ]
        monitor.nvml_sampler = mock_sampler
        monitor.gpu_available = True

        pods = [{
            'pod_name': 'gpu-pod-1',
            'namespace': 'default',
            'node_name': 'gpu-node-1',
            'gpu_requests': 1
        }]

        metrics = monitor.get_gpu_utilization(pods)

        assert len(metrics) == 1
        assert metrics[0]['pod_name'] == 'gpu-pod-1'
        assert metrics[0]['gpu_utilization'] == 85.5
        assert metrics[0]['is_idle'] == False

    @patch('monitors.kubernetes_monitor.config.load_kube_config')
    @patch('monitors.kubernetes_monitor.client.CoreV1Api')
    @patch('monitors.kubernetes_monitor.client.AppsV1Api')
    @patch('monitors.kubernetes_monitor.client.CustomObjectsApi')
    @patch('os.path.exists')
    @patch('os.uname')
    def test_get_gpu_utilization_inline_read(self, mock_uname, mock_exists, mock_custom, mock_apps, mock_core, mock_load_config):
        """Test get_gpu_utilization with inline pynvml read."""
        mock_exists.return_value = True
        mock_uname.return_value = Mock(nodename='gpu-node-1')

        with patch('monitors.kubernetes_monitor.PYNVML_AVAILABLE', True), \
             patch('monitors.kubernetes_monitor.pynvml') as mock_pynvml:
            monitor = KubernetesMonitor()
            monitor.gpu_available = True

            # Mock pynvml functions
            mock_handle = Mock()
            mock_pynvml.nvmlDeviceGetCount.return_value = 1
            mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = mock_handle
            mock_pynvml.nvmlDeviceGetUtilizationRates.return_value = Mock(gpu=75.5, memory=50.0)
            mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = Mock(
                used=4*1024*1024*1024,
                total=8*1024*1024*1024
            )
            mock_pynvml.nvmlDeviceGetTemperature.return_value = 70
            mock_pynvml.nvmlDeviceGetPowerUsage.return_value = 150000

            pods = [{
                'pod_name': 'gpu-pod-1',
                'namespace': 'default',
                'node_name': 'gpu-node-1'
            }]

            metrics = monitor.get_gpu_utilization(pods)

            assert len(metrics) == 1
            assert metrics[0]['gpu_utilization'] == 75.5

    @patch('monitors.kubernetes_monitor.config.load_kube_config')
    @patch('monitors.kubernetes_monitor.client.CoreV1Api')
    @patch('monitors.kubernetes_monitor.client.AppsV1Api')
    @patch('monitors.kubernetes_monitor.client.CustomObjectsApi')
    @patch('os.path.exists')
    @patch('os.uname')
    def test_get_gpu_utilization_error(self, mock_uname, mock_exists, mock_custom, mock_apps, mock_core, mock_load_config):
        """Test error handling in get_gpu_utilization."""
        mock_exists.return_value = True
        mock_uname.return_value = Mock(nodename='gpu-node-1')

        monitor = KubernetesMonitor()
        monitor.gpu_available = True
        monitor.nvml_sampler = Mock()
        monitor.nvml_sampler.is_available.return_value = False

        # Simulate error in reading GPUs
        monitor._read_gpus_inline = Mock(side_effect=Exception('GPU read error'))

        pods = [{'pod_name': 'gpu-pod-1', 'node_name': 'gpu-node-1'}]

        metrics = monitor.get_gpu_utilization(pods)

        assert len(metrics) == 0


class TestReadGPUsInline:
    """Test cases for _read_gpus_inline method."""

    @patch('monitors.kubernetes_monitor.config.load_kube_config')
    @patch('monitors.kubernetes_monitor.client.CoreV1Api')
    @patch('monitors.kubernetes_monitor.client.AppsV1Api')
    @patch('monitors.kubernetes_monitor.client.CustomObjectsApi')
    @patch('os.path.exists')
    def test_read_gpus_inline_success(self, mock_exists, mock_custom, mock_apps, mock_core, mock_load_config):
        """Test inline GPU reading."""
        mock_exists.return_value = True

        with patch('monitors.kubernetes_monitor.PYNVML_AVAILABLE', True), \
             patch('monitors.kubernetes_monitor.pynvml') as mock_pynvml:
            monitor = KubernetesMonitor()

            # Mock pynvml
            mock_handle = Mock()
            mock_pynvml.nvmlDeviceGetCount.return_value = 2
            mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = mock_handle
            mock_pynvml.nvmlDeviceGetUtilizationRates.side_effect = [
                Mock(gpu=80.0, memory=60.0),
                Mock(gpu=5.0, memory=5.0)
            ]
            mock_pynvml.nvmlDeviceGetMemoryInfo.side_effect = [
                Mock(used=6*1024*1024*1024, total=8*1024*1024*1024),
                Mock(used=1*1024*1024*1024, total=8*1024*1024*1024)
            ]
            mock_pynvml.nvmlDeviceGetTemperature.side_effect = [70, 50]
            mock_pynvml.nvmlDeviceGetPowerUsage.side_effect = [150000, 50000]

            samples = monitor._read_gpus_inline()

            assert len(samples) == 2
            assert samples[0]['gpu_index'] == 0
            assert samples[0]['gpu_utilization'] == 80.0
            assert samples[0]['is_idle'] == False
            assert samples[1]['gpu_index'] == 1
            assert samples[1]['is_idle'] == True

    @patch('monitors.kubernetes_monitor.config.load_kube_config')
    @patch('monitors.kubernetes_monitor.client.CoreV1Api')
    @patch('monitors.kubernetes_monitor.client.AppsV1Api')
    @patch('monitors.kubernetes_monitor.client.CustomObjectsApi')
    @patch('os.path.exists')
    def test_read_gpus_inline_no_devices(self, mock_exists, mock_custom, mock_apps, mock_core, mock_load_config):
        """Test inline GPU reading with no devices."""
        mock_exists.return_value = True

        with patch('monitors.kubernetes_monitor.PYNVML_AVAILABLE', True), \
             patch('monitors.kubernetes_monitor.pynvml') as mock_pynvml:
            monitor = KubernetesMonitor()

            mock_pynvml.nvmlDeviceGetCount.return_value = 0

            samples = monitor._read_gpus_inline()

            assert len(samples) == 0

    @patch('monitors.kubernetes_monitor.config.load_kube_config')
    @patch('monitors.kubernetes_monitor.client.CoreV1Api')
    @patch('monitors.kubernetes_monitor.client.AppsV1Api')
    @patch('monitors.kubernetes_monitor.client.CustomObjectsApi')
    @patch('os.path.exists')
    def test_read_gpus_inline_temperature_error(self, mock_exists, mock_custom, mock_apps, mock_core, mock_load_config):
        """Test inline GPU reading with temperature error."""
        mock_exists.return_value = True

        with patch('monitors.kubernetes_monitor.PYNVML_AVAILABLE', True), \
             patch('monitors.kubernetes_monitor.pynvml') as mock_pynvml:
            monitor = KubernetesMonitor()

            mock_handle = Mock()
            mock_pynvml.nvmlDeviceGetCount.return_value = 1
            mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = mock_handle
            mock_pynvml.nvmlDeviceGetUtilizationRates.return_value = Mock(gpu=50.0, memory=30.0)
            mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = Mock(
                used=2*1024*1024*1024,
                total=8*1024*1024*1024
            )
            mock_pynvml.nvmlDeviceGetTemperature.side_effect = Exception('Temp error')
            mock_pynvml.nvmlDeviceGetPowerUsage.return_value = 100000

            samples = monitor._read_gpus_inline()

            assert len(samples) == 1
            assert samples[0]['temperature_c'] is None
            assert samples[0]['power_usage_w'] == 100.0


class TestGetCostData:
    """Test cases for get_cost_data method."""

    @patch('monitors.kubernetes_monitor.config.load_kube_config')
    @patch('monitors.kubernetes_monitor.client.CoreV1Api')
    @patch('monitors.kubernetes_monitor.client.AppsV1Api')
    @patch('monitors.kubernetes_monitor.client.CustomObjectsApi')
    @patch('os.path.exists')
    def test_get_cost_data_success(self, mock_exists, mock_custom, mock_apps, mock_core, mock_load_config):
        """Test getting cost data successfully."""
        mock_exists.return_value = True

        monitor = KubernetesMonitor()

        # Mock node info
        mock_node = Mock()
        mock_node.metadata.labels = {
            'node.kubernetes.io/instance-type': 't3.medium',
            'accelerator': 'nvidia-tesla-t4'
        }
        monitor.v1.read_node = Mock(return_value=mock_node)

        pods = [{
            'pod_name': 'gpu-pod-1',
            'namespace': 'default',
            'node_name': 'gpu-node-1',
            'gpu_requests': 1
        }]

        cost_data = monitor.get_cost_data(pods)

        assert len(cost_data) == 1
        assert cost_data[0]['pod_name'] == 'gpu-pod-1'
        assert cost_data[0]['cloud_provider'] == 'kubernetes'
        assert cost_data[0]['cost'] > 0

    @patch('monitors.kubernetes_monitor.config.load_kube_config')
    @patch('monitors.kubernetes_monitor.client.CoreV1Api')
    @patch('monitors.kubernetes_monitor.client.AppsV1Api')
    @patch('monitors.kubernetes_monitor.client.CustomObjectsApi')
    @patch('os.path.exists')
    def test_get_cost_data_empty(self, mock_exists, mock_custom, mock_apps, mock_core, mock_load_config):
        """Test getting cost data with no pods."""
        mock_exists.return_value = True

        monitor = KubernetesMonitor()

        cost_data = monitor.get_cost_data([])

        assert len(cost_data) == 0

    @patch('monitors.kubernetes_monitor.config.load_kube_config')
    @patch('monitors.kubernetes_monitor.client.CoreV1Api')
    @patch('monitors.kubernetes_monitor.client.AppsV1Api')
    @patch('monitors.kubernetes_monitor.client.CustomObjectsApi')
    @patch('os.path.exists')
    def test_get_cost_data_error(self, mock_exists, mock_custom, mock_apps, mock_core, mock_load_config):
        """Test error handling in get_cost_data."""
        mock_exists.return_value = True

        monitor = KubernetesMonitor()
        monitor._get_node_info = Mock(side_effect=Exception('Node error'))

        pods = [{'pod_name': 'gpu-pod-1', 'node_name': 'gpu-node-1'}]

        cost_data = monitor.get_cost_data(pods)

        assert len(cost_data) == 0


class TestGetNodeInfo:
    """Test cases for _get_node_info method."""

    @patch('monitors.kubernetes_monitor.config.load_kube_config')
    @patch('monitors.kubernetes_monitor.client.CoreV1Api')
    @patch('monitors.kubernetes_monitor.client.AppsV1Api')
    @patch('monitors.kubernetes_monitor.client.CustomObjectsApi')
    @patch('os.path.exists')
    def test_get_node_info_success(self, mock_exists, mock_custom, mock_apps, mock_core, mock_load_config):
        """Test getting node info successfully."""
        mock_exists.return_value = True

        monitor = KubernetesMonitor()

        mock_node = Mock()
        mock_node.metadata.labels = {
            'node.kubernetes.io/instance-type': 'm5.xlarge',
            'topology.kubernetes.io/zone': 'us-east-1b'
        }

        monitor.v1.read_node = Mock(return_value=mock_node)

        node_info = monitor._get_node_info('gpu-node-1')

        assert node_info['instance_type'] == 'm5.xlarge'
        assert node_info['zone'] == 'us-east-1b'

    @patch('monitors.kubernetes_monitor.config.load_kube_config')
    @patch('monitors.kubernetes_monitor.client.CoreV1Api')
    @patch('monitors.kubernetes_monitor.client.AppsV1Api')
    @patch('monitors.kubernetes_monitor.client.CustomObjectsApi')
    @patch('os.path.exists')
    def test_get_node_info_error(self, mock_exists, mock_custom, mock_apps, mock_core, mock_load_config):
        """Test error handling in _get_node_info."""
        mock_exists.return_value = True

        monitor = KubernetesMonitor()
        monitor.v1.read_node = Mock(side_effect=Exception('Node read error'))

        node_info = monitor._get_node_info('gpu-node-1')

        assert len(node_info) == 0


class TestEstimatePodCost:
    """Test cases for _estimate_pod_cost method."""

    @patch('monitors.kubernetes_monitor.config.load_kube_config')
    @patch('monitors.kubernetes_monitor.client.CoreV1Api')
    @patch('monitors.kubernetes_monitor.client.AppsV1Api')
    @patch('monitors.kubernetes_monitor.client.CustomObjectsApi')
    @patch('os.path.exists')
    def test_estimate_pod_cost_with_gpu(self, mock_exists, mock_custom, mock_apps, mock_core, mock_load_config):
        """Test pod cost estimation with GPU."""
        mock_exists.return_value = True

        monitor = KubernetesMonitor()

        pod = {'gpu_requests': 1}
        node_info = {
            'instance_type': 't3.medium',
            'labels': {'accelerator': 'nvidia-tesla-t4'}
        }

        cost = monitor._estimate_pod_cost(pod, node_info)

        assert cost > 0
        assert isinstance(cost, float)

    @patch('monitors.kubernetes_monitor.config.load_kube_config')
    @patch('monitors.kubernetes_monitor.client.CoreV1Api')
    @patch('monitors.kubernetes_monitor.client.AppsV1Api')
    @patch('monitors.kubernetes_monitor.client.CustomObjectsApi')
    @patch('os.path.exists')
    def test_estimate_pod_cost_without_gpu(self, mock_exists, mock_custom, mock_apps, mock_core, mock_load_config):
        """Test pod cost estimation without GPU."""
        mock_exists.return_value = True

        monitor = KubernetesMonitor()

        pod = {'gpu_requests': 0}
        node_info = {'instance_type': 't3.medium', 'labels': {}}

        cost = monitor._estimate_pod_cost(pod, node_info)

        assert cost == 0.04  # Rounded from 0.0416

    @patch('monitors.kubernetes_monitor.config.load_kube_config')
    @patch('monitors.kubernetes_monitor.client.CoreV1Api')
    @patch('monitors.kubernetes_monitor.client.AppsV1Api')
    @patch('monitors.kubernetes_monitor.client.CustomObjectsApi')
    @patch('os.path.exists')
    def test_estimate_pod_cost_unknown_instance(self, mock_exists, mock_custom, mock_apps, mock_core, mock_load_config):
        """Test pod cost estimation with unknown instance type."""
        mock_exists.return_value = True

        monitor = KubernetesMonitor()

        pod = {'gpu_requests': 0}
        node_info = {'instance_type': 'unknown-type', 'labels': {}}

        cost = monitor._estimate_pod_cost(pod, node_info)

        assert cost == 0.04  # Default, rounded from 0.0416


class TestAdditionalCoverage:
    """Additional test cases to improve coverage."""

    @patch('monitors.kubernetes_monitor.config.load_kube_config')
    @patch('monitors.kubernetes_monitor.client.CoreV1Api')
    @patch('monitors.kubernetes_monitor.client.AppsV1Api')
    @patch('monitors.kubernetes_monitor.client.CustomObjectsApi')
    @patch('os.path.exists')
    def test_get_gpu_pods_generic_exception(self, mock_exists, mock_custom, mock_apps, mock_core, mock_load_config):
        """Test error handling for generic exceptions in get_gpu_pods."""
        mock_exists.return_value = True

        monitor = KubernetesMonitor()
        monitor.v1.list_namespaced_pod = Mock(side_effect=RuntimeError('Unexpected error'))

        pods = monitor.get_gpu_pods()

        assert len(pods) == 0

    @patch('monitors.kubernetes_monitor.config.load_kube_config')
    @patch('monitors.kubernetes_monitor.client.CoreV1Api')
    @patch('monitors.kubernetes_monitor.client.AppsV1Api')
    @patch('monitors.kubernetes_monitor.client.CustomObjectsApi')
    @patch('os.path.exists')
    def test_get_cost_data_generic_exception(self, mock_exists, mock_custom, mock_apps, mock_core, mock_load_config):
        """Test generic exception handling in get_cost_data."""
        mock_exists.return_value = True

        monitor = KubernetesMonitor()
        monitor._get_node_info = Mock(side_effect=RuntimeError('Node error'))

        pods = [{'pod_name': 'gpu-pod-1', 'node_name': 'gpu-node-1'}]

        cost_data = monitor.get_cost_data(pods)

        assert len(cost_data) == 0


    @patch('monitors.kubernetes_monitor.config.load_kube_config')
    @patch('monitors.kubernetes_monitor.client.CoreV1Api')
    @patch('monitors.kubernetes_monitor.client.AppsV1Api')
    @patch('monitors.kubernetes_monitor.client.CustomObjectsApi')
    @patch('os.path.exists')
    @patch('os.uname')
    def test_get_gpu_utilization_multiple_gpus(self, mock_uname, mock_exists, mock_custom, mock_apps, mock_core, mock_load_config):
        """Test get_gpu_utilization with multiple GPUs."""
        mock_exists.return_value = True
        mock_uname.return_value = Mock(nodename='gpu-node-1')

        monitor = KubernetesMonitor()

        # Setup mock sampler with multiple GPUs
        mock_sampler = Mock()
        mock_sampler.is_available.return_value = True
        mock_sampler.get_latest_metrics.return_value = [
            {
                'gpu_index': 0,
                'gpu_utilization': 85.0,
                'memory_utilization': 60.0,
                'memory_used_mb': 4096,
                'memory_total_mb': 8192,
                'temperature_c': 65,
                'power_usage_w': 120.0,
                'is_idle': False,
                'timestamp': datetime.now(timezone.utc).isoformat()
            },
            {
                'gpu_index': 1,
                'gpu_utilization': 40.0,
                'memory_utilization': 30.0,
                'memory_used_mb': 2048,
                'memory_total_mb': 8192,
                'temperature_c': 55,
                'power_usage_w': 80.0,
                'is_idle': False,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
        ]
        monitor.nvml_sampler = mock_sampler
        monitor.gpu_available = True

        pods = [{
            'pod_name': 'gpu-pod-1',
            'namespace': 'default',
            'node_name': 'gpu-node-1'
        }]

        metrics = monitor.get_gpu_utilization(pods)

        assert len(metrics) == 2
        assert metrics[0]['gpu_index'] == 0
        assert metrics[1]['gpu_index'] == 1

    @patch('monitors.kubernetes_monitor.config.load_kube_config')
    @patch('monitors.kubernetes_monitor.client.CoreV1Api')
    @patch('monitors.kubernetes_monitor.client.AppsV1Api')
    @patch('monitors.kubernetes_monitor.client.CustomObjectsApi')
    @patch('os.path.exists')
    @patch('os.uname')
    def test_get_gpu_utilization_pod_node_mismatch(self, mock_uname, mock_exists, mock_custom, mock_apps, mock_core, mock_load_config):
        """Test get_gpu_utilization when GPU node doesn't match pod node."""
        mock_exists.return_value = True
        mock_uname.return_value = Mock(nodename='different-node')

        monitor = KubernetesMonitor()

        mock_sampler = Mock()
        mock_sampler.is_available.return_value = True
        mock_sampler.get_latest_metrics.return_value = [
            {
                'gpu_index': 0,
                'gpu_utilization': 85.0,
                'memory_utilization': 60.0,
                'memory_used_mb': 4096,
                'memory_total_mb': 8192,
                'temperature_c': 65,
                'power_usage_w': 120.0,
                'is_idle': False,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
        ]
        monitor.nvml_sampler = mock_sampler
        monitor.gpu_available = True

        pods = [{
            'pod_name': 'gpu-pod-1',
            'namespace': 'default',
            'node_name': 'gpu-node-1'  # Different from uname
        }]

        metrics = monitor.get_gpu_utilization(pods)

        # No metrics should be returned if node doesn't match
        assert len(metrics) == 0

    @patch('monitors.kubernetes_monitor.config.load_kube_config')
    @patch('monitors.kubernetes_monitor.client.CoreV1Api')
    @patch('monitors.kubernetes_monitor.client.AppsV1Api')
    @patch('monitors.kubernetes_monitor.client.CustomObjectsApi')
    @patch('os.path.exists')
    def test_estimate_pod_cost_with_multiple_gpus(self, mock_exists, mock_custom, mock_apps, mock_core, mock_load_config):
        """Test pod cost estimation with multiple GPUs."""
        mock_exists.return_value = True

        monitor = KubernetesMonitor()

        pod = {'gpu_requests': 4}
        node_info = {
            'instance_type': 'g4dn.xlarge',
            'labels': {'accelerator': 'nvidia-tesla-a100'}
        }

        cost = monitor._estimate_pod_cost(pod, node_info)

        assert cost > 0
        # Should include both base cost and 4x GPU cost
        assert cost > 11.0  # 4 * 2.93 = 11.72

    @patch('monitors.kubernetes_monitor.config.load_kube_config')
    @patch('monitors.kubernetes_monitor.client.CoreV1Api')
    @patch('monitors.kubernetes_monitor.client.AppsV1Api')
    @patch('monitors.kubernetes_monitor.client.CustomObjectsApi')
    @patch('os.path.exists')
    def test_get_node_info_with_partial_labels(self, mock_exists, mock_custom, mock_apps, mock_core, mock_load_config):
        """Test _get_node_info with node having partial labels."""
        mock_exists.return_value = True

        monitor = KubernetesMonitor()

        mock_node = Mock()
        mock_node.metadata.labels = {
            'node.kubernetes.io/instance-type': 'c5.xlarge'
        }

        monitor.v1.read_node = Mock(return_value=mock_node)

        node_info = monitor._get_node_info('gpu-node-1')

        assert 'instance_type' in node_info
        assert 'zone' in node_info
        assert node_info['instance_type'] == 'c5.xlarge'
        assert node_info['zone'] == 'unknown'

    @patch('monitors.kubernetes_monitor.config.load_kube_config')
    @patch('monitors.kubernetes_monitor.client.CoreV1Api')
    @patch('monitors.kubernetes_monitor.client.AppsV1Api')
    @patch('monitors.kubernetes_monitor.client.CustomObjectsApi')
    @patch('os.path.exists')
    def test_get_namespace_resource_usage_multiple_containers(self, mock_exists, mock_custom, mock_apps, mock_core, mock_load_config):
        """Test namespace resource usage with multiple GPU containers."""
        mock_exists.return_value = True

        monitor = KubernetesMonitor()

        mock_container1 = Mock()
        mock_container1.resources.requests = {'nvidia.com/gpu': '2'}
        mock_container1.resources.limits = {'nvidia.com/gpu': '2'}

        mock_container2 = Mock()
        mock_container2.resources.requests = {'nvidia.com/gpu': '1'}
        mock_container2.resources.limits = {'nvidia.com/gpu': '1'}

        mock_pod = Mock()
        mock_pod.spec.containers = [mock_container1, mock_container2]
        mock_pod.status.phase = 'Running'

        mock_pod_list = Mock()
        mock_pod_list.items = [mock_pod]

        monitor.v1.list_namespaced_pod = Mock(return_value=mock_pod_list)

        usage = monitor.get_namespace_resource_usage('default')

        assert usage['total_gpu_requests'] == 3
        assert usage['total_gpu_limits'] == 3


class TestGetNamespaceResourceUsage:
    """Test cases for get_namespace_resource_usage method."""

    @patch('monitors.kubernetes_monitor.config.load_kube_config')
    @patch('monitors.kubernetes_monitor.client.CoreV1Api')
    @patch('monitors.kubernetes_monitor.client.AppsV1Api')
    @patch('monitors.kubernetes_monitor.client.CustomObjectsApi')
    @patch('os.path.exists')
    def test_get_namespace_resource_usage_success(self, mock_exists, mock_custom, mock_apps, mock_core, mock_load_config):
        """Test getting namespace resource usage."""
        mock_exists.return_value = True

        monitor = KubernetesMonitor()

        # Mock pod with GPU
        mock_container = Mock()
        mock_container.resources.requests = {'nvidia.com/gpu': '2'}
        mock_container.resources.limits = {'nvidia.com/gpu': '2'}

        mock_pod = Mock()
        mock_pod.metadata.name = 'gpu-pod-1'
        mock_pod.spec.containers = [mock_container]
        mock_pod.status.phase = 'Running'

        mock_pod_list = Mock()
        mock_pod_list.items = [mock_pod]

        monitor.v1.list_namespaced_pod = Mock(return_value=mock_pod_list)

        usage = monitor.get_namespace_resource_usage('default')

        assert usage['namespace'] == 'default'
        assert usage['pod_count'] == 1
        assert usage['total_gpu_requests'] == 2
        assert usage['total_gpu_limits'] == 2

    @patch('monitors.kubernetes_monitor.config.load_kube_config')
    @patch('monitors.kubernetes_monitor.client.CoreV1Api')
    @patch('monitors.kubernetes_monitor.client.AppsV1Api')
    @patch('monitors.kubernetes_monitor.client.CustomObjectsApi')
    @patch('os.path.exists')
    def test_get_namespace_resource_usage_no_running_pods(self, mock_exists, mock_custom, mock_apps, mock_core, mock_load_config):
        """Test getting namespace resource usage with no running pods."""
        mock_exists.return_value = True

        monitor = KubernetesMonitor()

        mock_container = Mock()
        mock_container.resources = None

        mock_pod = Mock()
        mock_pod.spec.containers = [mock_container]
        mock_pod.status.phase = 'Pending'

        mock_pod_list = Mock()
        mock_pod_list.items = [mock_pod]

        monitor.v1.list_namespaced_pod = Mock(return_value=mock_pod_list)

        usage = monitor.get_namespace_resource_usage('default')

        assert usage['pod_count'] == 0

    @patch('monitors.kubernetes_monitor.config.load_kube_config')
    @patch('monitors.kubernetes_monitor.client.CoreV1Api')
    @patch('monitors.kubernetes_monitor.client.AppsV1Api')
    @patch('monitors.kubernetes_monitor.client.CustomObjectsApi')
    @patch('os.path.exists')
    def test_get_namespace_resource_usage_error(self, mock_exists, mock_custom, mock_apps, mock_core, mock_load_config):
        """Test error handling in get_namespace_resource_usage."""
        mock_exists.return_value = True

        monitor = KubernetesMonitor()
        monitor.v1.list_namespaced_pod = Mock(side_effect=Exception('API error'))

        usage = monitor.get_namespace_resource_usage('default')

        # Should return empty dict when there's an error
        assert isinstance(usage, dict)
        assert len(usage) == 0
