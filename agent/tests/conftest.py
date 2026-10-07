import pytest
import os
import sys
from pathlib import Path

_AGENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_AGENT_ROOT))
sys.path.insert(0, str(_AGENT_ROOT / "src"))
import tempfile
from unittest.mock import Mock, patch
try:
    from moto import mock_ec2, mock_cloudwatch, mock_costexplorer
except ImportError:
    # moto v5+ uses mock_aws for all services
    from moto import mock_aws as mock_ec2
    mock_cloudwatch = mock_ec2
    mock_costexplorer = mock_ec2
import boto3
from datetime import datetime, timedelta

# Set test environment variables
os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'
os.environ['AWS_ACCESS_KEY_ID'] = 'testing'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'testing'
os.environ['AWS_SECURITY_TOKEN'] = 'testing'
os.environ['AWS_SESSION_TOKEN'] = 'testing'
os.environ['DASHBOARD_API_URL'] = 'http://localhost:8000'
os.environ['DASHBOARD_API_KEY'] = 'test-api-key'
os.environ['DB_TYPE'] = 'sqlite'
os.environ['LOG_LEVEL'] = 'ERROR'


@pytest.fixture(autouse=True)
def _stub_nvml_init():
    """NVML init can block on GPU-less runners when cluster tests construct many monitors."""
    patches = [
        patch("pynvml.nvmlInit", return_value=None),
        patch("pynvml.nvmlShutdown", return_value=None),
    ]
    try:
        import monitors.gpu_cluster_monitor as gcm  # noqa: WPS433

        if getattr(gcm, "PYNVML_AVAILABLE", False):
            patches.append(
                patch.object(gcm.pynvml, "nvmlInit", return_value=None),
            )
            patches.append(
                patch.object(gcm.pynvml, "nvmlShutdown", return_value=None),
            )
    except ImportError:
        pass

    for p in patches:
        p.start()
    try:
        yield
    finally:
        for p in reversed(patches):
            p.stop()


@pytest.fixture
def mock_aws_services():
    """Mock AWS services using moto."""
    with mock_ec2(), mock_cloudwatch(), mock_costexplorer():
        # Create mock EC2 client
        ec2_client = boto3.client('ec2', region_name='us-east-1')
        
        # Create mock CloudWatch client
        cloudwatch_client = boto3.client('cloudwatch', region_name='us-east-1')
        
        # Create mock Cost Explorer client
        costexplorer_client = boto3.client('ce', region_name='us-east-1')
        
        yield {
            'ec2': ec2_client,
            'cloudwatch': cloudwatch_client,
            'costexplorer': costexplorer_client
        }


@pytest.fixture
def sample_gpu_instance():
    """Sample GPU instance data for testing."""
    return {
        'instance_id': 'i-1234567890abcdef0',
        'instance_type': 'g4dn.xlarge',
        'status': 'running',
        'region': 'us-east-1',
        'experiment_id': 'exp-test-001',
        'metadata': {
            'launch_time': '2024-01-01T00:00:00Z',
            'tags': {'Environment': 'test'}
        }
    }


@pytest.fixture
def sample_metrics():
    """Sample GPU metrics data for testing."""
    return {
        'gpu_utilization': 85.5,
        'cpu_utilization': 72.3,
        'memory_utilization': 68.1,
        'gpu_memory_utilization': 45.2,
        'temperature': 65.0,
        'power_consumption': 120.5
    }


@pytest.fixture
def sample_cost_data():
    """Sample cost data for testing."""
    return {
        'date': '2024-01-01',
        'hourly_cost': 0.526,
        'daily_cost': 12.624,
        'monthly_cost': 383.52,
        'currency': 'USD'
    }


@pytest.fixture
def mock_dashboard_api():
    """Mock the dashboard API responses."""
    with patch('requests.post') as mock_post, \
         patch('requests.get') as mock_get:
        
        # Mock successful API responses
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {'success': True}
        
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {'success': True, 'data': []}
        
        yield {
            'post': mock_post,
            'get': mock_get
        }


@pytest.fixture
def mock_gpu_monitoring():
    """Mock GPU monitoring functions."""
    with patch('pynvml.nvmlInit'), \
         patch('pynvml.nvmlDeviceGetCount') as mock_count, \
         patch('pynvml.nvmlDeviceGetHandleByIndex') as mock_handle, \
         patch('pynvml.nvmlDeviceGetUtilizationRates') as mock_util, \
         patch('pynvml.nvmlDeviceGetMemoryInfo') as mock_memory, \
         patch('pynvml.nvmlDeviceGetTemperature') as mock_temp, \
         patch('pynvml.nvmlDeviceGetPowerUsage') as mock_power:
        
        # Mock GPU count
        mock_count.return_value = 1
        
        # Mock GPU handle
        mock_handle.return_value = Mock()
        
        # Mock utilization rates
        mock_util.return_value = Mock(gpu=85, memory=45)
        
        # Mock memory info
        mock_memory.return_value = Mock(used=1024*1024*1024, total=8*1024*1024*1024)
        
        # Mock temperature
        mock_temp.return_value = 65
        
        # Mock power usage
        mock_power.return_value = 120500  # in milliwatts
        
        yield {
            'count': mock_count,
            'handle': mock_handle,
            'utilization': mock_util,
            'memory': mock_memory,
            'temperature': mock_temp,
            'power': mock_power
        }


@pytest.fixture
def mock_system_monitoring():
    """Mock system monitoring functions."""
    with patch('psutil.cpu_percent') as mock_cpu, \
         patch('psutil.virtual_memory') as mock_memory:
        
        # Mock CPU usage
        mock_cpu.return_value = 72.3
        
        # Mock memory usage
        mock_memory.return_value = Mock(percent=68.1)
        
        yield {
            'cpu': mock_cpu,
            'memory': mock_memory
        }


@pytest.fixture
def mock_time():
    """Mock time for consistent testing."""
    from freezegun import freeze_time
    with freeze_time('2024-01-01 12:00:00'):
        yield


@pytest.fixture
def mock_logger():
    """Mock logger to avoid log output during tests."""
    with patch('src.aws_monitor.logger') as mock_log:
        yield mock_log
