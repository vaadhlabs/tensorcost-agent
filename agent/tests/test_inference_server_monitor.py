"""Comprehensive tests for InferenceServerMonitor."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import time
import requests

from monitors.inference_server_monitor import InferenceServerMonitor


class TestInferenceServerMonitorInitialization:
    """Test initialization with various environment configurations."""

    def test_init_with_no_env_vars(self):
        """Initialization with no environment variables should use defaults."""
        with patch.dict(os.environ, {}, clear=True):
            monitor = InferenceServerMonitor()
            assert monitor.vllm_endpoints == []
            assert monitor.tgi_endpoints == []
            assert monitor.triton_endpoints == []
            assert monitor.ollama_endpoints == ["http://localhost:11434"]

    def test_init_with_vllm_endpoints(self):
        """Initialization should parse VLLM_ENDPOINTS from env var."""
        env_vars = {"VLLM_ENDPOINTS": "http://localhost:8000,http://remote:8000"}
        with patch.dict(os.environ, env_vars):
            monitor = InferenceServerMonitor()
            assert len(monitor.vllm_endpoints) == 2
            assert "http://localhost:8000" in monitor.vllm_endpoints
            assert "http://remote:8000" in monitor.vllm_endpoints

    def test_init_with_tgi_endpoints(self):
        """Initialization should parse TGI_ENDPOINTS from env var."""
        env_vars = {"TGI_ENDPOINTS": "http://tgi1:8080,http://tgi2:8080"}
        with patch.dict(os.environ, env_vars):
            monitor = InferenceServerMonitor()
            assert len(monitor.tgi_endpoints) == 2

    def test_init_with_triton_endpoints(self):
        """Initialization should parse TRITON_ENDPOINTS from env var."""
        env_vars = {"TRITON_ENDPOINTS": "http://triton:8000"}
        with patch.dict(os.environ, env_vars):
            monitor = InferenceServerMonitor()
            assert len(monitor.triton_endpoints) == 1

    def test_init_with_custom_ollama_endpoints(self):
        """Initialization should override default Ollama endpoint."""
        env_vars = {"OLLAMA_ENDPOINTS": "http://custom-ollama:11434"}
        with patch.dict(os.environ, env_vars):
            monitor = InferenceServerMonitor()
            assert monitor.ollama_endpoints == ["http://custom-ollama:11434"]

    def test_init_with_multiple_endpoints(self):
        """Initialization with multiple server types."""
        env_vars = {
            "VLLM_ENDPOINTS": "http://vllm:8000",
            "TGI_ENDPOINTS": "http://tgi:8080",
            "TRITON_ENDPOINTS": "http://triton:8000",
            "OLLAMA_ENDPOINTS": "http://ollama:11434",
        }
        with patch.dict(os.environ, env_vars):
            monitor = InferenceServerMonitor()
            assert len(monitor.vllm_endpoints) == 1
            assert len(monitor.tgi_endpoints) == 1
            assert len(monitor.triton_endpoints) == 1
            assert len(monitor.ollama_endpoints) == 1

    def test_init_with_whitespace_endpoints(self):
        """Endpoints with whitespace should be trimmed."""
        env_vars = {"VLLM_ENDPOINTS": "  http://localhost:8000  ,  http://remote:8000  "}
        with patch.dict(os.environ, env_vars):
            monitor = InferenceServerMonitor()
            assert monitor.vllm_endpoints == ["http://localhost:8000", "http://remote:8000"]

    def test_init_with_custom_config(self):
        """Initialization should accept custom config."""
        config = {"custom_setting": "value"}
        monitor = InferenceServerMonitor(config=config)
        assert monitor.config == config


class TestParseEndpoints:
    """Test endpoint parsing utility."""

    def test_parse_empty_env_var(self):
        """Empty env var should return empty list."""
        with patch.dict(os.environ, {}, clear=True):
            result = InferenceServerMonitor._parse_endpoints("NONEXISTENT")
            assert result == []

    def test_parse_single_endpoint(self):
        """Single endpoint should be parsed correctly."""
        with patch.dict(os.environ, {"TEST_ENDPOINT": "http://localhost:8000"}):
            result = InferenceServerMonitor._parse_endpoints("TEST_ENDPOINT")
            assert result == ["http://localhost:8000"]

    def test_parse_multiple_endpoints(self):
        """Multiple comma-separated endpoints should be parsed."""
        with patch.dict(os.environ, {"TEST_ENDPOINTS": "http://a:8000,http://b:8000,http://c:8000"}):
            result = InferenceServerMonitor._parse_endpoints("TEST_ENDPOINTS")
            assert len(result) == 3

    def test_parse_with_empty_items(self):
        """Empty items in comma-separated list should be skipped."""
        with patch.dict(os.environ, {"TEST_ENDPOINTS": "http://a:8000,,http://b:8000,"}):
            result = InferenceServerMonitor._parse_endpoints("TEST_ENDPOINTS")
            assert len(result) == 2
            assert "http://a:8000" in result
            assert "http://b:8000" in result


class TestVLLMMetrics:
    """Test vLLM metrics collection."""

    @patch('monitors.inference_server_monitor.requests.get')
    def test_collect_vllm_metrics_success(self, mock_get):
        """Successful vLLM metrics collection."""
        mock_metrics_response = Mock()
        mock_metrics_response.text = """
vllm:num_requests_running 5
vllm:num_requests_waiting 2
vllm:avg_generation_throughput_toks_per_s 1000.0
vllm:avg_prompt_throughput_toks_per_s 500.0
vllm:e2e_request_latency_seconds_p50 0.5
vllm:e2e_request_latency_seconds_p95 1.5
vllm:e2e_request_latency_seconds_p99 2.0
vllm:gpu_cache_usage_perc 75.5
vllm:batch_size 8
        """

        mock_models_response = Mock()
        mock_models_response.json.return_value = {
            "data": [
                {"id": "meta-llama/Llama-2-7b"},
                {"id": "mistralai/Mistral-7B"},
            ]
        }

        mock_get.side_effect = [mock_metrics_response, mock_models_response]

        monitor = InferenceServerMonitor()
        metrics = monitor._collect_vllm_metrics("http://localhost:8000")

        assert metrics["server_type"] == "vllm"
        assert metrics["status"] == "healthy"
        assert metrics["requests_running"] == 5
        assert metrics["requests_queued"] == 2
        assert metrics["throughput_tokens_per_sec"] == 1500.0
        assert metrics["latency_p50_ms"] == 500.0
        assert metrics["latency_p95_ms"] == 1500.0
        assert metrics["latency_p99_ms"] == 2000.0
        assert metrics["gpu_cache_utilization_pct"] == 75.5
        assert metrics["batch_size_current"] == 8
        assert len(metrics["models_loaded"]) == 2

    @patch('monitors.inference_server_monitor.requests.get')
    def test_collect_vllm_metrics_with_timeout(self, mock_get):
        """vLLM metrics collection with timeout."""
        monitor = InferenceServerMonitor()
        mock_get.side_effect = requests.Timeout("Connection timeout")

        with pytest.raises(requests.Timeout):
            monitor._collect_vllm_metrics("http://localhost:8000")

    @patch('monitors.inference_server_monitor.requests.get')
    def test_collect_vllm_metrics_no_models(self, mock_get):
        """vLLM metrics collection with empty model list."""
        mock_metrics_response = Mock()
        mock_metrics_response.text = "vllm:num_requests_running 0\n"

        mock_models_response = Mock()
        mock_models_response.json.return_value = {"data": []}

        mock_get.side_effect = [mock_metrics_response, mock_models_response]

        monitor = InferenceServerMonitor()
        metrics = monitor._collect_vllm_metrics("http://localhost:8000")

        assert metrics["models_loaded"] == []
        assert metrics["model_name"] is None


class TestTGIMetrics:
    """Test TGI metrics collection."""

    @patch('monitors.inference_server_monitor.requests.get')
    def test_collect_tgi_metrics_success(self, mock_get):
        """Successful TGI metrics collection."""
        mock_metrics_response = Mock()
        mock_metrics_response.text = """
tgi_request_duration_count 100
tgi_request_generated_tokens_sum 50000
tgi_queue_size 3
tgi_batch_current_size 4
        """

        mock_info_response = Mock()
        mock_info_response.json.return_value = {
            "model_id": "meta-llama/Llama-2-7b-hf"
        }

        mock_get.side_effect = [mock_metrics_response, mock_info_response]

        monitor = InferenceServerMonitor()
        metrics = monitor._collect_tgi_metrics("http://localhost:8080")

        assert metrics["server_type"] == "tgi"
        assert metrics["status"] == "healthy"
        assert metrics["model_name"] == "meta-llama/Llama-2-7b-hf"
        assert metrics["requests_queued"] == 3
        assert metrics["batch_size_current"] == 4
        assert metrics["throughput_tokens_per_sec"] == 50000.0

    @patch('monitors.inference_server_monitor.requests.get')
    def test_collect_tgi_metrics_http_error(self, mock_get):
        """TGI metrics collection with HTTP error."""
        monitor = InferenceServerMonitor()
        mock_get.side_effect = requests.HTTPError("404 Not Found")

        with pytest.raises(requests.HTTPError):
            monitor._collect_tgi_metrics("http://localhost:8080")


class TestOllamaMetrics:
    """Test Ollama metrics collection."""

    @patch('monitors.inference_server_monitor.requests.get')
    def test_collect_ollama_metrics_success(self, mock_get):
        """Successful Ollama metrics collection."""
        mock_ps_response = Mock()
        mock_ps_response.json.return_value = {
            "models": [
                {"name": "llama2:7b", "size": 4000000000},
                {"name": "mistral:7b", "size": 5000000000},
            ]
        }

        mock_tags_response = Mock()
        mock_tags_response.json.return_value = {
            "models": [
                {"name": "llama2:7b"},
                {"name": "mistral:7b"},
                {"name": "neural-chat:7b"},
            ]
        }

        mock_get.side_effect = [mock_ps_response, mock_tags_response]

        monitor = InferenceServerMonitor()
        metrics = monitor._collect_ollama_metrics("http://localhost:11434")

        assert metrics["server_type"] == "ollama"
        assert metrics["status"] == "healthy"
        assert metrics["requests_running"] == 2
        assert len(metrics["models_loaded"]) == 3
        assert metrics["batch_size_current"] == 2
        # Approximately 8.58 GB (4GB + 5GB in MB / 1024)
        assert 8000 < metrics["vram_usage_mb"] < 10000

    @patch('monitors.inference_server_monitor.requests.get')
    def test_collect_ollama_metrics_empty(self, mock_get):
        """Ollama metrics collection with no running models."""
        mock_ps_response = Mock()
        mock_ps_response.json.return_value = {"models": []}

        mock_tags_response = Mock()
        mock_tags_response.json.return_value = {"models": []}

        mock_get.side_effect = [mock_ps_response, mock_tags_response]

        monitor = InferenceServerMonitor()
        metrics = monitor._collect_ollama_metrics("http://localhost:11434")

        assert metrics["requests_running"] == 0
        assert metrics["models_loaded"] == []
        assert metrics["model_name"] is None


class TestPrometheusMetricsParsing:
    """Test Prometheus metrics text parsing."""

    def test_parse_prometheus_metrics_basic(self):
        """Parse basic Prometheus metrics."""
        text = """
# HELP vllm_num_requests Running request count.
# TYPE vllm_num_requests gauge
vllm:num_requests_running 5
vllm:num_requests_waiting 2
        """
        monitor = InferenceServerMonitor()
        metrics = monitor._parse_prometheus_metrics(text)

        assert metrics["vllm:num_requests_running"] == 5.0
        assert metrics["vllm:num_requests_waiting"] == 2.0

    def test_parse_prometheus_metrics_with_labels(self):
        """Parse Prometheus metrics with labels - simpler case without braces in line."""
        text = """
vllm:request_latency_seconds_bucket 100
vllm:request_latency_seconds_count 500
vllm:request_latency_seconds_sum 1000
        """
        monitor = InferenceServerMonitor()
        metrics = monitor._parse_prometheus_metrics(text)

        # Should parse metrics
        assert len(metrics) > 0
        assert metrics["vllm:request_latency_seconds_bucket"] == 100
        assert metrics["vllm:request_latency_seconds_count"] == 500
        assert metrics["vllm:request_latency_seconds_sum"] == 1000

    def test_parse_prometheus_metrics_scientific_notation(self):
        """Parse Prometheus metrics with scientific notation."""
        text = """
metric_one 1.5e-3
metric_two 2.5E+2
metric_three 1e10
        """
        monitor = InferenceServerMonitor()
        metrics = monitor._parse_prometheus_metrics(text)

        assert metrics["metric_one"] == 1.5e-3
        assert metrics["metric_two"] == 2.5e2
        assert metrics["metric_three"] == 1e10

    def test_parse_prometheus_metrics_skip_comments(self):
        """Skip comment lines in Prometheus metrics."""
        text = """
# This is a comment
vllm:metric 10
# Another comment
vllm:other 20
        """
        monitor = InferenceServerMonitor()
        metrics = monitor._parse_prometheus_metrics(text)

        assert len(metrics) == 2
        assert "vllm:metric" in metrics
        assert "vllm:other" in metrics

    def test_parse_prometheus_metrics_empty_string(self):
        """Parse empty Prometheus metrics string."""
        monitor = InferenceServerMonitor()
        metrics = monitor._parse_prometheus_metrics("")
        assert metrics == {}

    def test_parse_prometheus_metrics_malformed_values(self):
        """Skip malformed metric values."""
        text = """
valid:metric 10
invalid:metric not_a_number
another:metric 20
        """
        monitor = InferenceServerMonitor()
        metrics = monitor._parse_prometheus_metrics(text)

        assert metrics["valid:metric"] == 10.0
        assert metrics["another:metric"] == 20.0
        assert "invalid:metric" not in metrics


class TestGetServerMetrics:
    """Test aggregated server metrics collection."""

    @patch('monitors.inference_server_monitor.InferenceServerMonitor._collect_vllm_metrics')
    @patch('monitors.inference_server_monitor.InferenceServerMonitor._collect_tgi_metrics')
    @patch('monitors.inference_server_monitor.InferenceServerMonitor._collect_ollama_metrics')
    def test_get_server_metrics_all_healthy(self, mock_ollama, mock_tgi, mock_vllm):
        """All servers healthy."""
        mock_vllm.return_value = {
            "server_type": "vllm",
            "endpoint": "http://vllm:8000",
            "status": "healthy",
            "models_loaded": ["llama2"],
            "requests_running": 2,
        }
        mock_tgi.return_value = {
            "server_type": "tgi",
            "endpoint": "http://tgi:8080",
            "status": "healthy",
            "models_loaded": ["mistral"],
            "requests_running": 1,
        }
        mock_ollama.return_value = {
            "server_type": "ollama",
            "endpoint": "http://ollama:11434",
            "status": "healthy",
            "models_loaded": ["neural-chat"],
            "requests_running": 0,
        }

        env_vars = {
            "VLLM_ENDPOINTS": "http://vllm:8000",
            "TGI_ENDPOINTS": "http://tgi:8080",
            "OLLAMA_ENDPOINTS": "http://ollama:11434",
        }
        with patch.dict(os.environ, env_vars):
            monitor = InferenceServerMonitor()
            metrics = monitor.get_server_metrics()

        assert len(metrics) == 3
        assert all(m["status"] == "healthy" for m in metrics)
        assert all("timestamp" in m for m in metrics)

    @patch('monitors.inference_server_monitor.InferenceServerMonitor._collect_vllm_metrics')
    def test_get_server_metrics_with_errors(self, mock_vllm):
        """Server error handling in aggregated metrics."""
        mock_vllm.side_effect = Exception("Connection refused")

        env_vars = {"VLLM_ENDPOINTS": "http://vllm:8000"}
        with patch.dict(os.environ, env_vars, clear=False):
            # Remove OLLAMA_ENDPOINTS if it exists to prevent default
            if "OLLAMA_ENDPOINTS" in os.environ:
                del os.environ["OLLAMA_ENDPOINTS"]
            monitor = InferenceServerMonitor()
            # Mock ollama collection to also fail
            monitor.ollama_endpoints = []  # Override to empty list
            metrics = monitor.get_server_metrics()

        assert len(metrics) == 1
        assert metrics[0]["status"] == "error"
        assert "Connection refused" in metrics[0]["error"]
        assert metrics[0]["endpoint"] == "http://vllm:8000"

    @patch('monitors.inference_server_monitor.InferenceServerMonitor._collect_vllm_metrics')
    @patch('monitors.inference_server_monitor.InferenceServerMonitor._collect_tgi_metrics')
    def test_get_server_metrics_partial_failure(self, mock_tgi, mock_vllm):
        """Partial server failures are reported."""
        mock_vllm.return_value = {
            "server_type": "vllm",
            "status": "healthy",
            "models_loaded": [],
        }
        mock_tgi.side_effect = Exception("Timeout")

        env_vars = {
            "VLLM_ENDPOINTS": "http://vllm:8000",
            "TGI_ENDPOINTS": "http://tgi:8080",
        }
        with patch.dict(os.environ, env_vars, clear=False):
            # Remove OLLAMA_ENDPOINTS if it exists to prevent default
            if "OLLAMA_ENDPOINTS" in os.environ:
                del os.environ["OLLAMA_ENDPOINTS"]
            monitor = InferenceServerMonitor()
            # Override ollama to empty list
            monitor.ollama_endpoints = []
            metrics = monitor.get_server_metrics()

        assert len(metrics) == 2
        healthy = [m for m in metrics if m["status"] == "healthy"]
        failed = [m for m in metrics if m["status"] == "error"]
        assert len(healthy) == 1
        assert len(failed) == 1


class TestGetLoadedModels:
    """Test model aggregation across servers."""

    @patch('monitors.inference_server_monitor.InferenceServerMonitor.get_server_metrics')
    def test_get_loaded_models_success(self, mock_get_metrics):
        """Get models from all servers."""
        mock_get_metrics.return_value = [
            {
                "server_type": "vllm",
                "endpoint": "http://vllm:8000",
                "status": "healthy",
                "models_loaded": ["model-a", "model-b"],
            },
            {
                "server_type": "tgi",
                "endpoint": "http://tgi:8080",
                "status": "healthy",
                "models_loaded": ["model-c"],
            },
        ]

        monitor = InferenceServerMonitor()
        models = monitor.get_loaded_models()

        assert len(models) == 3
        assert all(m["status"] == "loaded" for m in models)
        model_names = [m["model_name"] for m in models]
        assert "model-a" in model_names
        assert "model-b" in model_names
        assert "model-c" in model_names

    @patch('monitors.inference_server_monitor.InferenceServerMonitor.get_server_metrics')
    def test_get_loaded_models_skips_errors(self, mock_get_metrics):
        """Skip failed servers when collecting models."""
        mock_get_metrics.return_value = [
            {
                "server_type": "vllm",
                "endpoint": "http://vllm:8000",
                "status": "healthy",
                "models_loaded": ["model-a"],
            },
            {
                "server_type": "tgi",
                "endpoint": "http://tgi:8080",
                "status": "error",
                "error": "Connection refused",
            },
        ]

        monitor = InferenceServerMonitor()
        models = monitor.get_loaded_models()

        assert len(models) == 1
        assert models[0]["model_name"] == "model-a"


class TestGetServerHealth:
    """Test health check aggregation."""

    @patch('monitors.inference_server_monitor.InferenceServerMonitor._check_vllm_health')
    @patch('monitors.inference_server_monitor.InferenceServerMonitor._check_tgi_health')
    @patch('monitors.inference_server_monitor.InferenceServerMonitor._check_ollama_health')
    def test_get_server_health_all_healthy(self, mock_ollama_health, mock_tgi_health, mock_vllm_health):
        """All servers report healthy."""
        mock_vllm_health.return_value = {
            "endpoint": "http://vllm:8000",
            "server_type": "vllm",
            "healthy": True,
            "response_time_ms": 50.0,
        }
        mock_tgi_health.return_value = {
            "endpoint": "http://tgi:8080",
            "server_type": "tgi",
            "healthy": True,
            "response_time_ms": 45.0,
        }
        mock_ollama_health.return_value = {
            "endpoint": "http://ollama:11434",
            "server_type": "ollama",
            "healthy": True,
            "response_time_ms": 30.0,
        }

        env_vars = {
            "VLLM_ENDPOINTS": "http://vllm:8000",
            "TGI_ENDPOINTS": "http://tgi:8080",
            "OLLAMA_ENDPOINTS": "http://ollama:11434",
        }
        with patch.dict(os.environ, env_vars):
            monitor = InferenceServerMonitor()
            health = monitor.get_server_health()

        assert len(health) == 3
        assert all(h["healthy"] for h in health)

    @patch('monitors.inference_server_monitor.InferenceServerMonitor._check_vllm_health')
    def test_get_server_health_with_failures(self, mock_vllm_health):
        """Server health check failures."""
        mock_vllm_health.return_value = {
            "endpoint": "http://vllm:8000",
            "server_type": "vllm",
            "healthy": False,
            "error": "Connection timeout",
            "response_time_ms": 5000.0,
        }

        env_vars = {"VLLM_ENDPOINTS": "http://vllm:8000"}
        with patch.dict(os.environ, env_vars, clear=False):
            # Remove OLLAMA_ENDPOINTS if it exists to prevent default
            if "OLLAMA_ENDPOINTS" in os.environ:
                del os.environ["OLLAMA_ENDPOINTS"]
            monitor = InferenceServerMonitor()
            # Override ollama to empty list
            monitor.ollama_endpoints = []
            health = monitor.get_server_health()

        assert len(health) == 1
        assert not health[0]["healthy"]
        assert health[0]["response_time_ms"] == 5000.0


class TestHealthCheckMethods:
    """Test individual health check implementations."""

    @patch('monitors.inference_server_monitor.requests.get')
    def test_check_vllm_health_success(self, mock_get):
        """vLLM health check success."""
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        monitor = InferenceServerMonitor()
        health = monitor._check_vllm_health("http://localhost:8000")

        assert health["healthy"] is True
        assert health["server_type"] == "vllm"
        assert "response_time_ms" in health

    @patch('monitors.inference_server_monitor.requests.get')
    def test_check_vllm_health_timeout(self, mock_get):
        """vLLM health check timeout."""
        mock_get.side_effect = requests.Timeout("Timeout")

        monitor = InferenceServerMonitor()
        health = monitor._check_vllm_health("http://localhost:8000")

        assert health["healthy"] is False
        assert "Timeout" in health["error"]

    @patch('monitors.inference_server_monitor.requests.get')
    def test_check_tgi_health_success(self, mock_get):
        """TGI health check success."""
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        monitor = InferenceServerMonitor()
        health = monitor._check_tgi_health("http://localhost:8080")

        assert health["healthy"] is True
        assert health["server_type"] == "tgi"

    @patch('monitors.inference_server_monitor.requests.get')
    def test_check_ollama_health_success(self, mock_get):
        """Ollama health check success."""
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        monitor = InferenceServerMonitor()
        health = monitor._check_ollama_health("http://localhost:11434")

        assert health["healthy"] is True
        assert health["server_type"] == "ollama"


class TestTritonMetrics:
    """Test Triton Inference Server metrics collection."""

    @patch('monitors.inference_server_monitor.requests.get')
    def test_collect_triton_metrics_success(self, mock_get):
        """Successful Triton metrics collection."""
        mock_health_response = Mock()
        mock_health_response.raise_for_status = Mock()

        mock_metrics_response = Mock()
        mock_metrics_response.text = """
nv_inference_request_success 1000
nv_inference_queue_duration 0.05
nv_gpu_utilization 85.5
        """

        mock_get.side_effect = [mock_health_response, mock_metrics_response]

        monitor = InferenceServerMonitor()
        metrics = monitor._collect_triton_metrics("http://localhost:8000")

        assert metrics["server_type"] == "triton"
        assert metrics["status"] == "healthy"
        assert metrics["throughput_tokens_per_sec"] == 1000.0
        assert metrics["latency_p50_ms"] == 50.0
        assert metrics["gpu_cache_utilization_pct"] == 85.5

    @patch('monitors.inference_server_monitor.requests.get')
    def test_check_triton_health_success(self, mock_get):
        """Triton health check success."""
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        monitor = InferenceServerMonitor()
        health = monitor._check_triton_health("http://localhost:8000")

        assert health["healthy"] is True
        assert health["server_type"] == "triton"


class TestErrorHandling:
    """Test error handling in various scenarios."""

    @patch('monitors.inference_server_monitor.requests.get')
    def test_server_down(self, mock_get):
        """Handle completely down server."""
        mock_get.side_effect = requests.ConnectionError("Cannot connect")

        monitor = InferenceServerMonitor()
        with pytest.raises(requests.ConnectionError):
            monitor._collect_vllm_metrics("http://down:8000")

    @patch('monitors.inference_server_monitor.requests.get')
    def test_malformed_json_response(self, mock_get):
        """Handle malformed JSON responses."""
        mock_response = Mock()
        mock_response.json.side_effect = ValueError("Invalid JSON")
        mock_get.return_value = mock_response

        monitor = InferenceServerMonitor()
        with pytest.raises(ValueError):
            monitor._collect_vllm_metrics("http://localhost:8000")

    @patch('monitors.inference_server_monitor.requests.get')
    def test_http_status_error(self, mock_get):
        """Handle HTTP status errors."""
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = requests.HTTPError("500 Server Error")
        mock_get.return_value = mock_response

        monitor = InferenceServerMonitor()
        with pytest.raises(requests.HTTPError):
            monitor._collect_vllm_metrics("http://localhost:8000")


class TestTritonAndOllamaMonitoring:
    """Test monitoring of Triton and Ollama servers."""

    def test_init_with_triton_endpoints(self):
        """Should initialize with Triton endpoints."""
        env_vars = {"TRITON_ENDPOINTS": "http://triton1:8000,http://triton2:8001"}
        with patch.dict(os.environ, env_vars):
            monitor = InferenceServerMonitor()
            assert len(monitor.triton_endpoints) == 2
            assert "http://triton1:8000" in monitor.triton_endpoints

    def test_init_with_ollama_endpoints(self):
        """Should initialize with Ollama endpoints."""
        env_vars = {"OLLAMA_ENDPOINTS": "http://ollama1:11434,http://ollama2:11434"}
        with patch.dict(os.environ, env_vars):
            monitor = InferenceServerMonitor()
            assert len(monitor.ollama_endpoints) == 2
            assert "http://ollama1:11434" in monitor.ollama_endpoints

    @patch('monitors.inference_server_monitor.requests.get')
    def test_collect_triton_metrics_success(self, mock_get):
        """Should collect metrics from Triton successfully."""
        def get_side_effect(url, timeout):
            mock_response = Mock()
            if "/health/ready" in url:
                mock_response.raise_for_status = Mock()
            elif "/metrics" in url:
                mock_response.text = "nv_inference_request_success 100\nnv_inference_queue_duration 0.5"
                mock_response.raise_for_status = Mock()
            return mock_response

        mock_get.side_effect = get_side_effect

        monitor = InferenceServerMonitor()
        metrics = monitor._collect_triton_metrics("http://localhost:8000")

        assert metrics["server_type"] == "triton"
        assert metrics["endpoint"] == "http://localhost:8000"
        assert metrics["status"] == "healthy"

    @patch('monitors.inference_server_monitor.requests.get')
    def test_collect_triton_metrics_error(self, mock_get):
        """Should handle errors when collecting Triton metrics."""
        mock_get.side_effect = Exception("Connection failed")

        monitor = InferenceServerMonitor()
        with pytest.raises(Exception):
            monitor._collect_triton_metrics("http://localhost:8000")

    @patch('monitors.inference_server_monitor.requests.get')
    def test_collect_ollama_metrics_success(self, mock_get):
        """Should collect metrics from Ollama successfully."""
        def get_side_effect(url, timeout):
            mock_response = Mock()
            if "/api/ps" in url:
                mock_response.json.return_value = {
                    "models": [{"name": "mistral", "size": 4096000000}]
                }
            elif "/api/tags" in url:
                mock_response.json.return_value = {
                    "models": [{"name": "mistral", "modified_at": "2024-01-01T00:00:00Z"}]
                }
            mock_response.raise_for_status = Mock()
            return mock_response

        mock_get.side_effect = get_side_effect

        monitor = InferenceServerMonitor()
        metrics = monitor._collect_ollama_metrics("http://localhost:11434")

        assert metrics["server_type"] == "ollama"
        assert metrics["endpoint"] == "http://localhost:11434"
        assert "models_loaded" in metrics
        assert len(metrics["models_loaded"]) > 0

    @patch('monitors.inference_server_monitor.requests.get')
    def test_collect_ollama_metrics_error(self, mock_get):
        """Should handle errors when collecting Ollama metrics."""
        mock_get.side_effect = Exception("Connection failed")

        monitor = InferenceServerMonitor()
        with pytest.raises(Exception):
            monitor._collect_ollama_metrics("http://localhost:11434")

    @patch('monitors.inference_server_monitor.requests.get')
    def test_check_triton_health_failure(self, mock_get):
        """Should handle health check failure for Triton."""
        mock_get.side_effect = Exception("Cannot reach server")

        monitor = InferenceServerMonitor()
        health = monitor._check_triton_health("http://localhost:8000")

        assert health["healthy"] is False
        assert health["server_type"] == "triton"
        assert "error" in health

    @patch('monitors.inference_server_monitor.requests.get')
    def test_check_ollama_health_success(self, mock_get):
        """Should check health of Ollama server."""
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        monitor = InferenceServerMonitor()
        health = monitor._check_ollama_health("http://localhost:11434")

        assert health["healthy"] is True
        assert health["server_type"] == "ollama"

    @patch('monitors.inference_server_monitor.requests.get')
    def test_check_ollama_health_failure(self, mock_get):
        """Should handle health check failure for Ollama."""
        mock_get.side_effect = Exception("Cannot reach server")

        monitor = InferenceServerMonitor()
        health = monitor._check_ollama_health("http://localhost:11434")

        assert health["healthy"] is False
        assert health["server_type"] == "ollama"
        assert "error" in health

    @patch('monitors.inference_server_monitor.requests.get')
    def test_get_server_metrics_includes_triton(self, mock_get):
        """Should include Triton endpoints in metrics collection."""
        mock_response = Mock()
        mock_response.json.return_value = {"ready": True}
        mock_get.return_value = mock_response

        env_vars = {"TRITON_ENDPOINTS": "http://triton:8000"}
        with patch.dict(os.environ, env_vars):
            monitor = InferenceServerMonitor()
            metrics = monitor.get_server_metrics()

            triton_metrics = [m for m in metrics if m.get("server_type") == "triton"]
            assert len(triton_metrics) > 0

    @patch('monitors.inference_server_monitor.requests.get')
    def test_get_server_metrics_includes_ollama(self, mock_get):
        """Should include Ollama endpoints in metrics collection."""
        mock_response = Mock()
        mock_response.json.return_value = {"models": []}
        mock_get.return_value = mock_response

        env_vars = {"OLLAMA_ENDPOINTS": "http://ollama:11434"}
        with patch.dict(os.environ, env_vars):
            monitor = InferenceServerMonitor()
            metrics = monitor.get_server_metrics()

            ollama_metrics = [m for m in metrics if m.get("server_type") == "ollama"]
            assert len(ollama_metrics) > 0

    @patch('monitors.inference_server_monitor.requests.get')
    def test_get_server_health_includes_triton(self, mock_get):
        """Should include Triton in health check results."""
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        env_vars = {"TRITON_ENDPOINTS": "http://triton:8000"}
        with patch.dict(os.environ, env_vars):
            monitor = InferenceServerMonitor()
            health = monitor.get_server_health()

            triton_health = [h for h in health if h.get("server_type") == "triton"]
            assert len(triton_health) > 0

    @patch('monitors.inference_server_monitor.requests.get')
    def test_get_server_health_includes_ollama(self, mock_get):
        """Should include Ollama in health check results."""
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        env_vars = {"OLLAMA_ENDPOINTS": "http://ollama:11434"}
        with patch.dict(os.environ, env_vars):
            monitor = InferenceServerMonitor()
            health = monitor.get_server_health()

            ollama_health = [h for h in health if h.get("server_type") == "ollama"]
            assert len(ollama_health) > 0

    @patch('monitors.inference_server_monitor.requests.get')
    def test_collect_tgi_metrics_error_handling(self, mock_get):
        """Should handle errors when collecting TGI metrics."""
        mock_get.side_effect = Exception("Connection failed")

        monitor = InferenceServerMonitor()
        with pytest.raises(Exception):
            monitor._collect_tgi_metrics("http://localhost:8080")

    @patch('monitors.inference_server_monitor.requests.get')
    def test_check_tgi_health_failure(self, mock_get):
        """Should handle health check failure for TGI."""
        mock_get.side_effect = Exception("Cannot reach server")

        monitor = InferenceServerMonitor()
        health = monitor._check_tgi_health("http://localhost:8080")

        assert health["healthy"] is False
        assert health["server_type"] == "tgi"
        assert "error" in health

    @patch('monitors.inference_server_monitor.requests.get')
    def test_collect_vllm_metrics_prometheus_parsing_error(self, mock_get):
        """Should handle ValueError in Prometheus metric parsing."""
        def get_side_effect(url, timeout):
            mock_response = Mock()
            if "/metrics" in url:
                mock_response.text = "invalid_metric_line_without_proper_format"
            elif "/v1/models" in url:
                mock_response.json.return_value = {"data": [{"id": "mistral"}]}
            mock_response.raise_for_status = Mock()
            return mock_response

        mock_get.side_effect = get_side_effect

        monitor = InferenceServerMonitor()
        metrics = monitor._collect_vllm_metrics("http://localhost:8000")

        # Should return metrics dict but with parsing error handled
        assert metrics["server_type"] == "vllm"
        assert metrics["models_loaded"] == ["mistral"]
