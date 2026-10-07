"""Monitor for self-hosted LLM inference servers.

Supports monitoring of vLLM, Text Generation Inference (TGI), Triton Inference Server,
and Ollama. Auto-discovers servers from environment variables and collects metrics
from each server type's respective endpoints.
"""

import logging
import os
import re
import time
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


class InferenceServerMonitor:
    """Monitor for self-hosted LLM inference servers.

    Supports:
    - vLLM (OpenAI-compatible with Prometheus metrics)
    - Text Generation Inference (TGI)
    - Triton Inference Server
    - Ollama

    Auto-discovers servers from environment variables and provides methods to:
    - Collect metrics from all configured servers
    - Get loaded models across all servers
    - Perform health checks
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the inference server monitor.

        Args:
            config: Optional configuration dictionary. If not provided, servers are
                    auto-discovered from environment variables.

                    Supported env vars:
                    - VLLM_ENDPOINTS: comma-separated URLs (e.g., http://host:8000)
                    - TGI_ENDPOINTS: comma-separated URLs
                    - TRITON_ENDPOINTS: comma-separated URLs
                    - OLLAMA_ENDPOINTS: comma-separated URLs (default: http://localhost:11434)
        """
        self.config = config or {}
        self.request_timeout = 5  # seconds
        self._discover_servers()

    def _discover_servers(self) -> None:
        """Auto-discover inference servers from environment variables."""
        self.vllm_endpoints = self._parse_endpoints("VLLM_ENDPOINTS")
        self.tgi_endpoints = self._parse_endpoints("TGI_ENDPOINTS")
        self.triton_endpoints = self._parse_endpoints("TRITON_ENDPOINTS")

        # Ollama defaults to localhost:11434 if not specified
        ollama_env = os.getenv("OLLAMA_ENDPOINTS", "").strip()
        if ollama_env:
            self.ollama_endpoints = self._parse_endpoints("OLLAMA_ENDPOINTS")
        else:
            self.ollama_endpoints = ["http://localhost:11434"]

        logger.info(
            f"Discovered inference servers: vLLM={len(self.vllm_endpoints)}, "
            f"TGI={len(self.tgi_endpoints)}, Triton={len(self.triton_endpoints)}, "
            f"Ollama={len(self.ollama_endpoints)}"
        )

    @staticmethod
    def _parse_endpoints(env_var: str) -> List[str]:
        """Parse comma-separated endpoints from environment variable.

        Args:
            env_var: Environment variable name

        Returns:
            List of endpoint URLs
        """
        value = os.getenv(env_var, "").strip()
        if not value:
            return []
        return [url.strip() for url in value.split(",") if url.strip()]

    def get_server_metrics(self) -> List[Dict[str, Any]]:
        """Collect metrics from all configured inference servers.

        Returns a list of metrics dictionaries, one per server. Each dictionary contains:
        - server_type: 'vllm', 'tgi', 'triton', or 'ollama'
        - endpoint: Server URL
        - status: 'healthy' or 'error'
        - error: Error message if status is 'error'
        - model_name: Primary model name (if available)
        - models_loaded: List of loaded model names
        - requests_running: Number of requests currently being processed
        - requests_queued: Number of requests waiting in queue
        - throughput_tokens_per_sec: Tokens generated per second (input + output)
        - latency_p50_ms: 50th percentile latency in milliseconds
        - latency_p95_ms: 95th percentile latency in milliseconds
        - latency_p99_ms: 99th percentile latency in milliseconds
        - gpu_cache_utilization_pct: KV cache utilization percentage (if available)
        - batch_size_current: Current batch size
        - vram_usage_mb: GPU VRAM usage in MB
        - vram_total_mb: Total GPU VRAM in MB
        - timestamp: Collection timestamp (seconds since epoch)

        Returns:
            List of metrics dictionaries from all servers. Failed servers are included
            with status='error' and error message, to allow partial monitoring.
        """
        all_metrics = []
        timestamp = time.time()

        for endpoint in self.vllm_endpoints:
            try:
                metrics = self._collect_vllm_metrics(endpoint)
                metrics["timestamp"] = timestamp
                all_metrics.append(metrics)
            except Exception as e:
                logger.error(f"Error collecting vLLM metrics from {endpoint}: {e}")
                all_metrics.append({
                    "server_type": "vllm",
                    "endpoint": endpoint,
                    "status": "error",
                    "error": str(e),
                    "timestamp": timestamp,
                })

        for endpoint in self.tgi_endpoints:
            try:
                metrics = self._collect_tgi_metrics(endpoint)
                metrics["timestamp"] = timestamp
                all_metrics.append(metrics)
            except Exception as e:
                logger.error(f"Error collecting TGI metrics from {endpoint}: {e}")
                all_metrics.append({
                    "server_type": "tgi",
                    "endpoint": endpoint,
                    "status": "error",
                    "error": str(e),
                    "timestamp": timestamp,
                })

        for endpoint in self.triton_endpoints:
            try:
                metrics = self._collect_triton_metrics(endpoint)
                metrics["timestamp"] = timestamp
                all_metrics.append(metrics)
            except Exception as e:
                logger.error(f"Error collecting Triton metrics from {endpoint}: {e}")
                all_metrics.append({
                    "server_type": "triton",
                    "endpoint": endpoint,
                    "status": "error",
                    "error": str(e),
                    "timestamp": timestamp,
                })

        for endpoint in self.ollama_endpoints:
            try:
                metrics = self._collect_ollama_metrics(endpoint)
                metrics["timestamp"] = timestamp
                all_metrics.append(metrics)
            except Exception as e:
                logger.error(f"Error collecting Ollama metrics from {endpoint}: {e}")
                all_metrics.append({
                    "server_type": "ollama",
                    "endpoint": endpoint,
                    "status": "error",
                    "error": str(e),
                    "timestamp": timestamp,
                })

        return all_metrics

    def get_loaded_models(self) -> List[Dict[str, Any]]:
        """Get all loaded models across all servers.

        Returns a list of model dictionaries, each containing:
        - model_name: Name of the model
        - server_type: Type of server hosting the model
        - endpoint: Server endpoint URL
        - size_gb: Model size in gigabytes (if available)
        - quantization: Quantization type/bits (if available)
        - status: 'loaded' or 'error'

        Returns:
            List of model dictionaries
        """
        models = []

        for metrics in self.get_server_metrics():
            if metrics.get("status") == "error":
                continue

            server_type = metrics.get("server_type")
            endpoint = metrics.get("endpoint")

            for model_name in metrics.get("models_loaded", []):
                models.append({
                    "model_name": model_name,
                    "server_type": server_type,
                    "endpoint": endpoint,
                    "size_gb": None,
                    "quantization": None,
                    "status": "loaded",
                })

        return models

    def get_server_health(self) -> List[Dict[str, Any]]:
        """Perform health checks on all configured servers.

        Returns a list of health check results, each containing:
        - endpoint: Server URL
        - server_type: Type of server
        - healthy: Boolean indicating if server is healthy
        - response_time_ms: Response time in milliseconds
        - error: Error message if not healthy

        Returns:
            List of health check dictionaries
        """
        health = []

        for endpoint in self.vllm_endpoints:
            health.append(self._check_vllm_health(endpoint))

        for endpoint in self.tgi_endpoints:
            health.append(self._check_tgi_health(endpoint))

        for endpoint in self.triton_endpoints:
            health.append(self._check_triton_health(endpoint))

        for endpoint in self.ollama_endpoints:
            health.append(self._check_ollama_health(endpoint))

        return health

    def _collect_vllm_metrics(self, endpoint: str) -> Dict[str, Any]:
        """Collect metrics from vLLM server.

        Args:
            endpoint: vLLM server endpoint URL

        Returns:
            Dictionary of collected metrics

        Raises:
            RequestException: If HTTP request fails
            ValueError: If response parsing fails
        """
        start_time = time.time()

        # Collect metrics from /metrics endpoint (Prometheus format)
        metrics_response = requests.get(
            f"{endpoint}/metrics",
            timeout=self.request_timeout
        )
        metrics_response.raise_for_status()
        metrics_text = metrics_response.text

        # Collect model list from /v1/models endpoint
        models_response = requests.get(
            f"{endpoint}/v1/models",
            timeout=self.request_timeout
        )
        models_response.raise_for_status()
        models_data = models_response.json()

        # Parse Prometheus metrics
        parsed_metrics = self._parse_prometheus_metrics(metrics_text)

        # Extract model names
        models_loaded = [
            m.get("id", "unknown")
            for m in models_data.get("data", [])
        ]

        # Calculate latencies from histogram (vLLM provides e2e_request_latency_seconds)
        latency_p50_ms = parsed_metrics.get("vllm:e2e_request_latency_seconds_p50", 0) * 1000
        latency_p95_ms = parsed_metrics.get("vllm:e2e_request_latency_seconds_p95", 0) * 1000
        latency_p99_ms = parsed_metrics.get("vllm:e2e_request_latency_seconds_p99", 0) * 1000

        return {
            "server_type": "vllm",
            "endpoint": endpoint,
            "status": "healthy",
            "model_name": models_loaded[0] if models_loaded else None,
            "models_loaded": models_loaded,
            "requests_running": int(parsed_metrics.get("vllm:num_requests_running", 0)),
            "requests_queued": int(parsed_metrics.get("vllm:num_requests_waiting", 0)),
            "throughput_tokens_per_sec": float(
                parsed_metrics.get("vllm:avg_generation_throughput_toks_per_s", 0)
                + parsed_metrics.get("vllm:avg_prompt_throughput_toks_per_s", 0)
            ),
            "latency_p50_ms": latency_p50_ms,
            "latency_p95_ms": latency_p95_ms,
            "latency_p99_ms": latency_p99_ms,
            "gpu_cache_utilization_pct": float(
                parsed_metrics.get("vllm:gpu_cache_usage_perc", 0)
            ),
            "batch_size_current": int(parsed_metrics.get("vllm:batch_size", 0)),
            "vram_usage_mb": 0,  # Not directly available from vLLM metrics
            "vram_total_mb": 0,  # Not directly available from vLLM metrics
        }

    def _collect_tgi_metrics(self, endpoint: str) -> Dict[str, Any]:
        """Collect metrics from Text Generation Inference (TGI) server.

        Args:
            endpoint: TGI server endpoint URL

        Returns:
            Dictionary of collected metrics

        Raises:
            RequestException: If HTTP request fails
            ValueError: If response parsing fails
        """
        # Collect metrics from /metrics endpoint (Prometheus format)
        metrics_response = requests.get(
            f"{endpoint}/metrics",
            timeout=self.request_timeout
        )
        metrics_response.raise_for_status()
        metrics_text = metrics_response.text

        # Collect model info from /info endpoint
        info_response = requests.get(
            f"{endpoint}/info",
            timeout=self.request_timeout
        )
        info_response.raise_for_status()
        info_data = info_response.json()

        # Parse Prometheus metrics
        parsed_metrics = self._parse_prometheus_metrics(metrics_text)

        model_name = info_data.get("model_id", "unknown")

        # Calculate throughput from metrics
        request_count = parsed_metrics.get("tgi_request_duration_count", 0)
        generated_tokens = parsed_metrics.get("tgi_request_generated_tokens_sum", 0)
        throughput = generated_tokens if request_count > 0 else 0

        return {
            "server_type": "tgi",
            "endpoint": endpoint,
            "status": "healthy",
            "model_name": model_name,
            "models_loaded": [model_name],
            "requests_running": 0,  # Not directly available from TGI metrics
            "requests_queued": int(parsed_metrics.get("tgi_queue_size", 0)),
            "throughput_tokens_per_sec": float(throughput),
            "latency_p50_ms": 0,  # Would need histogram buckets
            "latency_p95_ms": 0,
            "latency_p99_ms": 0,
            "gpu_cache_utilization_pct": 0,  # Not available from TGI
            "batch_size_current": int(parsed_metrics.get("tgi_batch_current_size", 0)),
            "vram_usage_mb": 0,  # Not directly available from TGI metrics
            "vram_total_mb": 0,
        }

    def _collect_triton_metrics(self, endpoint: str) -> Dict[str, Any]:
        """Collect metrics from Triton Inference Server.

        Args:
            endpoint: Triton server endpoint URL

        Returns:
            Dictionary of collected metrics

        Raises:
            RequestException: If HTTP request fails
            ValueError: If response parsing fails
        """
        # Check server health first
        health_response = requests.get(
            f"{endpoint}/v2/health/ready",
            timeout=self.request_timeout
        )
        health_response.raise_for_status()

        # Collect metrics from /metrics endpoint (Prometheus format)
        metrics_response = requests.get(
            f"{endpoint}/metrics",
            timeout=self.request_timeout
        )
        metrics_response.raise_for_status()
        metrics_text = metrics_response.text

        # Parse Prometheus metrics
        parsed_metrics = self._parse_prometheus_metrics(metrics_text)

        # Extract model statistics if available
        # This is a simplified collection; a full implementation would parse
        # /v2/models/{model}/stats for each loaded model
        infer_request_success = int(
            parsed_metrics.get("nv_inference_request_success", 0)
        )
        queue_duration = float(
            parsed_metrics.get("nv_inference_queue_duration", 0)
        )

        return {
            "server_type": "triton",
            "endpoint": endpoint,
            "status": "healthy",
            "model_name": None,
            "models_loaded": [],
            "requests_running": 0,  # Not directly available
            "requests_queued": 0,  # Would need to poll /v2/models
            "throughput_tokens_per_sec": float(infer_request_success),
            "latency_p50_ms": queue_duration * 1000,
            "latency_p95_ms": 0,
            "latency_p99_ms": 0,
            "gpu_cache_utilization_pct": float(
                parsed_metrics.get("nv_gpu_utilization", 0)
            ),
            "batch_size_current": 0,
            "vram_usage_mb": 0,
            "vram_total_mb": 0,
        }

    def _collect_ollama_metrics(self, endpoint: str) -> Dict[str, Any]:
        """Collect metrics from Ollama server.

        Args:
            endpoint: Ollama server endpoint URL

        Returns:
            Dictionary of collected metrics

        Raises:
            RequestException: If HTTP request fails
            ValueError: If response parsing fails
        """
        # Get running models
        ps_response = requests.get(
            f"{endpoint}/api/ps",
            timeout=self.request_timeout
        )
        ps_response.raise_for_status()
        ps_data = ps_response.json()

        # Get available models
        tags_response = requests.get(
            f"{endpoint}/api/tags",
            timeout=self.request_timeout
        )
        tags_response.raise_for_status()
        tags_data = tags_response.json()

        # Extract model information
        models_loaded = [m.get("name", "unknown") for m in tags_data.get("models", [])]
        running_models = ps_data.get("models", [])

        # Calculate total VRAM usage
        total_vram_mb = sum(
            m.get("size", 0) / (1024 * 1024)
            for m in running_models
        )

        primary_model = running_models[0].get("name", None) if running_models else None

        return {
            "server_type": "ollama",
            "endpoint": endpoint,
            "status": "healthy",
            "model_name": primary_model,
            "models_loaded": models_loaded,
            "requests_running": len(running_models),
            "requests_queued": 0,  # Ollama doesn't expose queue
            "throughput_tokens_per_sec": 0,  # Not available from Ollama API
            "latency_p50_ms": 0,
            "latency_p95_ms": 0,
            "latency_p99_ms": 0,
            "gpu_cache_utilization_pct": 0,
            "batch_size_current": len(running_models),
            "vram_usage_mb": total_vram_mb,
            "vram_total_mb": 0,  # Not available from Ollama API
        }

    def _parse_prometheus_metrics(self, text: str) -> Dict[str, float]:
        """Parse Prometheus-format metrics text.

        Extracts metric name and value pairs from Prometheus text format.
        Handles gauges and counters, extracts histogram percentiles.

        Args:
            text: Prometheus format metrics text

        Returns:
            Dictionary mapping metric names to numeric values
        """
        metrics = {}

        for line in text.split("\n"):
            # Skip comments and empty lines
            if not line or line.startswith("#"):
                continue

            # Match metric line: metric_name{labels} value [timestamp]
            # For simplicity, we match basic patterns and extract value
            match = re.match(r"^([a-zA-Z_:][a-zA-Z0-9_:\.]*)\s+([0-9.eE+-]+)", line)
            if match:
                metric_name = match.group(1)
                try:
                    value = float(match.group(2))
                    metrics[metric_name] = value
                except ValueError:
                    continue

        return metrics

    def _check_vllm_health(self, endpoint: str) -> Dict[str, Any]:
        """Check health of vLLM server.

        Args:
            endpoint: vLLM server endpoint URL

        Returns:
            Health check result dictionary
        """
        start_time = time.time()
        try:
            response = requests.get(
                f"{endpoint}/health",
                timeout=self.request_timeout
            )
            response.raise_for_status()
            response_time_ms = (time.time() - start_time) * 1000
            return {
                "endpoint": endpoint,
                "server_type": "vllm",
                "healthy": True,
                "response_time_ms": response_time_ms,
            }
        except Exception as e:
            response_time_ms = (time.time() - start_time) * 1000
            return {
                "endpoint": endpoint,
                "server_type": "vllm",
                "healthy": False,
                "response_time_ms": response_time_ms,
                "error": str(e),
            }

    def _check_tgi_health(self, endpoint: str) -> Dict[str, Any]:
        """Check health of TGI server.

        Args:
            endpoint: TGI server endpoint URL

        Returns:
            Health check result dictionary
        """
        start_time = time.time()
        try:
            response = requests.get(
                f"{endpoint}/info",
                timeout=self.request_timeout
            )
            response.raise_for_status()
            response_time_ms = (time.time() - start_time) * 1000
            return {
                "endpoint": endpoint,
                "server_type": "tgi",
                "healthy": True,
                "response_time_ms": response_time_ms,
            }
        except Exception as e:
            response_time_ms = (time.time() - start_time) * 1000
            return {
                "endpoint": endpoint,
                "server_type": "tgi",
                "healthy": False,
                "response_time_ms": response_time_ms,
                "error": str(e),
            }

    def _check_triton_health(self, endpoint: str) -> Dict[str, Any]:
        """Check health of Triton server.

        Args:
            endpoint: Triton server endpoint URL

        Returns:
            Health check result dictionary
        """
        start_time = time.time()
        try:
            response = requests.get(
                f"{endpoint}/v2/health/ready",
                timeout=self.request_timeout
            )
            response.raise_for_status()
            response_time_ms = (time.time() - start_time) * 1000
            return {
                "endpoint": endpoint,
                "server_type": "triton",
                "healthy": True,
                "response_time_ms": response_time_ms,
            }
        except Exception as e:
            response_time_ms = (time.time() - start_time) * 1000
            return {
                "endpoint": endpoint,
                "server_type": "triton",
                "healthy": False,
                "response_time_ms": response_time_ms,
                "error": str(e),
            }

    def _check_ollama_health(self, endpoint: str) -> Dict[str, Any]:
        """Check health of Ollama server.

        Args:
            endpoint: Ollama server endpoint URL

        Returns:
            Health check result dictionary
        """
        start_time = time.time()
        try:
            response = requests.get(
                f"{endpoint}/api/tags",
                timeout=self.request_timeout
            )
            response.raise_for_status()
            response_time_ms = (time.time() - start_time) * 1000
            return {
                "endpoint": endpoint,
                "server_type": "ollama",
                "healthy": True,
                "response_time_ms": response_time_ms,
            }
        except Exception as e:
            response_time_ms = (time.time() - start_time) * 1000
            return {
                "endpoint": endpoint,
                "server_type": "ollama",
                "healthy": False,
                "response_time_ms": response_time_ms,
                "error": str(e),
            }
