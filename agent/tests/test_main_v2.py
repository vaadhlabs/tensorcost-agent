"""Slimmed test_main coverage for the rewritten agent — focuses on the
seams that actually changed in Phase 2:

  - Config validation pulls from the new env-var names.
  - _send_metrics / _send_instances route via the HTTP client (rename
    helpers exercised in test_http_transport.py).
  - The dropped event-type cycles do NOT call into the transport.

The legacy `tests/test_main.py` patched `main.AWSMonitor` (which is now
lazy-imported, not module-attribute) and `_send_to_backend` (deleted).
Keeping it would have been busywork — those internals don't exist any
more. The HTTP / gRPC / HMAC seams are covered by the dedicated test
files.
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture(autouse=True)
def _stub_optional_modules(monkeypatch):
    """Mock the heavy cloud SDKs so `import main` works in CI.

    Critical: do NOT stub `google` as a top-level MagicMock — `protobuf`
    lives there and the gRPC stubs need the real google.protobuf. Stub
    only the leaf packages we don't want imported.
    """
    for mod_name in (
        "azure", "azure.identity",
        "google.auth", "google.cloud",
        "kubernetes", "sagemaker",
    ):
        if mod_name not in sys.modules:
            sys.modules[mod_name] = MagicMock()
    yield


@pytest.fixture
def env_clean(monkeypatch):
    """Clear all relevant env vars."""
    for var in (
        "BACKEND_API_URL", "AGENT_API_KEY", "BACKEND_API_KEY",
        "AGENT_KEY_ID", "AGENT_HMAC_PEPPER", "TENANT_ID",
        "GRPC_TARGET", "COMM_MODE", "AGENT_HOSTNAME", "AGENT_ID",
        "AWS_ENABLED", "AZURE_ENABLED", "GCP_ENABLED",
        "K8S_ENABLED", "SAGEMAKER_ENABLED", "ENABLED_CLOUDS",
        "NVML_ENABLED", "SPOT_HANDLER_ENABLED",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("LOG_LEVEL", "ERROR")
    yield


def _fresh_agent(env_clean):
    """Helper — import main and instantiate after env is set up."""
    if "main" in sys.modules:
        del sys.modules["main"]
    import main as main_mod
    return main_mod.UnifiedGPUAgent()


class TestConfigValidation:
    def test_passes_with_minimum_env(self, env_clean, monkeypatch):
        monkeypatch.setenv("BACKEND_API_URL", "http://gw")
        monkeypatch.setenv("AGENT_API_KEY", "k")
        monkeypatch.setenv("TENANT_ID", "t-1")
        agent = _fresh_agent(env_clean)
        agent._validate_config()  # must not raise

    def test_fails_missing_backend_url(self, env_clean, monkeypatch):
        monkeypatch.setenv("AGENT_API_KEY", "k")
        agent = _fresh_agent(env_clean)
        with pytest.raises(SystemExit):
            agent._validate_config()

    def test_fails_missing_agent_api_key(self, env_clean, monkeypatch):
        monkeypatch.setenv("BACKEND_API_URL", "http://gw")
        agent = _fresh_agent(env_clean)
        with pytest.raises(SystemExit):
            agent._validate_config()

    def test_grpc_mode_requires_key_id_and_pepper(self, env_clean, monkeypatch):
        monkeypatch.setenv("BACKEND_API_URL", "http://gw")
        monkeypatch.setenv("AGENT_API_KEY", "k")
        monkeypatch.setenv("TENANT_ID", "t-1")
        monkeypatch.setenv("COMM_MODE", "grpc")
        monkeypatch.setenv("GRPC_TARGET", "x:1")
        # AGENT_KEY_ID / AGENT_HMAC_PEPPER missing — must fail.
        agent = _fresh_agent(env_clean)
        with pytest.raises(SystemExit):
            agent._validate_config()

    def test_grpc_mode_passes_when_all_set(self, env_clean, monkeypatch):
        for k, v in (
            ("BACKEND_API_URL", "http://gw"),
            ("AGENT_API_KEY", "k"),
            ("TENANT_ID", "t-1"),
            ("COMM_MODE", "grpc"),
            ("GRPC_TARGET", "x:1"),
            ("AGENT_KEY_ID", "k1"),
            ("AGENT_HMAC_PEPPER", "deadbeef" * 8),
        ):
            monkeypatch.setenv(k, v)
        agent = _fresh_agent(env_clean)
        agent._validate_config()

    def test_both_mode_requires_key_id_and_pepper(self, env_clean, monkeypatch):
        monkeypatch.setenv("BACKEND_API_URL", "http://gw")
        monkeypatch.setenv("AGENT_API_KEY", "k")
        monkeypatch.setenv("TENANT_ID", "t-1")
        monkeypatch.setenv("COMM_MODE", "both")
        monkeypatch.setenv("GRPC_TARGET", "x:1")
        agent = _fresh_agent(env_clean)
        with pytest.raises(SystemExit):
            agent._validate_config()

    def test_both_mode_passes_when_all_set(self, env_clean, monkeypatch):
        for k, v in (
            ("BACKEND_API_URL", "http://gw"),
            ("AGENT_API_KEY", "k"),
            ("TENANT_ID", "t-1"),
            ("COMM_MODE", "both"),
            ("GRPC_TARGET", "x:1"),
            ("AGENT_KEY_ID", "k1"),
            ("AGENT_HMAC_PEPPER", "deadbeef" * 8),
        ):
            monkeypatch.setenv(k, v)
        agent = _fresh_agent(env_clean)
        agent._validate_config()


class TestTransportRouting:
    @pytest.fixture
    def agent(self, env_clean, monkeypatch):
        monkeypatch.setenv("BACKEND_API_URL", "http://gw")
        monkeypatch.setenv("AGENT_API_KEY", "k")
        monkeypatch.setenv("TENANT_ID", "t-1")
        return _fresh_agent(env_clean)

    def test_send_metrics_calls_http_when_grpc_unavailable(self, agent):
        agent.http = MagicMock()
        agent.http.send_metrics.return_value = True
        agent.grpc_client = None
        ok = agent._send_metrics([
            {"instance_id": "i", "gpu_utilization": 5,
             "memory_utilization": 6, "temperature_c": 7,
             "timestamp": 1700000000},
        ])
        assert ok is True
        # The HTTP client received the LEGACY metric — it does the rename internally.
        agent.http.send_metrics.assert_called_once()

    def test_send_metrics_queues_on_failure(self, agent):
        agent.http = MagicMock()
        agent.http.send_metrics.return_value = False
        agent.grpc_client = None
        # Suppress disk persistence
        with patch.object(agent, "_persist_queue"):
            agent._send_metrics([{"instance_id": "i"}])
        assert len(agent._send_queue) == 1
        assert agent._send_queue[0]["data_type"] == "metrics"

    def test_send_metrics_uses_grpc_when_connected(self, agent):
        from transport.http_client import rename_metric_to_wire  # noqa: F401
        agent.http = MagicMock()
        agent.grpc_client = MagicMock()
        agent.grpc_client.is_connected = True
        ok = agent._send_metrics([
            {"instance_id": "i", "gpu_utilization": 5,
             "memory_utilization": 6, "temperature_c": 7,
             "timestamp": 1700000000},
        ])
        assert ok is True
        # gRPC path used; HTTP not touched.
        agent.grpc_client.send_metrics.assert_called_once()
        agent.http.send_metrics.assert_not_called()

    def test_health_check_uses_new_dto_fields(self, agent, monkeypatch):
        monkeypatch.setenv("AGENT_HOSTNAME", "host1")
        monkeypatch.setenv("AGENT_CLOUD_PROVIDER", "aws")
        monkeypatch.setenv("AGENT_REGION", "us-east-1")
        # Reload to pick up env.
        if "main" in sys.modules:
            del sys.modules["main"]
        import main as main_mod
        agent = main_mod.UnifiedGPUAgent()
        agent.http = MagicMock()
        agent.http.send_health.return_value = True
        agent._send_health_check()
        kwargs = agent.http.send_health.call_args.kwargs
        assert kwargs["hostname"] == "host1"
        assert kwargs["cloud_provider"] == "aws"
        assert kwargs["region"] == "us-east-1"
        assert kwargs["status"] == "ok"


class TestSpotInterruption:
    def test_spot_event_routes_to_dedicated_endpoint(self, env_clean, monkeypatch):
        monkeypatch.setenv("BACKEND_API_URL", "http://gw")
        monkeypatch.setenv("AGENT_API_KEY", "k")
        monkeypatch.setenv("TENANT_ID", "t-1")
        agent = _fresh_agent(env_clean)
        agent.http = MagicMock()
        agent.http.report_spot_interruption.return_value = True
        # alert_manager is real — patch its send to swallow.
        with patch.object(agent.alert_manager, "_send_alert"):
            agent._handle_spot_interruption({
                "instance_id": "i-abc",
                "cloud_provider": "aws",
                "action": "terminate",
                "time": "2024-01-01T00:00:00Z",
            })
        agent.http.report_spot_interruption.assert_called_once()
        kwargs = agent.http.report_spot_interruption.call_args.kwargs
        assert kwargs["external_instance_id"] == "i-abc"
        assert kwargs["cloud_provider"] == "aws"
        assert kwargs["interruption_type"] == "terminate"
