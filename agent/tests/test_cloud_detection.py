"""
Tests for the unified host-cloud-detection model in `main.py` (audit
§8.3 "two parallel models of what cloud are we on" follow-up).

The agent's host identity (`self.cloud_provider`) is resolved from a
single source of truth — operator env wins, IMDS auto-detection second,
``onprem`` fallback last. Per-cloud monitor-enable flags
(``AWS_ENABLED`` etc.) are an INDEPENDENT concern: they decide which
external clouds the agent SCRAPES, not what the agent IS. These tests
pin both halves so a future refactor can't reintroduce the parallel
model the audit flagged.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture(autouse=True)
def _stub_optional_modules():
    """Mock heavy cloud SDKs the same way test_main_v2 does."""
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
    for var in (
        "AGENT_CLOUD_PROVIDER", "AGENT_REGION", "AGENT_CLOUD_ID",
        "AWS_REGION",
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


def _fresh_main_module():
    """Force a clean re-import of `main` so module-level state resets."""
    if "main" in sys.modules:
        del sys.modules["main"]
    import main as main_mod  # noqa: WPS433
    return main_mod


# --- host-identity resolution (priority order) ----------------------- #


class TestHostIdentitySingleSourceOfTruth:
    def test_env_var_wins_imds_not_called(self, env_clean, monkeypatch):
        """When AGENT_CLOUD_PROVIDER is set, IMDS must NOT be probed."""
        monkeypatch.setenv("AGENT_CLOUD_PROVIDER", "gcp")
        main_mod = _fresh_main_module()
        with patch("src.cloud_identity.detect_cloud_identity") as imds:
            agent = main_mod.UnifiedGPUAgent()
            assert agent.cloud_provider == "gcp"
            imds.assert_not_called()

    def test_imds_aws_when_env_unset(self, env_clean, monkeypatch):
        """No env var, IMDS reports AWS — agent identifies as aws."""
        from src.cloud_identity import CloudIdentity

        main_mod = _fresh_main_module()
        with patch(
            "src.cloud_identity.detect_cloud_identity",
            return_value=CloudIdentity(
                cloud_provider="aws",
                cloud_id="i-0abc",
                region="us-east-1",
            ),
        ):
            agent = main_mod.UnifiedGPUAgent()
        assert agent.cloud_provider == "aws"
        assert agent.cloud_id == "i-0abc"
        assert agent.region == "us-east-1"

    def test_imds_timeout_falls_back_to_onprem(self, env_clean, monkeypatch):
        """No env var, IMDS times out / returns None — onprem fallback."""
        main_mod = _fresh_main_module()
        with patch(
            "src.cloud_identity.detect_cloud_identity",
            return_value=None,
        ):
            agent = main_mod.UnifiedGPUAgent()
        assert agent.cloud_provider == "onprem"

    def test_imds_exception_falls_back_to_onprem(self, env_clean):
        """IMDS raising must not block boot — falls back to onprem."""
        main_mod = _fresh_main_module()
        with patch(
            "src.cloud_identity.detect_cloud_identity",
            side_effect=RuntimeError("network down"),
        ):
            agent = main_mod.UnifiedGPUAgent()
        assert agent.cloud_provider == "onprem"


# --- monitor-enable flags are INDEPENDENT of host identity ----------- #


class TestMonitorEnableIsIndependentOfHostIdentity:
    def test_env_aws_plus_yaml_aws_enabled_loads_aws_monitor(
        self, env_clean, monkeypatch
    ):
        """Both env says AWS and AWS_ENABLED=true: identity=aws AND
        AWS monitor loaded (the typical single-cloud case)."""
        monkeypatch.setenv("AGENT_CLOUD_PROVIDER", "aws")
        monkeypatch.setenv("AWS_ENABLED", "true")
        main_mod = _fresh_main_module()
        # Stub the heavy boto3 imports inside AWSMonitor
        fake_aws_monitor = MagicMock()
        with patch.dict(
            sys.modules,
            {"monitors.aws_monitor": MagicMock(AWSMonitor=fake_aws_monitor)},
        ):
            agent = main_mod.UnifiedGPUAgent()
        assert agent.cloud_provider == "aws"
        assert "aws" in agent.monitors
        fake_aws_monitor.assert_called_once()

    def test_env_azure_plus_aws_enabled_cross_cloud(self, env_clean, monkeypatch):
        """Customer running on Azure, sampling AWS resources from that
        Azure host: identity remains azure, AWS monitor still loads.
        This is the documented host-identity-vs-monitor-enable split.
        """
        monkeypatch.setenv("AGENT_CLOUD_PROVIDER", "azure")
        monkeypatch.setenv("AWS_ENABLED", "true")
        main_mod = _fresh_main_module()
        fake_aws_monitor = MagicMock()
        with patch.dict(
            sys.modules,
            {"monitors.aws_monitor": MagicMock(AWSMonitor=fake_aws_monitor)},
        ):
            agent = main_mod.UnifiedGPUAgent()
        # Host identity reflects WHERE the agent runs, not what it scrapes.
        assert agent.cloud_provider == "azure"
        # Cross-cloud monitor still loads.
        assert "aws" in agent.monitors
        fake_aws_monitor.assert_called_once()

    def test_no_yaml_enable_means_no_monitor_even_if_imds_says_aws(
        self, env_clean, monkeypatch
    ):
        """IMDS detection alone does NOT auto-load any monitor — the
        operator must opt in via the YAML/env enable flag. This
        prevents an unexpected boto3 call against the host's IMDS role
        on a vanilla install."""
        from src.cloud_identity import CloudIdentity

        main_mod = _fresh_main_module()
        with patch(
            "src.cloud_identity.detect_cloud_identity",
            return_value=CloudIdentity(
                cloud_provider="aws", cloud_id="i-0abc", region="us-east-1",
            ),
        ):
            agent = main_mod.UnifiedGPUAgent()
        assert agent.cloud_provider == "aws"
        assert agent.monitors == {}


# --- send-queue overflow telemetry (audit §8.3) ---------------------- #


class TestSendQueueOverflowTelemetry:
    def _agent(self, env_clean):
        main_mod = _fresh_main_module()
        with patch(
            "src.cloud_identity.detect_cloud_identity",
            return_value=None,
        ):
            return main_mod.UnifiedGPUAgent()

    def test_below_threshold_no_log(self, env_clean, caplog):
        agent = self._agent(env_clean)
        # Shrink for fast assertion.
        agent._send_queue_maxlen = 100
        with caplog.at_level("WARNING"):
            agent._send_queue_append({"data_type": "metrics"})
        assert not any(
            "send queue" in r.getMessage() for r in caplog.records
        )

    def test_warn_at_80_percent(self, env_clean, caplog):
        agent = self._agent(env_clean)
        agent._send_queue_maxlen = 10
        # Pre-fill to 7 (next append → depth 8 = 80%).
        for _ in range(7):
            agent._send_queue.append({"data_type": "metrics"})
        with caplog.at_level("WARNING"):
            agent._send_queue_append({"data_type": "metrics"})
        warns = [r for r in caplog.records if r.levelname == "WARNING"]
        assert any("80%" in r.getMessage() for r in warns)

    def test_error_at_95_percent(self, env_clean, caplog):
        agent = self._agent(env_clean)
        agent._send_queue_maxlen = 100
        for _ in range(94):
            agent._send_queue.append({"data_type": "metrics"})
        with caplog.at_level("ERROR"):
            agent._send_queue_append({"data_type": "metrics"})
        errs = [r for r in caplog.records if r.levelname == "ERROR"]
        assert any("near saturation" in r.getMessage() for r in errs)

    def test_saturation_drops_oldest_and_logs(self, env_clean, caplog):
        # The deque is constructed in __init__ with maxlen=10000; we
        # rebuild it small so the test runs in O(1).
        agent = self._agent(env_clean)
        from collections import deque
        agent._send_queue_maxlen = 3
        agent._send_queue = deque(maxlen=3)
        for i in range(3):
            agent._send_queue.append({"data_type": "metrics", "n": i})
        with caplog.at_level("ERROR"):
            agent._send_queue_append({"data_type": "metrics", "n": 99})
        errs = [r for r in caplog.records if r.levelname == "ERROR"]
        assert any("saturated" in r.getMessage() for r in errs)
        assert agent._send_queue_dropped_total == 1
        # New item present, oldest evicted.
        items = list(agent._send_queue)
        assert items[-1]["n"] == 99
        assert items[0]["n"] == 1

    def test_threshold_log_is_rate_limited(self, env_clean, caplog):
        """Repeated appends in the warn band do NOT flood the log."""
        agent = self._agent(env_clean)
        agent._send_queue_maxlen = 100
        agent._send_queue_log_interval_s = 60.0
        # depth 80 = 80% threshold, 95 = 95% threshold. Pre-fill to 80
        # so the first append crosses the warn-band exactly.
        for _ in range(80):
            agent._send_queue.append({"data_type": "metrics"})
        with caplog.at_level("WARNING"):
            # Fire 5 appends — only the first should log.
            for _ in range(5):
                # Keep depth in the 80–94% band by popping after each
                # append so we re-cross the 80% threshold without
                # crossing 95%.
                agent._send_queue_append({"data_type": "metrics"})
                agent._send_queue.pop()
        warn_logs = [
            r for r in caplog.records
            if r.levelname == "WARNING" and "80%" in r.getMessage()
        ]
        assert len(warn_logs) == 1
