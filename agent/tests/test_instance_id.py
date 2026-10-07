"""
Tests for the instance_id plumbing introduced in the multi-instance
deployment feature.

Coverage:
  - HTTP heartbeat body includes instance_id when set
  - Default-generation path: AGENT_INSTANCE_ID unset → hostname-8hex
  - Persistence path: file present → reuse; absent → generate + write
  - Backward compat: old SaaS (no instance_id) is unaffected
  - GrpcClient accepts instance_id kwarg
"""

import json
import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

import pytest
import responses

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# HTTP heartbeat body
# ---------------------------------------------------------------------------

class TestHttpSendHealthInstanceId:
    """send_health includes instance_id in the request body."""

    @pytest.fixture
    def client(self):
        from transport.http_client import HttpSyncClient
        return HttpSyncClient(
            base_url="http://gw",
            api_key="k",
            instance_id="my-host-abc12345",
        )

    @responses.activate
    def test_instance_id_in_body_when_set_on_client(self, client):
        responses.add(
            responses.POST,
            "http://gw/api/gpu/sync/agent/health",
            json={"success": True},
            status=200,
        )
        client.send_health(hostname="h", version="v")
        body = json.loads(responses.calls[0].request.body)
        assert body["instance_id"] == "my-host-abc12345"

    @responses.activate
    def test_instance_id_kwarg_overrides_client_attribute(self, client):
        """Caller can pass a per-call instance_id that takes precedence."""
        responses.add(
            responses.POST,
            "http://gw/api/gpu/sync/agent/health",
            json={"success": True},
            status=200,
        )
        client.send_health(hostname="h", version="v", instance_id="override-99")
        body = json.loads(responses.calls[0].request.body)
        assert body["instance_id"] == "override-99"

    @responses.activate
    def test_instance_id_absent_when_not_set(self):
        """When no instance_id is provided, the field must not appear in the body
        so older SaaS versions see exactly the same payload shape as before."""
        from transport.http_client import HttpSyncClient
        client = HttpSyncClient(base_url="http://gw", api_key="k")
        responses.add(
            responses.POST,
            "http://gw/api/gpu/sync/agent/health",
            json={"success": True},
            status=200,
        )
        client.send_health(hostname="h", version="v")
        body = json.loads(responses.calls[0].request.body)
        assert "instance_id" not in body

    @responses.activate
    def test_other_health_fields_unaffected(self, client):
        """Adding instance_id must not disturb the existing DTO fields."""
        responses.add(
            responses.POST,
            "http://gw/api/gpu/sync/agent/health",
            json={"success": True},
            status=200,
        )
        client.send_health(
            hostname="host1",
            version="2.0.0",
            status="ok",
            cloud_provider="aws",
            region="us-east-1",
        )
        body = json.loads(responses.calls[0].request.body)
        assert body["hostname"] == "host1"
        assert body["version"] == "2.0.0"
        assert body["status"] == "ok"
        assert body["cloud_provider"] == "aws"
        assert body["region"] == "us-east-1"


# ---------------------------------------------------------------------------
# _resolve_instance_id — auto-generation and persistence
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _stub_optional_modules(monkeypatch):
    """Prevent heavy cloud SDK imports from blowing up in CI."""
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
    """Clear all agent env vars for each test."""
    for var in (
        "BACKEND_API_URL", "AGENT_API_KEY", "BACKEND_API_KEY",
        "AGENT_KEY_ID", "AGENT_HMAC_PEPPER", "TENANT_ID",
        "GRPC_TARGET", "COMM_MODE", "AGENT_HOSTNAME", "AGENT_ID",
        "AGENT_INSTANCE_ID", "AGENT_CLOUD_PROVIDER",
        "AWS_ENABLED", "AZURE_ENABLED", "GCP_ENABLED",
        "K8S_ENABLED", "SAGEMAKER_ENABLED", "ENABLED_CLOUDS",
        "NVML_ENABLED", "SPOT_HANDLER_ENABLED",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("LOG_LEVEL", "ERROR")
    yield


def _fresh_agent_class():
    """Return a freshly imported UnifiedGPUAgent class."""
    if "main" in sys.modules:
        del sys.modules["main"]
    import main as main_mod
    return main_mod.UnifiedGPUAgent


class TestResolveInstanceId:
    """Unit-test _resolve_instance_id directly via a minimal agent instance."""

    def _make_agent(self, monkeypatch, env_clean, extra_env=None):
        """Build a minimal agent with env set up."""
        monkeypatch.setenv("BACKEND_API_URL", "http://gw")
        monkeypatch.setenv("AGENT_API_KEY", "k")
        monkeypatch.setenv("TENANT_ID", "t-1")
        if extra_env:
            for k, v in extra_env.items():
                monkeypatch.setenv(k, v)
        cls = _fresh_agent_class()
        return cls()

    def test_env_var_wins_over_everything(self, monkeypatch, env_clean, tmp_path):
        """AGENT_INSTANCE_ID env var is used as-is; file and generation are skipped."""
        agent = self._make_agent(
            monkeypatch, env_clean,
            extra_env={"AGENT_INSTANCE_ID": "operator-explicit-id"},
        )
        assert agent.instance_id == "operator-explicit-id"

    def test_file_reused_on_subsequent_boot(self, monkeypatch, env_clean, tmp_path):
        """When the persistence file already exists, its content is returned."""
        instance_file = tmp_path / "instance_id"
        instance_file.write_text("persisted-from-last-boot")
        cls = _fresh_agent_class()
        # Override the class-level path for this test.
        monkeypatch.setattr(cls, "_INSTANCE_ID_FILE", str(instance_file))
        monkeypatch.setenv("BACKEND_API_URL", "http://gw")
        monkeypatch.setenv("AGENT_API_KEY", "k")
        monkeypatch.setenv("TENANT_ID", "t-1")
        agent = cls()
        assert agent.instance_id == "persisted-from-last-boot"

    def test_auto_generate_writes_file(self, monkeypatch, env_clean, tmp_path):
        """With no env var and no file, a value is generated and written."""
        instance_file = tmp_path / "instance_id"
        cls = _fresh_agent_class()
        monkeypatch.setattr(cls, "_INSTANCE_ID_FILE", str(instance_file))
        monkeypatch.setenv("BACKEND_API_URL", "http://gw")
        monkeypatch.setenv("AGENT_API_KEY", "k")
        monkeypatch.setenv("TENANT_ID", "t-1")
        agent = cls()
        # A value was chosen.
        assert agent.instance_id
        # It was written to disk.
        assert instance_file.read_text() == agent.instance_id

    def test_auto_generated_pattern(self, monkeypatch, env_clean, tmp_path):
        """Auto-generated id matches {hostname}-{8-hex-chars}."""
        import re
        instance_file = tmp_path / "instance_id"
        cls = _fresh_agent_class()
        monkeypatch.setattr(cls, "_INSTANCE_ID_FILE", str(instance_file))
        monkeypatch.setenv("BACKEND_API_URL", "http://gw")
        monkeypatch.setenv("AGENT_API_KEY", "k")
        monkeypatch.setenv("TENANT_ID", "t-1")
        agent = cls()
        # Pattern: <non-empty-hostname>-<exactly-8-hex-chars>
        assert re.match(r'^.+\-[0-9a-f]{8}$', agent.instance_id), (
            f"Generated instance_id {agent.instance_id!r} doesn't match expected pattern"
        )

    def test_auto_generate_uses_os_hostname(self, monkeypatch, env_clean, tmp_path):
        """The auto-generated prefix comes from socket.gethostname()."""
        instance_file = tmp_path / "instance_id"
        cls = _fresh_agent_class()
        monkeypatch.setattr(cls, "_INSTANCE_ID_FILE", str(instance_file))
        monkeypatch.setenv("BACKEND_API_URL", "http://gw")
        monkeypatch.setenv("AGENT_API_KEY", "k")
        monkeypatch.setenv("TENANT_ID", "t-1")
        import socket
        expected_prefix = socket.gethostname()
        agent = cls()
        assert agent.instance_id.startswith(expected_prefix + "-")

    def test_file_absent_then_present_on_next_agent_init(self, monkeypatch, env_clean, tmp_path):
        """A second agent init against the same file path reuses what the first wrote."""
        instance_file = tmp_path / "instance_id"
        cls = _fresh_agent_class()
        monkeypatch.setattr(cls, "_INSTANCE_ID_FILE", str(instance_file))
        monkeypatch.setenv("BACKEND_API_URL", "http://gw")
        monkeypatch.setenv("AGENT_API_KEY", "k")
        monkeypatch.setenv("TENANT_ID", "t-1")
        # First boot — generates and writes.
        first = cls()
        first_id = first.instance_id
        # Second boot — reads from file.
        if "main" in sys.modules:
            del sys.modules["main"]
        import main as main_mod
        monkeypatch.setattr(main_mod.UnifiedGPUAgent, "_INSTANCE_ID_FILE", str(instance_file))
        second = main_mod.UnifiedGPUAgent()
        assert second.instance_id == first_id

    def test_info_logged_when_auto_generated(self, monkeypatch, env_clean, tmp_path, caplog):
        """Operators must be able to see the chosen id in the logs."""
        import logging
        instance_file = tmp_path / "instance_id"
        cls = _fresh_agent_class()
        monkeypatch.setattr(cls, "_INSTANCE_ID_FILE", str(instance_file))
        monkeypatch.setenv("BACKEND_API_URL", "http://gw")
        monkeypatch.setenv("AGENT_API_KEY", "k")
        monkeypatch.setenv("TENANT_ID", "t-1")
        monkeypatch.setenv("LOG_LEVEL", "INFO")
        with caplog.at_level(logging.INFO):
            agent = cls()
        assert agent.instance_id in caplog.text


# ---------------------------------------------------------------------------
# Backward compatibility — no instance_id field → old SaaS still works
# ---------------------------------------------------------------------------

class TestBackwardCompat:
    @responses.activate
    def test_send_health_without_instance_id_matches_prior_shape(self):
        """When instance_id is absent the body is identical to the pre-feature wire."""
        from transport.http_client import HttpSyncClient
        client = HttpSyncClient(base_url="http://gw", api_key="k")
        responses.add(
            responses.POST,
            "http://gw/api/gpu/sync/agent/health",
            json={"success": True},
            status=200,
        )
        client.send_health(
            hostname="h", version="v", status="ok",
            cloud_provider="aws", region="us-east-1",
        )
        body = json.loads(responses.calls[0].request.body)
        # Exactly the five pre-feature fields — no extras that could trip
        # an old SaaS running strict DTO validation.
        assert set(body.keys()) == {"hostname", "version", "status", "cloud_provider", "region"}


# ---------------------------------------------------------------------------
# GrpcClient accepts instance_id
# ---------------------------------------------------------------------------

class TestGrpcClientInstanceId:
    def test_grpc_client_stores_instance_id(self):
        from grpc_client import GrpcClient
        gc = GrpcClient(
            grpc_target="localhost:50051",
            agent_id="host1",
            tenant_id="t-1",
            key_id="k1",
            hmac_pepper="deadbeef" * 8,
            instance_id="my-instance-abc12345",
        )
        assert gc.instance_id == "my-instance-abc12345"

    def test_grpc_client_instance_id_defaults_none(self):
        from grpc_client import GrpcClient
        gc = GrpcClient(
            grpc_target="localhost:50051",
            agent_id="host1",
            tenant_id="t-1",
            key_id="k1",
            hmac_pepper="deadbeef" * 8,
        )
        assert gc.instance_id is None
