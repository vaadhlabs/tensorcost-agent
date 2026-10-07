"""Tests for the new HTTP transport — focuses on the metric / instance
field renames and the auth-header swap. These shapes MUST match the
DTO classes at apps-new/backend/services/gpu-service/src/sync/dto/."""

import os
import sys

import pytest
import responses

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from transport.http_client import (  # noqa: E402
    HttpSyncClient,
    rename_metric_to_wire,
    rename_instance_to_wire,
    _coerce_ts_unix_ms,
)


class TestCoerceTimestamp:
    def test_iso_string(self):
        # 2024-01-01T00:00:00Z
        assert _coerce_ts_unix_ms("2024-01-01T00:00:00") == 1704067200000

    def test_iso_string_with_z_suffix(self):
        assert _coerce_ts_unix_ms("2024-01-01T00:00:00Z") == 1704067200000

    def test_float_seconds(self):
        assert _coerce_ts_unix_ms(1704067200.5) == 1704067200500

    def test_int_seconds(self):
        assert _coerce_ts_unix_ms(1704067200) == 1704067200000

    def test_int_already_ms(self):
        assert _coerce_ts_unix_ms(1704067200000) == 1704067200000

    def test_none_uses_now(self):
        out = _coerce_ts_unix_ms(None)
        # Just check it's a recent epoch-ms; don't pin exact wall clock.
        assert out > 1_700_000_000_000

    def test_unparseable_string_falls_back(self):
        out = _coerce_ts_unix_ms("not a date")
        assert out > 1_700_000_000_000


_UUID = "11111111-2222-3333-4444-555555555555"


class TestRenameMetric:
    def test_ecc_errors_uncorrected_alias_maps_to_total(self):
        legacy = {
            "instance_id": _UUID,
            "timestamp": "2024-01-01T00:00:00",
            "gpu_utilization": 10,
            "ecc_errors_uncorrected": 7,
        }
        wire = rename_metric_to_wire(legacy)
        assert wire["ecc_errors_total"] == 7

    def test_uuid_id_passes_through_as_instance_id(self):
        legacy = {
            "instance_id": _UUID,
            "timestamp": "2024-01-01T00:00:00",
            "gpu_utilization": 75.5,
            "memory_utilization": 60.0,
            "temperature_c": 65.0,
        }
        wire = rename_metric_to_wire(legacy)
        assert wire == {
            "instance_id": _UUID,
            "ts_unix_ms": 1704067200000,
            "util_pct": 75.5,
            "mem_pct": 60.0,
            "temp_c": 65.0,
            "gpu_index": 0,
        }

    def test_cloud_native_id_becomes_cloud_id_pair(self):
        """Server's MetricPointDto resolves (cloud_provider, cloud_id) ->
        UUID via the unique gpu.instance index. Cloud collectors only know
        the cloud-native id (e.g. `i-0abc...`), so the wire must carry it
        as `cloud_id` + `cloud_provider`, never as `instance_id`."""
        legacy = {
            "instance_id": "i-abc",
            "cloud_provider": "aws",
            "timestamp": "2024-01-01T00:00:00",
            "gpu_utilization": 75.5,
            "memory_utilization": 60.0,
            "temperature_c": 65.0,
        }
        wire = rename_metric_to_wire(legacy)
        assert wire == {
            "cloud_id": "i-abc",
            "cloud_provider": "aws",
            "ts_unix_ms": 1704067200000,
            "util_pct": 75.5,
            "mem_pct": 60.0,
            "temp_c": 65.0,
            "gpu_index": 0,
        }

    def test_drops_dropped_fields(self):
        """Backend doesn't accept cpu_utilization / is_idle / source / etc.
        power_usage_w is mapped to power_w (included when non-None), not dropped.
        memory_used_mb and memory_total_mb are forwarded verbatim (added in 025).
        gpu_index is always forwarded (defaults to 0)."""
        legacy = {
            "instance_id": _UUID,
            "gpu_utilization": 50,
            "cpu_utilization": 99,
            "power_usage_w": 350,
            "memory_used_mb": 8000,
            "memory_total_mb": 16000,
            "is_idle": True,
            "source": "cloudwatch",
            "experiment_id": "exp-1",
            "mlflow_run_id": "run-1",
        }
        wire = rename_metric_to_wire(legacy)
        assert wire is not None
        # power_w included (non-None); memory_used_mb + memory_total_mb forwarded (025);
        # gpu_index always present; cpu_utilization / is_idle / source / etc. dropped.
        assert set(wire.keys()) == {
            "instance_id", "ts_unix_ms", "util_pct", "mem_pct", "temp_c",
            "gpu_index", "power_w", "memory_used_mb", "memory_total_mb",
        }

    def test_returns_none_when_instance_id_missing(self):
        assert rename_metric_to_wire({"gpu_utilization": 1}) is None

    def test_returns_none_when_cloud_native_id_lacks_provider(self):
        """No cloud_provider means the server can't resolve the row —
        better to drop locally than send an unaddressable point."""
        assert rename_metric_to_wire({"instance_id": "i-abc"}) is None

    def test_handles_missing_metric_fields_gracefully(self):
        wire = rename_metric_to_wire({"instance_id": _UUID})
        assert wire is not None
        assert wire["util_pct"] == 0.0
        assert wire["mem_pct"] == 0.0
        assert wire["temp_c"] == 0.0

    def test_forwards_dark_signal_fields(self):
        legacy = {
            "instance_id": _UUID,
            "gpu_utilization": 55,
            "clock_mhz": 1410,
            "memory_clock_mhz": 1215,
            "nvlink_links_degraded": 2,
        }
        wire = rename_metric_to_wire(legacy)
        assert wire is not None
        assert wire["clock_mhz"] == 1410
        assert wire["memory_clock_mhz"] == 1215
        assert wire["nvlink_links_degraded"] == 2


class TestRenameInstance:
    def test_uuid_id_minimal_with_state(self):
        wire = rename_instance_to_wire({"instance_id": _UUID, "state": "running"})
        assert wire == {"instance_id": _UUID, "status": "running"}

    def test_cloud_native_id_becomes_cloud_id_pair(self):
        wire = rename_instance_to_wire(
            {"instance_id": "i-abc", "cloud_provider": "aws", "state": "running"}
        )
        assert wire == {"cloud_id": "i-abc", "cloud_provider": "aws", "status": "running"}

    def test_includes_tags(self):
        wire = rename_instance_to_wire(
            {
                "instance_id": "i-abc",
                "cloud_provider": "aws",
                "state": "running",
                "tags": {"team": "ml"},
            }
        )
        assert wire == {
            "cloud_id": "i-abc",
            "cloud_provider": "aws",
            "status": "running",
            "tags": {"team": "ml"},
        }

    def test_drops_extra_fields(self):
        legacy = {
            "instance_id": "i-abc",
            "cloud_provider": "aws",
            "state": "running",
            "instance_type": "p3.2xlarge",
            "region": "us-east-1",
            "availability_zone": "us-east-1a",
            "gpu_count": 1,
            "gpu_type": "V100",
            "launch_time": "2024-01-01",
        }
        wire = rename_instance_to_wire(legacy)
        assert wire is not None
        assert set(wire.keys()) == {"cloud_id", "cloud_provider", "status", "region"}
        assert wire["region"] == "us-east-1"

    def test_returns_none_when_instance_id_missing(self):
        assert rename_instance_to_wire({"state": "running"}) is None

    def test_returns_none_when_cloud_native_id_lacks_provider(self):
        assert rename_instance_to_wire({"instance_id": "i-abc", "state": "running"}) is None


class TestHttpSyncClient:
    @pytest.fixture
    def client(self):
        return HttpSyncClient(base_url="http://gw", api_key="plain-key")

    def test_requires_base_url(self):
        with pytest.raises(ValueError):
            HttpSyncClient(base_url="", api_key="k")

    def test_requires_api_key(self):
        with pytest.raises(ValueError):
            HttpSyncClient(base_url="http://gw", api_key="")

    def test_strips_trailing_slash(self):
        c = HttpSyncClient(base_url="http://gw/", api_key="k")
        assert c.base_url == "http://gw"

    @responses.activate
    def test_send_metrics_uses_x_agent_key_header(self, client):
        responses.add(responses.POST, "http://gw/api/gpu/sync/", json={"success": True}, status=200)
        ok = client.send_metrics([
            {"instance_id": "i-abc", "cloud_provider": "aws",
             "gpu_utilization": 50, "memory_utilization": 60,
             "temperature_c": 70, "timestamp": 1704067200},
        ])
        assert ok is True
        req = responses.calls[0].request
        assert req.headers["X-Agent-Key"] == "plain-key"
        # Critically: the LEGACY headers must NOT be sent.
        assert "X-API-Key" not in req.headers
        assert "X-Tenant-ID" not in req.headers

    @responses.activate
    def test_send_metrics_sends_renamed_payload(self, client):
        responses.add(responses.POST, "http://gw/api/gpu/sync/", json={"success": True}, status=200)
        client.send_metrics([
            {"instance_id": "i-abc", "cloud_provider": "aws",
             "gpu_utilization": 1, "memory_utilization": 2,
             "temperature_c": 3, "timestamp": 1704067200},
        ])
        body = responses.calls[0].request.body
        import json
        payload = json.loads(body)
        assert payload["data_type"] == "metrics"
        # Cloud-native ids ride the (cloud_id, cloud_provider) path so the
        # server can resolve to a UUID via gpu.instance's unique index.
        assert payload["metrics"][0] == {
            "cloud_id": "i-abc", "cloud_provider": "aws",
            "ts_unix_ms": 1704067200000,
            "util_pct": 1.0, "mem_pct": 2.0, "temp_c": 3.0,
            "gpu_index": 0,
        }

    @responses.activate
    def test_send_metrics_skips_when_all_filtered(self, client):
        # No instance_id -> filtered out; client never makes the HTTP call.
        ok = client.send_metrics([{"gpu_utilization": 50}])
        assert ok is True
        assert len(responses.calls) == 0

    @responses.activate
    def test_send_health_dto_shape(self, client):
        responses.add(
            responses.POST, "http://gw/api/gpu/sync/agent/health",
            json={"success": True, "message": "ok", "timestamp": "now"}, status=200,
        )
        client.send_health(
            hostname="host1", version="2.0.0", status="ok",
            cloud_provider="aws", region="us-east-1",
        )
        import json
        body = json.loads(responses.calls[0].request.body)
        # ONLY these five fields per AgentHealthDto:
        assert set(body.keys()) == {"hostname", "version", "status", "cloud_provider", "region"}

    @responses.activate
    def test_send_health_omits_optional_fields_when_unset(self, client):
        responses.add(
            responses.POST, "http://gw/api/gpu/sync/agent/health",
            json={"success": True, "message": "ok", "timestamp": "now"}, status=200,
        )
        client.send_health(hostname="h", version="v")
        import json
        body = json.loads(responses.calls[0].request.body)
        assert "cloud_provider" not in body
        assert "region" not in body

    @responses.activate
    def test_poll_commands_returns_data_array(self, client):
        responses.add(
            responses.GET, "http://gw/api/gpu/sync/agent/commands",
            json={"success": True, "data": [{"id": "c1", "type": "stop", "instance_id": "i", "params": {}}]},
            status=200,
        )
        cmds = client.poll_commands(agent_id="a1")
        assert len(cmds) == 1
        assert cmds[0]["id"] == "c1"

    @responses.activate
    def test_command_result_shape(self, client):
        responses.add(
            responses.POST, "http://gw/api/gpu/sync/agent/command-result",
            json={"success": True}, status=200,
        )
        client.report_command_result(command_id="c1", status="succeeded")
        import json
        body = json.loads(responses.calls[0].request.body)
        assert body == {"command_id": "c1", "status": "succeeded"}

    @responses.activate
    def test_spot_interruption_uses_dedicated_endpoint(self, client):
        responses.add(
            responses.POST, "http://gw/api/gpu/sync/spot-interruption",
            json={"success": True, "event_id": "e1", "alert_id": None, "fallback_action_id": None},
            status=200,
        )
        client.report_spot_interruption(
            external_instance_id="i-abc", cloud_provider="aws",
            interruption_type="terminate",
        )
        import json
        body = json.loads(responses.calls[0].request.body)
        assert body == {
            "external_instance_id": "i-abc",
            "cloud_provider": "aws",
            "interruption_type": "terminate",
        }

    @responses.activate
    def test_post_failure_returns_false(self, client):
        responses.add(responses.POST, "http://gw/api/gpu/sync/", status=500)
        ok = client.send_metrics(
            [{"instance_id": "i", "cloud_provider": "aws", "gpu_utilization": 1}]
        )
        assert ok is False
