"""
HTTP client for the TensorCost gpu-service /api/sync surface.

Owns auth header naming (`X-Agent-Key`), endpoint routing, and the
metric-field rename from the legacy collector shape to the new
`util_pct` / `mem_pct` / `temp_c` / `ts_unix_ms` shape that
`SyncBatchDto` requires. Per AGENT_COMPATIBILITY.md the new backend
accepts only metrics + instance_updates on the batch endpoint; spot
interruptions go to a dedicated controller.

Wire endpoints used (all `apps-new/backend/services/gpu-service/src/sync/
sync.controller.ts`):

  POST /api/gpu/sync/                  metrics + instance_updates batch
  POST /api/gpu/sync/agent/health      agent heartbeat (DTO=AgentHealthDto)
  GET  /api/gpu/sync/agent/commands    poll for pending instance commands
  POST /api/gpu/sync/agent/command-result  command execution outcome
  POST /api/gpu/sync/spot-interruption     dedicated spot-event endpoint

DROPPED — these endpoints don't exist on apps-new and the agent must
not send them. See AGENT_COMPATIBILITY.md §2.3 "Dropped Event Types".

  costs / ai_spend / inference_metrics / training_runs / gateway_metrics
  cluster_health / recommendations / errors / generic alerts
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests

from transport.tls_pinning import TlsConfig, build_pinned_https_adapter


logger = logging.getLogger(__name__)


# ----- tag egress filter ----------------------------------------------------
#
# Audit §8.2 Blocker #5 — instance tags were forwarded full-fidelity to
# the backend, no allowlist, no length cap, no PII filter. Tags routinely
# contain owner emails, project codenames, and occasionally secrets.
# Default-deny allowlist, with two opt-in env knobs:
#
#   AGENT_TAG_ALLOWLIST       comma-separated keys; replaces the default
#                             when set. Empty string = empty allowlist
#                             (drop everything).
#   AGENT_TAG_HASH_UNKNOWN    if "true" (case-insensitive), retain
#                             disallowed keys but replace value with the
#                             first 16 hex chars of sha256(value). Lets
#                             ops see tag *cardinality* without exfilling
#                             the literal value.
#
# Length cap: any value over 256 chars is truncated to value[:253] + "..."
# regardless of whether it matches the allowlist or is being hashed.

_DEFAULT_TAG_ALLOWLIST = frozenset({
    "cost-center",
    "cost_center",
    "environment",
    "env",
    "team",
    "owner",
    "project",
    "service",
    "tier",
})

_TAG_VALUE_MAX_LEN = 256
_TAG_VALUE_TRUNC_SUFFIX = "..."


def _tag_allowlist() -> frozenset:
    raw = os.getenv("AGENT_TAG_ALLOWLIST")
    if raw is None:
        return _DEFAULT_TAG_ALLOWLIST
    # Empty string is an explicit "drop everything" — distinct from unset.
    return frozenset(part.strip() for part in raw.split(",") if part.strip())


def _tag_hash_unknown() -> bool:
    return os.getenv("AGENT_TAG_HASH_UNKNOWN", "").strip().lower() == "true"


def _truncate_tag_value(value: str) -> str:
    if len(value) <= _TAG_VALUE_MAX_LEN:
        return value
    return value[: _TAG_VALUE_MAX_LEN - len(_TAG_VALUE_TRUNC_SUFFIX)] + _TAG_VALUE_TRUNC_SUFFIX


def _hash_tag_value(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()[:16]


def filter_tags(tags: Optional[Dict[Any, Any]]) -> Dict[str, str]:
    """
    Apply the agent's tag-egress posture to a legacy collector tag dict.

    * Keys not in the allowlist are dropped, OR — if
      AGENT_TAG_HASH_UNKNOWN=true — retained with their value replaced
      by sha256(value)[:16].
    * Allowlisted values are passed through, length-capped to 256 chars
      with a "..." suffix.
    * None / empty / non-dict input returns {}.

    The function is read at call time, not at module import — this lets
    tests and operators flip the env without restarting the agent.
    """
    if not tags or not isinstance(tags, dict):
        return {}
    allow = _tag_allowlist()
    hash_unknown = _tag_hash_unknown()
    out: Dict[str, str] = {}
    for k, v in tags.items():
        key = str(k)
        value = str(v) if v is not None else ""
        if key in allow:
            out[key] = _truncate_tag_value(value)
        elif hash_unknown:
            out[key] = _hash_tag_value(_truncate_tag_value(value))
        # else: silently drop
    return out


# ----- metric-rename helper -------------------------------------------------

def _coerce_ts_unix_ms(legacy_ts: Any) -> int:
    """
    Convert a legacy `timestamp` field — accepts:
      - ISO 8601 string with optional 'Z' suffix
      - float / int seconds since epoch
      - missing / None  → current wall clock
    Returns integer epoch milliseconds.
    """
    if legacy_ts is None or legacy_ts == "":
        return int(time.time() * 1000)
    if isinstance(legacy_ts, (int, float)):
        # Heuristic: if value > 1e11 assume already ms.
        return int(legacy_ts) if legacy_ts > 1e11 else int(legacy_ts * 1000)
    if isinstance(legacy_ts, str):
        try:
            # Strip a trailing 'Z' since fromisoformat accepts +00:00 not Z
            # until Python 3.11. Be conservative.
            s = legacy_ts.rstrip("Z")
            dt = datetime.fromisoformat(s)
            # The legacy collectors emit `datetime.utcnow().isoformat()`
            # which is timezone-naive UTC. .timestamp() on naive datetimes
            # interprets them as LOCAL time — wrong. Force UTC if naive.
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return int(dt.timestamp() * 1000)
        except ValueError:
            logger.warning("could not parse timestamp %r — using now()", legacy_ts)
            return int(time.time() * 1000)
    return int(time.time() * 1000)


_UUID_RE = re.compile(r"^[0-9a-f-]{36}$", re.IGNORECASE)


def _identity_fields(legacy: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Resolve a legacy `{instance_id, cloud_provider}` pair into the
    server's identity shape:

      * UUID-shaped legacy id  -> `{instance_id: <uuid>}`
      * cloud-native legacy id -> `{cloud_id: <i-…>, cloud_provider: <aws|gcp|azure|k8s>}`

    Server-side `MetricPointDto` / `InstanceUpdateDto` accept either path
    and `SyncService` resolves cloud-id pairs to UUIDs via the unique
    `(tenant_id, cloud_provider, instance_id)` index. Returns None if
    we have neither a UUID nor a cloud_id+cloud_provider pair — pushing
    an unaddressable metric just creates a server-side warn log.
    """
    raw = legacy.get("instance_id")
    if not raw:
        return None
    raw = str(raw)
    if _UUID_RE.match(raw):
        return {"instance_id": raw}
    cloud_provider = legacy.get("cloud_provider")
    if not cloud_provider:
        return None
    return {"cloud_id": raw, "cloud_provider": str(cloud_provider)}


def rename_metric_to_wire(legacy: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Map a legacy collector metric dict to the new
    `MetricPointDto` shape. Returns None if no resolvable identity
    (no UUID and no cloud_id+cloud_provider pair).

    Field map (per AGENT_COMPATIBILITY.md §2.2):
      gpu_utilization     -> util_pct
      memory_utilization  -> mem_pct
      temperature_c       -> temp_c
      timestamp (ISO/sec) -> ts_unix_ms (epoch ms int)
      gpu_index           -> gpu_index (forwarded verbatim, default 0)
      power_usage_w       -> power_w   (None -> omitted; migration 022 DEFAULT NULL)

    Identity carried through (UUID path or cloud_id+cloud_provider).
    Dropped: cpu_utilization, memory_used_mb, memory_total_mb, source,
    is_idle, experiment_id, mlflow_run_id, sagemaker_job_name,
    gpu_utilization_local, memory_utilization_local.
    """
    identity = _identity_fields(legacy)
    if identity is None:
        return None
    out: Dict[str, Any] = {
        **identity,
        "ts_unix_ms": _coerce_ts_unix_ms(legacy.get("timestamp")),
        "util_pct": float(legacy.get("gpu_utilization", 0) or 0),
        "mem_pct": float(legacy.get("memory_utilization", 0) or 0),
        "temp_c": float(legacy.get("temperature_c", 0) or 0),
        "gpu_index": int(legacy.get("gpu_index", 0) or 0),
    }
    # power_usage_w is None when the GPU doesn't support power readings.
    # Only include the key when the value is non-None so the server can
    # distinguish "not reported" (key absent / proto3 zero-default) from
    # "genuinely 0 W" — the server writes NULL for missing values.
    raw_power = legacy.get("power_usage_w")
    if raw_power is not None:
        out["power_w"] = float(raw_power)
    # mem_bw_pct: only include when explicitly non-None. The server stores
    # NULL for the 0.0 proto3 default (migration 023 conflation note), so
    # omitting the key here causes the gRPC/HTTP path to send 0.0, which
    # the sync service translates to NULL — same NULL semantics as power_w.
    raw_mbw = legacy.get("mem_bw_pct")
    if raw_mbw is not None:
        out["mem_bw_pct"] = float(raw_mbw)
    # ecc_errors_total: cumulative uncorrected ECC count. Unlike power_w and
    # mem_bw_pct, 0 is a legitimate value (no errors since boot) and is NOT
    # mapped to NULL by the sync service. Include the key only when non-None;
    # absent key = pre-Volta GPU or agent predating migration 024.
    raw_ecc = legacy.get("ecc_errors_total")
    if raw_ecc is None and legacy.get("ecc_errors_uncorrected") is not None:
        raw_ecc = legacy.get("ecc_errors_uncorrected")
    if raw_ecc is not None:
        out["ecc_errors_total"] = int(raw_ecc)
    # memory_used_mb: 0.0 IS a valid value (GPU with nothing loaded yet). Do
    # not apply a zero-sentinel — the server stores 0 as 0. Include only
    # when non-None; absent = agent predating migration 025 or NVML failure.
    raw_mem_used = legacy.get("memory_used_mb")
    if raw_mem_used is not None:
        out["memory_used_mb"] = float(raw_mem_used)
    # memory_total_mb: 0.0 is NOT a valid value (any real GPU has positive
    # capacity). Include only when non-None so the server can apply the
    # NULLIF(x, 0) sentinel pattern from migration 025 — 0.0 on the wire
    # means "not reported" just like power_w.
    raw_mem_total = legacy.get("memory_total_mb")
    if raw_mem_total is not None:
        out["memory_total_mb"] = float(raw_mem_total)
    raw_clock = legacy.get("clock_mhz")
    if raw_clock is not None:
        out["clock_mhz"] = int(raw_clock)
    raw_mem_clock = legacy.get("memory_clock_mhz")
    if raw_mem_clock is not None:
        out["memory_clock_mhz"] = int(raw_mem_clock)
    raw_nvlink = legacy.get("nvlink_links_degraded")
    if raw_nvlink is not None:
        out["nvlink_links_degraded"] = int(raw_nvlink)
    return out


def rename_instance_to_wire(legacy: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Map a legacy collector instance dict to the new `InstanceUpdateDto`.
    Identity follows the same rule as `rename_metric_to_wire`:
    UUID-shaped legacy id wins; otherwise cloud_id+cloud_provider.
    """
    identity = _identity_fields(legacy)
    if identity is None:
        return None
    out: Dict[str, Any] = dict(identity)
    state = legacy.get("state") or legacy.get("status")
    if state:
        out["status"] = str(state)
    region = legacy.get("region")
    if region:
        out["region"] = str(region).strip()
    filtered = filter_tags(legacy.get("tags"))
    if filtered:
        out["tags"] = filtered
    parts = legacy.get("mig_partitions")
    if isinstance(parts, list) and len(parts) > 0:
        out["mig_enabled"] = bool(legacy.get("mig_enabled", True))
        out["mig_partitions"] = [
            {
                "partition_id": str(p.get("partition_id") or ""),
                "profile": str(p.get("profile") or "unknown"),
                "memory_mb": int(p.get("memory_mb") or 0),
                "compute_slice": int(p.get("compute_slice") or 0),
                "gpu_index": int(p.get("gpu_index") or 0),
            }
            for p in parts
            if isinstance(p, dict)
        ]
    return out


# ----- the client -----------------------------------------------------------

class HttpSyncClient:
    """
    HTTP client for the TensorCost gpu-service /api/sync surface.

    Auth: header `X-Agent-Key: <plain-api-key>` validated server-side
    against `gpu.agent.api_key_hash` (bcrypt). The plain key is the
    output of the provisioning CLI's `api_key` field, stored in env as
    `AGENT_API_KEY`.

    All HTTP calls are best-effort — failures are logged + bubbled up
    via a bool return so callers can decide whether to enqueue for
    retry.
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        timeout_seconds: int = 30,
        tls_config: Optional[TlsConfig] = None,
        instance_id: Optional[str] = None,
    ):
        if not base_url:
            raise ValueError("base_url is required")
        if not api_key:
            raise ValueError("api_key is required (set AGENT_API_KEY)")
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key
        self.timeout = timeout_seconds
        self._tls_config = tls_config
        self.instance_id = instance_id

        # Build a requests.Session so we can attach a pinning adapter +
        # mTLS client cert + CA bundle once. All HTTP calls go through
        # this session. When no hardening is configured, this is a
        # vanilla session that behaves identically to module-level
        # requests.* — the prior default.
        self._session = requests.Session()
        self._verify: Any = True
        self._cert: Optional[Tuple[str, str]] = None
        if tls_config is not None:
            if tls_config.ca_bundle_path:
                # Replaces system roots — both for HTTPS pinning and for
                # the BYOC topology where the customer's internal CA
                # issued the gateway cert.
                self._verify = tls_config.ca_bundle_path
            if tls_config.mtls_enabled:
                self._cert = (
                    tls_config.mtls_cert_path,
                    tls_config.mtls_key_path,
                )
            if tls_config.https_pinning_enabled:
                adapter = build_pinned_https_adapter(tls_config)
                self._session.mount("https://", adapter)

    def _headers(self) -> Dict[str, str]:
        return {
            "X-Agent-Key": self._api_key,
            "Content-Type": "application/json",
        }

    def _request_kwargs(self) -> Dict[str, Any]:
        """Common per-request kwargs for verify / cert when TLS hardening
        is configured. Empty dict on default deploys."""
        out: Dict[str, Any] = {}
        if self._verify is not True:
            out["verify"] = self._verify
        if self._cert is not None:
            out["cert"] = self._cert
        return out

    # ----- POST /api/gpu/sync/ ------------------------------------------------

    def send_metrics(self, legacy_metrics: Iterable[Dict[str, Any]]) -> bool:
        """Translate + POST a metrics batch. No-op for empty input."""
        wire_points = [
            p for p in (rename_metric_to_wire(m) for m in legacy_metrics) if p is not None
        ]
        if not wire_points:
            return True
        payload = {"data_type": "metrics", "metrics": wire_points}
        return self._post_sync(payload)

    def send_instance_updates(self, legacy_instances: Iterable[Dict[str, Any]]) -> bool:
        wire_updates = [
            i for i in (rename_instance_to_wire(x) for x in legacy_instances) if i is not None
        ]
        if not wire_updates:
            return True
        payload = {"data_type": "instance_updates", "instance_updates": wire_updates}
        return self._post_sync(payload)

    def _post_sync(self, payload: Dict[str, Any]) -> bool:
        try:
            r = self._session.post(
                f"{self.base_url}/api/gpu/sync/",
                headers=self._headers(),
                json=payload,
                timeout=self.timeout,
                **self._request_kwargs(),
            )
            r.raise_for_status()
            return True
        except requests.RequestException as e:
            logger.warning("POST /api/sync failed: %s", e)
            return False

    # ----- POST /api/gpu/sync/agent/health -----------------------------------

    def send_health(
        self,
        *,
        hostname: str,
        version: str,
        status: str = "ok",
        cloud_provider: Optional[str] = None,
        region: Optional[str] = None,
        instance_id: Optional[str] = None,
    ) -> bool:
        """
        Heartbeat the gpu-service agent table.

        `instance_id` is the stable per-host identity introduced for
        multi-instance deployments (k8s DaemonSets, autoscale groups)
        where multiple agents share one credential bundle. When present,
        the SaaS writes a row in `gpu.agent_instance` keyed
        `(agent_id, instance_id)` instead of updating the parent agent
        row directly. Backward-compatible — old SaaS versions that
        don't know the field ignore it.

        Falls back to `self.instance_id` when the caller doesn't
        supply one explicitly (supports call sites that were written
        before instance_id existed).
        """
        effective_instance_id = instance_id or self.instance_id
        body: Dict[str, Any] = {
            "hostname": hostname,
            "version": version,
            "status": status,
        }
        if cloud_provider:
            body["cloud_provider"] = cloud_provider
        if region:
            body["region"] = region
        if effective_instance_id:
            body["instance_id"] = effective_instance_id
        try:
            r = self._session.post(
                f"{self.base_url}/api/gpu/sync/agent/health",
                headers=self._headers(),
                json=body,
                timeout=10,
                **self._request_kwargs(),
            )
            r.raise_for_status()
            return True
        except requests.RequestException as e:
            logger.warning("POST /api/gpu/sync/agent/health failed: %s", e)
            return False

    # ----- GET /api/gpu/sync/agent/commands ---------------------------------

    def poll_commands(self, agent_id: str) -> List[Dict[str, Any]]:
        """
        Returns up to 10 pending agent-executed commands. The server
        atomically flips them to in_progress + pins agent ownership, so
        the caller MUST send a command-result for each item to clear
        the row (else the row stays in_progress forever).
        """
        try:
            r = self._session.get(
                f"{self.base_url}/api/gpu/sync/agent/commands",
                headers=self._headers(),
                params={"agent_id": agent_id},
                timeout=10,
                **self._request_kwargs(),
            )
            if r.status_code != 200:
                logger.warning("GET /api/gpu/sync/agent/commands -> %d", r.status_code)
                return []
            return r.json().get("data", []) or []
        except requests.RequestException as e:
            logger.warning("GET /api/gpu/sync/agent/commands failed: %s", e)
            return []

    # ----- POST /api/gpu/sync/agent/command-result --------------------------

    def report_command_result(
        self,
        *,
        command_id: str,
        status: str,
        error: Optional[str] = None,
        result: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        DTO is {command_id, status, error?, result?}. status must be
        one of `succeeded` / `failed` / `in_progress`.
        """
        body: Dict[str, Any] = {"command_id": command_id, "status": status}
        if error is not None:
            body["error"] = error
        if result is not None:
            body["result"] = result
        try:
            r = self._session.post(
                f"{self.base_url}/api/gpu/sync/agent/command-result",
                headers=self._headers(),
                json=body,
                timeout=10,
                **self._request_kwargs(),
            )
            r.raise_for_status()
            return True
        except requests.RequestException as e:
            logger.warning("POST /api/gpu/sync/agent/command-result failed: %s", e)
            return False

    # ----- POST /api/gpu/sync/spot-interruption -----------------------------

    def report_spot_interruption(
        self,
        *,
        external_instance_id: str,
        cloud_provider: str,
        interruption_type: str,
        checkpoint_status: Optional[str] = None,
        workload_metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Dedicated spot-event endpoint — the legacy `alerts` envelope
        with `alert_type=spot_interruption` is no longer accepted.
        Per SpotInterruptionDto, cloud_provider must be
        aws / azure / gcp.
        """
        body: Dict[str, Any] = {
            "external_instance_id": external_instance_id,
            "cloud_provider": cloud_provider,
            "interruption_type": interruption_type,
        }
        if checkpoint_status:
            body["checkpoint_status"] = checkpoint_status
        if workload_metadata:
            body["workload_metadata"] = workload_metadata
        try:
            r = self._session.post(
                f"{self.base_url}/api/gpu/sync/spot-interruption",
                headers=self._headers(),
                json=body,
                timeout=10,
                **self._request_kwargs(),
            )
            r.raise_for_status()
            return True
        except requests.RequestException as e:
            logger.warning("POST /api/gpu/sync/spot-interruption failed: %s", e)
            return False
