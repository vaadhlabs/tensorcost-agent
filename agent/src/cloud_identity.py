"""
Cloud-identity auto-detection (26-FCR §3.5 / §6.3).

When the agent boots without `AGENT_CLOUD_PROVIDER` / `AGENT_REGION`, fall
back to a best-effort discovery against the local instance metadata
service. Without this, an operator who forgets to wire the env var at
provisioning time produces metric streams tagged `unknown` for both
provider and instance id — which downstream causes the per-instance UNIQUE
constraint on `(tenant_id, cloud_provider, instance_id)` to collide across
genuinely-different VMs and silently drop ingest.

Each probe has a 1-second timeout so a non-cloud host (laptop / on-prem)
adds at most ~3 seconds of startup overhead before falling through to
`onprem`. Probes hit the link-local 169.254.169.254 metadata endpoint;
those packets never leave the host.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# Link-local metadata endpoints. Identical IP across AWS / Azure / GCP —
# the URL paths and required headers differ.
_AWS_IMDS = "http://169.254.169.254/latest"
_AZURE_IMDS = "http://169.254.169.254/metadata/instance"
_GCP_IMDS = "http://169.254.169.254/computeMetadata/v1"

# Each metadata probe gets a tight timeout. Off-cloud hosts return ECONNREFUSED
# or hit the connect-timeout fast; the value is small enough not to noticeably
# delay startup but large enough to tolerate a slow IMDSv2 token round-trip
# under high CPU contention.
_PROBE_TIMEOUT_SECONDS = 1.0


@dataclass(frozen=True)
class CloudIdentity:
    """Discovered cloud-identity tuple. `None` fields stay `None` if
    the probe couldn't determine that piece (e.g. region unset on a
    badly-configured EC2 instance)."""
    cloud_provider: str
    cloud_id: Optional[str]
    region: Optional[str]


def detect_cloud_identity() -> Optional[CloudIdentity]:
    """Try AWS, then Azure, then GCP. Return the first that succeeds.

    Returns None when no probe succeeds — caller can default to
    `onprem` or whatever fallback its UX requires.
    """
    for probe, label in (
        (_detect_aws, "AWS"),
        (_detect_azure, "Azure"),
        (_detect_gcp, "GCP"),
    ):
        try:
            ident = probe()
        except Exception as exc:  # noqa: BLE001 — IMDS probes are best-effort
            logger.debug("cloud-identity probe %s raised: %s", label, exc)
            continue
        if ident is not None:
            logger.info(
                "cloud-identity detected: provider=%s id=%s region=%s",
                ident.cloud_provider,
                ident.cloud_id,
                ident.region,
            )
            return ident
    logger.info("cloud-identity: no IMDS responded; defaulting to operator-provided env")
    return None


def _detect_aws() -> Optional[CloudIdentity]:
    """AWS IMDSv2: PUT a token request, then GET metadata with the token.

    IMDSv1 (no token) was deprecated in favour of v2 to defend against
    SSRF; we use v2 unconditionally. v2-disabled instances reject the PUT
    with 405 — we treat that as not-AWS and fall through.
    """
    token_resp = requests.put(
        f"{_AWS_IMDS}/api/token",
        headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"},
        timeout=_PROBE_TIMEOUT_SECONDS,
    )
    if token_resp.status_code != 200:
        return None
    token = token_resp.text.strip()
    headers = {"X-aws-ec2-metadata-token": token}

    instance_id_resp = requests.get(
        f"{_AWS_IMDS}/meta-data/instance-id",
        headers=headers,
        timeout=_PROBE_TIMEOUT_SECONDS,
    )
    if instance_id_resp.status_code != 200:
        return None
    instance_id = instance_id_resp.text.strip()
    if not instance_id:
        return None

    # Region: best-effort. Older AMIs expose `placement/availability-zone`
    # which we trim to a region; newer ones have `placement/region` directly.
    region: Optional[str] = None
    region_resp = requests.get(
        f"{_AWS_IMDS}/meta-data/placement/region",
        headers=headers,
        timeout=_PROBE_TIMEOUT_SECONDS,
    )
    if region_resp.status_code == 200:
        region = region_resp.text.strip()
    else:
        az_resp = requests.get(
            f"{_AWS_IMDS}/meta-data/placement/availability-zone",
            headers=headers,
            timeout=_PROBE_TIMEOUT_SECONDS,
        )
        if az_resp.status_code == 200:
            az = az_resp.text.strip()
            # Trim trailing AZ letter: us-east-1a → us-east-1
            region = az[:-1] if az and az[-1].isalpha() else az

    return CloudIdentity(cloud_provider="aws", cloud_id=instance_id, region=region)


def _detect_azure() -> Optional[CloudIdentity]:
    """Azure IMDS: GET with the `Metadata: true` header.

    Azure returns the full JSON document; we read just `vmId` and
    `location`. `vmId` is the canonical immutable identifier; `name` is
    operator-changeable and unsuitable for the dedup key.
    """
    resp = requests.get(
        _AZURE_IMDS,
        params={"api-version": "2021-02-01"},
        headers={"Metadata": "true"},
        timeout=_PROBE_TIMEOUT_SECONDS,
    )
    if resp.status_code != 200:
        return None
    body = resp.json()
    compute = body.get("compute", {}) if isinstance(body, dict) else {}
    vm_id = compute.get("vmId")
    if not vm_id:
        return None
    region = compute.get("location") or None
    return CloudIdentity(cloud_provider="azure", cloud_id=str(vm_id), region=region)


def _detect_gcp() -> Optional[CloudIdentity]:
    """GCP metadata: GET with `Metadata-Flavor: Google` header.

    GCE returns `instance/id` as a numeric string; we use it as `cloud_id`.
    `zone` returns a path like `projects/<num>/zones/us-central1-a`; we
    strip to the tail and remove the trailing AZ letter to get the region.
    """
    headers = {"Metadata-Flavor": "Google"}
    id_resp = requests.get(
        f"{_GCP_IMDS}/instance/id",
        headers=headers,
        timeout=_PROBE_TIMEOUT_SECONDS,
    )
    if id_resp.status_code != 200:
        return None
    instance_id = id_resp.text.strip()
    if not instance_id:
        return None

    region: Optional[str] = None
    zone_resp = requests.get(
        f"{_GCP_IMDS}/instance/zone",
        headers=headers,
        timeout=_PROBE_TIMEOUT_SECONDS,
    )
    if zone_resp.status_code == 200:
        zone_path = zone_resp.text.strip()
        # `projects/123/zones/us-central1-a` → `us-central1-a` → `us-central1`
        zone = zone_path.rsplit("/", 1)[-1] if "/" in zone_path else zone_path
        region = zone[:-2] if len(zone) > 2 and zone[-2] == "-" and zone[-1].isalpha() else zone

    return CloudIdentity(cloud_provider="gcp", cloud_id=instance_id, region=region)
