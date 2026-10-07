"""
Tests for `src/cloud_identity.py` (26-FCR §3.5 / §6.3).

The module probes link-local IMDS endpoints; we patch `requests.put` and
`requests.get` so the tests don't actually hit 169.254.169.254.
"""

from __future__ import annotations

import sys
import os
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cloud_identity import (  # noqa: E402
    CloudIdentity,
    detect_cloud_identity,
)


def _resp(
    status: int = 200,
    *,
    text: str = "",
    json_body: Optional[Dict[str, Any]] = None,
) -> MagicMock:
    mock = MagicMock()
    mock.status_code = status
    mock.text = text
    if json_body is not None:
        mock.json = MagicMock(return_value=json_body)
    return mock


# --- AWS path ---------------------------------------------------------- #


@patch("src.cloud_identity.requests.get")
@patch("src.cloud_identity.requests.put")
def test_aws_full_path(put_mock: MagicMock, get_mock: MagicMock) -> None:
    put_mock.return_value = _resp(200, text="aws-token-abc")

    def aws_get(url: str, **_kw: Any) -> MagicMock:
        if url.endswith("/meta-data/instance-id"):
            return _resp(200, text="i-0123456789abcdef0")
        if url.endswith("/meta-data/placement/region"):
            return _resp(200, text="us-east-1")
        return _resp(404)

    get_mock.side_effect = aws_get
    out = detect_cloud_identity()
    assert out == CloudIdentity(
        cloud_provider="aws",
        cloud_id="i-0123456789abcdef0",
        region="us-east-1",
    )


@patch("src.cloud_identity.requests.get")
@patch("src.cloud_identity.requests.put")
def test_aws_az_fallback_when_region_endpoint_404s(
    put_mock: MagicMock, get_mock: MagicMock,
) -> None:
    """Older AMIs only expose availability-zone — must trim to region."""
    put_mock.return_value = _resp(200, text="aws-token-abc")

    def aws_get(url: str, **_kw: Any) -> MagicMock:
        if url.endswith("/meta-data/instance-id"):
            return _resp(200, text="i-old")
        if url.endswith("/meta-data/placement/region"):
            return _resp(404)
        if url.endswith("/meta-data/placement/availability-zone"):
            return _resp(200, text="us-east-1c")
        return _resp(404)

    get_mock.side_effect = aws_get
    out = detect_cloud_identity()
    assert out is not None
    assert out.cloud_provider == "aws"
    assert out.cloud_id == "i-old"
    assert out.region == "us-east-1"  # trailing 'c' trimmed


@patch("src.cloud_identity.requests.get")
@patch("src.cloud_identity.requests.put")
def test_aws_imdsv2_disabled_falls_through(
    put_mock: MagicMock, get_mock: MagicMock,
) -> None:
    """IMDSv2-disabled instances reject the token PUT with 405."""
    put_mock.return_value = _resp(405)

    def azure_get(url: str, **_kw: Any) -> MagicMock:
        if "metadata/instance" in url:
            return _resp(404)  # also not Azure
        return _resp(404)

    get_mock.side_effect = azure_get
    out = detect_cloud_identity()
    assert out is None


# --- Azure path -------------------------------------------------------- #


@patch("src.cloud_identity.requests.get")
@patch("src.cloud_identity.requests.put")
def test_azure_path(put_mock: MagicMock, get_mock: MagicMock) -> None:
    """AWS PUT returns 405; Azure GET returns the metadata document."""
    put_mock.return_value = _resp(405)

    def by_url(url: str, **_kw: Any) -> MagicMock:
        if "metadata/instance" in url:
            return _resp(
                200,
                json_body={
                    "compute": {
                        "vmId": "11111111-2222-3333-4444-555555555555",
                        "location": "eastus",
                    },
                },
            )
        return _resp(404)

    get_mock.side_effect = by_url
    out = detect_cloud_identity()
    assert out == CloudIdentity(
        cloud_provider="azure",
        cloud_id="11111111-2222-3333-4444-555555555555",
        region="eastus",
    )


@patch("src.cloud_identity.requests.get")
@patch("src.cloud_identity.requests.put")
def test_azure_missing_vm_id_falls_through(
    put_mock: MagicMock, get_mock: MagicMock,
) -> None:
    put_mock.return_value = _resp(405)
    get_mock.return_value = _resp(200, json_body={"compute": {"location": "eastus"}})
    # AWS fails → Azure has no vmId → GCP returns 404 below; must be None.
    out = detect_cloud_identity()
    assert out is None


# --- GCP path ---------------------------------------------------------- #


@patch("src.cloud_identity.requests.get")
@patch("src.cloud_identity.requests.put")
def test_gcp_path(put_mock: MagicMock, get_mock: MagicMock) -> None:
    put_mock.return_value = _resp(405)

    def by_url(url: str, **_kw: Any) -> MagicMock:
        if "metadata/instance" in url:
            return _resp(404)  # not Azure
        if url.endswith("/computeMetadata/v1/instance/id"):
            return _resp(200, text="6543210987654321")
        if url.endswith("/computeMetadata/v1/instance/zone"):
            return _resp(200, text="projects/123456789/zones/us-central1-a")
        return _resp(404)

    get_mock.side_effect = by_url
    out = detect_cloud_identity()
    assert out == CloudIdentity(
        cloud_provider="gcp",
        cloud_id="6543210987654321",
        region="us-central1",
    )


# --- Off-cloud / network failure paths --------------------------------- #


@patch("src.cloud_identity.requests.get")
@patch("src.cloud_identity.requests.put")
def test_no_imds_returns_none(put_mock: MagicMock, get_mock: MagicMock) -> None:
    """Laptop / on-prem: every call raises a connection error."""
    import requests as real_requests

    put_mock.side_effect = real_requests.exceptions.ConnectionError("refused")
    get_mock.side_effect = real_requests.exceptions.ConnectionError("refused")
    assert detect_cloud_identity() is None


@patch("src.cloud_identity.requests.get")
@patch("src.cloud_identity.requests.put")
def test_partial_aws_response_falls_through(
    put_mock: MagicMock, get_mock: MagicMock,
) -> None:
    """AWS token works but instance-id 404s → we don't claim AWS."""
    put_mock.return_value = _resp(200, text="tkn")

    def by_url(url: str, **_kw: Any) -> MagicMock:
        if url.endswith("/meta-data/instance-id"):
            return _resp(404)
        return _resp(404)

    get_mock.side_effect = by_url
    assert detect_cloud_identity() is None


@patch("src.cloud_identity.requests.get")
@patch("src.cloud_identity.requests.put")
def test_aws_probe_raises_then_azure_succeeds(
    put_mock: MagicMock, get_mock: MagicMock,
) -> None:
    """A raised exception on one probe must NOT short-circuit later probes."""
    import requests as real_requests

    put_mock.side_effect = real_requests.exceptions.Timeout("aws timeout")

    def by_url(url: str, **_kw: Any) -> MagicMock:
        if "metadata/instance" in url:
            return _resp(
                200,
                json_body={"compute": {"vmId": "abcd", "location": "westus2"}},
            )
        return _resp(404)

    get_mock.side_effect = by_url
    out = detect_cloud_identity()
    assert out is not None
    assert out.cloud_provider == "azure"
