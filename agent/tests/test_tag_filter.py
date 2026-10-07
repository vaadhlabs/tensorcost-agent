"""
Tests for the tag-egress filter introduced for §8.2 Blocker #5.

The filter is the agent's only choke point between legacy collector
tag dicts (which routinely contain owner emails, project codenames,
and occasionally secrets) and the wire — both the HTTP transport
(`rename_instance_to_wire`) and the gRPC path (`main._send_instances`)
both call `filter_tags`.

Posture under test:
  * Default-deny allowlist keeps the obvious cost-attribution keys.
  * AGENT_TAG_ALLOWLIST replaces the default at call time.
  * AGENT_TAG_HASH_UNKNOWN=true keeps disallowed keys but hashes values.
  * Values longer than 256 chars are truncated with "..." suffix.
  * None / empty / non-dict input returns {}.
"""

from __future__ import annotations

import hashlib
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from transport.http_client import (  # noqa: E402
    _DEFAULT_TAG_ALLOWLIST,
    filter_tags,
    rename_instance_to_wire,
)


_UUID = "11111111-1111-1111-1111-111111111111"


class TestDefaultAllowlist:
    def test_passes_through_costcenter_environment_team(self, monkeypatch):
        monkeypatch.delenv("AGENT_TAG_ALLOWLIST", raising=False)
        monkeypatch.delenv("AGENT_TAG_HASH_UNKNOWN", raising=False)
        out = filter_tags({
            "cost-center": "ml-research",
            "environment": "prod",
            "team": "platform",
        })
        assert out == {
            "cost-center": "ml-research",
            "environment": "prod",
            "team": "platform",
        }

    def test_passes_through_underscore_costcenter_variant(self, monkeypatch):
        monkeypatch.delenv("AGENT_TAG_ALLOWLIST", raising=False)
        out = filter_tags({"cost_center": "abc"})
        assert out == {"cost_center": "abc"}

    def test_drops_unknown_keys_by_default(self, monkeypatch):
        monkeypatch.delenv("AGENT_TAG_ALLOWLIST", raising=False)
        monkeypatch.delenv("AGENT_TAG_HASH_UNKNOWN", raising=False)
        out = filter_tags({
            "team": "ml",
            "email_owner": "alice@example.com",
            "internal_codename": "project-thunder",
            "secret_token": "ghp_xxxxxxxxxxxx",
        })
        assert out == {"team": "ml"}

    def test_default_includes_audit_named_keys(self):
        # The audit explicitly names cost-center / environment / team as
        # the recommended allowlist. Pin that into the default.
        for k in ("cost-center", "environment", "team"):
            assert k in _DEFAULT_TAG_ALLOWLIST


class TestEnvOverride:
    def test_env_override_replaces_default(self, monkeypatch):
        monkeypatch.setenv("AGENT_TAG_ALLOWLIST", "foo,bar")
        monkeypatch.delenv("AGENT_TAG_HASH_UNKNOWN", raising=False)
        out = filter_tags({"foo": "1", "bar": "2", "team": "ml"})
        # team is in the *default* allowlist but not the override; must
        # be dropped because the override REPLACES the default.
        assert out == {"foo": "1", "bar": "2"}

    def test_env_override_tolerates_whitespace(self, monkeypatch):
        monkeypatch.setenv("AGENT_TAG_ALLOWLIST", "  foo , bar  ")
        out = filter_tags({"foo": "1", "bar": "2"})
        assert out == {"foo": "1", "bar": "2"}

    def test_empty_env_override_drops_everything(self, monkeypatch):
        monkeypatch.setenv("AGENT_TAG_ALLOWLIST", "")
        out = filter_tags({"team": "ml", "cost-center": "abc"})
        assert out == {}


class TestHashUnknownOptIn:
    def test_hash_unknown_true_retains_key_with_hex_hash(self, monkeypatch):
        monkeypatch.delenv("AGENT_TAG_ALLOWLIST", raising=False)
        monkeypatch.setenv("AGENT_TAG_HASH_UNKNOWN", "true")
        out = filter_tags({
            "team": "ml",  # allowlisted — value passes through
            "email_owner": "alice@example.com",  # not allowed — hashed
        })
        expected_hash = hashlib.sha256(b"alice@example.com").hexdigest()[:16]
        assert out["team"] == "ml"
        assert out["email_owner"] == expected_hash
        assert len(out["email_owner"]) == 16

    def test_hash_unknown_case_insensitive(self, monkeypatch):
        monkeypatch.setenv("AGENT_TAG_HASH_UNKNOWN", "TRUE")
        out = filter_tags({"unknown": "v"})
        assert out["unknown"] == hashlib.sha256(b"v").hexdigest()[:16]

    def test_hash_unknown_false_drops(self, monkeypatch):
        monkeypatch.setenv("AGENT_TAG_HASH_UNKNOWN", "false")
        out = filter_tags({"unknown": "v"})
        assert out == {}


class TestLengthCap:
    def test_long_allowlisted_value_is_truncated(self, monkeypatch):
        monkeypatch.delenv("AGENT_TAG_ALLOWLIST", raising=False)
        long_val = "x" * 1000
        out = filter_tags({"team": long_val})
        assert len(out["team"]) == 256
        assert out["team"].endswith("...")
        assert out["team"][:253] == "x" * 253

    def test_value_at_cap_is_passed_through(self, monkeypatch):
        monkeypatch.delenv("AGENT_TAG_ALLOWLIST", raising=False)
        val = "y" * 256
        out = filter_tags({"team": val})
        assert out["team"] == val

    def test_value_just_over_cap_is_truncated(self, monkeypatch):
        monkeypatch.delenv("AGENT_TAG_ALLOWLIST", raising=False)
        val = "z" * 257
        out = filter_tags({"team": val})
        assert len(out["team"]) == 256
        assert out["team"].endswith("...")


class TestEdgeInputs:
    def test_none_returns_empty_dict(self):
        assert filter_tags(None) == {}

    def test_empty_dict_returns_empty_dict(self):
        assert filter_tags({}) == {}

    def test_non_dict_returns_empty_dict(self):
        assert filter_tags("not-a-dict") == {}  # type: ignore[arg-type]
        assert filter_tags(["k", "v"]) == {}  # type: ignore[arg-type]

    def test_none_value_is_stringified_to_empty(self, monkeypatch):
        monkeypatch.delenv("AGENT_TAG_ALLOWLIST", raising=False)
        out = filter_tags({"team": None})
        assert out == {"team": ""}

    def test_int_value_is_stringified(self, monkeypatch):
        monkeypatch.delenv("AGENT_TAG_ALLOWLIST", raising=False)
        out = filter_tags({"team": 42})
        assert out == {"team": "42"}


class TestRenameInstanceWiring:
    """The HTTP transport's instance-rename path must use the filter."""

    def test_rename_instance_drops_unknown_tag(self, monkeypatch):
        monkeypatch.delenv("AGENT_TAG_ALLOWLIST", raising=False)
        monkeypatch.delenv("AGENT_TAG_HASH_UNKNOWN", raising=False)
        wire = rename_instance_to_wire({
            "instance_id": "i-abc",
            "cloud_provider": "aws",
            "state": "running",
            "tags": {
                "team": "ml",
                "email_owner": "alice@example.com",
            },
        })
        assert wire is not None
        assert wire["tags"] == {"team": "ml"}

    def test_rename_instance_omits_tags_when_all_dropped(self, monkeypatch):
        monkeypatch.setenv("AGENT_TAG_ALLOWLIST", "")
        wire = rename_instance_to_wire({
            "instance_id": "i-abc",
            "cloud_provider": "aws",
            "state": "running",
            "tags": {"team": "ml"},
        })
        assert wire is not None
        assert "tags" not in wire
