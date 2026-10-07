"""Tests for agent-run attribution — agent_id + workflow_id."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from tensorcost import wrap
from tensorcost._applied_mode import build_proxy_headers
from tensorcost._config import MissingConfigError, resolve_config
from tensorcost._providers import _openai


class TestAgentRunConfig:
    def test_wrap_time_agent_and_workflow(self, monkeypatch):
        monkeypatch.setenv("TENSORCOST_API_KEY", "k")
        cfg = resolve_config(agent_id="support-bot", workflow_id="run-1")
        assert cfg.agent_id == "support-bot"
        assert cfg.workflow_id == "run-1"

    def test_env_vars_resolve(self, monkeypatch):
        monkeypatch.setenv("TENSORCOST_API_KEY", "k")
        monkeypatch.setenv("TENSORCOST_AGENT_ID", "env-agent")
        monkeypatch.setenv("TENSORCOST_WORKFLOW_ID", "env-workflow")
        cfg = resolve_config()
        assert cfg.agent_id == "env-agent"
        assert cfg.workflow_id == "env-workflow"

    def test_max_length_128(self, monkeypatch):
        monkeypatch.setenv("TENSORCOST_API_KEY", "k")
        with pytest.raises(MissingConfigError, match="agent_id"):
            resolve_config(agent_id="x" * 129)


class TestAgentRunObservation:
    def test_openai_observation_carries_agent_and_workflow(
        self, openai_client, recording_transport
    ):
        _openai.install(
            openai_client,
            recording_transport,
            tenant_id=None,
            environment=None,
            connection_id=None,
            agent_id="code-review",
            workflow_id="run-abc",
        )
        openai_client.chat.completions.create(
            model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}]
        )
        obs = recording_transport.observations[0]
        assert obs["agent_id"] == "code-review"
        assert obs["workflow_id"] == "run-abc"

    def test_with_meta_overrides_workflow(self, openai_client, recording_transport):
        _openai.install(
            openai_client,
            recording_transport,
            tenant_id=None,
            environment=None,
            connection_id=None,
            agent_id="planner",
        )
        from tensorcost._attribution import attach_with_meta

        attach_with_meta(openai_client)
        scoped = openai_client.with_meta(workflow_id="task-99")
        scoped.chat.completions.create(
            model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}]
        )
        obs = recording_transport.observations[0]
        assert obs["agent_id"] == "planner"
        assert obs["workflow_id"] == "task-99"

    def test_with_meta_leaves_unwrapped_methods_callable(
        self, openai_client, recording_transport
    ):
        _openai.install(
            openai_client,
            recording_transport,
            tenant_id=None,
            environment=None,
            connection_id=None,
        )
        from tensorcost._attribution import attach_with_meta

        openai_client.models = MagicMock()
        openai_client.models.list = MagicMock(return_value=["model-a"])
        attach_with_meta(openai_client)
        scoped = openai_client.with_meta(workflow_id="task-1")
        assert scoped.models.list() == ["model-a"]
        openai_client.models.list.assert_called_once()


class TestAnthropicWithMeta:
    def test_with_meta_overrides_workflow(self, anthropic_client, recording_transport):
        from tensorcost._providers import _anthropic
        from tensorcost._attribution import attach_with_meta

        _anthropic.install(
            anthropic_client,
            recording_transport,
            tenant_id=None,
            environment=None,
            connection_id=None,
            agent_id="planner",
        )
        attach_with_meta(anthropic_client)
        scoped = anthropic_client.with_meta(workflow_id="task-anth-99")
        scoped.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=10,
            messages=[{"role": "user", "content": "hi"}],
        )
        obs = recording_transport.observations[0]
        assert obs["agent_id"] == "planner"
        assert obs["workflow_id"] == "task-anth-99"


class TestAgentRunProxyHeaders:
    def test_build_proxy_headers_includes_agent_fields(self):
        headers = build_proxy_headers(
            provider_url="https://api.openai.com",
            provider_auth="Bearer sk-abc",
            agent_id="a-1",
            workflow_id="w-1",
        )
        assert headers["x-tc-agent-id"] == "a-1"
        assert headers["x-tc-workflow-id"] == "w-1"
