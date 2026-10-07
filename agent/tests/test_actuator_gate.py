"""
Tests for the fail-closed actuator gate added to main.py:_execute_command_inner.

The agent has a code path that can stop / start / resize / restart / terminate
customer cloud instances. Per the 2026-05-07 audit §8.2 Blocker #7, this path
is now gated behind two env vars (`AGENT_ACTUATOR_ENABLED` and
`AGENT_ACTUATOR_ALLOWED_ACTIONS`). These tests pin the contract.
"""

import os
import sys
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import main  # noqa: E402


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    monkeypatch.delenv('AGENT_ACTUATOR_ENABLED', raising=False)
    monkeypatch.delenv('AGENT_ACTUATOR_ALLOWED_ACTIONS', raising=False)
    yield


class TestActuatorEnabledHelper:
    def test_default_is_false(self):
        assert main._actuator_enabled() is False

    def test_true_lowercase(self, monkeypatch):
        monkeypatch.setenv('AGENT_ACTUATOR_ENABLED', 'true')
        assert main._actuator_enabled() is True

    def test_true_uppercase(self, monkeypatch):
        monkeypatch.setenv('AGENT_ACTUATOR_ENABLED', 'TRUE')
        assert main._actuator_enabled() is True

    def test_with_whitespace(self, monkeypatch):
        monkeypatch.setenv('AGENT_ACTUATOR_ENABLED', '  true  ')
        assert main._actuator_enabled() is True

    @pytest.mark.parametrize('val', ['1', 'yes', 'on', 'enabled', '', 'false'])
    def test_anything_else_is_false(self, monkeypatch, val):
        """Only the exact string 'true' enables the gate.
        '1' / 'yes' style toggles deliberately do NOT — operators
        copy-pasting from a CI template should not accidentally
        flip this on."""
        monkeypatch.setenv('AGENT_ACTUATOR_ENABLED', val)
        assert main._actuator_enabled() is False


class TestActuatorAllowedActionsHelper:
    def test_default_is_empty(self):
        assert main._actuator_allowed_actions() == frozenset()

    def test_csv_parses(self, monkeypatch):
        monkeypatch.setenv('AGENT_ACTUATOR_ALLOWED_ACTIONS', 'stop,start')
        assert main._actuator_allowed_actions() == {'stop', 'start'}

    def test_whitespace_tolerated(self, monkeypatch):
        monkeypatch.setenv('AGENT_ACTUATOR_ALLOWED_ACTIONS', ' stop , start ')
        assert main._actuator_allowed_actions() == {'stop', 'start'}

    def test_unknown_actions_silently_dropped(self, monkeypatch):
        """Typos in the env var should not extend the allowlist beyond
        the five real actions. Better to silently drop than to error
        on agent boot — the gate is fail-closed anyway."""
        monkeypatch.setenv('AGENT_ACTUATOR_ALLOWED_ACTIONS', 'stop,delete,terminate')
        assert main._actuator_allowed_actions() == {'stop', 'terminate'}

    def test_case_insensitive(self, monkeypatch):
        monkeypatch.setenv('AGENT_ACTUATOR_ALLOWED_ACTIONS', 'STOP,Start')
        assert main._actuator_allowed_actions() == {'stop', 'start'}


def _agent_with_monitor():
    """Build a partial UnifiedGPUAgent stand-in with a mock monitor.
    We avoid the full __init__ to skip config loading, IMDS probes,
    etc. — only the dispatch method is under test here."""
    agent = main.UnifiedGPUAgent.__new__(main.UnifiedGPUAgent)
    monitor = MagicMock()
    monitor.stop_instance.return_value = {'state': 'stopping'}
    monitor.terminate_instance.return_value = {'state': 'terminating'}
    agent.monitors = {'aws': monitor}
    return agent, monitor


class TestActuatorDispatch:
    def test_rejected_when_disabled(self):
        agent, monitor = _agent_with_monitor()
        result = agent._execute_command_inner({
            'command_id': 'c1', 'type': 'stop',
            'instance_id': 'i-abc', 'cloud_provider': 'aws',
        })
        assert result['status'] == 'rejected'
        assert 'AGENT_ACTUATOR_ENABLED' in result['error']
        monitor.stop_instance.assert_not_called()

    def test_rejected_when_action_not_in_allowlist(self, monkeypatch):
        monkeypatch.setenv('AGENT_ACTUATOR_ENABLED', 'true')
        monkeypatch.setenv('AGENT_ACTUATOR_ALLOWED_ACTIONS', 'stop')

        agent, monitor = _agent_with_monitor()
        result = agent._execute_command_inner({
            'command_id': 'c1', 'type': 'terminate',
            'instance_id': 'i-abc', 'cloud_provider': 'aws',
        })
        assert result['status'] == 'rejected'
        assert "'terminate'" in result['error']
        monitor.terminate_instance.assert_not_called()

    def test_dispatched_when_enabled_and_allowed(self, monkeypatch):
        monkeypatch.setenv('AGENT_ACTUATOR_ENABLED', 'true')
        monkeypatch.setenv('AGENT_ACTUATOR_ALLOWED_ACTIONS', 'stop')

        agent, monitor = _agent_with_monitor()
        result = agent._execute_command_inner({
            'command_id': 'c1', 'type': 'stop',
            'instance_id': 'i-abc', 'cloud_provider': 'aws',
        })
        assert result['status'] == 'succeeded'
        monitor.stop_instance.assert_called_once_with('i-abc')

    def test_terminate_requires_explicit_terminate_in_allowlist(self, monkeypatch):
        """Even with start+stop+resize+restart enabled, terminate stays
        gated. Catching the operator who ticks 'enable everything except
        terminate' rather than enumerating positively."""
        monkeypatch.setenv('AGENT_ACTUATOR_ENABLED', 'true')
        monkeypatch.setenv(
            'AGENT_ACTUATOR_ALLOWED_ACTIONS', 'stop,start,resize,restart',
        )

        agent, monitor = _agent_with_monitor()
        result = agent._execute_command_inner({
            'command_id': 'c1', 'type': 'terminate',
            'instance_id': 'i-abc', 'cloud_provider': 'aws',
        })
        assert result['status'] == 'rejected'
        monitor.terminate_instance.assert_not_called()
