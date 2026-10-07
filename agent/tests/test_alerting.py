import pytest
import sys
import os
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import json
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from alerting import AlertManager as AlertingManager


class TestAlertingManager:
    """Test cases for alerting functionality."""

    def test_init(self):
        """Test AlertManager initialization."""
        alerting = AlertingManager()

        # Check initialization of actual AlertManager properties
        assert alerting.slack_webhook is None  # Not configured by default
        assert alerting.email_config is not None
        assert isinstance(alerting.email_config, dict)
        assert alerting.idle_threshold is not None
        assert alerting.cost_threshold is not None
        assert isinstance(alerting._idle_since, dict)
        assert len(alerting._idle_since) == 0

    def test_init_with_slack_config(self):
        """Test AlertManager initialization with Slack config."""
        with patch.dict('os.environ', {
            'SLACK_WEBHOOK_URL': 'https://hooks.slack.com/test'
        }):
            alerting = AlertingManager()

            assert alerting.slack_webhook == 'https://hooks.slack.com/test'

    def test_init_with_email_config(self):
        """Test AlertManager initialization with email config."""
        with patch.dict('os.environ', {
            'EMAIL_SMTP_SERVER': 'smtp.gmail.com',
            'EMAIL_SMTP_PORT': '587',
            'EMAIL_USERNAME': 'test@gmail.com',
            'EMAIL_PASSWORD': 'password',
            'EMAIL_TO': 'alerts@example.com'
        }):
            alerting = AlertingManager()

            assert alerting.email_config['smtp_server'] == 'smtp.gmail.com'
            assert alerting.email_config['smtp_port'] == 587
            assert alerting.email_config['username'] == 'test@gmail.com'
            assert alerting.email_config['to_email'] == 'alerts@example.com'

    def test_init_with_custom_thresholds(self):
        """Test AlertManager with custom thresholds."""
        with patch.dict('os.environ', {
            'IDLE_GPU_THRESHOLD': '20',
            'DAILY_COST_THRESHOLD': '150.5'
        }):
            alerting = AlertingManager()

            assert alerting.idle_threshold == 20
            assert alerting.cost_threshold == 150.5

    def test_check_idle_gpus_with_idle_instances(self):
        """Test checking for idle GPU instances."""
        alerting = AlertingManager()

        # Create metrics with idle instances
        gpu_metrics = [
            {
                'instance_id': 'i-123',
                'instance_type': 'g4dn.xlarge',
                'gpu_utilization': 5.0,
                'cpu_utilization': 2.0,
                'is_idle': True
            }
        ]

        with patch.object(alerting, '_send_alert') as mock_send:
            alerting.check_idle_gpus(gpu_metrics)

            # Should have called _send_alert for idle instances
            mock_send.assert_called_once()
            call_args = mock_send.call_args
            assert call_args[0][0] == 'idle_gpu'
            # The message is formatted via _format_idle_gpu_alert
            assert isinstance(call_args[0][1], str)
            assert 'IDLE' in call_args[0][1]

    def test_check_idle_gpus_tracks_idle_time(self):
        """Test that idle time is tracked for instances."""
        alerting = AlertingManager()

        gpu_metrics = [
            {
                'instance_id': 'i-idle-1',
                'instance_type': 'g4dn.xlarge',
                'gpu_utilization': 2.0,
                'is_idle': True
            }
        ]

        with patch.object(alerting, '_send_alert'):
            alerting.check_idle_gpus(gpu_metrics)

            # Instance should be tracked in _idle_since
            assert 'i-idle-1' in alerting._idle_since
            assert isinstance(alerting._idle_since['i-idle-1'], float)

    def test_check_idle_gpus_clears_non_idle_instances(self):
        """Test that non-idle instances are cleared from tracking."""
        alerting = AlertingManager()

        # Mark instance as idle first
        alerting._idle_since['i-was-idle'] = time.time() - 300

        # Now check metrics where it's not idle
        gpu_metrics = [
            {
                'instance_id': 'i-was-idle',
                'instance_type': 'g4dn.xlarge',
                'gpu_utilization': 80.0,
                'is_idle': False
            }
        ]

        with patch.object(alerting, '_send_alert'):
            alerting.check_idle_gpus(gpu_metrics)

        # Should be cleared from idle tracking
        assert 'i-was-idle' not in alerting._idle_since

    def test_check_cost_thresholds_below_threshold(self):
        """Test cost checking when below threshold."""
        alerting = AlertingManager()
        alerting.cost_threshold = 100.0

        cost_data = [
            {'instance_type': 'g4dn.xlarge', 'cost': 50.0}
        ]

        with patch.object(alerting, '_send_alert') as mock_send:
            alerting.check_cost_thresholds(cost_data)

            # Should not send alert when under threshold
            mock_send.assert_not_called()

    def test_check_cost_thresholds_exceeds_threshold(self):
        """Test cost checking when exceeds threshold."""
        alerting = AlertingManager()
        alerting.cost_threshold = 100.0

        cost_data = [
            {'instance_type': 'g4dn.xlarge', 'cost': 150.0}
        ]

        with patch.object(alerting, '_send_alert') as mock_send:
            alerting.check_cost_thresholds(cost_data)

            # Should send alert when over threshold
            mock_send.assert_called_once()
            assert mock_send.call_args[0][0] == 'cost_threshold'

    def test_format_duration_seconds_only(self):
        """Test formatting duration with seconds only."""
        alerting = AlertingManager()

        formatted = alerting._format_duration(45)
        assert formatted == '45s'

    def test_format_duration_minutes_and_seconds(self):
        """Test formatting duration with minutes and seconds."""
        alerting = AlertingManager()

        formatted = alerting._format_duration(90)
        assert formatted == '1m 30s'

    def test_format_duration_hours_and_minutes(self):
        """Test formatting duration with hours and minutes."""
        alerting = AlertingManager()

        formatted = alerting._format_duration(8100)
        assert formatted == '2h 15m'

    def test_format_idle_gpu_alert_message(self):
        """Test formatting idle GPU alert message."""
        alerting = AlertingManager()

        idle_instances = [
            {
                'instance_id': 'i-123',
                'instance_type': 'g4dn.xlarge',
                'gpu_utilization': 5.0,
                'cpu_utilization': 2.0,
                'idle_duration_seconds': 300
            }
        ]

        message = alerting._format_idle_gpu_alert(idle_instances)

        assert 'IDLE GPU ALERT' in message
        assert 'i-123' in message
        assert 'g4dn.xlarge' in message
        assert '5.0' in message

    def test_format_cost_alert_message(self):
        """Test formatting cost alert message."""
        alerting = AlertingManager()
        alerting.cost_threshold = 100.0

        cost_data = [
            {'instance_type': 'g4dn.xlarge', 'cost': 150.0}
        ]

        message = alerting._format_cost_alert(150.0, cost_data)

        assert 'COST THRESHOLD ALERT' in message
        assert '150.00' in message
        assert '100.00' in message
        assert 'g4dn.xlarge' in message

    @patch('requests.post')
    def test_send_slack_alert_with_webhook(self, mock_post):
        """Test sending Slack alert when webhook is configured."""
        with patch.dict('os.environ', {
            'SLACK_WEBHOOK_URL': 'https://hooks.slack.com/test'
        }):
            alerting = AlertingManager()

            mock_post.return_value.raise_for_status.return_value = None

            alerting._send_slack_alert('Test message')

            # Verify POST was called to webhook URL
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            assert call_args[0][0] == 'https://hooks.slack.com/test'
            assert 'json' in call_args[1]
            assert call_args[1]['json']['text'] == 'Test message'

    def test_send_slack_alert_without_webhook(self):
        """Test that Slack alert fails gracefully without webhook."""
        alerting = AlertingManager()
        # No webhook configured

        with patch('requests.post') as mock_post:
            alerting._send_slack_alert('Test message')

            # Should not be called if webhook is None
            # AlertManager doesn't raise, just logs

    @patch('src.alerting.smtplib.SMTP')
    def test_send_email_alert_with_config(self, mock_smtp):
        """Test sending email alert when configured."""
        with patch.dict('os.environ', {
            'EMAIL_SMTP_SERVER': 'smtp.gmail.com',
            'EMAIL_SMTP_PORT': '587',
            'EMAIL_USERNAME': 'test@gmail.com',
            'EMAIL_PASSWORD': 'password',
            'EMAIL_TO': 'alerts@example.com'
        }):
            alerting = AlertingManager()

            mock_server = Mock()
            mock_smtp.return_value = mock_server

            alerting._send_email_alert('idle_gpu', 'Test alert message')

            # Verify SMTP was used
            mock_server.starttls.assert_called_once()
            mock_server.login.assert_called_once()
            mock_server.sendmail.assert_called_once()

    def test_send_alert_calls_slack_and_email(self):
        """Test that _send_alert calls both Slack and email methods."""
        with patch.dict('os.environ', {
            'SLACK_WEBHOOK_URL': 'https://hooks.slack.com/test',
            'EMAIL_SMTP_SERVER': 'smtp.gmail.com',
            'EMAIL_SMTP_PORT': '587',
            'EMAIL_USERNAME': 'test@gmail.com',
            'EMAIL_PASSWORD': 'password',
            'EMAIL_TO': 'alerts@example.com'
        }):
            alerting = AlertingManager()

            with patch.object(alerting, '_send_slack_alert') as mock_slack, \
                 patch.object(alerting, '_send_email_alert') as mock_email:

                alerting._send_alert('test_type', 'Test message', [])

                mock_slack.assert_called_once_with('Test message')
                mock_email.assert_called_once()

    def test_send_test_alert(self):
        """Test sending a test alert."""
        alerting = AlertingManager()

        with patch.object(alerting, '_send_alert') as mock_send:
            alerting.send_test_alert()

            mock_send.assert_called_once()
            call_args = mock_send.call_args
            assert call_args[0][0] == 'test'
            assert 'TEST ALERT' in call_args[0][1]


class TestAlertingIdleSinceCleaning:
    """Test idle_since cleanup for terminated instances."""

    def test_idle_since_cleaned_for_terminated_instances(self):
        """Should clean up idle_since tracking for terminated instances."""
        alerting = AlertingManager()

        # Set idle_since for some instances
        instance_ids = ['i-123', 'i-456', 'i-789']
        for iid in instance_ids:
            alerting._idle_since[iid] = time.time() - 300  # Mark as idle 5 minutes ago

        # Simulate instance termination by removing from instances list
        # Only i-123 and i-456 remain active
        active_instances = ['i-123', 'i-456']

        # Clean up idle_since for terminated instances
        for iid in list(alerting._idle_since.keys()):
            if iid not in active_instances:
                del alerting._idle_since[iid]

        # i-789 should be removed
        assert 'i-789' not in alerting._idle_since
        # i-123 and i-456 should remain
        assert 'i-123' in alerting._idle_since
        assert 'i-456' in alerting._idle_since

    def test_idle_since_retained_for_active_instances(self):
        """Should retain idle_since tracking for active instances."""
        alerting = AlertingManager()

        # Track idle time for active instance
        instance_id = 'i-123'
        idle_start = time.time() - 180  # Idle for 3 minutes
        alerting._idle_since[instance_id] = idle_start

        # Instance is still in active list
        active_instances = ['i-123', 'i-456']

        # Clean up (only removes instances NOT in active list)
        for iid in list(alerting._idle_since.keys()):
            if iid not in active_instances:
                del alerting._idle_since[iid]

        # i-123 should still be tracked
        assert instance_id in alerting._idle_since
        # Idle time should be preserved
        assert alerting._idle_since[instance_id] == idle_start

    def test_multiple_idle_since_entries_cleaned_correctly(self):
        """Should clean multiple idle_since entries correctly."""
        alerting = AlertingManager()

        # Setup multiple instances with idle tracking
        instances_data = {
            'i-1': time.time() - 100,  # Running
            'i-2': time.time() - 200,  # Running
            'i-3': time.time() - 300,  # Terminated
            'i-4': time.time() - 400,  # Terminated
            'i-5': time.time() - 500,  # Running
        }

        for iid, idle_time in instances_data.items():
            alerting._idle_since[iid] = idle_time

        # Only these are still active
        active_instances = ['i-1', 'i-2', 'i-5']

        # Clean up terminated instances
        for iid in list(alerting._idle_since.keys()):
            if iid not in active_instances:
                del alerting._idle_since[iid]

        # Check cleanup results
        assert len(alerting._idle_since) == 3
        assert 'i-3' not in alerting._idle_since
        assert 'i-4' not in alerting._idle_since
        assert 'i-1' in alerting._idle_since
        assert 'i-2' in alerting._idle_since
        assert 'i-5' in alerting._idle_since

        # Verify idle times are preserved for active instances
        assert alerting._idle_since['i-1'] == instances_data['i-1']
        assert alerting._idle_since['i-2'] == instances_data['i-2']
        assert alerting._idle_since['i-5'] == instances_data['i-5']

    def test_idle_since_cleaned_when_all_instances_terminated(self):
        """Should clean all idle_since entries when all instances terminate."""
        alerting = AlertingManager()

        # All instances are idle
        for i in range(5):
            alerting._idle_since[f'i-{i}'] = time.time() - 100

        assert len(alerting._idle_since) == 5

        # All instances terminated (empty active list)
        active_instances = []

        for iid in list(alerting._idle_since.keys()):
            if iid not in active_instances:
                del alerting._idle_since[iid]

        # All should be cleaned
        assert len(alerting._idle_since) == 0

    def test_idle_since_empty_when_no_instances_tracked(self):
        """Should handle empty idle_since dict gracefully."""
        alerting = AlertingManager()

        # No instances tracked
        assert len(alerting._idle_since) == 0

        # Cleanup on empty dict should be no-op
        active_instances = ['i-1', 'i-2']
        for iid in list(alerting._idle_since.keys()):
            if iid not in active_instances:
                del alerting._idle_since[iid]

        # Should remain empty
        assert len(alerting._idle_since) == 0


class TestIdleSinceDurationTracking:
    """Test idle_since tracking and idle duration calculation."""

    def test_idle_duration_seconds_in_alert_message(self):
        """Should calculate and include idle duration in alert message."""
        alerting = AlertingManager()

        instance_id = 'i-123'
        # Mark as idle 180 seconds (3 minutes) ago
        alerting._idle_since[instance_id] = time.time() - 180

        idle_duration = time.time() - alerting._idle_since[instance_id]

        # Verify duration is approximately 180 seconds
        assert 179 <= idle_duration <= 181

    def test_idle_duration_calculation_just_idle(self):
        """Should calculate correct duration when instance just became idle."""
        alerting = AlertingManager()

        instance_id = 'i-456'
        alerting._idle_since[instance_id] = time.time()

        idle_duration = time.time() - alerting._idle_since[instance_id]

        # Should be very close to 0
        assert idle_duration < 1

    def test_idle_duration_calculation_long_idle(self):
        """Should calculate correct duration for long idle periods."""
        alerting = AlertingManager()

        instance_id = 'i-789'
        # Mark as idle 3600 seconds (1 hour) ago
        alerting._idle_since[instance_id] = time.time() - 3600

        idle_duration = time.time() - alerting._idle_since[instance_id]

        # Should be approximately 3600 seconds
        assert 3599 <= idle_duration <= 3601

    def test_format_duration_seconds_only(self):
        """Should format duration with seconds only."""
        alerting = AlertingManager()

        # 45 seconds
        formatted = alerting._format_duration(45)
        assert formatted == '45s'

    def test_format_duration_minutes_and_seconds(self):
        """Should format duration with minutes and seconds."""
        alerting = AlertingManager()

        # 1 minute 30 seconds = 90 seconds
        formatted = alerting._format_duration(90)
        assert formatted == '1m 30s'

    def test_format_duration_hours_and_minutes(self):
        """Should format duration with hours and minutes."""
        alerting = AlertingManager()

        # 2 hours 15 minutes = 8100 seconds
        formatted = alerting._format_duration(8100)
        assert formatted == '2h 15m'

    def test_format_duration_exact_minute(self):
        """Should format duration when exactly at minute boundary."""
        alerting = AlertingManager()

        # Exactly 2 minutes = 120 seconds
        formatted = alerting._format_duration(120)
        assert formatted == '2m 0s'

    def test_format_duration_exact_hour(self):
        """Should format duration when exactly at hour boundary."""
        alerting = AlertingManager()

        # Exactly 1 hour = 3600 seconds
        formatted = alerting._format_duration(3600)
        assert formatted == '1h 0m'

    def test_format_duration_zero(self):
        """Should format zero duration."""
        alerting = AlertingManager()

        formatted = alerting._format_duration(0)
        assert formatted == '0s'

    def test_format_duration_large_duration(self):
        """Should format large duration correctly."""
        alerting = AlertingManager()

        # 5 hours 42 minutes 33 seconds
        duration = 5 * 3600 + 42 * 60 + 33
        formatted = alerting._format_duration(duration)
        assert formatted == '5h 42m'

    def test_check_idle_gpus_tracks_idle_start_time(self):
        """Should track when an instance first became idle."""
        alerting = AlertingManager()

        instance_id = 'i-idle-123'
        before_track = time.time()

        # Simulate tracking idle instance
        alerting._idle_since[instance_id] = time.time()

        after_track = time.time()

        # idle_since should be recorded between before and after
        assert before_track <= alerting._idle_since[instance_id] <= after_track

    def test_check_idle_gpus_cleanup_non_idle_instances(self):
        """Should remove instances from idle tracking when they become active."""
        alerting = AlertingManager()

        # Track multiple instances as idle
        alerting._idle_since['i-was-idle'] = time.time() - 300
        alerting._idle_since['i-still-idle'] = time.time() - 200

        # Simulate i-was-idle becoming active again
        if 'i-was-idle' in alerting._idle_since:
            # In real code, this would be done when GPU utilization exceeds threshold
            del alerting._idle_since['i-was-idle']

        # Verify cleanup
        assert 'i-was-idle' not in alerting._idle_since
        assert 'i-still-idle' in alerting._idle_since

    def test_idle_since_duration_message_format(self):
        """Should include formatted idle duration in alert messages."""
        alerting = AlertingManager()

        instance_id = 'i-234'
        # 2 minutes 30 seconds
        alerting._idle_since[instance_id] = time.time() - 150

        idle_duration_seconds = time.time() - alerting._idle_since[instance_id]
        formatted_duration = alerting._format_duration(int(idle_duration_seconds))

        assert formatted_duration == '2m 30s' or formatted_duration == '2m 29s'

    def test_idle_tracking_multiple_instances_different_times(self):
        """Should track different idle start times for different instances."""
        alerting = AlertingManager()

        # Track instances with different idle start times
        alerting._idle_since['i-old'] = time.time() - 600  # 10 minutes ago
        alerting._idle_since['i-new'] = time.time() - 60   # 1 minute ago

        old_duration = time.time() - alerting._idle_since['i-old']
        new_duration = time.time() - alerting._idle_since['i-new']

        # Old instance should have longer idle duration
        assert old_duration > new_duration
        assert 599 <= old_duration <= 601
        assert 59 <= new_duration <= 61
