import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import unittest
import time
import threading
from unittest.mock import Mock, patch, MagicMock, call
from datetime import datetime
import requests

from spot_handler import SpotInterruptionHandler


class TestSpotInterruptionHandler(unittest.TestCase):
    """Test suite for SpotInterruptionHandler"""

    def setUp(self):
        """Set up test fixtures"""
        self.on_interruption = Mock()
        self.on_checkpoint = Mock()

    def tearDown(self):
        """Clean up after tests"""
        # Ensure any threads are stopped
        pass

    # ============================================================================
    # Initialization Tests
    # ============================================================================

    def test_initialization_with_callbacks(self):
        """Test handler initializes with callbacks"""
        handler = SpotInterruptionHandler(
            on_interruption=self.on_interruption,
            on_checkpoint=self.on_checkpoint
        )
        self.assertEqual(handler.on_interruption, self.on_interruption)
        self.assertEqual(handler.on_checkpoint, self.on_checkpoint)
        self.assertEqual(handler.poll_interval, 5)

    def test_initialization_without_checkpoint_callback(self):
        """Test handler initializes without checkpoint callback"""
        handler = SpotInterruptionHandler(on_interruption=self.on_interruption)
        self.assertEqual(handler.on_interruption, self.on_interruption)
        self.assertIsNone(handler.on_checkpoint)

    def test_initialization_with_custom_poll_interval(self):
        """Test handler initializes with custom poll interval"""
        handler = SpotInterruptionHandler(
            on_interruption=self.on_interruption,
            poll_interval=10
        )
        self.assertEqual(handler.poll_interval, 10)

    def test_initialization_state(self):
        """Test handler initializes with correct state"""
        handler = SpotInterruptionHandler(on_interruption=self.on_interruption)
        self.assertIsNone(handler._imds_token)
        self.assertEqual(handler._imds_token_expiry, 0)
        self.assertIsNone(handler._instance_id)
        self.assertFalse(handler._already_notified)
        self.assertIsNone(handler._thread)
        self.assertFalse(handler._stop_event.is_set())

    # ============================================================================
    # IMDS Token Tests
    # ============================================================================

    @patch('spot_handler.requests.put')
    def test_get_imds_token_success(self, mock_put):
        """Test successful IMDS token acquisition"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = 'test-token-value'
        mock_put.return_value = mock_response

        handler = SpotInterruptionHandler(on_interruption=self.on_interruption)
        token = handler._get_imds_token()

        self.assertEqual(token, 'test-token-value')
        mock_put.assert_called_once_with(
            'http://169.254.169.254/latest/api/token',
            headers={'X-aws-ec2-metadata-token-ttl-seconds': '21600'},
            timeout=2
        )

    @patch('spot_handler.requests.put')
    def test_get_imds_token_failure_non_200_status(self, mock_put):
        """Test IMDS token acquisition fails with non-200 status"""
        mock_response = Mock()
        mock_response.status_code = 401
        mock_put.return_value = mock_response

        handler = SpotInterruptionHandler(on_interruption=self.on_interruption)
        token = handler._get_imds_token()

        self.assertIsNone(token)

    @patch('spot_handler.requests.put')
    def test_get_imds_token_failure_request_exception(self, mock_put):
        """Test IMDS token acquisition fails with request exception"""
        mock_put.side_effect = requests.exceptions.Timeout()

        handler = SpotInterruptionHandler(on_interruption=self.on_interruption)
        token = handler._get_imds_token()

        self.assertIsNone(token)

    @patch('spot_handler.requests.put')
    def test_get_imds_token_failure_connection_error(self, mock_put):
        """Test IMDS token acquisition fails with connection error"""
        mock_put.side_effect = requests.exceptions.ConnectionError()

        handler = SpotInterruptionHandler(on_interruption=self.on_interruption)
        token = handler._get_imds_token()

        self.assertIsNone(token)

    @patch('spot_handler.requests.put')
    def test_get_imds_token_caching(self, mock_put):
        """Test IMDS token is cached"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = 'cached-token'
        mock_put.return_value = mock_response

        handler = SpotInterruptionHandler(on_interruption=self.on_interruption)

        # First call should make a request
        token1 = handler._get_imds_token()
        self.assertEqual(token1, 'cached-token')
        self.assertEqual(mock_put.call_count, 1)

        # Second call should return cached token without making a request
        token2 = handler._get_imds_token()
        self.assertEqual(token2, 'cached-token')
        self.assertEqual(mock_put.call_count, 1)

    @patch('spot_handler.requests.put')
    @patch('spot_handler.time.time')
    def test_get_imds_token_cache_expiry(self, mock_time, mock_put):
        """Test IMDS token cache expires and refreshes"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = 'fresh-token'
        mock_put.return_value = mock_response

        handler = SpotInterruptionHandler(on_interruption=self.on_interruption)

        # First call at time 0
        mock_time.return_value = 0
        token1 = handler._get_imds_token()
        self.assertEqual(token1, 'fresh-token')

        # Second call at time 30000 (before expiry at 21000 seconds)
        mock_time.return_value = 15000
        token2 = handler._get_imds_token()
        self.assertEqual(token2, 'fresh-token')
        self.assertEqual(mock_put.call_count, 1)

        # Third call at time 25000 (after expiry)
        mock_time.return_value = 25000
        token3 = handler._get_imds_token()
        self.assertEqual(token3, 'fresh-token')
        self.assertEqual(mock_put.call_count, 2)

    # ============================================================================
    # Instance ID Tests
    # ============================================================================

    @patch('spot_handler.requests.get')
    @patch('spot_handler.requests.put')
    def test_get_instance_id_success(self, mock_put, mock_get):
        """Test successful instance ID retrieval"""
        token_response = Mock()
        token_response.status_code = 200
        token_response.text = 'test-token'
        mock_put.return_value = token_response

        instance_response = Mock()
        instance_response.status_code = 200
        instance_response.text = 'i-1234567890abcdef0\n'
        mock_get.return_value = instance_response

        handler = SpotInterruptionHandler(on_interruption=self.on_interruption)
        instance_id = handler._get_instance_id()

        self.assertEqual(instance_id, 'i-1234567890abcdef0')
        mock_get.assert_called_once_with(
            'http://169.254.169.254/latest/meta-data/instance-id',
            headers={'X-aws-ec2-metadata-token': 'test-token'},
            timeout=2
        )

    @patch('spot_handler.requests.get')
    @patch('spot_handler.requests.put')
    def test_get_instance_id_failure_no_token(self, mock_put, mock_get):
        """Test instance ID retrieval fails when token cannot be obtained"""
        mock_put.side_effect = requests.exceptions.ConnectionError()

        handler = SpotInterruptionHandler(on_interruption=self.on_interruption)
        instance_id = handler._get_instance_id()

        self.assertIsNone(instance_id)
        mock_get.assert_not_called()

    @patch('spot_handler.requests.get')
    @patch('spot_handler.requests.put')
    def test_get_instance_id_failure_non_200_status(self, mock_put, mock_get):
        """Test instance ID retrieval fails with non-200 status"""
        token_response = Mock()
        token_response.status_code = 200
        token_response.text = 'test-token'
        mock_put.return_value = token_response

        instance_response = Mock()
        instance_response.status_code = 404
        mock_get.return_value = instance_response

        handler = SpotInterruptionHandler(on_interruption=self.on_interruption)
        instance_id = handler._get_instance_id()

        self.assertIsNone(instance_id)

    @patch('spot_handler.requests.get')
    @patch('spot_handler.requests.put')
    def test_get_instance_id_failure_request_exception(self, mock_put, mock_get):
        """Test instance ID retrieval fails with request exception"""
        token_response = Mock()
        token_response.status_code = 200
        token_response.text = 'test-token'
        mock_put.return_value = token_response

        mock_get.side_effect = requests.exceptions.Timeout()

        handler = SpotInterruptionHandler(on_interruption=self.on_interruption)
        instance_id = handler._get_instance_id()

        self.assertIsNone(instance_id)

    @patch('spot_handler.requests.get')
    @patch('spot_handler.requests.put')
    def test_get_instance_id_caching(self, mock_put, mock_get):
        """Test instance ID is cached"""
        token_response = Mock()
        token_response.status_code = 200
        token_response.text = 'test-token'
        mock_put.return_value = token_response

        instance_response = Mock()
        instance_response.status_code = 200
        instance_response.text = 'i-cached-id'
        mock_get.return_value = instance_response

        handler = SpotInterruptionHandler(on_interruption=self.on_interruption)

        # First call should make a request
        id1 = handler._get_instance_id()
        self.assertEqual(id1, 'i-cached-id')
        self.assertEqual(mock_get.call_count, 1)

        # Second call should return cached ID without making a request
        id2 = handler._get_instance_id()
        self.assertEqual(id2, 'i-cached-id')
        self.assertEqual(mock_get.call_count, 1)

    # ============================================================================
    # Interruption Check Tests
    # ============================================================================

    @patch('spot_handler.requests.get')
    @patch('spot_handler.requests.put')
    def test_check_interruption_found(self, mock_put, mock_get):
        """Test interruption detection when spot notice is found"""
        token_response = Mock()
        token_response.status_code = 200
        token_response.text = 'test-token'
        mock_put.return_value = token_response

        def mock_get_side_effect(url, **kwargs):
            response = Mock()
            if 'instance-id' in url:
                response.status_code = 200
                response.text = 'i-test-instance'
            elif 'spot/instance-action' in url:
                response.status_code = 200
                response.json.return_value = {
                    'action': 'terminate',
                    'time': '2026-03-27T12:00:00Z'
                }
            return response

        mock_get.side_effect = mock_get_side_effect

        handler = SpotInterruptionHandler(on_interruption=self.on_interruption)
        info = handler._check_interruption()

        self.assertIsNotNone(info)
        self.assertEqual(info['instance_id'], 'i-test-instance')
        self.assertEqual(info['action'], 'terminate')
        self.assertEqual(info['time'], '2026-03-27T12:00:00Z')
        self.assertIn('detected_at', info)

    @patch('spot_handler.requests.get')
    @patch('spot_handler.requests.put')
    def test_check_interruption_not_found_404(self, mock_put, mock_get):
        """Test interruption check when spot notice not found (404)"""
        token_response = Mock()
        token_response.status_code = 200
        token_response.text = 'test-token'
        mock_put.return_value = token_response

        spot_response = Mock()
        spot_response.status_code = 404
        mock_get.return_value = spot_response

        handler = SpotInterruptionHandler(on_interruption=self.on_interruption)
        info = handler._check_interruption()

        self.assertIsNone(info)

    @patch('spot_handler.requests.get')
    @patch('spot_handler.requests.put')
    def test_check_interruption_no_token(self, mock_put, mock_get):
        """Test interruption check fails when token cannot be obtained"""
        mock_put.side_effect = requests.exceptions.ConnectionError()

        handler = SpotInterruptionHandler(on_interruption=self.on_interruption)
        info = handler._check_interruption()

        self.assertIsNone(info)
        mock_get.assert_not_called()

    @patch('spot_handler.requests.get')
    @patch('spot_handler.requests.put')
    def test_check_interruption_connection_error(self, mock_put, mock_get):
        """Test interruption check handles connection error gracefully"""
        token_response = Mock()
        token_response.status_code = 200
        token_response.text = 'test-token'
        mock_put.return_value = token_response

        mock_get.side_effect = requests.exceptions.ConnectionError()

        handler = SpotInterruptionHandler(on_interruption=self.on_interruption)
        info = handler._check_interruption()

        self.assertIsNone(info)

    @patch('spot_handler.requests.get')
    @patch('spot_handler.requests.put')
    def test_check_interruption_timeout_error(self, mock_put, mock_get):
        """Test interruption check handles timeout gracefully"""
        token_response = Mock()
        token_response.status_code = 200
        token_response.text = 'test-token'
        mock_put.return_value = token_response

        mock_get.side_effect = requests.exceptions.Timeout()

        handler = SpotInterruptionHandler(on_interruption=self.on_interruption)
        info = handler._check_interruption()

        self.assertIsNone(info)

    @patch('spot_handler.requests.get')
    @patch('spot_handler.requests.put')
    def test_check_interruption_json_decode_error(self, mock_put, mock_get):
        """Test interruption check handles JSON decode error"""
        token_response = Mock()
        token_response.status_code = 200
        token_response.text = 'test-token'
        mock_put.return_value = token_response

        def mock_get_side_effect(url, **kwargs):
            response = Mock()
            if 'instance-id' in url:
                response.status_code = 200
                response.text = 'i-test-instance'
            elif 'spot/instance-action' in url:
                response.status_code = 200
                response.json.side_effect = ValueError('Invalid JSON')
            return response

        mock_get.side_effect = mock_get_side_effect

        handler = SpotInterruptionHandler(on_interruption=self.on_interruption)
        info = handler._check_interruption()

        self.assertIsNone(info)

    @patch('spot_handler.requests.get')
    @patch('spot_handler.requests.put')
    def test_check_interruption_default_values(self, mock_put, mock_get):
        """Test interruption check uses default values for missing fields"""
        token_response = Mock()
        token_response.status_code = 200
        token_response.text = 'test-token'
        mock_put.return_value = token_response

        def mock_get_side_effect(url, **kwargs):
            response = Mock()
            if 'instance-id' in url:
                response.status_code = 200
                response.text = 'i-test-instance'
            elif 'spot/instance-action' in url:
                response.status_code = 200
                response.json.return_value = {}
            return response

        mock_get.side_effect = mock_get_side_effect

        handler = SpotInterruptionHandler(on_interruption=self.on_interruption)
        info = handler._check_interruption()

        self.assertIsNotNone(info)
        self.assertEqual(info['action'], 'terminate')
        self.assertIn('time', info)

    @patch('spot_handler.requests.get')
    @patch('spot_handler.requests.put')
    def test_check_interruption_instance_id_unknown_fallback(self, mock_put, mock_get):
        """Test interruption check uses 'unknown' when instance ID retrieval fails"""
        token_response = Mock()
        token_response.status_code = 200
        token_response.text = 'test-token'
        mock_put.return_value = token_response

        def mock_get_side_effect(url, **kwargs):
            response = Mock()
            if 'instance-id' in url:
                response.status_code = 404
            elif 'spot/instance-action' in url:
                response.status_code = 200
                response.json.return_value = {'action': 'terminate'}
            return response

        mock_get.side_effect = mock_get_side_effect

        handler = SpotInterruptionHandler(on_interruption=self.on_interruption)
        info = handler._check_interruption()

        self.assertIsNotNone(info)
        self.assertEqual(info['instance_id'], 'unknown')

    # ============================================================================
    # Poll Loop Tests
    # ============================================================================

    @patch('spot_handler.requests.get')
    @patch('spot_handler.requests.put')
    def test_poll_loop_triggers_callbacks(self, mock_put, mock_get):
        """Test poll loop triggers checkpoint then interruption callbacks"""
        token_response = Mock()
        token_response.status_code = 200
        token_response.text = 'test-token'
        mock_put.return_value = token_response

        def mock_get_side_effect(url, **kwargs):
            response = Mock()
            if 'instance-id' in url:
                response.status_code = 200
                response.text = 'i-test-instance'
            elif 'spot/instance-action' in url:
                response.status_code = 200
                response.json.return_value = {'action': 'terminate', 'time': '2026-03-27T12:00:00Z'}
            return response

        mock_get.side_effect = mock_get_side_effect

        handler = SpotInterruptionHandler(
            on_interruption=self.on_interruption,
            on_checkpoint=self.on_checkpoint,
            poll_interval=0.1
        )

        # Run poll loop once by starting and immediately stopping
        handler.start()
        time.sleep(0.3)
        handler.stop()

        # Both callbacks should have been called
        self.on_checkpoint.assert_called_once()
        self.on_interruption.assert_called_once()

        # Verify callback was called with correct info
        call_args = self.on_interruption.call_args[0][0]
        self.assertEqual(call_args['instance_id'], 'i-test-instance')
        self.assertEqual(call_args['action'], 'terminate')

    @patch('spot_handler.requests.get')
    @patch('spot_handler.requests.put')
    def test_poll_loop_without_checkpoint_callback(self, mock_put, mock_get):
        """Test poll loop works without checkpoint callback"""
        token_response = Mock()
        token_response.status_code = 200
        token_response.text = 'test-token'
        mock_put.return_value = token_response

        def mock_get_side_effect(url, **kwargs):
            response = Mock()
            if 'instance-id' in url:
                response.status_code = 200
                response.text = 'i-test-instance'
            elif 'spot/instance-action' in url:
                response.status_code = 200
                response.json.return_value = {'action': 'terminate', 'time': '2026-03-27T12:00:00Z'}
            return response

        mock_get.side_effect = mock_get_side_effect

        handler = SpotInterruptionHandler(
            on_interruption=self.on_interruption,
            on_checkpoint=None,
            poll_interval=0.1
        )

        handler.start()
        time.sleep(0.3)
        handler.stop()

        self.on_interruption.assert_called_once()

    # ============================================================================
    # Only Notify Once Tests
    # ============================================================================

    @patch('spot_handler.requests.get')
    @patch('spot_handler.requests.put')
    def test_already_notified_flag_prevents_duplicate_notifications(self, mock_put, mock_get):
        """Test _already_notified flag prevents duplicate notifications"""
        token_response = Mock()
        token_response.status_code = 200
        token_response.text = 'test-token'
        mock_put.return_value = token_response

        def mock_get_side_effect(url, **kwargs):
            response = Mock()
            if 'instance-id' in url:
                response.status_code = 200
                response.text = 'i-test-instance'
            elif 'spot/instance-action' in url:
                response.status_code = 200
                response.json.return_value = {'action': 'terminate', 'time': '2026-03-27T12:00:00Z'}
            return response

        mock_get.side_effect = mock_get_side_effect

        handler = SpotInterruptionHandler(
            on_interruption=self.on_interruption,
            on_checkpoint=self.on_checkpoint,
            poll_interval=0.1
        )

        handler.start()
        time.sleep(0.5)  # Allow multiple poll cycles
        handler.stop()

        # Should only be called once despite multiple poll cycles
        self.on_interruption.assert_called_once()
        self.on_checkpoint.assert_called_once()

    @patch('spot_handler.requests.get')
    @patch('spot_handler.requests.put')
    def test_no_notification_when_no_interruption(self, mock_put, mock_get):
        """Test no notification when no interruption is detected"""
        token_response = Mock()
        token_response.status_code = 200
        token_response.text = 'test-token'
        mock_put.return_value = token_response

        spot_response = Mock()
        spot_response.status_code = 404
        mock_get.return_value = spot_response

        handler = SpotInterruptionHandler(
            on_interruption=self.on_interruption,
            on_checkpoint=self.on_checkpoint,
            poll_interval=0.1
        )

        handler.start()
        time.sleep(0.3)
        handler.stop()

        self.on_interruption.assert_not_called()
        self.on_checkpoint.assert_not_called()

    # ============================================================================
    # Callback Error Handling Tests
    # ============================================================================

    @patch('spot_handler.requests.get')
    @patch('spot_handler.requests.put')
    def test_checkpoint_failure_does_not_prevent_interruption_callback(self, mock_put, mock_get):
        """Test interruption callback is called even if checkpoint callback fails"""
        token_response = Mock()
        token_response.status_code = 200
        token_response.text = 'test-token'
        mock_put.return_value = token_response

        def mock_get_side_effect(url, **kwargs):
            response = Mock()
            if 'instance-id' in url:
                response.status_code = 200
                response.text = 'i-test-instance'
            elif 'spot/instance-action' in url:
                response.status_code = 200
                response.json.return_value = {'action': 'terminate', 'time': '2026-03-27T12:00:00Z'}
            return response

        mock_get.side_effect = mock_get_side_effect

        checkpoint_callback = Mock(side_effect=Exception('Checkpoint failed'))
        interruption_callback = Mock()

        handler = SpotInterruptionHandler(
            on_interruption=interruption_callback,
            on_checkpoint=checkpoint_callback,
            poll_interval=0.1
        )

        handler.start()
        time.sleep(0.3)
        handler.stop()

        checkpoint_callback.assert_called_once()
        interruption_callback.assert_called_once()

    @patch('spot_handler.requests.get')
    @patch('spot_handler.requests.put')
    def test_interruption_callback_failure_is_caught(self, mock_put, mock_get):
        """Test interruption callback failure is caught and logged"""
        token_response = Mock()
        token_response.status_code = 200
        token_response.text = 'test-token'
        mock_put.return_value = token_response

        def mock_get_side_effect(url, **kwargs):
            response = Mock()
            if 'instance-id' in url:
                response.status_code = 200
                response.text = 'i-test-instance'
            elif 'spot/instance-action' in url:
                response.status_code = 200
                response.json.return_value = {'action': 'terminate', 'time': '2026-03-27T12:00:00Z'}
            return response

        mock_get.side_effect = mock_get_side_effect

        interruption_callback = Mock(side_effect=Exception('Callback error'))

        handler = SpotInterruptionHandler(
            on_interruption=interruption_callback,
            poll_interval=0.1
        )

        # Should not raise exception
        handler.start()
        time.sleep(0.3)
        handler.stop()

        interruption_callback.assert_called_once()

    # ============================================================================
    # Lifecycle Tests
    # ============================================================================

    def test_start_creates_daemon_thread(self):
        """Test start() creates a daemon thread"""
        handler = SpotInterruptionHandler(on_interruption=self.on_interruption)
        handler.start()

        self.assertIsNotNone(handler._thread)
        self.assertTrue(handler._thread.daemon)
        self.assertEqual(handler._thread.name, 'spot-handler')

        handler.stop()

    def test_start_multiple_times_does_not_create_multiple_threads(self):
        """Test start() multiple times doesn't create multiple threads"""
        handler = SpotInterruptionHandler(on_interruption=self.on_interruption)

        handler.start()
        thread1 = handler._thread

        handler.start()
        thread2 = handler._thread

        self.assertIs(thread1, thread2)

        handler.stop()

    def test_start_resets_already_notified_flag(self):
        """Test start() resets the already_notified flag"""
        handler = SpotInterruptionHandler(on_interruption=self.on_interruption)
        handler._already_notified = True

        handler.start()
        self.assertFalse(handler._already_notified)

        handler.stop()

    def test_start_clears_stop_event(self):
        """Test start() clears the stop event"""
        handler = SpotInterruptionHandler(on_interruption=self.on_interruption)
        handler._stop_event.set()

        handler.start()
        self.assertFalse(handler._stop_event.is_set())

        handler.stop()

    def test_stop_sets_stop_event(self):
        """Test stop() sets the stop event"""
        handler = SpotInterruptionHandler(on_interruption=self.on_interruption)
        handler.start()
        time.sleep(0.1)

        handler.stop()
        self.assertTrue(handler._stop_event.is_set())

    def test_stop_waits_for_thread(self):
        """Test stop() waits for thread to finish"""
        handler = SpotInterruptionHandler(on_interruption=self.on_interruption)
        handler.start()

        handler.stop()
        self.assertFalse(handler._thread.is_alive())

    def test_stop_with_timeout(self):
        """Test stop() uses timeout when joining thread"""
        handler = SpotInterruptionHandler(on_interruption=self.on_interruption)
        handler.start()

        # Should complete quickly with default timeout
        handler.stop()
        self.assertFalse(handler._thread.is_alive())

    # ============================================================================
    # is_running Property Tests
    # ============================================================================

    def test_is_running_false_before_start(self):
        """Test is_running is False before start()"""
        handler = SpotInterruptionHandler(on_interruption=self.on_interruption)
        self.assertFalse(handler.is_running)

    def test_is_running_true_after_start(self):
        """Test is_running is True after start()"""
        handler = SpotInterruptionHandler(on_interruption=self.on_interruption)
        handler.start()
        time.sleep(0.1)

        self.assertTrue(handler.is_running)

        handler.stop()

    def test_is_running_false_after_stop(self):
        """Test is_running is False after stop()"""
        handler = SpotInterruptionHandler(on_interruption=self.on_interruption)
        handler.start()
        time.sleep(0.1)

        handler.stop()
        self.assertFalse(handler.is_running)

    # ============================================================================
    # Thread Daemon Tests
    # ============================================================================

    def test_thread_is_daemon(self):
        """Test that the background thread is a daemon thread"""
        handler = SpotInterruptionHandler(on_interruption=self.on_interruption)
        handler.start()
        time.sleep(0.1)

        self.assertTrue(handler._thread.daemon)

        handler.stop()

    def test_daemon_thread_does_not_prevent_process_exit(self):
        """Test daemon thread flag allows process to exit"""
        handler = SpotInterruptionHandler(on_interruption=self.on_interruption)
        handler.start()

        # Daemon threads should not prevent process exit
        self.assertTrue(handler._thread.daemon)

        handler.stop()


if __name__ == '__main__':
    unittest.main()
