"""
AWS Spot Instance Interruption Handler

Polls the EC2 instance metadata service for spot interruption notices.
When an interruption is detected (2-minute warning), triggers a callback
to checkpoint workloads and optionally request an on-demand replacement.
"""

import os
import time
import logging
import threading
import requests
from typing import Callable, Optional, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)

# IMDSv2 token endpoint
_IMDS_TOKEN_URL = 'http://169.254.169.254/latest/api/token'
_IMDS_SPOT_URL = 'http://169.254.169.254/latest/meta-data/spot/instance-action'
_IMDS_INSTANCE_ID_URL = 'http://169.254.169.254/latest/meta-data/instance-id'

_POLL_INTERVAL = 5  # seconds


class SpotInterruptionHandler:
    """Monitors for AWS Spot interruption notices via IMDS.

    Args:
        on_interruption: Callback(info_dict) invoked when interruption detected.
            info_dict contains: instance_id, action, time (ISO), detected_at.
        on_checkpoint: Optional callback to trigger workload checkpointing.
        poll_interval: Seconds between metadata polls (default 5).
    """

    def __init__(
        self,
        on_interruption: Callable[[Dict[str, Any]], None],
        on_checkpoint: Optional[Callable[[], None]] = None,
        poll_interval: int = _POLL_INTERVAL,
    ):
        self.on_interruption = on_interruption
        self.on_checkpoint = on_checkpoint
        self.poll_interval = poll_interval
        self._stop_event = threading.Event()
        self._thread = None
        self._imds_token = None
        self._imds_token_expiry = 0
        self._instance_id = None
        self._already_notified = False

    def _get_imds_token(self) -> Optional[str]:
        """Get IMDSv2 session token (valid for 6 hours)."""
        now = time.time()
        if self._imds_token and now < self._imds_token_expiry:
            return self._imds_token
        try:
            resp = requests.put(
                _IMDS_TOKEN_URL,
                headers={'X-aws-ec2-metadata-token-ttl-seconds': '21600'},
                timeout=2,
            )
            if resp.status_code == 200:
                self._imds_token = resp.text
                self._imds_token_expiry = now + 21000  # refresh before expiry
                return self._imds_token
        except Exception:
            pass
        return None

    def _get_instance_id(self) -> Optional[str]:
        """Get this instance's ID from IMDS."""
        if self._instance_id:
            return self._instance_id
        token = self._get_imds_token()
        if not token:
            return None
        try:
            resp = requests.get(
                _IMDS_INSTANCE_ID_URL,
                headers={'X-aws-ec2-metadata-token': token},
                timeout=2,
            )
            if resp.status_code == 200:
                self._instance_id = resp.text.strip()
                return self._instance_id
        except Exception:
            pass
        return None

    def _check_interruption(self) -> Optional[Dict[str, Any]]:
        """Check IMDS for spot interruption notice. Returns info dict or None."""
        token = self._get_imds_token()
        if not token:
            return None
        try:
            resp = requests.get(
                _IMDS_SPOT_URL,
                headers={'X-aws-ec2-metadata-token': token},
                timeout=2,
            )
            if resp.status_code == 200:
                data = resp.json()
                return {
                    'instance_id': self._get_instance_id() or 'unknown',
                    'action': data.get('action', 'terminate'),
                    'time': data.get('time', datetime.utcnow().isoformat()),
                    'detected_at': datetime.utcnow().isoformat(),
                }
            # 404 = no interruption notice, which is normal
        except requests.exceptions.ConnectionError:
            # Not running on EC2, or IMDS disabled
            pass
        except Exception as e:
            logger.debug(f"Spot interruption check error: {e}")
        return None

    def _poll_loop(self):
        """Background polling loop."""
        logger.info("Spot interruption handler started")
        while not self._stop_event.is_set():
            try:
                info = self._check_interruption()
                if info and not self._already_notified:
                    self._already_notified = True
                    logger.warning(
                        f"SPOT INTERRUPTION DETECTED: instance={info['instance_id']} "
                        f"action={info['action']} time={info['time']}"
                    )
                    # Trigger checkpoint first
                    if self.on_checkpoint:
                        try:
                            logger.info("Triggering workload checkpoint...")
                            self.on_checkpoint()
                        except Exception as e:
                            logger.error(f"Checkpoint callback failed: {e}")
                    # Then notify
                    try:
                        self.on_interruption(info)
                    except Exception as e:
                        logger.error(f"Interruption callback failed: {e}")
            except Exception as e:
                logger.debug(f"Spot handler poll error: {e}")

            self._stop_event.wait(timeout=self.poll_interval)

        logger.info("Spot interruption handler stopped")

    def start(self):
        """Start the background polling thread (daemon)."""
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._already_notified = False
        self._thread = threading.Thread(
            target=self._poll_loop, name='spot-handler', daemon=True
        )
        self._thread.start()

    def stop(self):
        """Stop the polling thread."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=10)

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()
