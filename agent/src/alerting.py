"""
Alerting module for notifications via email and Slack
"""

import os
import logging
import smtplib
import time
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import List, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)

class AlertManager:
    def __init__(self):
        self.slack_webhook = os.getenv('SLACK_WEBHOOK_URL')
        self.email_config = {
            'smtp_server': os.getenv('EMAIL_SMTP_SERVER'),
            'smtp_port': int(os.getenv('EMAIL_SMTP_PORT', 587)),
            'username': os.getenv('EMAIL_USERNAME'),
            'password': os.getenv('EMAIL_PASSWORD'),
            'to_email': os.getenv('EMAIL_TO')
        }

        # Alert thresholds
        self.idle_threshold = int(os.getenv('IDLE_GPU_THRESHOLD', 10))  # minutes
        self.cost_threshold = float(os.getenv('DAILY_COST_THRESHOLD', 100))  # USD

        # Track when each instance first went idle (instance_id -> epoch timestamp)
        self._idle_since: Dict[str, float] = {}

    def check_idle_gpus(self, gpu_metrics: List[Dict[str, Any]]):
        """Check for idle GPU instances and send alerts.

        Tracks how long each instance has been continuously idle so alert
        messages include duration (e.g., 'idle for 3m 20s').
        """
        now = time.time()
        idle_instances = []
        currently_idle_ids = set()
        current_instance_ids = set()

        for m in gpu_metrics:
            # Track all instance IDs currently in metrics
            instance_id = m.get('instance_id', m.get('pod_name', 'unknown'))
            current_instance_ids.add(instance_id)

            if not m.get('is_idle'):
                continue

            currently_idle_ids.add(instance_id)

            if instance_id not in self._idle_since:
                self._idle_since[instance_id] = now

            idle_seconds = now - self._idle_since[instance_id]
            m_with_duration = dict(m)
            m_with_duration['idle_duration_seconds'] = idle_seconds
            idle_instances.append(m_with_duration)

        # Clear tracking for instances that are no longer idle
        for iid in list(self._idle_since.keys()):
            if iid not in currently_idle_ids:
                del self._idle_since[iid]

        # Cleanup stale entries for instances no longer in metrics
        for iid in list(self._idle_since.keys()):
            if iid not in current_instance_ids:
                del self._idle_since[iid]
                logger.debug(f"Cleaned up stale idle tracking for instance {iid}")

        if idle_instances:
            message = self._format_idle_gpu_alert(idle_instances)
            self._send_alert('idle_gpu', message, idle_instances)
    
    def check_cost_thresholds(self, cost_data: List[Dict[str, Any]]):
        """Check for cost threshold violations"""
        if not cost_data:
            return
            
        # Calculate total daily cost
        total_daily_cost = sum(cost['cost'] for cost in cost_data)
        
        if total_daily_cost > self.cost_threshold:
            message = self._format_cost_alert(total_daily_cost, cost_data)
            self._send_alert('cost_threshold', message, cost_data)
    
    @staticmethod
    def _format_duration(seconds: float) -> str:
        """Format seconds into a human-readable duration string."""
        seconds = int(seconds)
        if seconds < 60:
            return f"{seconds}s"
        minutes, secs = divmod(seconds, 60)
        if minutes < 60:
            return f"{minutes}m {secs}s"
        hours, minutes = divmod(minutes, 60)
        return f"{hours}h {minutes}m"

    def _format_idle_gpu_alert(self, idle_instances: List[Dict[str, Any]]) -> str:
        """Format idle GPU alert message"""
        message = f"🚨 IDLE GPU ALERT\n\n"
        message += f"Found {len(idle_instances)} idle GPU instances:\n\n"

        for instance in idle_instances:
            instance_id = instance.get('instance_id', instance.get('pod_name', 'Unknown'))
            message += f"• {instance_id} ({instance.get('instance_type', 'Unknown')})\n"
            message += f"  - GPU Utilization: {instance['gpu_utilization']:.1f}%\n"
            if 'cpu_utilization' in instance:
                message += f"  - CPU Utilization: {instance['cpu_utilization']:.1f}%\n"
            idle_secs = instance.get('idle_duration_seconds')
            if idle_secs is not None:
                message += f"  - Idle for: {self._format_duration(idle_secs)}\n"
            if instance.get('experiment_id'):
                message += f"  - Experiment: {instance['experiment_id']}\n"
            message += "\n"

        message += f"💡 Consider stopping these instances to save costs.\n"
        message += f"Threshold: {self.idle_threshold} minutes of low utilization"

        return message
    
    def _format_cost_alert(self, total_cost: float, cost_data: List[Dict[str, Any]]) -> str:
        """Format cost threshold alert message"""
        message = f"💰 COST THRESHOLD ALERT\n\n"
        message += f"Daily AWS costs have exceeded threshold:\n\n"
        message += f"• Total Cost: ${total_cost:.2f}\n"
        message += f"• Threshold: ${self.cost_threshold:.2f}\n\n"
        
        message += "Cost breakdown by instance type:\n"
        instance_costs = {}
        for cost in cost_data:
            instance_type = cost.get('instance_type', 'Unknown')
            if instance_type not in instance_costs:
                instance_costs[instance_type] = 0
            instance_costs[instance_type] += cost['cost']
        
        for instance_type, cost in instance_costs.items():
            message += f"• {instance_type}: ${cost:.2f}\n"
        
        return message
    
    def _send_alert(self, alert_type: str, message: str, data: List[Dict[str, Any]]):
        """Send alert via configured channels"""
        try:
            # Send Slack notification
            if self.slack_webhook:
                self._send_slack_alert(message)
            
            # Send email notification
            if all(self.email_config.values()):
                self._send_email_alert(alert_type, message)
            
            logger.info(f"Alert sent: {alert_type}")
            
        except Exception as e:
            logger.error(f"Error sending alert: {e}")
    
    def _send_slack_alert(self, message: str):
        """Send alert to Slack"""
        try:
            payload = {
                "text": message,
                "username": "GPU Cost Bot",
                "icon_emoji": ":robot_face:"
            }
            
            response = requests.post(self.slack_webhook, json=payload)
            response.raise_for_status()
            
        except Exception as e:
            logger.error(f"Error sending Slack alert: {e}")
    
    def _send_email_alert(self, alert_type: str, message: str):
        """Send alert via email"""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.email_config['username']
            msg['To'] = self.email_config['to_email']
            msg['Subject'] = f"GPU Cost Alert: {alert_type.replace('_', ' ').title()}"
            
            msg.attach(MIMEText(message, 'plain'))
            
            server = smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port'])
            server.starttls()
            server.login(self.email_config['username'], self.email_config['password'])
            
            text = msg.as_string()
            server.sendmail(self.email_config['username'], self.email_config['to_email'], text)
            server.quit()
            
        except Exception as e:
            logger.error(f"Error sending email alert: {e}")
    
    def send_test_alert(self):
        """Send a test alert to verify configuration"""
        test_message = "🧪 TEST ALERT\n\nThis is a test message to verify alerting configuration."
        self._send_alert('test', test_message, [])
