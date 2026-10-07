"""Always-on host identity monitor — no cloud discovery required."""

from __future__ import annotations

from typing import Any, Dict, List

from host_identity import build_host_instance


class HostIdentityMonitor:
    """Registers the agent host even when no cloud API returns instances."""

    def __init__(
        self,
        *,
        hostname: str,
        cloud_provider: str,
        instance_id: str,
    ) -> None:
        self.hostname = hostname
        self.cloud_provider = cloud_provider or "onprem"
        self.instance_id = instance_id or hostname

    def get_gpu_instances(self) -> List[Dict[str, Any]]:
        return [
            build_host_instance(
                hostname=self.hostname,
                cloud_provider=self.cloud_provider,
                instance_id=self.instance_id,
            )
        ]

    def get_gpu_utilization(
        self, instances: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        return []
