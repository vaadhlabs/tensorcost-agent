"""
GPU Cluster Health Monitor

Monitors GPU cluster health for multi-node training including:
- NVLink topology and status
- InfiniBand/RoCE fabric health
- NCCL communication diagnostics
- GPU-to-GPU bandwidth
- Cluster readiness for distributed training
"""

import os
import json
import logging
import subprocess
import re
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
import socket
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Optional imports
try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False

logger = logging.getLogger(__name__)


class NodeStatus(Enum):
    """Node health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    DOWN = "down"
    UNKNOWN = "unknown"


class InterconnectStatus(Enum):
    """Interconnect health status."""
    OK = "ok"
    DEGRADED = "degraded"
    DOWN = "down"
    UNKNOWN = "unknown"


@dataclass
class GPUInfo:
    """Information about a single GPU."""
    index: int
    name: str
    uuid: Optional[str] = None
    memory_total_mb: int = 0
    memory_free_mb: int = 0
    temperature_c: float = 0.0
    power_draw_w: float = 0.0
    clock_mhz: int = 0
    memory_clock_mhz: int = 0
    pcie_bandwidth_gbps: float = 0.0
    ecc_errors_uncorrected: int = 0
    ecc_errors_corrected: int = 0
    gpu_compute_cap: str = ""
    nvlink_count: int = 0


@dataclass
class NodeInfo:
    """Information about a cluster node."""
    hostname: str
    ip_address: Optional[str] = None
    status: str = NodeStatus.UNKNOWN.value
    gpu_count: int = 0
    gpus_healthy: int = 0
    gpus_with_errors: int = 0
    gpus: List[GPUInfo] = field(default_factory=list)
    nvlink_status: str = InterconnectStatus.UNKNOWN.value
    network_status: str = InterconnectStatus.UNKNOWN.value
    memory_total_gb: float = 0.0
    memory_free_gb: float = 0.0
    temperature_max_c: float = 0.0
    power_draw_total_w: float = 0.0
    ecc_errors_uncorrected_total: int = 0
    ecc_errors_corrected_total: int = 0
    last_heartbeat: Optional[str] = None
    uptime_hours: float = 0.0
    error_message: Optional[str] = None


@dataclass
class NVLinkInfo:
    """NVLink topology information."""
    source_gpu: int = 0
    target_gpu: int = 0
    link_type: str = ""
    bandwidth_gbps: float = 25.0  # NVLink 2.0 default
    status: str = InterconnectStatus.OK.value
    errors: int = 0


@dataclass
class InfiniBandPort:
    """InfiniBand port information."""
    port_id: str = ""
    status: str = InterconnectStatus.UNKNOWN.value
    link_speed_gbps: float = 0.0
    link_width: str = ""
    symbol_errors: int = 0
    link_downed_count: int = 0
    physical_state: str = ""
    logical_state: str = ""


class GPUClusterMonitor:
    """
    Monitors GPU cluster health for multi-node distributed training.

    Tracks NVLink topology, InfiniBand/RoCE fabric health, NCCL communication,
    GPU-to-GPU bandwidth, and overall cluster readiness.
    """

    def __init__(self):
        """Initialize cluster monitor from environment configuration."""
        self.enabled = os.getenv("CLUSTER_MONITOR_ENABLED", "true").lower() == "true"
        self.cluster_nodes = self._parse_cluster_nodes()
        self.nccl_debug_log_path = os.getenv("NCCL_DEBUG_LOG_PATH")
        self.fabric_manager_endpoint = os.getenv("FABRIC_MANAGER_ENDPOINT")
        self.slurm_enabled = os.getenv("SLURM_ENABLED", "false").lower() == "true"

        self._node_cache: Dict[str, NodeInfo] = {}
        self._cache_timestamp: Optional[datetime] = None
        self._cache_ttl_seconds = 30

        if not self.enabled:
            logger.info("GPU cluster monitor is disabled")
            return

        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                logger.info("PYNVML initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize PYNVML: {e}")

        logger.info(
            f"GPU Cluster Monitor initialized. Nodes: {self.cluster_nodes}, "
            f"SLURM: {self.slurm_enabled}, NCCL logs: {self.nccl_debug_log_path}"
        )

    def _parse_cluster_nodes(self) -> List[str]:
        """Parse cluster nodes from environment variable."""
        nodes_str = os.getenv("CLUSTER_NODES", "localhost")

        if nodes_str.lower() == "auto":
            return self._discover_nodes()

        nodes = [node.strip() for node in nodes_str.split(",") if node.strip()]
        return nodes if nodes else ["localhost"]

    def _discover_nodes(self) -> List[str]:
        """Discover cluster nodes via SLURM or Kubernetes."""
        nodes = []

        # Try SLURM first
        if self.slurm_enabled:
            try:
                result = subprocess.run(
                    ["sinfo", "-h", "-o", "%N"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    node_str = result.stdout.strip()
                    # Handle SLURM node ranges like node[0-3]
                    nodes = self._expand_slurm_node_ranges(node_str)
                    if nodes:
                        logger.info(f"Discovered {len(nodes)} nodes via SLURM")
                        return nodes
            except Exception as e:
                logger.warning(f"Failed to discover nodes via SLURM: {e}")

        # Try Kubernetes
        try:
            result = subprocess.run(
                ["kubectl", "get", "nodes", "-o", "jsonpath={.items[*].metadata.name}"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                nodes = result.stdout.strip().split()
                if nodes:
                    logger.info(f"Discovered {len(nodes)} nodes via Kubernetes")
                    return nodes
        except Exception as e:
            logger.debug(f"Kubernetes discovery unavailable: {e}")

        logger.info("Falling back to localhost")
        return ["localhost"]

    @staticmethod
    def _expand_slurm_node_ranges(node_str: str) -> List[str]:
        """Expand SLURM node range notation like 'node[0-3]' to individual nodes."""
        nodes = []
        # Simple expansion for patterns like node[0-3] or node[01-04]
        pattern = r"(\w+)\[(\d+)-(\d+)\]"
        matches = re.findall(pattern, node_str)

        if matches:
            for prefix, start, end in matches:
                start_int = int(start)
                end_int = int(end)
                pad_width = len(start) if start[0] == "0" else 0
                for i in range(start_int, end_int + 1):
                    if pad_width:
                        nodes.append(f"{prefix}{i:0{pad_width}d}")
                    else:
                        nodes.append(f"{prefix}{i}")
        else:
            # No ranges, just comma-separated nodes
            nodes = [n.strip() for n in node_str.split(",") if n.strip()]

        return nodes

    def get_cluster_topology(self) -> Dict[str, Any]:
        """
        Get GPU cluster topology including NVLink and network fabric.

        Returns:
            Dictionary with keys:
            - nodes: list of node info dicts
            - total_gpus: total GPU count across cluster
            - gpu_type: GPU model name
            - nvlink_topology: NVLink connectivity info
            - network_fabric: fabric type and status
            - fabric_bandwidth_gbps: fabric bandwidth
            - cluster_status: overall status
        """
        if not self.enabled:
            return {"error": "Cluster monitor disabled"}

        try:
            node_infos = self.get_node_health()
            interconnect = self.get_interconnect_health()

            total_gpus = sum(node.gpu_count for node in node_infos)
            gpu_type = ""
            if node_infos and node_infos[0].gpus:
                gpu_type = node_infos[0].gpus[0].name

            # Determine overall status
            unhealthy_nodes = [n for n in node_infos if n.status != NodeStatus.HEALTHY.value]
            cluster_status = (
                NodeStatus.DOWN.value if any(n.status == NodeStatus.DOWN.value for n in node_infos)
                else NodeStatus.DEGRADED.value if unhealthy_nodes
                else NodeStatus.HEALTHY.value
            )

            return {
                "nodes": [asdict(n) for n in node_infos],
                "total_gpus": total_gpus,
                "gpu_type": gpu_type,
                "nvlink_topology": interconnect.get("nvlink_links_total", 0),
                "network_fabric": interconnect.get("fabric_type", "unknown"),
                "fabric_bandwidth_gbps": interconnect.get("ib_bandwidth_gbps", 0.0),
                "cluster_status": cluster_status,
                "timestamp": datetime.utcnow().isoformat(),
            }
        except Exception as e:
            logger.exception("Error getting cluster topology")
            return {"error": str(e), "timestamp": datetime.utcnow().isoformat()}

    def get_node_health(self) -> List[NodeInfo]:
        """
        Health check each node in the cluster.

        Returns:
            List of NodeInfo objects with detailed per-node health information.
        """
        if not self.enabled:
            return []

        # Check cache
        if self._cache_timestamp and (datetime.utcnow() - self._cache_timestamp).total_seconds() < self._cache_ttl_seconds:
            return list(self._node_cache.values())

        node_infos = []

        with ThreadPoolExecutor(max_workers=min(10, len(self.cluster_nodes))) as executor:
            futures = {
                executor.submit(self._check_node_health, node): node
                for node in self.cluster_nodes
            }

            for future in as_completed(futures):
                node = futures[future]
                try:
                    info = future.result(timeout=30)
                    node_infos.append(info)
                    self._node_cache[node] = info
                except Exception as e:
                    logger.error(f"Failed to check health of {node}: {e}")
                    node_info = NodeInfo(
                        hostname=node,
                        status=NodeStatus.DOWN.value,
                        error_message=str(e)
                    )
                    node_infos.append(node_info)
                    self._node_cache[node] = node_info

        self._cache_timestamp = datetime.utcnow()
        return node_infos

    def _check_node_health(self, hostname: str) -> NodeInfo:
        """Check health of a single node."""
        node_info = NodeInfo(hostname=hostname)

        try:
            # Resolve IP address
            try:
                node_info.ip_address = socket.gethostbyname(hostname)
            except socket.gaierror:
                node_info.ip_address = None

            if hostname.lower() == "localhost" or hostname == "127.0.0.1":
                # Local node
                return self._check_local_node_health()
            else:
                # Remote node - try SSH
                return self._check_remote_node_health(hostname)

        except Exception as e:
            logger.exception(f"Error checking node health for {hostname}")
            node_info.status = NodeStatus.DOWN.value
            node_info.error_message = str(e)
            return node_info

    def _check_local_node_health(self) -> NodeInfo:
        """Check health of the local node."""
        node_info = NodeInfo(hostname="localhost")
        node_info.ip_address = "127.0.0.1"

        try:
            # Get GPU info
            gpu_info_list = self._get_local_gpu_info()
            node_info.gpus = gpu_info_list
            node_info.gpu_count = len(gpu_info_list)

            if not gpu_info_list:
                node_info.status = NodeStatus.DEGRADED.value
                node_info.error_message = "No GPUs detected"
                return node_info

            # Calculate aggregates
            node_info.memory_total_gb = sum(g.memory_total_mb for g in gpu_info_list) / 1024.0
            node_info.memory_free_gb = sum(g.memory_free_mb for g in gpu_info_list) / 1024.0
            node_info.temperature_max_c = max((g.temperature_c for g in gpu_info_list), default=0.0)
            node_info.power_draw_total_w = sum(g.power_draw_w for g in gpu_info_list)
            node_info.ecc_errors_uncorrected_total = sum(g.ecc_errors_uncorrected for g in gpu_info_list)
            node_info.ecc_errors_corrected_total = sum(g.ecc_errors_corrected for g in gpu_info_list)

            # Check GPU health
            node_info.gpus_healthy = sum(1 for g in gpu_info_list if g.temperature_c < 85 and g.ecc_errors_uncorrected == 0)
            node_info.gpus_with_errors = sum(1 for g in gpu_info_list if g.ecc_errors_uncorrected > 0)

            # Check NVLink status
            nvlink_status = self._get_nvlink_status()
            node_info.nvlink_status = InterconnectStatus.OK.value if nvlink_status else InterconnectStatus.UNKNOWN.value

            # Determine overall status
            if node_info.gpus_with_errors > 0 or node_info.temperature_max_c > 85:
                node_info.status = NodeStatus.DEGRADED.value
            elif node_info.gpus_healthy == node_info.gpu_count:
                node_info.status = NodeStatus.HEALTHY.value
            else:
                node_info.status = NodeStatus.DEGRADED.value

            # Get uptime
            try:
                with open("/proc/uptime", "r") as f:
                    uptime_seconds = float(f.read().split()[0])
                    node_info.uptime_hours = uptime_seconds / 3600.0
            except Exception:
                node_info.uptime_hours = 0.0

            node_info.last_heartbeat = datetime.utcnow().isoformat()

        except Exception as e:
            logger.exception("Error checking local node health")
            node_info.status = NodeStatus.DOWN.value
            node_info.error_message = str(e)

        return node_info

    def _check_remote_node_health(self, hostname: str) -> NodeInfo:
        """Check health of a remote node via SSH."""
        node_info = NodeInfo(hostname=hostname)

        try:
            # Try to SSH and run nvidia-smi
            cmd = f"ssh -o ConnectTimeout=5 {hostname} nvidia-smi --query-gpu=index,name,memory.total,memory.free,temperature.gpu,power.draw --format=csv,noheader"
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode == 0:
                node_info.status = NodeStatus.HEALTHY.value
                # Parse GPU info from output
                lines = result.stdout.strip().split("\n")
                node_info.gpu_count = len(lines)
                node_info.gpus_healthy = len(lines)
            else:
                node_info.status = NodeStatus.DOWN.value
                node_info.error_message = result.stderr or "SSH connection failed"

        except Exception as e:
            logger.warning(f"Could not check remote node {hostname}: {e}")
            node_info.status = NodeStatus.DOWN.value
            node_info.error_message = str(e)

        node_info.last_heartbeat = datetime.utcnow().isoformat()
        return node_info

    def _get_local_gpu_info(self) -> List[GPUInfo]:
        """Get local GPU information via nvidia-smi or pynvml."""
        gpu_list = []

        # Try pynvml first if available
        if PYNVML_AVAILABLE:
            try:
                return self._get_gpu_info_pynvml()
            except Exception as e:
                logger.debug(f"PYNVML query failed, falling back to nvidia-smi: {e}")

        # Fallback to nvidia-smi
        return self._get_gpu_info_nvidia_smi()

    def _get_gpu_info_pynvml(self) -> List[GPUInfo]:
        """Get GPU info using pynvml."""
        gpu_list = []

        try:
            device_count = pynvml.nvmlDeviceGetCount()

            for i in range(device_count):
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    gpu_info = GPUInfo(index=i)

                    # Get name
                    gpu_info.name = pynvml.nvmlDeviceGetName(handle).decode("utf-8")

                    # Get UUID
                    try:
                        gpu_info.uuid = pynvml.nvmlDeviceGetUUID(handle).decode("utf-8")
                    except Exception:
                        pass

                    # Get memory info
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    gpu_info.memory_total_mb = mem_info.total // (1024 * 1024)
                    gpu_info.memory_free_mb = mem_info.free // (1024 * 1024)

                    # Get temperature
                    try:
                        gpu_info.temperature_c = float(pynvml.nvmlDeviceGetTemperature(handle, 0))
                    except Exception:
                        pass

                    # Get power draw
                    try:
                        power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
                        gpu_info.power_draw_w = power_mw / 1000.0
                    except Exception:
                        pass

                    # Get clock speeds
                    try:
                        gpu_info.clock_mhz = pynvml.nvmlDeviceGetClockInfo(handle, 0)
                        gpu_info.memory_clock_mhz = pynvml.nvmlDeviceGetClockInfo(handle, 1)
                    except Exception:
                        pass

                    # Get ECC errors (if supported)
                    try:
                        ecc_errors = pynvml.nvmlDeviceGetEccMode(handle)
                        gpu_info.ecc_errors_uncorrected = ecc_errors[0]
                    except Exception:
                        pass

                    gpu_list.append(gpu_info)

                except Exception as e:
                    logger.warning(f"Error querying GPU {i}: {e}")

        except Exception as e:
            logger.warning(f"PYNVML error: {e}")

        return gpu_list

    def _get_gpu_info_nvidia_smi(self) -> List[GPUInfo]:
        """Get GPU info using nvidia-smi command."""
        gpu_list = []

        try:
            # Query all GPU info
            cmd = [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,memory.total,memory.free,temperature.gpu,power.draw,clocks.current,clocks.mem",
                "--format=csv,noheader,nounits"
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode != 0:
                logger.warning(f"nvidia-smi failed: {result.stderr}")
                return gpu_list

            for line in result.stdout.strip().split("\n"):
                if not line.strip():
                    continue

                try:
                    parts = [p.strip() for p in line.split(",")]
                    gpu_info = GPUInfo(
                        index=int(parts[0]),
                        name=parts[1],
                        uuid=parts[2] if len(parts) > 2 else None,
                        memory_total_mb=int(float(parts[3])) if len(parts) > 3 else 0,
                        memory_free_mb=int(float(parts[4])) if len(parts) > 4 else 0,
                        temperature_c=float(parts[5]) if len(parts) > 5 else 0.0,
                        power_draw_w=float(parts[6]) if len(parts) > 6 else 0.0,
                        clock_mhz=int(float(parts[7])) if len(parts) > 7 else 0,
                        memory_clock_mhz=int(float(parts[8])) if len(parts) > 8 else 0,
                    )

                    gpu_list.append(gpu_info)
                except Exception as e:
                    logger.warning(f"Error parsing GPU info line: {line}: {e}")

            # Get ECC errors separately
            try:
                cmd_ecc = ["nvidia-smi", "-q", "-d", "ECC"]
                result_ecc = subprocess.run(
                    cmd_ecc,
                    capture_output=True,
                    text=True,
                    timeout=10
                )

                if result_ecc.returncode == 0:
                    self._parse_ecc_errors(result_ecc.stdout, gpu_list)
            except Exception as e:
                logger.debug(f"Could not get ECC errors: {e}")

        except Exception as e:
            logger.exception("Error getting GPU info via nvidia-smi")

        return gpu_list

    @staticmethod
    def _parse_ecc_errors(nvidia_smi_output: str, gpu_list: List[GPUInfo]) -> None:
        """Parse ECC error counts from nvidia-smi output."""
        gpu_idx = 0

        for line in nvidia_smi_output.split("\n"):
            line = line.strip()

            if line.startswith("GPU"):
                # Extract GPU index
                match = re.search(r"GPU (\d+)", line)
                if match:
                    gpu_idx = int(match.group(1))

            # Look for ECC error counts
            if "Single Bit Errors" in line or "Uncorrected" in line:
                match = re.search(r":\s*(\d+)", line)
                if match and gpu_idx < len(gpu_list):
                    gpu_list[gpu_idx].ecc_errors_uncorrected = int(match.group(1))

            if "Double Bit Errors" in line or "Corrected" in line:
                match = re.search(r":\s*(\d+)", line)
                if match and gpu_idx < len(gpu_list):
                    gpu_list[gpu_idx].ecc_errors_corrected = int(match.group(1))

    def get_interconnect_health(self) -> Dict[str, Any]:
        """
        Check NVLink and network fabric health.

        Returns:
            Dictionary with NVLink and InfiniBand status and metrics.
        """
        if not self.enabled:
            return {}

        result = {
            "nvlink_links_total": 0,
            "nvlink_links_active": 0,
            "nvlink_links_degraded": 0,
            "nvlink_bandwidth_achieved_gbps": 0.0,
            "nvlink_bandwidth_expected_gbps": 0.0,
            "ib_ports_total": 0,
            "ib_ports_active": 0,
            "ib_ports_down": 0,
            "ib_symbol_errors": 0,
            "ib_link_downed_count": 0,
            "nccl_all_reduce_bandwidth_gbps": 0.0,
            "fabric_type": "unknown",
            "overall_status": InterconnectStatus.UNKNOWN.value,
        }

        try:
            # Get NVLink status
            nvlink_info = self._get_nvlink_status()
            if nvlink_info:
                result["nvlink_links_total"] = len(nvlink_info)
                result["nvlink_links_active"] = sum(1 for n in nvlink_info if n.status == InterconnectStatus.OK.value)
                result["nvlink_links_degraded"] = sum(1 for n in nvlink_info if n.status == InterconnectStatus.DEGRADED.value)
                result["nvlink_bandwidth_expected_gbps"] = nvlink_info[0].bandwidth_gbps * len(nvlink_info) if nvlink_info else 0.0

            # Get InfiniBand status
            ib_info = self._get_infiniband_status()
            if ib_info:
                result["fabric_type"] = "infiniband"
                result["ib_ports_total"] = ib_info.get("ports_total", 0)
                result["ib_ports_active"] = ib_info.get("ports_active", 0)
                result["ib_ports_down"] = ib_info.get("ports_down", 0)
                result["ib_symbol_errors"] = ib_info.get("symbol_errors", 0)
                result["ib_link_downed_count"] = ib_info.get("link_downed_count", 0)

            # Parse NCCL logs if available
            if self.nccl_debug_log_path and os.path.exists(self.nccl_debug_log_path):
                nccl_info = self._parse_nccl_logs(self.nccl_debug_log_path)
                result["nccl_all_reduce_bandwidth_gbps"] = nccl_info.get("all_reduce_bandwidth_gbps", 0.0)

            # Determine overall status
            if result["ib_ports_down"] > 0 or result["nvlink_links_degraded"] > result["nvlink_links_total"] / 2:
                result["overall_status"] = InterconnectStatus.DEGRADED.value
            elif result["nvlink_links_total"] > 0 or result["ib_ports_total"] > 0:
                result["overall_status"] = InterconnectStatus.OK.value

            result["timestamp"] = datetime.utcnow().isoformat()

        except Exception as e:
            logger.exception("Error getting interconnect health")
            result["error"] = str(e)

        return result

    def _get_nvlink_status(self) -> List[NVLinkInfo]:
        """Query NVLink status via nvidia-smi nvlink."""
        nvlink_list = []

        try:
            cmd = ["nvidia-smi", "nvlink", "-s"]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode != 0:
                logger.debug(f"nvidia-smi nvlink failed: {result.stderr}")
                return nvlink_list

            # Parse output for NVLink status
            for line in result.stdout.split("\n"):
                line = line.strip()
                if "Link" in line and ("GPU" in line or "Down" in line or "Up" in line):
                    # Try to extract link info
                    nvlink_info = NVLinkInfo()

                    if "Down" in line:
                        nvlink_info.status = InterconnectStatus.DOWN.value
                    elif "Up" in line or "Ok" in line:
                        nvlink_info.status = InterconnectStatus.OK.value

                    nvlink_list.append(nvlink_info)

        except Exception as e:
            logger.warning(f"Error getting NVLink status: {e}")

        return nvlink_list

    def _get_infiniband_status(self) -> Dict[str, Any]:
        """Query InfiniBand status via ibstat/ibstatus commands."""
        ib_info = {
            "ports_total": 0,
            "ports_active": 0,
            "ports_down": 0,
            "symbol_errors": 0,
            "link_downed_count": 0,
            "ports": [],
        }

        try:
            # Try ibstat first
            cmd = ["ibstat"]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode == 0:
                ib_info = self._parse_ibstat_output(result.stdout, ib_info)
            else:
                # Try ibstatus
                cmd = ["ibstatus"]
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=10
                )

                if result.returncode == 0:
                    ib_info = self._parse_ibstatus_output(result.stdout, ib_info)

        except FileNotFoundError:
            logger.debug("InfiniBand tools (ibstat/ibstatus) not found")
        except Exception as e:
            logger.warning(f"Error getting InfiniBand status: {e}")

        return ib_info

    @staticmethod
    def _parse_ibstat_output(output: str, ib_info: Dict[str, Any]) -> Dict[str, Any]:
        """Parse ibstat command output."""
        for line in output.split("\n"):
            line = line.strip()

            if "State:" in line or "PhysicalState:" in line:
                if "Active" in line:
                    ib_info["ports_active"] += 1
                elif "Down" in line:
                    ib_info["ports_down"] += 1
                ib_info["ports_total"] += 1

            if "SymbolErrorCounter:" in line or "RcvErrors:" in line:
                match = re.search(r":\s*(\d+)", line)
                if match:
                    ib_info["symbol_errors"] += int(match.group(1))

        return ib_info

    @staticmethod
    def _parse_ibstatus_output(output: str, ib_info: Dict[str, Any]) -> Dict[str, Any]:
        """Parse ibstatus command output."""
        for line in output.split("\n"):
            line = line.strip()

            if "port_state:" in line.lower():
                if "active" in line.lower():
                    ib_info["ports_active"] += 1
                    ib_info["ports_total"] += 1
                elif "down" in line.lower():
                    ib_info["ports_down"] += 1
                    ib_info["ports_total"] += 1

        return ib_info

    def _parse_nccl_logs(self, log_path: str) -> Dict[str, Any]:
        """Parse NCCL debug logs for communication diagnostics."""
        nccl_info = {
            "all_reduce_bandwidth_gbps": 0.0,
            "timeout_count": 0,
            "errors": [],
            "messages": [],
        }

        try:
            if not os.path.exists(log_path):
                logger.debug(f"NCCL log not found: {log_path}")
                return nccl_info

            with open(log_path, "r") as f:
                for line in f:
                    line = line.strip()

                    # Look for bandwidth info
                    if "bandwidth" in line.lower():
                        match = re.search(r"([\d.]+)\s*(gb/s|gbps|gb/sec)", line, re.IGNORECASE)
                        if match:
                            nccl_info["all_reduce_bandwidth_gbps"] = float(match.group(1))

                    # Look for timeouts
                    if "timeout" in line.lower():
                        nccl_info["timeout_count"] += 1
                        nccl_info["messages"].append(line)

                    # Look for errors
                    if "error" in line.lower() or "fail" in line.lower():
                        nccl_info["errors"].append(line)

        except Exception as e:
            logger.warning(f"Error parsing NCCL logs: {e}")

        return nccl_info

    def _query_slurm_nodes(self) -> List[Dict[str, Any]]:
        """Query SLURM for node status if available."""
        nodes = []

        if not self.slurm_enabled:
            return nodes

        try:
            cmd = ["sinfo", "-h", "-o", "%N %t %e %A"]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode == 0:
                for line in result.stdout.split("\n"):
                    if not line.strip():
                        continue

                    parts = line.split()
                    if len(parts) >= 3:
                        nodes.append({
                            "hostname": parts[0],
                            "state": parts[1],
                            "reason": parts[2] if len(parts) > 2 else "",
                        })

        except Exception as e:
            logger.warning(f"Error querying SLURM: {e}")

        return nodes

    def _query_fabric_manager(self, endpoint: str) -> Dict[str, Any]:
        """Query NVIDIA Fabric Manager API for fabric health."""
        fabric_info = {
            "status": InterconnectStatus.UNKNOWN.value,
            "links": 0,
            "bandwidth_gbps": 0.0,
        }

        try:
            import requests
            response = requests.get(f"{endpoint}/api/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                fabric_info.update(data)
        except ImportError:
            logger.debug("requests library not available for Fabric Manager query")
        except Exception as e:
            logger.warning(f"Error querying Fabric Manager: {e}")

        return fabric_info

    def get_distributed_training_readiness(self) -> Dict[str, Any]:
        """
        Check if cluster is ready for distributed training.

        Returns:
            Dictionary with readiness status and recommendations.
        """
        if not self.enabled:
            return {"ready": False, "error": "Cluster monitor disabled"}

        try:
            node_health = self.get_node_health()
            interconnect = self.get_interconnect_health()

            issues = []
            available_gpus = sum(n.gpu_count for n in node_health if n.status == NodeStatus.HEALTHY.value)

            # Check node health
            unhealthy_nodes = [n.hostname for n in node_health if n.status != NodeStatus.HEALTHY.value]
            if unhealthy_nodes:
                issues.append(f"Unhealthy nodes: {', '.join(unhealthy_nodes)}")

            # Check interconnect
            if interconnect.get("overall_status") == InterconnectStatus.DOWN.value:
                issues.append("Interconnect is down")
            elif interconnect.get("overall_status") == InterconnectStatus.DEGRADED.value:
                issues.append("Interconnect is degraded")

            # Check NCCL
            try:
                nccl_version = self._get_nccl_version()
                cuda_version = self._get_cuda_version()
                driver_version = self._get_driver_version()
            except Exception as e:
                logger.warning(f"Could not get version info: {e}")
                nccl_version = "unknown"
                cuda_version = "unknown"
                driver_version = "unknown"

            # Get network latency if possible
            network_latency_us = self._measure_network_latency() if len(self.cluster_nodes) > 1 else 0.0

            ready = len(issues) == 0 and available_gpus > 0

            # Recommend parallelism strategy
            if available_gpus >= 8:
                parallelism = "distributed_data_parallel"
            elif available_gpus >= 4:
                parallelism = "distributed_data_parallel or tensor_parallel"
            else:
                parallelism = "single_node_data_parallel"

            return {
                "ready": ready,
                "issues": issues,
                "available_gpus": available_gpus,
                "healthy_nodes": len([n for n in node_health if n.status == NodeStatus.HEALTHY.value]),
                "nccl_version": nccl_version,
                "cuda_version": cuda_version,
                "driver_version": driver_version,
                "network_latency_us": network_latency_us,
                "recommended_parallelism_strategy": parallelism,
                "interconnect_status": interconnect.get("overall_status", "unknown"),
                "timestamp": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            logger.exception("Error checking distributed training readiness")
            return {
                "ready": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
            }

    def detect_cluster_issues(self) -> List[Dict[str, Any]]:
        """
        Detect hardware and communication issues in the cluster.

        Returns:
            List of detected issues with severity and recommendations.
        """
        if not self.enabled:
            return []

        issues = []

        try:
            node_health = self.get_node_health()
            interconnect = self.get_interconnect_health()

            # Check for GPU ECC errors trending up
            for node in node_health:
                if node.ecc_errors_uncorrected_total > 0:
                    issues.append({
                        "type": "gpu_ecc_errors",
                        "severity": "high" if node.ecc_errors_uncorrected_total > 10 else "medium",
                        "node": node.hostname,
                        "description": f"Uncorrected ECC errors detected: {node.ecc_errors_uncorrected_total}",
                        "recommendation": "Run comprehensive GPU diagnostics or replace GPU",
                    })

                # Check for thermal throttling
                if node.temperature_max_c > 85:
                    issues.append({
                        "type": "thermal_throttling",
                        "severity": "high" if node.temperature_max_c > 90 else "medium",
                        "node": node.hostname,
                        "description": f"GPU temperature high: {node.temperature_max_c:.1f}C",
                        "recommendation": "Check cooling system and reduce workload",
                    })

                # Check for unhealthy GPUs
                if node.gpus_with_errors > 0:
                    issues.append({
                        "type": "gpu_errors",
                        "severity": "medium",
                        "node": node.hostname,
                        "description": f"{node.gpus_with_errors} GPUs with errors",
                        "recommendation": "Investigate GPU errors or replace affected GPUs",
                    })

            # Check for NVLink degradation
            if interconnect.get("nvlink_links_degraded", 0) > 0:
                issues.append({
                    "type": "nvlink_degradation",
                    "severity": "medium",
                    "description": f"NVLink links degraded: {interconnect['nvlink_links_degraded']}/{interconnect['nvlink_links_total']}",
                    "recommendation": "Check GPU connectivity and reseat if necessary",
                })

            # Check for InfiniBand issues
            if interconnect.get("ib_ports_down", 0) > 0:
                issues.append({
                    "type": "infiniband_port_down",
                    "severity": "high",
                    "description": f"InfiniBand ports down: {interconnect['ib_ports_down']}",
                    "recommendation": "Check network cables and switch configuration",
                })

            if interconnect.get("ib_symbol_errors", 0) > 100:
                issues.append({
                    "type": "infiniband_symbol_errors",
                    "severity": "medium",
                    "description": f"InfiniBand symbol errors: {interconnect['ib_symbol_errors']}",
                    "recommendation": "Check network cables and signal quality",
                })

            # Check for straggler nodes
            if len(node_health) > 1:
                healthy_nodes = [n for n in node_health if n.status == NodeStatus.HEALTHY.value]
                if healthy_nodes:
                    avg_gpu_count = sum(n.gpu_count for n in healthy_nodes) / len(healthy_nodes)
                    for node in node_health:
                        if node.gpu_count < avg_gpu_count * 0.5:
                            issues.append({
                                "type": "straggler_node",
                                "severity": "low",
                                "node": node.hostname,
                                "description": f"Node has fewer GPUs than peers: {node.gpu_count}",
                                "recommendation": "Consider rebalancing workload",
                            })

            # Check NCCL logs
            if self.nccl_debug_log_path:
                nccl_info = self._parse_nccl_logs(self.nccl_debug_log_path)
                if nccl_info.get("timeout_count", 0) > 0:
                    issues.append({
                        "type": "nccl_timeout",
                        "severity": "high",
                        "description": f"NCCL timeouts detected: {nccl_info['timeout_count']}",
                        "recommendation": "Check network connectivity and NCCL timeout settings",
                    })

                if nccl_info.get("errors"):
                    issues.append({
                        "type": "nccl_error",
                        "severity": "high",
                        "description": f"NCCL errors detected: {len(nccl_info['errors'])} errors",
                        "recommendation": "Check NCCL logs and network configuration",
                    })

        except Exception as e:
            logger.exception("Error detecting cluster issues")
            issues.append({
                "type": "detection_error",
                "severity": "low",
                "description": f"Error during issue detection: {str(e)}",
                "recommendation": "Check monitor logs",
            })

        return issues

    def _get_nccl_version(self) -> str:
        """Get NCCL version."""
        try:
            result = subprocess.run(
                ["python", "-c", "import torch; print(torch.cuda.nccl.version())"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass

        return "unknown"

    def _get_cuda_version(self) -> str:
        """Get CUDA version."""
        try:
            result = subprocess.run(
                ["nvidia-smi"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                match = re.search(r"CUDA Version: ([\d.]+)", result.stdout)
                if match:
                    return match.group(1)
        except Exception:
            pass

        return "unknown"

    def _get_driver_version(self) -> str:
        """Get GPU driver version."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return result.stdout.strip().split("\n")[0]
        except Exception:
            pass

        return "unknown"

    def _measure_network_latency(self) -> float:
        """Measure inter-node network latency."""
        if len(self.cluster_nodes) < 2:
            return 0.0

        latencies = []

        for node in self.cluster_nodes[1:]:
            try:
                result = subprocess.run(
                    ["ping", "-c", "1", "-W", "1", node],
                    capture_output=True,
                    text=True,
                    timeout=5
                )

                if result.returncode == 0:
                    match = re.search(r"time=([\d.]+)\s*ms", result.stdout)
                    if match:
                        latencies.append(float(match.group(1)) * 1000)  # Convert to microseconds
            except Exception:
                pass

        return sum(latencies) / len(latencies) if latencies else 0.0

    def __del__(self):
        """Cleanup resources."""
        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass


def main():
    """Example usage of GPUClusterMonitor."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    monitor = GPUClusterMonitor()

    print("\n=== Cluster Topology ===")
    topology = monitor.get_cluster_topology()
    print(json.dumps(topology, indent=2))

    print("\n=== Node Health ===")
    node_health = monitor.get_node_health()
    for node in node_health:
        print(f"{node.hostname}: {node.status} ({node.gpu_count} GPUs)")

    print("\n=== Interconnect Health ===")
    interconnect = monitor.get_interconnect_health()
    print(json.dumps(interconnect, indent=2))

    print("\n=== Distributed Training Readiness ===")
    readiness = monitor.get_distributed_training_readiness()
    print(json.dumps(readiness, indent=2))

    print("\n=== Cluster Issues ===")
    issues = monitor.detect_cluster_issues()
    for issue in issues:
        print(json.dumps(issue, indent=2))


if __name__ == "__main__":
    main()
