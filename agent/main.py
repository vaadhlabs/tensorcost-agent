#!/usr/bin/env python3
"""
Unified GPU Cost Optimization Agent — TensorCost edition.

This file was rewritten as Phase 2 of the legacy → apps-new migration
("Path C" in `apps-new/docs/AGENT_COMPATIBILITY.md`). The new gateway
accepts a much narrower event surface than the legacy backend did, so
several monitoring cycles that the agent shipped have been switched
off here. The collectors still exist on disk (under `src/monitors/`)
so we can revive them if a future apps-new release adds the matching
ingest endpoints — but they no longer push to the wire.

Dropped event types (per AGENT_COMPATIBILITY.md §2.3 "Dropped Event
Types") — these are server-driven now or fed by separate ingest
pipelines:

  - costs                 cost ingest is server-pulled, not agent-pushed
  - ai_spend              ai-service has its own ingest controller
  - inference_metrics     no agent path on apps-new
  - training_runs         ai-service.training_run is filled by integration
  - gateway_metrics       no agent path on apps-new
  - cluster_health        monitoring-service is server-driven
  - alerts (generic)      apps-new evaluates alerts server-side
  - recommendations       cost-service computes them server-side
  - errors                no error-ingest endpoint; OTel traces only

Spot interruption events still flow — but via the dedicated
`/api/sync/spot-interruption` endpoint, NOT the legacy `alerts`
envelope. See AGENT_COMPATIBILITY.md §2.3 for the full table.
"""

import os
import socket
import sys
import time
import logging
import schedule
import uuid
import yaml
import json
import signal
import threading
import concurrent.futures
from collections import deque
from dotenv import load_dotenv
from typing import Dict, Any, List

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Monitor imports are intentionally LAZY — performed inside the init
# branches that actually instantiate each monitor, not here at module
# load. Each monitor pulls in a cloud-specific SDK (boto3, azure-*,
# google-cloud-*, kubernetes, etc.), and the slim per-cloud container
# variants (tensorcost/gpu-agent:{aws,azure,gcp,node}) only install one
# of those SDK sets. Importing everything at the top would blow up
# `import main` on every variant except `:full` — which is exactly the
# smoke test failure we fixed here.
from alerting import AlertManager
from nvml_sampler import NvmlSampler
from grpc_client import GrpcClient
from spot_handler import SpotInterruptionHandler
from transport.http_client import HttpSyncClient, filter_tags
from transport.tls_pinning import ConfigError as TlsConfigError, load_tls_config_or_die

# Load environment variables
load_dotenv()

# Initialize OpenTelemetry tracing before other imports that get instrumented
if os.getenv('OTEL_ENABLED', '').lower() == 'true':
    from tracing import init_tracing, shutdown_tracing
    init_tracing()

# Configure logging. Ensure the LOG_FILE's parent dir exists before the
# FileHandler is constructed — otherwise importing this module on a
# fresh checkout (CI runners, test collection, `python -c "import main"`)
# crashes with FileNotFoundError before any test can mock the handler.
# The runtime `start()` path also calls os.makedirs('./logs', exist_ok=True),
# but that's too late for module-load-time handler construction.
_log_file = os.getenv('LOG_FILE', './logs/unified_agent.log')
_log_dir = os.path.dirname(_log_file) or '.'
if _log_dir and _log_dir != '.':
    os.makedirs(_log_dir, exist_ok=True)
logging.basicConfig(
    level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO')),
    format=os.getenv('LOG_FORMAT', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
    handlers=[
        logging.FileHandler(_log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

AGENT_VERSION = "unified-gpu-agent/2.0.0"

# Actuator gates (see _execute_command_inner). Both checks fail closed.
_ACTUATOR_ACTION_NAMES = frozenset({'stop', 'start', 'resize', 'restart', 'terminate'})


def _actuator_enabled() -> bool:
    return os.getenv('AGENT_ACTUATOR_ENABLED', '').strip().lower() == 'true'


def _actuator_allowed_actions() -> frozenset:
    raw = os.getenv('AGENT_ACTUATOR_ALLOWED_ACTIONS', '')
    parts = {p.strip().lower() for p in raw.split(',') if p.strip()}
    return frozenset(parts & _ACTUATOR_ACTION_NAMES)


class UnifiedGPUAgent:
    def __init__(self, config_path: str = None):
        """Initialize unified GPU agent"""
        self.config = self._load_config(config_path)
        self.monitors = {}
        self.alert_manager = AlertManager()
        self.ai_spend_monitor = None

        # Backend wire config — env-driven.
        self.backend_api_url = os.getenv('BACKEND_API_URL')
        # NEW header: X-Agent-Key (plaintext, bcrypt-validated by gpu-service).
        # Provisioning CLI emits this as AGENT_API_KEY.
        self.agent_api_key = os.getenv('AGENT_API_KEY') or os.getenv('BACKEND_API_KEY')
        self.tenant_id = os.getenv('TENANT_ID', 'default')
        # Hostname — server keys gpu.agent rows on (tenant_id, hostname).
        # Falls back to AGENT_ID for legacy compose files.
        self.hostname = (
            os.getenv('AGENT_HOSTNAME')
            or os.getenv('AGENT_ID')
            or f'agent-{self.tenant_id}'
        )

        # Per-host instance identity for multi-instance deployments. One
        # credential bundle can run on N hosts (k8s DaemonSet, autoscale);
        # each host needs a stable, unique id so the SaaS can track them
        # individually in gpu.agent_instance(agent_id, instance_id).
        #
        # Resolution order:
        #   1. AGENT_INSTANCE_ID env var (operator-explicit)
        #   2. /app/data/instance_id file (persisted from a prior boot)
        #   3. Auto-generate: "{hostname}-{8-char uuid hex}" and persist it
        #
        # The auto-generated value blends the OS hostname with a short
        # random suffix so containers sharing a host (same HOSTNAME env)
        # still get distinct ids. The 8-char hex gives ~4 billion distinct
        # values per hostname, enough collision-resistance for any single
        # tenant fleet. The id is written to /app/data/instance_id on
        # first boot so restarts reuse the same identity.
        self.instance_id = self._resolve_instance_id()

        # Cloud-provider context for the health DTO — the agent host's
        # identity. SINGLE SOURCE OF TRUTH (audit §8.3 fix): resolved
        # exactly once here, in priority order
        #   1. operator env ``AGENT_CLOUD_PROVIDER`` (explicit wins),
        #   2. IMDS auto-detection (`src.cloud_identity`),
        #   3. ``onprem`` fallback when IMDS times out / no env set.
        # Downstream code reads ``self.cloud_provider`` and never
        # re-derives host identity from the YAML monitor enable flags
        # (which describe which EXTERNAL clouds to scrape, not what the
        # agent IS — see `_initialize_monitors` docstring). Without this
        # discipline a customer with `AWS_ENABLED=true` running on Azure
        # would silently mis-stamp their own heartbeat as `aws`.
        # 26-FCR §3.5 — without the IMDS fallback, the operator who
        # forgets the env at provisioning time produces metric streams
        # tagged "unknown" for both provider AND instance_id, which
        # collides downstream against the (tenant_id, cloud_provider,
        # instance_id) UNIQUE index and silently drops ingest.
        self.cloud_provider = os.getenv('AGENT_CLOUD_PROVIDER')
        self.region = os.getenv('AGENT_REGION') or os.getenv('AWS_REGION')
        self.cloud_id = os.getenv('AGENT_CLOUD_ID')

        if not self.cloud_provider:
            try:
                from src.cloud_identity import detect_cloud_identity
                discovered = detect_cloud_identity()
            except Exception:  # noqa: BLE001 — never block startup on this
                discovered = None
            if discovered:
                self.cloud_provider = discovered.cloud_provider
                # Don't overwrite operator-set region/id — only fill gaps.
                self.cloud_id = self.cloud_id or discovered.cloud_id
                self.region = self.region or discovered.region
            else:
                self.cloud_provider = 'onprem'

        # Track previously seen instances per cloud to detect disappearances
        self._known_instances = {}  # {cloud_name: set(instance_ids)}

        # Graceful shutdown event
        self._shutdown_event = None

        # HTTP failure queue for metric buffering. The deque silently
        # drops the oldest item when `maxlen` is hit — historically that
        # was operationally invisible (audit §8.3). `_send_queue_append`
        # wraps every append site so we emit threshold logs at 80% / 95%
        # / 100% of capacity, rate-limited per-threshold so a sustained
        # outage doesn't flood the log file.
        self._send_queue_maxlen = 10000
        self._send_queue = deque(maxlen=self._send_queue_maxlen)
        self._queue_file = os.path.join(os.path.dirname(__file__), '.send_queue.json')
        # Per-threshold (key) last-emitted monotonic timestamp; the
        # rate-limit window is `_send_queue_log_interval_s`.
        self._send_queue_threshold_last_log = {'warn80': 0.0, 'err95': 0.0, 'sat100': 0.0}
        self._send_queue_log_interval_s = float(
            os.getenv('AGENT_SEND_QUEUE_LOG_INTERVAL_SEC', '60')
        )
        # Counter of oldest-item evictions caused by overflow. Surfaced
        # as an OTel span attribute when tracing is wired.
        self._send_queue_dropped_total = 0
        self._load_queue()

        from exporters.orchestrator import build_metric_exporter

        self.metric_exporter = build_metric_exporter(
            region=self.region or "",
            provider=self.cloud_provider or "",
        )

        # Communication mode: "http" (default), "grpc", or "both"
        self.comm_mode = os.getenv('COMM_MODE', 'http').lower()

        # TLS / mTLS posture (§8.2 Blocker #6). load_tls_config_or_die
        # parses AGENT_TLS_PIN_SHA256_HTTPS / _GRPC, AGENT_TLS_CA_BUNDLE,
        # AGENT_MTLS_CLIENT_CERT, AGENT_MTLS_CLIENT_KEY. With nothing
        # set, returns an empty TlsConfig and behaviour is identical to
        # pre-blocker. With pins set, both planes enforce SPKI pinning;
        # bad config (invalid base64, missing files, half-set mTLS pair,
        # http:// URL with pinning on) fails fast before either client
        # is built.
        try:
            self._tls_config = load_tls_config_or_die(
                backend_api_url=self.backend_api_url,
                grpc_target=os.getenv('GRPC_TARGET'),
            )
        except TlsConfigError:
            # Re-raise — TLS misconfiguration is a hard boot failure;
            # falling back to system roots when the operator asked for
            # pinning would defeat the point.
            raise

        # HTTP transport client (used as primary in `http`/`both` modes,
        # fallback in `grpc` mode).
        self.http = None
        if self.backend_api_url and self.agent_api_key:
            try:
                self.http = HttpSyncClient(
                    base_url=self.backend_api_url,
                    api_key=self.agent_api_key,
                    tls_config=self._tls_config,
                    instance_id=self.instance_id,
                )
            except ValueError as e:
                logger.error(f"HTTP client init failed: {e}")

        # ACTIVE_MONITORS may set NVML_ENABLED before sampler init.
        self._apply_active_monitors(self.config.get('monitoring', {}))

        # NVML sampler for local GPU monitoring
        self.nvml_sampler = None
        if os.getenv('NVML_ENABLED', 'true').lower() == 'true':
            try:
                self.nvml_sampler = NvmlSampler(
                    sample_interval=int(os.getenv('NVML_SAMPLE_INTERVAL', 10)),
                    idle_gpu_threshold=float(os.getenv('NVML_IDLE_GPU_THRESHOLD', 10)),
                    idle_memory_threshold=float(os.getenv('NVML_IDLE_MEMORY_THRESHOLD', 10)),
                    idle_duration_threshold=int(os.getenv('NVML_IDLE_DURATION', 120)),
                )
                logger.info("NVML sampler initialized")
            except Exception as e:
                logger.error(f"Failed to initialize NVML sampler: {e}")

        # gRPC client for the new TensorCost AgentService.Connect stream.
        self.grpc_client = None
        if self.comm_mode in ('grpc', 'both'):
            grpc_target = os.getenv('GRPC_TARGET')
            key_id = os.getenv('AGENT_KEY_ID')
            hmac_pepper = os.getenv('AGENT_HMAC_PEPPER')
            if grpc_target and key_id and hmac_pepper:
                try:
                    # use_tls=None hands off to GrpcClient's TLS auto-detect:
                    # honors TC_AGENT_TLS=true|false|auto (default auto =
                    # TLS unless target is a loopback host). The legacy
                    # GRPC_USE_TLS=false default would silently fail
                    # against any production NLB with ACM certs (which
                    # is every customer deployment). Operators who need
                    # plaintext (stunnel sidecar etc.) can still set
                    # TC_AGENT_TLS=false explicitly.
                    self.grpc_client = GrpcClient(
                        grpc_target=grpc_target,
                        agent_id=self.hostname,
                        tenant_id=self.tenant_id,
                        key_id=key_id,
                        hmac_pepper=hmac_pepper,
                        on_command=self._execute_command,
                        use_tls=None,
                        ca_cert_path=os.getenv('GRPC_CA_CERT') or None,
                        tls_config=self._tls_config,
                        instance_id=self.instance_id,
                    )
                    logger.info(f"gRPC client initialized, target={grpc_target}")
                except Exception as e:
                    logger.error(f"Failed to initialize gRPC client: {e}")
            else:
                missing = [
                    name for name, val in
                    (('GRPC_TARGET', grpc_target), ('AGENT_KEY_ID', key_id),
                     ('AGENT_HMAC_PEPPER', hmac_pepper))
                    if not val
                ]
                logger.warning(
                    "COMM_MODE includes grpc but missing %s — gRPC disabled",
                    ', '.join(missing),
                )

        # Spot Interruption Handler (AWS only, when running on EC2)
        self.spot_handler = None
        if os.getenv('SPOT_HANDLER_ENABLED', 'false').lower() == 'true':
            try:
                self.spot_handler = SpotInterruptionHandler(
                    on_interruption=self._handle_spot_interruption,
                    on_checkpoint=self._handle_spot_checkpoint,
                    poll_interval=int(os.getenv('SPOT_POLL_INTERVAL', 5)),
                )
                logger.info("Spot interruption handler initialized")
            except Exception as e:
                logger.error(f"Failed to initialize spot handler: {e}")

        # Initialize monitors based on configuration
        self._initialize_monitors()

        # AI spend / inference / training / LLM-gateway / cluster monitors —
        # the collectors still load (so customer config doesn't blow up on
        # boot), but their output is no longer pushed anywhere. See the
        # top-of-file comment block + AGENT_COMPATIBILITY.md §2.3 for why.
        # When/if apps-new adds the matching ingest endpoints, re-enable
        # the corresponding cycles inside `run_monitoring_cycle()`.

        if os.getenv('AI_SPEND_ENABLED', 'false').lower() == 'true':
            try:
                from monitors.ai_spend_monitor import AISpendMonitor
                self.ai_spend_monitor = AISpendMonitor()
                logger.info("AI Spend Monitor initialized (collector-only — no ingest)")
            except Exception as e:
                logger.error(f"Failed to initialize AI Spend Monitor: {e}")

        self.inference_server_monitor = None
        if os.getenv('INFERENCE_SERVER_ENABLED', 'false').lower() == 'true':
            try:
                from monitors.inference_server_monitor import InferenceServerMonitor
                self.inference_server_monitor = InferenceServerMonitor()
                logger.info("Inference Server Monitor initialized (collector-only — no ingest)")
            except Exception as e:
                logger.error(f"Failed to initialize Inference Server Monitor: {e}")

        self.training_run_monitor = None
        if os.getenv('TRAINING_MONITOR_ENABLED', 'false').lower() == 'true':
            try:
                from monitors.training_run_monitor import TrainingRunMonitor
                self.training_run_monitor = TrainingRunMonitor()
                logger.info("Training Run Monitor initialized (collector-only — no ingest)")
            except Exception as e:
                logger.error(f"Failed to initialize Training Run Monitor: {e}")

        self.llm_gateway_monitor = None
        if os.getenv('LLM_GATEWAY_ENABLED', 'false').lower() == 'true':
            try:
                from monitors.llm_gateway_monitor import LLMGatewayMonitor
                self.llm_gateway_monitor = LLMGatewayMonitor()
                logger.info("LLM Gateway Monitor initialized (collector-only — no ingest)")
            except Exception as e:
                logger.error(f"Failed to initialize LLM Gateway Monitor: {e}")

        self.gpu_cluster_monitor = None
        if os.getenv('CLUSTER_MONITOR_ENABLED', 'false').lower() == 'true':
            try:
                from monitors.gpu_cluster_monitor import GPUClusterMonitor
                self.gpu_cluster_monitor = GPUClusterMonitor()
                logger.info("GPU Cluster Monitor initialized (collector-only — no ingest)")
            except Exception as e:
                logger.error(f"Failed to initialize GPU Cluster Monitor: {e}")

    # ------------------------------------------------------------------
    # Instance-id resolution
    # ------------------------------------------------------------------

    _INSTANCE_ID_FILE = '/app/data/instance_id'

    def _resolve_instance_id(self) -> str:
        """Return the stable per-host instance id, creating it if needed.

        Persistence contract:
          - If AGENT_INSTANCE_ID is set, use it (skip the file entirely
            so the operator's explicit value always wins on every boot).
          - If /app/data/instance_id exists, read and return its content
            so container restarts reuse the same identity.
          - Otherwise generate "{hostname}-{8-char uuid hex}", write it to
            the file, log it at INFO so operators can see what their fleet
            members will report under, and return it.

        Write failures are non-fatal — the generated value is still used
        for the current run; it just won't survive the next restart.
        """
        explicit = os.getenv('AGENT_INSTANCE_ID')
        if explicit:
            return explicit.strip()

        # Try to read from the persistence file.
        try:
            with open(self._INSTANCE_ID_FILE, 'r') as fh:
                stored = fh.read().strip()
            if stored:
                return stored
        except FileNotFoundError:
            pass
        except Exception as e:
            logger.warning("Could not read instance_id file %s: %s", self._INSTANCE_ID_FILE, e)

        # Auto-generate and persist.
        generated = f"{socket.gethostname()}-{uuid.uuid4().hex[:8]}"
        logger.info(
            "AGENT_INSTANCE_ID not set — using auto-generated instance_id=%s "
            "(set AGENT_INSTANCE_ID to make this explicit)",
            generated,
        )
        try:
            os.makedirs(os.path.dirname(self._INSTANCE_ID_FILE), exist_ok=True)
            with open(self._INSTANCE_ID_FILE, 'w') as fh:
                fh.write(generated)
        except Exception as e:
            logger.warning("Could not persist instance_id to %s: %s", self._INSTANCE_ID_FILE, e)

        return generated

    def _load_config(self, config_path: str = None) -> Dict[str, Any]:
        """Load configuration from file or environment"""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        else:
            config = self._load_config_from_env()
        return config

    def _load_config_from_env(self) -> Dict[str, Any]:
        """Load configuration from environment variables"""
        config = {
            'monitoring': {
                'enabled_clouds': os.getenv('ENABLED_CLOUDS', 'aws').split(','),
                'interval': int(os.getenv('MONITORING_INTERVAL', 300)),
                'aws': {
                    'enabled': os.getenv('AWS_ENABLED', 'false').lower() == 'true',
                    'region': os.getenv('AWS_REGION', 'us-east-1'),
                    'services': os.getenv('AWS_SERVICES', 'ec2,sagemaker').split(',')
                },
                'azure': {
                    'enabled': os.getenv('AZURE_ENABLED', 'false').lower() == 'true',
                    'subscription_id': os.getenv('AZURE_SUBSCRIPTION_ID'),
                    'resource_group': os.getenv('AZURE_RESOURCE_GROUP'),
                    'services': os.getenv('AZURE_SERVICES', 'virtual_machines').split(',')
                },
                'gcp': {
                    'enabled': os.getenv('GCP_ENABLED', 'false').lower() == 'true',
                    'project_id': os.getenv('GCP_PROJECT_ID'),
                    'services': os.getenv('GCP_SERVICES', 'compute_engine').split(',')
                },
                'kubernetes': {
                    'enabled': os.getenv('K8S_ENABLED', 'false').lower() == 'true',
                    'kubeconfig_path': os.getenv('KUBECONFIG_PATH', '~/.kube/config'),
                    'namespaces': os.getenv('K8S_NAMESPACES', 'default').split(',')
                },
                'sagemaker': {
                    'enabled': os.getenv('SAGEMAKER_ENABLED', 'false').lower() == 'true',
                    'region': os.getenv('AWS_REGION', 'us-east-1'),
                    'services': os.getenv('SAGEMAKER_SERVICES', 'training_jobs,endpoints').split(',')
                }
            },
            'dashboard': {
                'api_url': os.getenv('DASHBOARD_API_URL'),
                'api_key': os.getenv('DASHBOARD_API_KEY'),
                'sync_enabled': os.getenv('REMOTE_SYNC_ENABLED', 'true').lower() == 'true'
            },
            'alerting': {
                'slack_webhook': os.getenv('SLACK_WEBHOOK_URL'),
                'email_to': os.getenv('EMAIL_TO'),
                'idle_threshold': int(os.getenv('IDLE_GPU_THRESHOLD', 10)),
                'cost_threshold': float(os.getenv('DAILY_COST_THRESHOLD', 100))
            }
        }
        return config

    def _apply_active_monitors(self, monitoring_config: Dict[str, Any]) -> None:
        """Honor ACTIVE_MONITORS when set — comma-separated monitor keys from deploy.

        Keys: aws, azure, gcp, kubernetes/k8s, sagemaker, nvml. When unset,
        per-cloud *_ENABLED flags from config/env remain authoritative.
        """
        raw = os.getenv('ACTIVE_MONITORS', '').strip()
        if not raw:
            return
        keys = {k.strip().lower() for k in raw.split(',') if k.strip()}
        if not keys:
            return
        alias_map = {
            'aws': 'aws',
            'azure': 'azure',
            'gcp': 'gcp',
            'kubernetes': 'kubernetes',
            'k8s': 'kubernetes',
            'sagemaker': 'sagemaker',
        }
        enabled_sections: set[str] = set()
        for key in keys:
            section = alias_map.get(key)
            if section:
                enabled_sections.add(section)
        for section in ('aws', 'azure', 'gcp', 'kubernetes', 'sagemaker'):
            if section in enabled_sections:
                monitoring_config.setdefault(section, {})['enabled'] = True
            elif section in monitoring_config:
                monitoring_config[section]['enabled'] = False
        if 'nvml' in keys:
            os.environ['NVML_ENABLED'] = 'true'
        elif 'nvml' not in keys and raw:
            os.environ['NVML_ENABLED'] = 'false'

    def _initialize_monitors(self):
        """
        Initialize external-resource monitors based on YAML/env config.

        Unified cloud-identity model (audit §8.3 "two parallel models of
        what cloud are we on"):

        * ``self.cloud_provider`` is the **agent host's** identity, set
          once in ``__init__`` from a single source of truth — operator
          env (``AGENT_CLOUD_PROVIDER``) wins, otherwise IMDS detection,
          otherwise ``onprem``. It stamps the ``cloud_provider`` field on
          every metric/instance/health DTO the agent emits and is the
          key for the server's ``(tenant_id, cloud_provider, instance_id)``
          UNIQUE index.

        * ``monitoring.{aws,azure,gcp,kubernetes,sagemaker}.enabled`` are
          per-cloud **resource-collection** flags. They decide which
          external clouds the agent SCRAPES — not what the agent claims
          to BE. This split is intentional: a customer running the agent
          on an Azure VM can legitimately enable the AWS monitor to
          sample EC2 instances they own from that Azure host (cross-cloud
          observability). In that case the host identity remains
          ``azure`` (so the agent's own heartbeat lands on the right
          row) while the AWS monitor still loads and pushes EC2 metrics
          tagged with ``cloud_provider=aws`` per metric.

        Therefore: this function consults the YAML/env enable flags for
        deciding which monitors to instantiate. It never derives the
        host identity from those flags — that responsibility lives
        exclusively in ``__init__``.
        """
        monitoring_config = self.config.get('monitoring', {})
        self._apply_active_monitors(monitoring_config)

        if monitoring_config.get('aws', {}).get('enabled', False):
            try:
                from monitors.aws_monitor import AWSMonitor
                self.monitors['aws'] = AWSMonitor(monitoring_config['aws'])
                logger.info("AWS monitor initialized")
            except Exception as e:
                logger.error(f"Failed to initialize AWS monitor: {e}")

        if monitoring_config.get('azure', {}).get('enabled', False):
            try:
                from monitors.azure_monitor import AzureMonitor
                self.monitors['azure'] = AzureMonitor(monitoring_config['azure'])
                logger.info("Azure monitor initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Azure monitor: {e}")

        if monitoring_config.get('gcp', {}).get('enabled', False):
            try:
                from monitors.gcp_monitor import GCPMonitor
                self.monitors['gcp'] = GCPMonitor(monitoring_config['gcp'])
                logger.info("Google Cloud monitor initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Google Cloud monitor: {e}")

        if monitoring_config.get('kubernetes', {}).get('enabled', False):
            try:
                from monitors.kubernetes_monitor import KubernetesMonitor
                k8s_cfg = monitoring_config.get('kubernetes', {})
                namespaces = k8s_cfg.get('namespaces') or ['default']
                self.monitors['kubernetes'] = KubernetesMonitor(
                    nvml_sampler=self.nvml_sampler,
                    namespaces=namespaces,
                )
                logger.info("Kubernetes monitor initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Kubernetes monitor: {e}")

        if monitoring_config.get('sagemaker', {}).get('enabled', False):
            try:
                from monitors.sagemaker_monitor import SageMakerMonitor
                self.monitors['sagemaker'] = SageMakerMonitor(monitoring_config['sagemaker'])
                logger.info("SageMaker monitor initialized")
            except Exception as e:
                logger.error(f"Failed to initialize SageMaker monitor: {e}")

        # Standalone host GPU telemetry — NVML on bare-metal / VM.
        # Coexists with K8s inventory; operators can force with HOST_GPU_TELEMETRY_ENABLED=true.
        host_gpu_explicit = os.getenv('HOST_GPU_TELEMETRY_ENABLED', '').lower() == 'true'
        nvml_enabled = os.getenv('NVML_ENABLED', 'false').lower() == 'true'
        if self.nvml_sampler and (host_gpu_explicit or nvml_enabled):
            try:
                from monitors.host_gpu_monitor import HostGpuMonitor
                self.monitors['host'] = HostGpuMonitor(
                    hostname=self.hostname,
                    cloud_provider=self.cloud_provider,
                    instance_id=self.instance_id,
                    nvml_sampler=self.nvml_sampler,
                )
                logger.info("Host GPU monitor initialized (NVML path)")
            except Exception as e:
                logger.error(f"Failed to initialize Host GPU monitor: {e}")

        # Agentless DCGM / Prometheus scrape — live sync path (not collector-only).
        dcgm_url = os.getenv('DCGM_EXPORTER_URL') or os.getenv('GPU_PROMETHEUS_URL')
        if dcgm_url:
            try:
                from monitors.dcgm_prometheus_monitor import DcgmPrometheusMonitor
                self.monitors['dcgm'] = DcgmPrometheusMonitor(
                    hostname=self.hostname,
                    cloud_provider=self.cloud_provider,
                    instance_id=self.instance_id,
                    metrics_url=dcgm_url,
                )
                logger.info("DCGM/Prometheus monitor initialized url=%s", dcgm_url)
            except Exception as e:
                logger.error(f"Failed to initialize DCGM monitor: {e}")

        # Always-on host identity for on-prem / neocloud when no hyperscaler scrape is enabled.
        host_identity_default = self.cloud_provider == 'onprem'
        host_identity_enabled = os.getenv(
            'HOST_IDENTITY_ENABLED',
            'true' if host_identity_default else 'false',
        ).lower() == 'true'
        cloud_scrape_enabled = any(
            monitoring_config.get(p, {}).get('enabled', False)
            for p in ('aws', 'azure', 'gcp', 'kubernetes', 'sagemaker')
        )
        if host_identity_enabled and not cloud_scrape_enabled:
            try:
                from monitors.host_identity_monitor import HostIdentityMonitor
                self.monitors['host_identity'] = HostIdentityMonitor(
                    hostname=self.hostname,
                    cloud_provider=self.cloud_provider,
                    instance_id=self.instance_id,
                )
                logger.info("Host identity monitor initialized (no cloud discovery)")
            except Exception as e:
                logger.error(f"Failed to initialize host identity monitor: {e}")

        logger.info(f"Initialized {len(self.monitors)} monitors: {list(self.monitors.keys())}")

    # ------------------------------------------------------------------
    # Persisted send queue — buffers metrics/instances on HTTP failure.
    # ------------------------------------------------------------------

    def _send_queue_append(self, item: Dict[str, Any]) -> None:
        """
        Append `item` to `self._send_queue` with overflow telemetry.

        The underlying `deque(maxlen=N)` silently drops the oldest item
        when full — historically that was operationally invisible
        (audit §8.3 "silent send-queue overflow"). This wrapper:

          * detects the pre-append saturation case and emits an ERROR
            log noting that the oldest item is being evicted;
          * after appending, emits threshold logs at 80% / 95% of
            capacity, rate-limited per-threshold (default: at most one
            log per threshold per ``AGENT_SEND_QUEUE_LOG_INTERVAL_SEC``)
            so a sustained gateway outage doesn't flood the log file;
          * surfaces queue depth + cumulative drops as OpenTelemetry
            span attributes when ``OTEL_ENABLED=true`` is wired (no-op
            otherwise — `tracing.py` keeps the import set lazy).
        """
        # Pre-append saturation — the deque is about to evict its
        # oldest entry to make room. Emit ERROR every time (no rate
        # limit) because each event represents a real data loss.
        if len(self._send_queue) >= self._send_queue_maxlen:
            self._send_queue_dropped_total += 1
            logger.error(
                "send queue saturated — dropping oldest item to accept new "
                "(maxlen=%d, dropped_total=%d, data_type=%s)",
                self._send_queue_maxlen,
                self._send_queue_dropped_total,
                item.get('data_type'),
            )
            self._emit_queue_otel_attrs(saturated=True)

        self._send_queue.append(item)

        depth = len(self._send_queue)
        warn_threshold = int(self._send_queue_maxlen * 0.80)
        err_threshold = int(self._send_queue_maxlen * 0.95)

        # 95% — ERROR, rate-limited.
        if depth >= err_threshold:
            if self._should_log_queue_threshold('err95'):
                logger.error(
                    "send queue near saturation, oldest items will be dropped "
                    "(depth=%d, maxlen=%d, %.0f%% full)",
                    depth, self._send_queue_maxlen,
                    100.0 * depth / self._send_queue_maxlen,
                )
                self._emit_queue_otel_attrs()
        # 80% — WARN, rate-limited. Suppress when 95% already fired.
        elif depth >= warn_threshold:
            if self._should_log_queue_threshold('warn80'):
                logger.warning(
                    "send queue at 80%% capacity "
                    "(depth=%d, maxlen=%d, %.0f%% full)",
                    depth, self._send_queue_maxlen,
                    100.0 * depth / self._send_queue_maxlen,
                )
                self._emit_queue_otel_attrs()

    def _should_log_queue_threshold(self, key: str) -> bool:
        """One-log-per-N-seconds gate for queue threshold messages."""
        now = time.monotonic()
        last = self._send_queue_threshold_last_log.get(key, 0.0)
        if now - last >= self._send_queue_log_interval_s:
            self._send_queue_threshold_last_log[key] = now
            return True
        return False

    def _emit_queue_otel_attrs(self, saturated: bool = False) -> None:
        """
        Emit queue-depth attributes onto the active OTel span if tracing
        is wired and a span is currently in scope. No-op otherwise — the
        OTel SDK is an optional install (see `src/tracing.py`) and we
        never want telemetry to crash the agent loop.
        """
        try:
            from opentelemetry import trace  # noqa: WPS433 — local import
            span = trace.get_current_span()
            if span is None or not span.is_recording():
                return
            span.set_attribute('agent.send_queue.depth', len(self._send_queue))
            span.set_attribute('agent.send_queue.maxlen', self._send_queue_maxlen)
            span.set_attribute(
                'agent.send_queue.dropped_total', self._send_queue_dropped_total
            )
            if saturated:
                span.set_attribute('agent.send_queue.saturated', True)
        except Exception:  # noqa: BLE001 — telemetry must never raise
            pass

    def _load_queue(self):
        """Load persisted queue from disk with corruption recovery."""
        for path in [self._queue_file, self._queue_file + '.bak']:
            try:
                if os.path.exists(path):
                    with open(path, 'r') as f:
                        items = json.load(f)
                    self._send_queue.extend(items)
                    logger.info(f"Loaded {len(self._send_queue)} queued items from {path}")
                    return
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Corrupt queue file {path}: {e}")
            except Exception as e:
                logger.warning(f"Failed to load send queue from {path}: {e}")

    def _persist_queue(self):
        """Persist queue to disk atomically for crash recovery."""
        tmp_path = self._queue_file + '.tmp'
        try:
            with open(tmp_path, 'w') as f:
                json.dump(list(self._send_queue), f)
                f.flush()
                os.fsync(f.fileno())
            if os.path.exists(self._queue_file):
                bak_path = self._queue_file + '.bak'
                try:
                    import shutil
                    shutil.copy2(self._queue_file, bak_path)
                except Exception:
                    pass
            os.replace(tmp_path, self._queue_file)
        except Exception as e:
            logger.warning(f"Failed to persist send queue: {e}")
            try:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            except Exception:
                pass

    def _drain_queue(self):
        """Try to send queued items via HTTP."""
        if not self._send_queue or not self.http:
            return
        sent = 0
        while self._send_queue:
            item = self._send_queue[0]
            data_type = item.get('data_type')
            ok = False
            if data_type == 'metrics':
                ok = self.http.send_metrics(item.get('legacy_metrics', []))
            elif data_type == 'instances':
                ok = self.http.send_instance_updates(item.get('legacy_instances', []))
            if ok:
                self._send_queue.popleft()
                sent += 1
            else:
                break
        if sent > 0:
            logger.info(f"Drained {sent} queued items")
            self._persist_queue()

    # ------------------------------------------------------------------
    # Wire-out helpers — single place that picks gRPC vs HTTP.
    # ------------------------------------------------------------------

    def _send_metrics(self, legacy_metrics: List[Dict[str, Any]]) -> bool:
        """
        Push a list of legacy-shape metric dicts. The transport layer
        translates to the new wire shape (`util_pct`, `mem_pct`,
        `temp_c`, `ts_unix_ms`) and stamps the (cloud_id, cloud_provider)
        identity pair so the server can resolve the metric to a row.
        """
        if not legacy_metrics:
            return True
        if self.metric_exporter:
            self.metric_exporter.on_metrics(legacy_metrics)
        # Try gRPC first when connected.
        if self.grpc_client and self.grpc_client.is_connected:
            try:
                # gRPC client expects already-renamed fields; do the
                # translation once, here, then ship.
                from transport.http_client import rename_metric_to_wire
                wire = [rename_metric_to_wire(m) for m in legacy_metrics]
                wire = [w for w in wire if w is not None]
                # Stamp identity. Each metric carries the agent's host
                # identity (cloud_provider from main.py's IMDS-or-env
                # detection) plus the per-instance cloud_id taken from
                # whichever id field the legacy collector populated —
                # we prefer an explicit `cloud_id`, fall back to the
                # already-translated `instance_id` (which for cloud
                # collectors IS the cloud-native id like `i-0abc…`),
                # finally fall back to self.cloud_id (single-host VM
                # case where the agent IS the instance).
                for w, src in zip(wire, legacy_metrics):
                    if not w.get("cloud_provider"):
                        w["cloud_provider"] = (
                            src.get("cloud_provider") or self.cloud_provider or ""
                        )
                    if not w.get("cloud_id"):
                        w["cloud_id"] = (
                            src.get("cloud_id")
                            or w.get("instance_id")
                            or self.cloud_id
                            or ""
                        )
                self.grpc_client.send_metrics(wire)
                return True
            except Exception as e:
                logger.warning(f"gRPC metric send failed: {e}, falling back to HTTP")
        # HTTP fallback. Stamp the agent's host cloud_provider on each
        # legacy dict so rename_metric_to_wire can produce the
        # (cloud_id, cloud_provider) identity pair the server's
        # MetricPointDto requires when the legacy `instance_id` is a
        # cloud-native id (e.g. `i-0abc…`) rather than our local UUID.
        # Without this stamp the rename returns None and the metric is
        # silently dropped — exactly the bug the post-agent-rewrite-c
        # README §"Known issues" admits today.
        if not self.http:
            return False
        for m in legacy_metrics:
            if not m.get("cloud_provider") and self.cloud_provider:
                m["cloud_provider"] = self.cloud_provider
        ok = self.http.send_metrics(legacy_metrics)
        if not ok:
            self._send_queue_append({'data_type': 'metrics', 'legacy_metrics': legacy_metrics})
            self._persist_queue()
        return ok

    def _send_instances(self, legacy_instances: List[Dict[str, Any]]) -> bool:
        if not legacy_instances:
            return True
        if self.grpc_client and self.grpc_client.is_connected:
            try:
                for inst in legacy_instances:
                    iid = inst.get('instance_id')
                    cloud_id = inst.get('cloud_id') or iid or self.cloud_id or ""
                    cloud_provider = (
                        inst.get('cloud_provider') or self.cloud_provider or ""
                    )
                    if not cloud_id and not iid:
                        continue
                    mig_kwargs: Dict[str, Any] = {}
                    mig_parts = inst.get('mig_partitions')
                    if isinstance(mig_parts, list) and len(mig_parts) > 0:
                        mig_kwargs['mig_partitions'] = mig_parts
                        mig_kwargs['mig_enabled'] = bool(inst.get('mig_enabled', True))
                    # Ship the (cloud_id, cloud_provider) pair so the
                    # server's UPSERT branch creates the gpu.instance
                    # row on first sight. instance_id (UUID) only set
                    # when the caller has it — cloud collectors don't.
                    self.grpc_client.send_instance_update(
                        instance_id="",
                        cloud_id=str(cloud_id),
                        cloud_provider=str(cloud_provider),
                        region=str(inst.get('region') or ''),
                        status=str(inst.get('state') or inst.get('status') or 'running'),
                        tags=filter_tags(inst.get('tags')),
                        **mig_kwargs,
                    )
                return True
            except Exception as e:
                logger.warning(f"gRPC instance send failed: {e}, falling back to HTTP")
        if not self.http:
            return False
        for inst in legacy_instances:
            if not inst.get("cloud_provider") and self.cloud_provider:
                inst["cloud_provider"] = self.cloud_provider
        ok = self.http.send_instance_updates(legacy_instances)
        if not ok:
            self._send_queue_append({'data_type': 'instances', 'legacy_instances': legacy_instances})
            self._persist_queue()
        return ok

    def _send_health_check(self) -> bool:
        # Touch a local heartbeat file before the network call so the Docker
        # HEALTHCHECK reflects loop-liveness even when the gateway is briefly
        # unreachable. Healthy = main loop has cycled in the last health
        # interval; the remote send_health is a separate signal.
        try:
            os.makedirs('/app/data', exist_ok=True)
            with open('/app/data/heartbeat', 'w') as fh:
                fh.write(str(time.time()))
        except Exception as e:
            logger.warning(f"Could not write heartbeat file: {e}")

        if not self.http:
            return False
        return self.http.send_health(
            hostname=self.hostname,
            version=AGENT_VERSION,
            status='ok',
            cloud_provider=self.cloud_provider,
            region=self.region,
            instance_id=self.instance_id,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _setup_signal_handlers(self):
        """Install signal handlers for graceful shutdown."""
        self._shutdown_event = threading.Event()

        def _handle_signal(signum, frame):
            sig_name = signal.Signals(signum).name
            logger.info(f"Received {sig_name}, initiating graceful shutdown...")
            self._shutdown_event.set()

        signal.signal(signal.SIGTERM, _handle_signal)
        signal.signal(signal.SIGINT, _handle_signal)

    def shutdown(self):
        """Gracefully shutdown the agent."""
        try:
            logger.info("Agent shutdown initiated...")

            if self.metric_exporter:
                try:
                    self.metric_exporter.stop()
                except Exception as e:
                    logger.warning(f"Error stopping metric exporters: {e}")

            if self.grpc_client:
                try:
                    self.grpc_client.stop()
                    logger.info("gRPC client stopped")
                except Exception as e:
                    logger.warning(f"Error stopping gRPC client: {e}")

            if self.spot_handler:
                try:
                    self.spot_handler.stop()
                    logger.info("Spot handler stopped")
                except Exception as e:
                    logger.warning(f"Error stopping spot handler: {e}")

            if self.nvml_sampler:
                try:
                    self.nvml_sampler.stop()
                    logger.info("NVML sampler stopped")
                except Exception as e:
                    logger.warning(f"Error stopping NVML sampler: {e}")

            # Final health check with status='down' so the dashboard
            # doesn't render a stale "ok" until heartbeat timeout.
            if self.http:
                try:
                    self.http.send_health(
                        hostname=self.hostname,
                        version=AGENT_VERSION,
                        status='down',
                        cloud_provider=self.cloud_provider,
                        region=self.region,
                        instance_id=self.instance_id,
                    )
                    logger.info("Final health check sent (status=down)")
                except Exception as e:
                    logger.warning(f"Error sending final health check: {e}")

            logger.info("Agent shutdown complete")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

    # ------------------------------------------------------------------
    # Command poll + execute (HTTP fallback path; gRPC pushes natively).
    # ------------------------------------------------------------------

    def poll_and_execute_commands(self):
        """Poll backend for pending action commands and execute them.

        When gRPC is connected, commands arrive via server push — skip HTTP polling.
        """
        if self.grpc_client and self.grpc_client.is_connected:
            logger.debug("gRPC connected — skipping HTTP command poll")
            return
        if not self.http:
            return
        commands = self.http.poll_commands(agent_id=self.hostname)
        if not commands:
            return
        logger.info(f"Received {len(commands)} commands from backend")
        for cmd in commands:
            # The HTTP poll surfaces the row id directly; gRPC surfaces
            # the same data inside an InstanceCommand. Normalise to a
            # dict with `command_id` so _execute_command is shape-agnostic.
            command_id = cmd.get('id') or cmd.get('command_id')
            params = cmd.get('params') or cmd.get('action_params') or {}
            normalised = {
                'command_id': command_id,
                'instance_id': cmd.get('instance_id'),
                'type': cmd.get('type') or cmd.get('action_type'),
                'params': params,
                'cloud_provider': cmd.get('cloud_provider') or params.get('cloud_provider'),
                'region': cmd.get('region') or params.get('region'),
            }
            result = self._execute_command(normalised)
            self._report_command_result(command_id, result)

    def _execute_command_inner(self, cmd):
        """Internal command execution logic.

        The agent can dispatch stop / start / resize / restart / terminate
        on customer cloud instances. This is gated fail-closed by two
        environment variables — both must be set or the command is
        rejected with status='rejected':

          AGENT_ACTUATOR_ENABLED          must equal "true" (case-
                                          insensitive). Default false.
          AGENT_ACTUATOR_ALLOWED_ACTIONS  comma-separated list of action
                                          names that may be dispatched.
                                          Default empty (nothing allowed).

        Two env vars rather than one because operators frequently want to
        permit `stop` and `start` for cost-savings rotations while never
        permitting `terminate`. Forcing both means there is no
        single-toggle that opens the full surface — the destructive
        action has to be named explicitly.

        Every reject is logged at WARN with the full (cloud_provider,
        instance_id, action_type, command_id) tuple so the operator-
        side audit trail is intact even when nothing actually fired.

        Cmd dict accepts BOTH the new-style {command_id, instance_id, type,
        params} and legacy-style {id, action_type, action_params,
        cloud_provider, region} so the same handler serves both
        transports.
        """
        command_id = cmd.get('command_id') or cmd.get('id')
        action_type = cmd.get('type') or cmd.get('action_type')
        instance_id = cmd.get('instance_id')
        params = cmd.get('params') or cmd.get('action_params') or {}
        cloud_provider = cmd.get('cloud_provider') or params.get('cloud_provider')
        if not cloud_provider:
            logger.warning(
                "actuator command rejected — cloud_provider is required",
                extra={
                    'command_id': command_id, 'action_type': action_type,
                    'instance_id': instance_id,
                },
            )
            return {
                'status': 'rejected',
                'error': 'cloud_provider is required on command params',
            }
        region = cmd.get('region') or params.get('region')

        if not _actuator_enabled():
            logger.warning(
                "actuator command rejected — AGENT_ACTUATOR_ENABLED is not 'true'",
                extra={
                    'command_id': command_id, 'action_type': action_type,
                    'instance_id': instance_id, 'cloud_provider': cloud_provider,
                },
            )
            return {'status': 'rejected', 'error': 'actuator disabled (AGENT_ACTUATOR_ENABLED!=true)'}

        allowed = _actuator_allowed_actions()
        if action_type not in allowed:
            logger.warning(
                "actuator command rejected — action not in AGENT_ACTUATOR_ALLOWED_ACTIONS",
                extra={
                    'command_id': command_id, 'action_type': action_type,
                    'instance_id': instance_id, 'cloud_provider': cloud_provider,
                    'allowed': sorted(allowed),
                },
            )
            return {
                'status': 'rejected',
                'error': f"action '{action_type}' not in allowlist {sorted(allowed)}",
            }

        logger.info(
            "actuator command dispatching",
            extra={
                'command_id': command_id, 'action_type': action_type,
                'instance_id': instance_id, 'cloud_provider': cloud_provider,
            },
        )

        try:
            monitor = self.monitors.get(cloud_provider)
            if not monitor:
                return {'status': 'failed', 'error': f'No monitor for provider: {cloud_provider}'}

            aws_region_kw = {'region': region} if cloud_provider == 'aws' and region else {}

            if action_type == 'stop':
                result = monitor.stop_instance(instance_id, **aws_region_kw)
            elif action_type == 'start':
                result = monitor.start_instance(instance_id, **aws_region_kw)
            elif action_type == 'resize':
                target = params.get('target_instance_type')
                if cloud_provider == 'aws':
                    result = monitor.resize_instance(instance_id, target, **aws_region_kw)
                else:
                    result = monitor.resize_instance(instance_id, target)
            elif action_type == 'restart':
                result = monitor.restart_instance(instance_id, **aws_region_kw)
            elif action_type == 'terminate':
                result = monitor.terminate_instance(instance_id, **aws_region_kw)
            else:
                return {'status': 'failed', 'error': f'Unknown action type: {action_type}'}

            logger.info(f"Command executed: {action_type} on {instance_id} -> success")
            return {'status': 'succeeded', 'result': result or {}}
        except Exception as e:
            logger.error(f"Command failed: {action_type} on {instance_id} -> {e}")
            return {'status': 'failed', 'error': str(e)}

    def _execute_command(self, cmd):
        """Execute a cloud action command with timeout."""
        # Per-command timeout, defaulting to 2 min — same as legacy.
        timeout = (cmd.get('params') or {}).get('timeout', 120)
        try:
            timeout = int(timeout)
        except (TypeError, ValueError):
            timeout = 120

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self._execute_command_inner, cmd)
            try:
                return future.result(timeout=timeout)
            except concurrent.futures.TimeoutError:
                cmd_id = cmd.get('command_id') or cmd.get('id', 'unknown')
                logger.error(f"Command {cmd_id} timed out after {timeout}s")
                return {'status': 'failed', 'error': f'Command timed out after {timeout}s'}

    def _report_command_result(self, command_id, result):
        """Report command execution result back to backend (HTTP path)."""
        if not self.http or not command_id:
            return
        self.http.report_command_result(
            command_id=str(command_id),
            status=result.get('status', 'failed'),
            error=result.get('error'),
            result=result.get('result'),
        )
        logger.info(f"Command result reported for {command_id}: {result.get('status')}")

    # ------------------------------------------------------------------
    # Spot interruption — dedicated endpoint, NOT the legacy alerts envelope.
    # ------------------------------------------------------------------

    def _handle_spot_interruption(self, info):
        """Handle spot instance interruption notice."""
        logger.warning(f"Handling spot interruption: {info}")
        if self.http:
            self.http.report_spot_interruption(
                external_instance_id=info.get('instance_id', ''),
                cloud_provider=info.get('cloud_provider', 'aws'),
                interruption_type=info.get('action', 'terminate'),
                checkpoint_status=info.get('checkpoint_status'),
                workload_metadata={'time': info.get('time'), 'raw': info},
            )
        # Local alert still fires — runbook + slack notifications.
        self.alert_manager._send_alert(
            'spot_interruption',
            f"SPOT INTERRUPTION: Instance {info.get('instance_id')} will be "
            f"{info.get('action')}d at {info.get('time')}. Checkpoint triggered.",
            [info],
        )

    def _handle_spot_checkpoint(self):
        """Trigger workload checkpoint before spot termination."""
        checkpoint_script = os.getenv('SPOT_CHECKPOINT_SCRIPT')
        if checkpoint_script and os.path.exists(checkpoint_script):
            import subprocess
            try:
                result = subprocess.run(
                    [checkpoint_script],
                    timeout=90,
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0:
                    logger.info(f"Checkpoint script succeeded: {result.stdout[:200]}")
                else:
                    logger.error(f"Checkpoint script failed (rc={result.returncode}): {result.stderr[:200]}")
            except subprocess.TimeoutExpired:
                logger.error("Checkpoint script timed out (90s)")
            except Exception as e:
                logger.error(f"Checkpoint script error: {e}")
        else:
            logger.info("No SPOT_CHECKPOINT_SCRIPT configured, skipping checkpoint")

    # ------------------------------------------------------------------
    # Monitoring cycle
    # ------------------------------------------------------------------

    def run_monitoring_cycle(self):
        """Run a complete monitoring cycle across all enabled clouds.

        Only metrics + instance-state updates are pushed; cost / ai-spend /
        inference / training / gateway / cluster cycles are dropped per
        AGENT_COMPATIBILITY.md §2.3.
        """
        try:
            logger.info("Starting unified monitoring cycle...")
            current_instance_ids = set()

            for cloud_name, monitor in self.monitors.items():
                try:
                    logger.info(f"Collecting data from {cloud_name}...")
                    instances = monitor.get_gpu_instances()

                    current_ids = {i['instance_id'] for i in instances}
                    previous_ids = self._known_instances.get(cloud_name, set())
                    disappeared_ids = previous_ids - current_ids
                    if disappeared_ids:
                        logger.info(f"{len(disappeared_ids)} instances disappeared from {cloud_name}: {disappeared_ids}")
                        for gone_id in disappeared_ids:
                            instances.append({
                                'instance_id': gone_id,
                                'instance_type': 'unknown',
                                'state': 'terminated',
                                'cloud_provider': cloud_name,
                                'tags': {},
                                'availability_zone': None,
                                'launch_time': None,
                                'gpu_count': 0,
                                'gpu_type': None,
                            })

                    self._known_instances[cloud_name] = {
                        i['instance_id'] for i in instances
                        if i['state'] not in ('terminated', 'shutting-down')
                    }
                    for inst in instances:
                        current_instance_ids.add(inst['instance_id'])

                    if instances:
                        self._send_instances(instances)
                        logger.info(f"Sent {len(instances)} instances from {cloud_name}")

                    metrics = monitor.get_gpu_utilization(instances)

                    # Enrich with NVML when available.
                    #
                    # Before 2026-05-15 this block merged all NVML devices
                    # into a single per-instance aggregate metric — every
                    # heatmap row showed identical data because the backend
                    # had no gpu_index column. Now we fan-out: for each
                    # (instance, gpu_index) pair reported by NVML we emit
                    # one MetricPoint with that gpu_index explicitly set.
                    # The backend stores it in gpu.metric.gpu_index (migration
                    # 021) and the heatmap buckets by (gpu_index, ts_bucket)
                    # to render distinct per-device colour series.
                    if self.nvml_sampler and self.nvml_sampler.is_available():
                        from host_identity import (
                            build_host_instance,
                            merge_nvml_metrics,
                            metric_template_from_instance,
                        )
                        nvml_metrics = self.nvml_sampler.get_latest_metrics()
                        idle_devices = set(self.nvml_sampler.get_idle_devices())
                        if nvml_metrics:
                            if metrics:
                                instance_template = metric_template_from_instance(
                                    {
                                        "instance_id": metrics[0].get("instance_id", self.instance_id),
                                        "cloud_id": metrics[0].get("cloud_id", self.hostname),
                                        "cloud_provider": metrics[0].get(
                                            "cloud_provider", self.cloud_provider
                                        ),
                                    }
                                )
                            elif instances:
                                instance_template = metric_template_from_instance(instances[0])
                            else:
                                instance_template = metric_template_from_instance(
                                    build_host_instance(
                                        hostname=self.hostname,
                                        cloud_provider=self.cloud_provider,
                                        instance_id=self.instance_id,
                                    )
                                )
                            per_device_metrics = merge_nvml_metrics(
                                instance_template, nvml_metrics, idle_devices
                            )
                            if per_device_metrics:
                                metrics = per_device_metrics

                    # Phase 4 dark signals — stamp host NVLink degradation on
                    # gpu_index=0 samples when the cluster monitor is enabled.
                    if metrics and self.gpu_cluster_monitor:
                        try:
                            interconnect = self.gpu_cluster_monitor.get_interconnect_health()
                            nvlink_degraded = interconnect.get("nvlink_links_degraded")
                            if nvlink_degraded is not None:
                                for m in metrics:
                                    if int(m.get("gpu_index", 0) or 0) == 0:
                                        m["nvlink_links_degraded"] = int(nvlink_degraded)
                        except Exception as exc:
                            logger.debug("NVLink dark-signal stamp skipped: %s", exc)

                    if metrics:
                        self._send_metrics(metrics)
                        logger.info(f"Sent {len(metrics)} metrics from {cloud_name}")
                        self.alert_manager.check_idle_gpus(metrics)

                    # ---- DROPPED: cost cycle. apps-new ingests cost data
                    # via cost-service's own pipeline, not from agents.
                    # See AGENT_COMPATIBILITY.md §2.3 "Dropped Event Types".
                    # Original code:
                    #   costs = monitor.get_cost_data()
                    #   if costs:
                    #       self._send_to_backend('costs', costs)
                    #   self.alert_manager.check_cost_thresholds(costs)

                    del instances, metrics
                except Exception as e:
                    logger.error(f"Error collecting data from {cloud_name}: {e}")
                    # ---- DROPPED: error-event ingest. apps-new has no
                    # /api/sync/errors endpoint; OTel traces are the
                    # sanctioned channel. Surface locally only.

            # ---- DROPPED: ai_spend cycle.
            # apps-new ai-service has its own ingest controller; agents
            # do not push ai_spend to /api/sync. Collector still runs
            # so customer config doesn't break, but no wire push here.
            # if self.ai_spend_monitor: ...

            # ---- DROPPED: inference_metrics cycle.
            # No agent path on apps-new. See AGENT_COMPATIBILITY.md §2.3.
            # if self.inference_server_monitor: ...

            # ---- DROPPED: training_runs cycle.
            # ai-service.training_run is filled by integration; not by
            # agent push.
            # if self.training_run_monitor: ...

            # ---- DROPPED: gateway_metrics cycle.
            # No agent path on apps-new.
            # if self.llm_gateway_monitor: ...

            # ---- DROPPED: cluster_health cycle.
            # monitoring-service is server-driven on apps-new.
            # if self.gpu_cluster_monitor: ...

            for provider in list(self._known_instances.keys()):
                stale = [iid for iid in self._known_instances[provider] if iid not in current_instance_ids]
                for iid in stale:
                    # _known_instances is set-of-id, not dict — discard, not del.
                    self._known_instances[provider].discard(iid)
                    logger.debug(f"Cleaned up stale instance {iid} from {provider}")

            logger.info("Unified monitoring cycle completed successfully")
        except Exception as e:
            logger.error(f"Error in unified monitoring cycle: {e}")
            # ---- DROPPED: error-event ingest (see above).

    def _validate_config(self):
        """Validate required configuration at startup."""
        errors = []
        warnings = []

        if not self.backend_api_url:
            errors.append("BACKEND_API_URL not configured")
        if not self.agent_api_key:
            errors.append("AGENT_API_KEY not configured (legacy: BACKEND_API_KEY)")
        if not self.tenant_id or self.tenant_id == 'default':
            warnings.append("TENANT_ID not configured (using 'default' — server will reject)")

        clouds = self.config.get('monitoring', {}).get('enabled_clouds', [])
        if not clouds:
            warnings.append("No cloud providers enabled in monitoring.enabled_clouds")

        if self.comm_mode in ('grpc', 'both'):
            if not os.getenv('GRPC_TARGET'):
                errors.append("GRPC_TARGET not set but COMM_MODE includes grpc")
            if not os.getenv('AGENT_KEY_ID'):
                errors.append("AGENT_KEY_ID not set but COMM_MODE includes grpc")
            if not os.getenv('AGENT_HMAC_PEPPER'):
                errors.append("AGENT_HMAC_PEPPER not set but COMM_MODE includes grpc")

        for w in warnings:
            logger.warning(f"Config warning: {w}")
        if errors:
            for e in errors:
                logger.error(f"Config error: {e}")
            raise SystemExit(f"Configuration validation failed: {'; '.join(errors)}")
        logger.info("Configuration validation passed")

    # ---- DROPPED: _run_recommendations_cycle.
    # apps-new cost-service computes RI / Savings Plan / spot recommendations
    # server-side. The collector hooks (`get_recommendations`, `get_spot_pricing`,
    # `get_spot_candidates`) are still on the monitors but no longer invoked.

    def start(self):
        """Start the agent with scheduled monitoring."""
        self._validate_config()

        logger.info("Starting Unified GPU Cost Optimization Agent...")
        logger.info(f"Communication mode: {self.comm_mode}")

        self._setup_signal_handlers()
        os.makedirs('./logs', exist_ok=True)

        if self.metric_exporter:
            self.metric_exporter.start()
        if self.nvml_sampler:
            self.nvml_sampler.start()
        if self.grpc_client:
            self.grpc_client.start()
        if self.spot_handler:
            self.spot_handler.start()

        monitoring_config = self.config.get('monitoring', {})
        interval = monitoring_config.get('interval', 300)

        schedule.every(interval).seconds.do(self.run_monitoring_cycle)
        schedule.every(interval).seconds.do(self._drain_queue)

        # ---- DROPPED: recommendations cycle. See note above.
        # if os.getenv('ENABLE_RECOMMENDATIONS', 'false').lower() == 'true':
        #     rec_interval = int(os.getenv('RECOMMENDATION_INTERVAL', 21600))
        #     schedule.every(rec_interval).seconds.do(self._run_recommendations_cycle)

        command_poll_interval = monitoring_config.get('command_poll_interval', 15)
        schedule.every(command_poll_interval).seconds.do(self.poll_and_execute_commands)

        # Health check on a separate cadence so the dashboard "agent up"
        # indicator updates faster than the monitoring cycle.
        health_interval = int(os.getenv('HEALTH_INTERVAL', 60))
        schedule.every(health_interval).seconds.do(self._send_health_check)

        self.run_monitoring_cycle()
        self._send_health_check()
        self.poll_and_execute_commands()

        try:
            while not self._shutdown_event.is_set():
                schedule.run_pending()
                self._shutdown_event.wait(timeout=1)
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        finally:
            self.shutdown()

    def get_status(self) -> Dict[str, Any]:
        """Get agent status"""
        return {
            'status': 'running',
            'monitors': list(self.monitors.keys()),
            'config': self.config,
            'timestamp': time.time()
        }

    def get_health(self) -> Dict[str, Any]:
        """Get agent health"""
        health = {
            'status': 'healthy',
            'monitors': {},
            'timestamp': time.time()
        }
        for cloud_name, monitor in self.monitors.items():
            try:
                instances = monitor.get_instances()
                health['monitors'][cloud_name] = {
                    'status': 'healthy',
                    'instance_count': len(instances)
                }
            except Exception as e:
                health['monitors'][cloud_name] = {
                    'status': 'unhealthy',
                    'error': str(e)
                }
                health['status'] = 'degraded'
        return health


def main():
    """Main entry point"""
    import argparse
    parser = argparse.ArgumentParser(description='Unified GPU Cost Optimization Agent')
    parser.add_argument('--config', '-c', help='Configuration file path')
    parser.add_argument('--status', action='store_true', help='Show agent status')
    parser.add_argument('--health', action='store_true', help='Show agent health')
    args = parser.parse_args()

    agent = UnifiedGPUAgent(args.config)
    if args.status:
        status = agent.get_status()
        print(f"Agent Status: {status['status']}")
        print(f"Monitors: {', '.join(status['monitors'])}")
        return
    if args.health:
        health = agent.get_health()
        print(f"Agent Health: {health['status']}")
        for monitor, info in health['monitors'].items():
            print(f"  {monitor}: {info['status']}")
        return
    agent.start()


if __name__ == "__main__":
    main()
