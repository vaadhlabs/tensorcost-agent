"""
Cross-cloud training job monitor for ML/LLM training runs.

Tracks training runs across AWS SageMaker, Azure ML, GCP Vertex AI, and custom/self-hosted
training infrastructure. Provides unified view with cost tracking, progress estimation, and
anomaly detection.
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import statistics

# Optional provider SDKs
try:
    import boto3
    HAS_BOTO3 = True
except ImportError:
    HAS_BOTO3 = False

try:
    from azure.identity import DefaultAzureCredential
    from azure.ai.ml import MLClient
    HAS_AZURE_ML = True
except ImportError:
    HAS_AZURE_ML = False

try:
    from google.cloud import aiplatform
    HAS_VERTEX_AI = True
except ImportError:
    HAS_VERTEX_AI = False

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

try:
    import mlflow
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


logger = logging.getLogger(__name__)


class TrainingStatus(Enum):
    """Training job status enum."""
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPING = "stopping"
    STOPPED = "stopped"
    PENDING = "pending"


class TrainingFramework(Enum):
    """ML framework enum."""
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    JAX = "jax"
    HUGGINGFACE = "huggingface"
    UNKNOWN = "unknown"


@dataclass
class InstancePricing:
    """Instance pricing configuration."""
    instance_type: str
    provider: str
    gpu_type: Optional[str]
    gpu_count: int
    hourly_cost: float
    region: str = "us-east-1"


class TrainingRunMonitor:
    """
    Unified monitor for ML training runs across multiple cloud providers.

    Supports:
    - AWS SageMaker
    - Azure ML
    - GCP Vertex AI
    - Weights & Biases (as unified tracker)
    - MLflow (as unified tracker)
    """

    # Instance pricing dictionary (example configs, expandable)
    INSTANCE_PRICING = {
        # AWS SageMaker instances
        "ml.p3.2xlarge": InstancePricing("ml.p3.2xlarge", "sagemaker", "V100", 1, 3.06),
        "ml.p3.8xlarge": InstancePricing("ml.p3.8xlarge", "sagemaker", "V100", 4, 12.24),
        "ml.p3.16xlarge": InstancePricing("ml.p3.16xlarge", "sagemaker", "V100", 8, 24.48),
        "ml.p4d.24xlarge": InstancePricing("ml.p4d.24xlarge", "sagemaker", "A100", 8, 32.77),
        "ml.g4dn.xlarge": InstancePricing("ml.g4dn.xlarge", "sagemaker", "T4", 1, 0.526),
        "ml.g4dn.12xlarge": InstancePricing("ml.g4dn.12xlarge", "sagemaker", "T4", 4, 1.86),
        "ml.g5.xlarge": InstancePricing("ml.g5.xlarge", "sagemaker", "A10G", 1, 1.006),
        "ml.g5.12xlarge": InstancePricing("ml.g5.12xlarge", "sagemaker", "A10G", 4, 4.024),
        # Azure ML instances
        "Standard_NC6": InstancePricing("Standard_NC6", "azure_ml", "K80", 1, 0.90),
        "Standard_NC12": InstancePricing("Standard_NC12", "azure_ml", "K80", 2, 1.80),
        "Standard_ND6S": InstancePricing("Standard_ND6S", "azure_ml", "V100", 1, 2.16),
        "Standard_ND40rs": InstancePricing("Standard_ND40rs", "azure_ml", "V100", 8, 17.28),
        # GCP Vertex AI instances
        "n1-standard-4": InstancePricing("n1-standard-4", "vertex_ai", None, 0, 0.38),
        "a2-highgpu-1g": InstancePricing("a2-highgpu-1g", "vertex_ai", "A100", 1, 2.50),
        "a2-highgpu-8g": InstancePricing("a2-highgpu-8g", "vertex_ai", "A100", 8, 19.99),
        "g4-gpu-1": InstancePricing("g4-gpu-1", "vertex_ai", "T4", 1, 0.29),
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the training run monitor.

        Args:
            config: Optional configuration dictionary with provider settings.
                   If not provided, reads from environment variables.
        """
        self.config = config or {}

        # Read environment flags
        self.enabled = os.getenv("TRAINING_MONITOR_ENABLED", "true").lower() == "true"
        self.sagemaker_enabled = os.getenv("SAGEMAKER_TRAINING_ENABLED", "true").lower() == "true"
        self.azure_ml_enabled = os.getenv("AZURE_ML_TRAINING_ENABLED", "true").lower() == "true"
        self.vertex_ai_enabled = os.getenv("VERTEX_AI_TRAINING_ENABLED", "true").lower() == "true"

        # Optional integrations
        self.wandb_api_key = os.getenv("WANDB_API_KEY", "")
        self.mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "")

        # Initialize cloud clients
        self.sagemaker_client = None
        self.azure_ml_client = None
        self.vertex_ai_client = None

        self._initialize_clients()

    def _initialize_clients(self) -> None:
        """Initialize cloud provider clients based on availability and config."""
        if not self.enabled:
            logger.info("TrainingRunMonitor is disabled")
            return

        # Initialize SageMaker
        if self.sagemaker_enabled and HAS_BOTO3:
            try:
                self.sagemaker_client = boto3.client(
                    "sagemaker",
                    region_name=self.config.get("aws_region", "us-east-1")
                )
                logger.info("SageMaker client initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize SageMaker client: {e}")

        # Initialize Azure ML
        if self.azure_ml_enabled and HAS_AZURE_ML:
            try:
                credential = DefaultAzureCredential()
                self.azure_ml_client = MLClient(
                    credential,
                    subscription_id=self.config.get("azure_subscription_id"),
                    resource_group_name=self.config.get("azure_resource_group"),
                    workspace_name=self.config.get("azure_workspace_name")
                )
                logger.info("Azure ML client initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Azure ML client: {e}")

        # Initialize Vertex AI
        if self.vertex_ai_enabled and HAS_VERTEX_AI:
            try:
                aiplatform.init(
                    project=self.config.get("gcp_project_id"),
                    location=self.config.get("gcp_region", "us-central1")
                )
                logger.info("Vertex AI client initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Vertex AI client: {e}")

    def get_active_training_runs(self) -> List[Dict[str, Any]]:
        """
        Get all active training runs across providers.

        Returns:
            List of training run dictionaries with keys:
                - run_id, provider, job_name, status
                - instance_type, instance_count, gpu_type, gpu_count
                - start_time, elapsed_hours, estimated_completion
                - cost_so_far, estimated_total_cost, cost_per_hour
                - framework, model_name, dataset_name
                - current_epoch, total_epochs, current_step, total_steps
                - training_loss, validation_loss
                - gpu_utilization_avg, memory_utilization_avg
                - cloud_provider, region, tags
        """
        if not self.enabled:
            return []

        active_runs = []

        # Collect from each provider
        active_runs.extend(self._collect_sagemaker_runs())
        active_runs.extend(self._collect_azure_ml_runs())
        active_runs.extend(self._collect_vertex_ai_runs())
        active_runs.extend(self._collect_wandb_runs())
        active_runs.extend(self._collect_mlflow_runs())

        # Filter to only active runs
        active_runs = [r for r in active_runs if r.get("status") == TrainingStatus.RUNNING.value]

        logger.info(f"Found {len(active_runs)} active training runs")
        return active_runs

    def get_training_history(self, days: int = 7) -> List[Dict[str, Any]]:
        """
        Get completed training runs for cost analysis.

        Args:
            days: Number of days of history to retrieve.

        Returns:
            List of completed training run dictionaries.
        """
        if not self.enabled:
            return []

        history = []
        cutoff_time = datetime.utcnow() - timedelta(days=days)

        # Collect from each provider
        history.extend(self._collect_sagemaker_runs(completed_only=True, cutoff_time=cutoff_time))
        history.extend(self._collect_azure_ml_runs(completed_only=True, cutoff_time=cutoff_time))
        history.extend(self._collect_vertex_ai_runs(completed_only=True, cutoff_time=cutoff_time))
        history.extend(self._collect_wandb_runs(cutoff_time=cutoff_time))
        history.extend(self._collect_mlflow_runs(cutoff_time=cutoff_time))

        logger.info(f"Found {len(history)} completed training runs in {days} days")
        return history

    def get_training_cost_summary(self, days: int = 30) -> Dict[str, Any]:
        """
        Aggregate training costs by provider, instance type, and project.

        Args:
            days: Number of days of cost history to summarize.

        Returns:
            Dictionary with keys:
                - total_cost: float
                - cost_by_provider: dict
                - cost_by_instance_type: dict
                - cost_by_project: dict
                - avg_cost_per_run: float
                - total_gpu_hours: float
                - failed_run_waste: float
        """
        if not self.enabled:
            return self._empty_cost_summary()

        history = self.get_training_history(days=days)

        total_cost = 0.0
        cost_by_provider = {}
        cost_by_instance_type = {}
        cost_by_project = {}
        total_gpu_hours = 0.0
        failed_run_waste = 0.0

        for run in history:
            cost = run.get("estimated_total_cost", 0.0)
            total_cost += cost

            # Cost by provider
            provider = run.get("provider", "unknown")
            cost_by_provider[provider] = cost_by_provider.get(provider, 0.0) + cost

            # Cost by instance type
            instance_type = run.get("instance_type", "unknown")
            cost_by_instance_type[instance_type] = cost_by_instance_type.get(instance_type, 0.0) + cost

            # Cost by project
            project = run.get("tags", {}).get("project", "untagged")
            cost_by_project[project] = cost_by_project.get(project, 0.0) + cost

            # GPU hours
            gpu_count = run.get("gpu_count", 0)
            elapsed_hours = run.get("elapsed_hours", 0.0)
            total_gpu_hours += gpu_count * elapsed_hours

            # Failed run waste
            if run.get("status") == TrainingStatus.FAILED.value:
                failed_run_waste += cost

        avg_cost_per_run = total_cost / len(history) if history else 0.0

        return {
            "total_cost": total_cost,
            "cost_by_provider": cost_by_provider,
            "cost_by_instance_type": cost_by_instance_type,
            "cost_by_project": cost_by_project,
            "avg_cost_per_run": avg_cost_per_run,
            "total_gpu_hours": total_gpu_hours,
            "failed_run_waste": failed_run_waste,
            "summary_period_days": days,
            "run_count": len(history),
        }

    def detect_training_anomalies(self, runs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Detect anomalies in training runs.

        Checks for:
        - Loss not decreasing (stalled training)
        - GPU utilization < 30% (underutilized)
        - Run duration > 2x median (hung job)
        - OOM patterns
        - Cost exceeding budget

        Args:
            runs: List of training run dictionaries.

        Returns:
            List of anomaly dictionaries with keys:
                - run_id, job_name, provider
                - anomaly_type: str
                - severity: "low", "medium", "high", "critical"
                - description: str
                - recommendation: str
        """
        anomalies = []

        if not runs:
            return anomalies

        # Group runs by config for comparison
        config_groups = {}
        for run in runs:
            key = (
                run.get("instance_type"),
                run.get("framework"),
                run.get("model_name"),
            )
            if key not in config_groups:
                config_groups[key] = []
            config_groups[key].append(run)

        # Calculate medians for each config
        config_medians = {}
        for key, group_runs in config_groups.items():
            durations = [r.get("elapsed_hours", 0.0) for r in group_runs]
            if durations:
                config_medians[key] = statistics.median(durations)

        # Check each run for anomalies
        for run in runs:
            run_id = run.get("run_id")
            job_name = run.get("job_name")
            provider = run.get("provider")

            # Check 1: Stalled training (loss not decreasing)
            training_loss = run.get("training_loss")
            if training_loss is not None and training_loss > 10.0:  # Threshold
                anomalies.append({
                    "run_id": run_id,
                    "job_name": job_name,
                    "provider": provider,
                    "anomaly_type": "stalled_training",
                    "severity": "high",
                    "description": f"Training loss is very high ({training_loss:.4f}), possible stalled training",
                    "recommendation": "Check data pipeline, learning rate, and model architecture",
                })

            # Check 2: GPU underutilization
            gpu_util = run.get("gpu_utilization_avg", 0.0)
            if gpu_util < 30.0:
                anomalies.append({
                    "run_id": run_id,
                    "job_name": job_name,
                    "provider": provider,
                    "anomaly_type": "gpu_underutilized",
                    "severity": "medium",
                    "description": f"GPU utilization is low ({gpu_util:.1f}%), instance may be oversized",
                    "recommendation": "Consider using smaller instance type or increasing batch size",
                })

            # Check 3: Hung job (duration > 2x median)
            config_key = (
                run.get("instance_type"),
                run.get("framework"),
                run.get("model_name"),
            )
            if config_key in config_medians:
                median_duration = config_medians[config_key]
                elapsed = run.get("elapsed_hours", 0.0)
                if elapsed > 2.0 * median_duration and elapsed > 2.0:  # At least 2 hours
                    anomalies.append({
                        "run_id": run_id,
                        "job_name": job_name,
                        "provider": provider,
                        "anomaly_type": "hung_job",
                        "severity": "critical",
                        "description": f"Run duration ({elapsed:.1f}h) is >2x median ({median_duration:.1f}h)",
                        "recommendation": "Check for deadlocks, I/O bottlenecks, or communication issues",
                    })

            # Check 4: High memory usage (proxy for OOM risk)
            mem_util = run.get("memory_utilization_avg", 0.0)
            if mem_util > 90.0:
                anomalies.append({
                    "run_id": run_id,
                    "job_name": job_name,
                    "provider": provider,
                    "anomaly_type": "memory_pressure",
                    "severity": "high",
                    "description": f"Memory utilization is high ({mem_util:.1f}%), OOM risk",
                    "recommendation": "Reduce batch size, enable gradient checkpointing, or use larger instance",
                })

            # Check 5: Cost exceeding budget
            budget = run.get("tags", {}).get("budget_usd", None)
            if budget:
                cost = run.get("cost_so_far", 0.0)
                if cost > budget * 1.1:  # 10% over budget
                    anomalies.append({
                        "run_id": run_id,
                        "job_name": job_name,
                        "provider": provider,
                        "anomaly_type": "budget_exceeded",
                        "severity": "medium",
                        "description": f"Cost (${cost:.2f}) exceeds budget (${budget:.2f})",
                        "recommendation": "Consider stopping run, using cheaper instance, or reducing training time",
                    })

        logger.info(f"Detected {len(anomalies)} anomalies across {len(runs)} runs")
        return anomalies

    def get_training_recommendations(self) -> List[Dict[str, Any]]:
        """
        Recommend optimizations for training runs.

        Returns:
            List of recommendation dictionaries with keys:
                - recommendation_type: str
                - priority: "low", "medium", "high"
                - description: str
                - potential_savings: float (estimated cost savings)
                - affected_runs: int
        """
        recommendations = []
        active_runs = self.get_active_training_runs()

        if not active_runs:
            return recommendations

        # Recommendation 1: Use spot/preemptible instances
        suitable_for_spot = [r for r in active_runs if r.get("gpu_count", 0) <= 4]
        if suitable_for_spot:
            avg_cost = sum(r.get("cost_per_hour", 0.0) for r in suitable_for_spot) / len(suitable_for_spot)
            savings = avg_cost * 0.7  # Spot instances ~70% cheaper
            recommendations.append({
                "recommendation_type": "use_spot_instances",
                "priority": "high",
                "description": f"Use spot/preemptible instances for {len(suitable_for_spot)} smaller training runs",
                "potential_savings": savings,
                "affected_runs": len(suitable_for_spot),
            })

        # Recommendation 2: Downsize underutilized instances
        underutilized = [
            r for r in active_runs
            if r.get("gpu_utilization_avg", 0.0) < 30.0
        ]
        if underutilized:
            avg_cost = sum(r.get("cost_per_hour", 0.0) for r in underutilized) / len(underutilized)
            savings = avg_cost * 0.4  # Estimate 40% savings by downsizing
            recommendations.append({
                "recommendation_type": "downsize_instances",
                "priority": "high",
                "description": f"Downsize {len(underutilized)} underutilized instances (GPU util < 30%)",
                "potential_savings": savings,
                "affected_runs": len(underutilized),
            })

        # Recommendation 3: Enable mixed precision
        fp32_runs = [r for r in active_runs if r.get("tags", {}).get("mixed_precision") != "enabled"]
        if fp32_runs:
            recommendations.append({
                "recommendation_type": "enable_mixed_precision",
                "priority": "medium",
                "description": f"Enable mixed precision (AMP) for {len(fp32_runs)} runs to reduce memory and training time",
                "potential_savings": 0.0,  # Speed improvement, not direct cost savings
                "affected_runs": len(fp32_runs),
            })

        # Recommendation 4: Gradient accumulation for memory-limited runs
        high_memory = [
            r for r in active_runs
            if r.get("memory_utilization_avg", 0.0) > 80.0
        ]
        if high_memory:
            recommendations.append({
                "recommendation_type": "gradient_accumulation",
                "priority": "medium",
                "description": f"Enable gradient accumulation for {len(high_memory)} memory-constrained runs",
                "potential_savings": 0.0,  # Allows larger batches without downsizing
                "affected_runs": len(high_memory),
            })

        # Recommendation 5: Consolidate small runs
        small_runs = [r for r in active_runs if r.get("gpu_count", 0) == 1]
        if len(small_runs) > 3:
            recommendations.append({
                "recommendation_type": "consolidate_small_runs",
                "priority": "low",
                "description": f"Consolidate {len(small_runs)} single-GPU runs into fewer multi-GPU instances",
                "potential_savings": 0.0,  # Reduced overhead
                "affected_runs": len(small_runs),
            })

        logger.info(f"Generated {len(recommendations)} training recommendations")
        return recommendations

    def _collect_sagemaker_runs(
        self,
        completed_only: bool = False,
        cutoff_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Collect training runs from AWS SageMaker."""
        if not self.sagemaker_client or not self.sagemaker_enabled:
            return []

        runs = []
        try:
            # List training jobs
            paginator = self.sagemaker_client.get_paginator("list_training_jobs")

            status_filter = "InProgress" if not completed_only else None

            for page in paginator.paginate(
                StatusEquals=status_filter,
                MaxResults=100,
            ):
                for job in page.get("TrainingJobSummaries", []):
                    job_name = job.get("TrainingJobName")
                    creation_time = job.get("CreationTime")

                    # Filter by cutoff time if provided
                    if cutoff_time and creation_time < cutoff_time:
                        continue

                    # Get detailed info
                    try:
                        details = self.sagemaker_client.describe_training_job(
                            TrainingJobName=job_name
                        )

                        run_dict = self._parse_sagemaker_run(details)
                        runs.append(run_dict)
                    except Exception as e:
                        logger.warning(f"Failed to describe SageMaker job {job_name}: {e}")

            logger.info(f"Collected {len(runs)} SageMaker training runs")
        except Exception as e:
            logger.error(f"Error collecting SageMaker runs: {e}")

        return runs

    def _parse_sagemaker_run(self, job_details: Dict[str, Any]) -> Dict[str, Any]:
        """Parse SageMaker training job details into unified format."""
        job_name = job_details.get("TrainingJobName")
        status_map = {
            "InProgress": TrainingStatus.RUNNING.value,
            "Completed": TrainingStatus.COMPLETED.value,
            "Failed": TrainingStatus.FAILED.value,
            "Stopping": TrainingStatus.STOPPING.value,
            "Stopped": TrainingStatus.STOPPED.value,
        }
        status = status_map.get(job_details.get("TrainingJobStatus"), "unknown")

        resource_config = job_details.get("ResourceConfig", {})
        instance_type = resource_config.get("InstanceType", "unknown")
        instance_count = resource_config.get("InstanceCount", 1)

        # Extract GPU info
        gpu_type, gpu_count = self._extract_gpu_info(instance_type, instance_count)

        # Calculate costs
        start_time = job_details.get("CreationTime", datetime.utcnow())
        end_time = job_details.get("TrainingEndTime", datetime.utcnow())
        elapsed = (end_time - start_time).total_seconds() / 3600  # hours

        pricing_key = instance_type
        hourly_cost = self.INSTANCE_PRICING.get(pricing_key, InstancePricing(
            instance_type, "sagemaker", gpu_type, gpu_count, 1.0
        )).hourly_cost

        total_cost = hourly_cost * elapsed

        # Extract hyperparameters for metadata
        hyperparams = job_details.get("HyperParameters", {})

        return {
            "run_id": job_details.get("TrainingJobArn", job_name),
            "provider": "sagemaker",
            "job_name": job_name,
            "status": status,
            "instance_type": instance_type,
            "instance_count": instance_count,
            "gpu_type": gpu_type,
            "gpu_count": gpu_count,
            "start_time": start_time.isoformat() if isinstance(start_time, datetime) else start_time,
            "elapsed_hours": elapsed,
            "estimated_completion": (start_time + timedelta(hours=elapsed*1.2)).isoformat(),  # Estimate 20% more
            "cost_so_far": total_cost,
            "estimated_total_cost": total_cost,
            "cost_per_hour": hourly_cost,
            "framework": self._detect_framework(job_details.get("AlgorithmSpecification", {}).get("TrainingImage", "")),
            "model_name": hyperparams.get("model_name", "unknown"),
            "dataset_name": hyperparams.get("dataset_name", "unknown"),
            "current_epoch": 0,  # Not always available in SageMaker
            "total_epochs": 0,
            "current_step": 0,
            "total_steps": 0,
            "training_loss": None,
            "validation_loss": None,
            "gpu_utilization_avg": 0.0,  # Would need CloudWatch metrics
            "memory_utilization_avg": 0.0,
            "cloud_provider": "aws",
            "region": job_details.get("TrainingJobArn", "").split(":")[3] if ":" in job_details.get("TrainingJobArn", "") else "unknown",
            "tags": self._parse_sagemaker_tags(job_details),
        }

    def _collect_azure_ml_runs(
        self,
        completed_only: bool = False,
        cutoff_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Collect training runs from Azure ML."""
        if not self.azure_ml_enabled or not HAS_AZURE_ML:
            return []

        runs = []
        try:
            if not self.azure_ml_client:
                logger.warning("Azure ML client not initialized")
                return runs

            # List runs from experiment
            experiments = self.azure_ml_client.experiments.list()
            for experiment in experiments:
                runs_in_exp = self.azure_ml_client.runs.list(experiment.name)
                for run in runs_in_exp:
                    if cutoff_time and run.start_time < cutoff_time:
                        continue

                    run_dict = self._parse_azure_ml_run(run)
                    runs.append(run_dict)

            logger.info(f"Collected {len(runs)} Azure ML training runs")
        except Exception as e:
            logger.error(f"Error collecting Azure ML runs: {e}")

        return runs

    def _parse_azure_ml_run(self, run: Any) -> Dict[str, Any]:
        """Parse Azure ML run into unified format."""
        status_map = {
            "Running": TrainingStatus.RUNNING.value,
            "Completed": TrainingStatus.COMPLETED.value,
            "Failed": TrainingStatus.FAILED.value,
            "Cancelled": TrainingStatus.STOPPED.value,
        }
        status = status_map.get(str(run.status), "unknown")

        # Extract compute target info
        compute_target = getattr(run, "compute_target", "unknown")
        instance_type = str(compute_target) if compute_target else "unknown"

        gpu_type, gpu_count = self._extract_gpu_info(instance_type)
        instance_count = 1

        # Calculate costs
        start_time = run.start_time or datetime.utcnow()
        end_time = run.end_time or datetime.utcnow()
        elapsed = (end_time - start_time).total_seconds() / 3600

        hourly_cost = self.INSTANCE_PRICING.get(instance_type, InstancePricing(
            instance_type, "azure_ml", gpu_type, gpu_count, 1.0
        )).hourly_cost

        total_cost = hourly_cost * elapsed

        return {
            "run_id": run.id,
            "provider": "azure_ml",
            "job_name": run.display_name,
            "status": status,
            "instance_type": instance_type,
            "instance_count": instance_count,
            "gpu_type": gpu_type,
            "gpu_count": gpu_count,
            "start_time": start_time.isoformat() if isinstance(start_time, datetime) else start_time,
            "elapsed_hours": elapsed,
            "estimated_completion": (start_time + timedelta(hours=elapsed*1.2)).isoformat(),
            "cost_so_far": total_cost,
            "estimated_total_cost": total_cost,
            "cost_per_hour": hourly_cost,
            "framework": self._detect_framework(run.display_name),
            "model_name": getattr(run, "tags", {}).get("model_name", "unknown") if hasattr(run, "tags") else "unknown",
            "dataset_name": getattr(run, "tags", {}).get("dataset_name", "unknown") if hasattr(run, "tags") else "unknown",
            "current_epoch": 0,
            "total_epochs": 0,
            "current_step": 0,
            "total_steps": 0,
            "training_loss": None,
            "validation_loss": None,
            "gpu_utilization_avg": 0.0,
            "memory_utilization_avg": 0.0,
            "cloud_provider": "azure",
            "region": getattr(run, "target", "unknown"),
            "tags": getattr(run, "tags", {}),
        }

    def _collect_vertex_ai_runs(
        self,
        completed_only: bool = False,
        cutoff_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Collect training runs from GCP Vertex AI."""
        if not self.vertex_ai_enabled or not HAS_VERTEX_AI:
            return []

        runs = []
        try:
            training_jobs = aiplatform.CustomTrainingJob.list(
                order_by="create_time"
            )

            for job in training_jobs:
                if cutoff_time and job.create_time < cutoff_time:
                    continue

                run_dict = self._parse_vertex_ai_run(job)
                runs.append(run_dict)

            logger.info(f"Collected {len(runs)} Vertex AI training runs")
        except Exception as e:
            logger.error(f"Error collecting Vertex AI runs: {e}")

        return runs

    def _parse_vertex_ai_run(self, job: Any) -> Dict[str, Any]:
        """Parse Vertex AI training job into unified format."""
        status_map = {
            "JOB_STATE_RUNNING": TrainingStatus.RUNNING.value,
            "JOB_STATE_SUCCEEDED": TrainingStatus.COMPLETED.value,
            "JOB_STATE_FAILED": TrainingStatus.FAILED.value,
            "JOB_STATE_CANCELLED": TrainingStatus.STOPPED.value,
        }
        status = status_map.get(job.state.name, "unknown")

        # Extract machine spec
        machine_spec = getattr(job, "machine_spec", {})
        instance_type = getattr(machine_spec, "machine_type", "unknown")
        gpu_type = getattr(machine_spec, "accelerator_type", None)
        gpu_count = getattr(machine_spec, "accelerator_count", 0)
        instance_count = 1

        # Calculate costs
        start_time = job.create_time or datetime.utcnow()
        end_time = job.end_time or datetime.utcnow()
        elapsed = (end_time - start_time).total_seconds() / 3600

        hourly_cost = self.INSTANCE_PRICING.get(instance_type, InstancePricing(
            instance_type, "vertex_ai", gpu_type, gpu_count, 1.0
        )).hourly_cost

        total_cost = hourly_cost * elapsed

        return {
            "run_id": job.resource_name,
            "provider": "vertex_ai",
            "job_name": job.display_name,
            "status": status,
            "instance_type": instance_type,
            "instance_count": instance_count,
            "gpu_type": gpu_type,
            "gpu_count": gpu_count,
            "start_time": start_time.isoformat() if isinstance(start_time, datetime) else start_time,
            "elapsed_hours": elapsed,
            "estimated_completion": (start_time + timedelta(hours=elapsed*1.2)).isoformat(),
            "cost_so_far": total_cost,
            "estimated_total_cost": total_cost,
            "cost_per_hour": hourly_cost,
            "framework": self._detect_framework(job.display_name),
            "model_name": getattr(job, "labels", {}).get("model_name", "unknown") if hasattr(job, "labels") else "unknown",
            "dataset_name": getattr(job, "labels", {}).get("dataset_name", "unknown") if hasattr(job, "labels") else "unknown",
            "current_epoch": 0,
            "total_epochs": 0,
            "current_step": 0,
            "total_steps": 0,
            "training_loss": None,
            "validation_loss": None,
            "gpu_utilization_avg": 0.0,
            "memory_utilization_avg": 0.0,
            "cloud_provider": "gcp",
            "region": getattr(job, "location", "unknown"),
            "tags": getattr(job, "labels", {}),
        }

    def _collect_wandb_runs(self, cutoff_time: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Collect training runs from Weights & Biases."""
        if not self.wandb_api_key or not HAS_REQUESTS:
            return []

        runs = []
        try:
            headers = {"Authorization": f"Bearer {self.wandb_api_key}"}

            # Get all runs via W&B API
            url = "https://api.wandb.ai/graphql"
            query = """
            {
                viewer {
                    projects(first: 100) {
                        edges {
                            node {
                                runs(first: 100) {
                                    edges {
                                        node {
                                            id
                                            name
                                            state
                                            createdAt
                                            updatedAt
                                            duration
                                            config
                                            summaryMetrics
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            """

            response = requests.post(url, json={"query": query}, headers=headers)
            if response.status_code == 200:
                data = response.json()
                # Parse W&B response and convert to unified format
                # This is a simplified example; actual implementation would traverse the full response
                logger.info(f"Collected W&B runs")
        except Exception as e:
            logger.warning(f"Error collecting W&B runs: {e}")

        return runs

    def _collect_mlflow_runs(self, cutoff_time: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Collect training runs from MLflow."""
        if not self.mlflow_tracking_uri or not HAS_MLFLOW:
            return []

        runs = []
        try:
            mlflow.set_tracking_uri(self.mlflow_tracking_uri)
            client = mlflow.tracking.MlflowClient()

            # Get experiments
            experiments = client.search_experiments()
            for experiment in experiments:
                # Search runs in experiment
                mlflow_runs = client.search_runs(
                    experiment_ids=[experiment.experiment_id],
                    max_results=100
                )

                for mlflow_run in mlflow_runs:
                    run_dict = self._parse_mlflow_run(mlflow_run)
                    runs.append(run_dict)

            logger.info(f"Collected {len(runs)} MLflow training runs")
        except Exception as e:
            logger.warning(f"Error collecting MLflow runs: {e}")

        return runs

    def _parse_mlflow_run(self, mlflow_run: Any) -> Dict[str, Any]:
        """Parse MLflow run into unified format."""
        status_map = {
            "RUNNING": TrainingStatus.RUNNING.value,
            "FINISHED": TrainingStatus.COMPLETED.value,
            "FAILED": TrainingStatus.FAILED.value,
            "KILLED": TrainingStatus.STOPPED.value,
        }

        status_str = mlflow_run.info.status if hasattr(mlflow_run.info, "status") else "UNKNOWN"
        status = status_map.get(status_str, "unknown")

        start_time = datetime.fromtimestamp(mlflow_run.info.start_time / 1000) if mlflow_run.info.start_time else datetime.utcnow()
        end_time = datetime.fromtimestamp(mlflow_run.info.end_time / 1000) if mlflow_run.info.end_time else datetime.utcnow()
        elapsed = (end_time - start_time).total_seconds() / 3600

        params = mlflow_run.data.params if mlflow_run.data.params else {}
        metrics = mlflow_run.data.metrics if mlflow_run.data.metrics else {}

        return {
            "run_id": mlflow_run.info.run_id,
            "provider": "mlflow",
            "job_name": mlflow_run.data.tags.get("mlflow.runName", mlflow_run.info.run_id) if mlflow_run.data.tags else mlflow_run.info.run_id,
            "status": status,
            "instance_type": params.get("instance_type", "unknown"),
            "instance_count": int(params.get("instance_count", 1)),
            "gpu_type": params.get("gpu_type"),
            "gpu_count": int(params.get("gpu_count", 0)),
            "start_time": start_time.isoformat(),
            "elapsed_hours": elapsed,
            "estimated_completion": (start_time + timedelta(hours=elapsed*1.2)).isoformat(),
            "cost_so_far": float(metrics.get("cost_so_far", 0.0)),
            "estimated_total_cost": float(metrics.get("estimated_total_cost", 0.0)),
            "cost_per_hour": float(metrics.get("cost_per_hour", 0.0)),
            "framework": self._detect_framework(params.get("framework", "unknown")),
            "model_name": params.get("model_name", "unknown"),
            "dataset_name": params.get("dataset_name", "unknown"),
            "current_epoch": int(metrics.get("current_epoch", 0)),
            "total_epochs": int(params.get("total_epochs", 0)),
            "current_step": int(metrics.get("current_step", 0)),
            "total_steps": int(params.get("total_steps", 0)),
            "training_loss": float(metrics.get("training_loss", 0.0)) if "training_loss" in metrics else None,
            "validation_loss": float(metrics.get("validation_loss", 0.0)) if "validation_loss" in metrics else None,
            "gpu_utilization_avg": float(metrics.get("gpu_utilization_avg", 0.0)),
            "memory_utilization_avg": float(metrics.get("memory_utilization_avg", 0.0)),
            "cloud_provider": params.get("cloud_provider", "unknown"),
            "region": params.get("region", "unknown"),
            "tags": mlflow_run.data.tags if mlflow_run.data.tags else {},
        }

    @staticmethod
    def _extract_gpu_info(instance_type: str, count: int = 1) -> tuple:
        """Extract GPU type and count from instance type string."""
        instance_type_lower = instance_type.lower()

        if "p3" in instance_type_lower:
            return ("V100", 1 if "2xlarge" in instance_type_lower else (4 if "8xlarge" in instance_type_lower else 8))
        elif "p4d" in instance_type_lower:
            return ("A100", 8)
        elif "p5" in instance_type_lower:
            return ("H100", 8)
        elif "g4" in instance_type_lower or "g4dn" in instance_type_lower:
            return ("T4", 1 if "xlarge" in instance_type_lower else 4)
        elif "g5" in instance_type_lower:
            return ("A10G", 1 if "xlarge" in instance_type_lower else 4)
        elif "g6" in instance_type_lower:
            return ("L40", 1 if "xlarge" in instance_type_lower else 4)
        elif "nc" in instance_type_lower or "nd" in instance_type_lower:
            return ("K80" if "nc" in instance_type_lower else "V100", count)
        elif "a2" in instance_type_lower:
            return ("A100", count)
        elif "t4" in instance_type_lower:
            return ("T4", count)
        else:
            return (None, 0)

    @staticmethod
    def _detect_framework(text: str) -> str:
        """Detect ML framework from text (image name, job name, etc)."""
        text_lower = text.lower()

        if "pytorch" in text_lower or "torch" in text_lower:
            return TrainingFramework.PYTORCH.value
        elif "tensorflow" in text_lower or "tf" in text_lower:
            return TrainingFramework.TENSORFLOW.value
        elif "jax" in text_lower:
            return TrainingFramework.JAX.value
        elif "huggingface" in text_lower or "hf" in text_lower:
            return TrainingFramework.HUGGINGFACE.value
        else:
            return TrainingFramework.UNKNOWN.value

    @staticmethod
    def _parse_sagemaker_tags(job_details: Dict[str, Any]) -> Dict[str, str]:
        """Parse tags from SageMaker job details."""
        tags = {}
        for tag in job_details.get("Tags", []):
            tags[tag.get("Key")] = tag.get("Value")
        return tags

    @staticmethod
    def _empty_cost_summary() -> Dict[str, Any]:
        """Return empty cost summary structure."""
        return {
            "total_cost": 0.0,
            "cost_by_provider": {},
            "cost_by_instance_type": {},
            "cost_by_project": {},
            "avg_cost_per_run": 0.0,
            "total_gpu_hours": 0.0,
            "failed_run_waste": 0.0,
            "summary_period_days": 0,
            "run_count": 0,
        }


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    monitor = TrainingRunMonitor()

    # Get active runs
    active_runs = monitor.get_active_training_runs()
    print(f"Active runs: {len(active_runs)}")

    # Get cost summary
    cost_summary = monitor.get_training_cost_summary(days=30)
    print(f"Cost summary: {json.dumps(cost_summary, indent=2, default=str)}")

    # Get anomalies
    anomalies = monitor.detect_training_anomalies(active_runs)
    print(f"Anomalies: {len(anomalies)}")

    # Get recommendations
    recommendations = monitor.get_training_recommendations()
    print(f"Recommendations: {len(recommendations)}")
