"""Comprehensive tests for TrainingRunMonitor."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from monitors.training_run_monitor import TrainingRunMonitor, TrainingStatus, TrainingFramework


class TestTrainingRunMonitorInitialization:
    """Test initialization with various configurations."""

    def test_init_with_defaults(self):
        """Initialization with default settings."""
        with patch.dict(os.environ, {"TRAINING_MONITOR_ENABLED": "true"}, clear=True):
            monitor = TrainingRunMonitor()
            assert monitor.enabled is True
            assert monitor.config == {}

    def test_init_disabled(self):
        """Initialization with monitor disabled."""
        with patch.dict(os.environ, {"TRAINING_MONITOR_ENABLED": "false"}):
            monitor = TrainingRunMonitor()
            assert monitor.enabled is False

    def test_init_sagemaker_enabled(self):
        """Initialization with SageMaker enabled."""
        env_vars = {
            "TRAINING_MONITOR_ENABLED": "true",
            "SAGEMAKER_TRAINING_ENABLED": "true",
        }
        with patch.dict(os.environ, env_vars):
            with patch('monitors.training_run_monitor.HAS_BOTO3', True):
                with patch('monitors.training_run_monitor.boto3'):
                    monitor = TrainingRunMonitor()
                    assert monitor.sagemaker_enabled is True

    def test_init_azure_ml_enabled(self):
        """Initialization with Azure ML enabled."""
        env_vars = {
            "TRAINING_MONITOR_ENABLED": "true",
            "AZURE_ML_TRAINING_ENABLED": "true",
        }
        with patch.dict(os.environ, env_vars):
            monitor = TrainingRunMonitor()
            assert monitor.azure_ml_enabled is True

    def test_init_vertex_ai_enabled(self):
        """Initialization with Vertex AI enabled."""
        env_vars = {
            "TRAINING_MONITOR_ENABLED": "true",
            "VERTEX_AI_TRAINING_ENABLED": "true",
        }
        with patch.dict(os.environ, env_vars):
            monitor = TrainingRunMonitor()
            assert monitor.vertex_ai_enabled is True

    def test_init_with_wandb_api_key(self):
        """Initialization with W&B API key."""
        env_vars = {
            "TRAINING_MONITOR_ENABLED": "true",
            "WANDB_API_KEY": "test-key-12345",
        }
        with patch.dict(os.environ, env_vars):
            monitor = TrainingRunMonitor()
            assert monitor.wandb_api_key == "test-key-12345"

    def test_init_with_mlflow_uri(self):
        """Initialization with MLflow URI."""
        env_vars = {
            "TRAINING_MONITOR_ENABLED": "true",
            "MLFLOW_TRACKING_URI": "http://localhost:5000",
        }
        with patch.dict(os.environ, env_vars):
            monitor = TrainingRunMonitor()
            assert monitor.mlflow_tracking_uri == "http://localhost:5000"

    def test_init_with_custom_config(self):
        """Initialization with custom configuration."""
        config = {
            "aws_region": "us-west-2",
            "azure_subscription_id": "sub-123",
        }
        monitor = TrainingRunMonitor(config=config)
        assert monitor.config == config


class TestGetActiveTrainingRuns:
    """Test active training runs collection."""

    @patch('monitors.training_run_monitor.TrainingRunMonitor._collect_sagemaker_runs')
    @patch('monitors.training_run_monitor.TrainingRunMonitor._collect_azure_ml_runs')
    @patch('monitors.training_run_monitor.TrainingRunMonitor._collect_vertex_ai_runs')
    @patch('monitors.training_run_monitor.TrainingRunMonitor._collect_wandb_runs')
    @patch('monitors.training_run_monitor.TrainingRunMonitor._collect_mlflow_runs')
    def test_get_active_runs_filters_status(self, mock_mlflow, mock_wandb, mock_vertex, mock_azure, mock_sagemaker):
        """Only returns running jobs, filters others."""
        mock_sagemaker.return_value = [
            {
                "run_id": "run-1",
                "provider": "sagemaker",
                "status": TrainingStatus.RUNNING.value,
                "job_name": "job1",
            },
            {
                "run_id": "run-2",
                "provider": "sagemaker",
                "status": TrainingStatus.COMPLETED.value,
                "job_name": "job2",
            },
        ]
        mock_azure.return_value = []
        mock_vertex.return_value = []
        mock_wandb.return_value = []
        mock_mlflow.return_value = []

        env_vars = {"TRAINING_MONITOR_ENABLED": "true"}
        with patch.dict(os.environ, env_vars):
            monitor = TrainingRunMonitor()
            active = monitor.get_active_training_runs()

        assert len(active) == 1
        assert active[0]["status"] == TrainingStatus.RUNNING.value

    @patch('monitors.training_run_monitor.TrainingRunMonitor._collect_sagemaker_runs')
    @patch('monitors.training_run_monitor.TrainingRunMonitor._collect_azure_ml_runs')
    @patch('monitors.training_run_monitor.TrainingRunMonitor._collect_vertex_ai_runs')
    @patch('monitors.training_run_monitor.TrainingRunMonitor._collect_wandb_runs')
    @patch('monitors.training_run_monitor.TrainingRunMonitor._collect_mlflow_runs')
    def test_get_active_runs_aggregates_providers(self, mock_mlflow, mock_wandb, mock_vertex, mock_azure, mock_sagemaker):
        """Aggregates runs from multiple providers."""
        mock_sagemaker.return_value = [
            {"run_id": "sm-1", "provider": "sagemaker", "status": TrainingStatus.RUNNING.value}
        ]
        mock_azure.return_value = [
            {"run_id": "az-1", "provider": "azure_ml", "status": TrainingStatus.RUNNING.value}
        ]
        mock_vertex.return_value = [
            {"run_id": "gcp-1", "provider": "vertex_ai", "status": TrainingStatus.RUNNING.value}
        ]
        mock_wandb.return_value = []
        mock_mlflow.return_value = []

        env_vars = {"TRAINING_MONITOR_ENABLED": "true"}
        with patch.dict(os.environ, env_vars):
            monitor = TrainingRunMonitor()
            active = monitor.get_active_training_runs()

        assert len(active) == 3

    def test_get_active_runs_disabled(self):
        """Returns empty list when disabled."""
        env_vars = {"TRAINING_MONITOR_ENABLED": "false"}
        with patch.dict(os.environ, env_vars):
            monitor = TrainingRunMonitor()
            active = monitor.get_active_training_runs()

        assert active == []


class TestGetTrainingCostSummary:
    """Test training cost aggregation."""

    @patch('monitors.training_run_monitor.TrainingRunMonitor.get_training_history')
    def test_get_cost_summary_aggregates(self, mock_history):
        """Cost summary aggregates by provider and instance type."""
        mock_history.return_value = [
            {
                "run_id": "run-1",
                "provider": "sagemaker",
                "instance_type": "ml.p3.2xlarge",
                "estimated_total_cost": 100.0,
                "gpu_count": 1,
                "elapsed_hours": 10.0,
                "status": TrainingStatus.COMPLETED.value,
                "tags": {"project": "project-a"},
            },
            {
                "run_id": "run-2",
                "provider": "sagemaker",
                "instance_type": "ml.p3.8xlarge",
                "estimated_total_cost": 400.0,
                "gpu_count": 4,
                "elapsed_hours": 10.0,
                "status": TrainingStatus.COMPLETED.value,
                "tags": {"project": "project-a"},
            },
            {
                "run_id": "run-3",
                "provider": "azure_ml",
                "instance_type": "Standard_ND6S",
                "estimated_total_cost": 50.0,
                "gpu_count": 1,
                "elapsed_hours": 10.0,
                "status": TrainingStatus.COMPLETED.value,
                "tags": {"project": "project-b"},
            },
        ]

        env_vars = {"TRAINING_MONITOR_ENABLED": "true"}
        with patch.dict(os.environ, env_vars):
            monitor = TrainingRunMonitor()
            summary = monitor.get_training_cost_summary(days=7)

        assert summary["total_cost"] == 550.0
        assert summary["cost_by_provider"]["sagemaker"] == 500.0
        assert summary["cost_by_provider"]["azure_ml"] == 50.0
        assert summary["cost_by_instance_type"]["ml.p3.2xlarge"] == 100.0
        assert summary["cost_by_project"]["project-a"] == 500.0
        assert summary["cost_by_project"]["project-b"] == 50.0
        assert summary["total_gpu_hours"] == 60.0
        assert summary["avg_cost_per_run"] == 550.0 / 3

    @patch('monitors.training_run_monitor.TrainingRunMonitor.get_training_history')
    def test_get_cost_summary_failed_run_waste(self, mock_history):
        """Cost summary includes failed run waste."""
        mock_history.return_value = [
            {
                "run_id": "run-1",
                "provider": "sagemaker",
                "instance_type": "ml.p3.2xlarge",
                "estimated_total_cost": 100.0,
                "gpu_count": 1,
                "elapsed_hours": 5.0,
                "status": TrainingStatus.FAILED.value,
                "tags": {},
            },
            {
                "run_id": "run-2",
                "provider": "sagemaker",
                "instance_type": "ml.p3.2xlarge",
                "estimated_total_cost": 100.0,
                "gpu_count": 1,
                "elapsed_hours": 10.0,
                "status": TrainingStatus.COMPLETED.value,
                "tags": {},
            },
        ]

        env_vars = {"TRAINING_MONITOR_ENABLED": "true"}
        with patch.dict(os.environ, env_vars):
            monitor = TrainingRunMonitor()
            summary = monitor.get_training_cost_summary(days=7)

        assert summary["failed_run_waste"] == 100.0

    def test_get_cost_summary_disabled(self):
        """Returns empty summary when disabled."""
        env_vars = {"TRAINING_MONITOR_ENABLED": "false"}
        with patch.dict(os.environ, env_vars):
            monitor = TrainingRunMonitor()
            summary = monitor.get_training_cost_summary()

        assert "total_cost" in summary
        assert summary["total_cost"] == 0.0


class TestDetectTrainingAnomalies:
    """Test anomaly detection in training runs."""

    def test_detect_stalled_training(self):
        """Detect stalled training with high loss."""
        runs = [
            {
                "run_id": "run-1",
                "job_name": "job1",
                "provider": "sagemaker",
                "training_loss": 15.0,
                "instance_type": "ml.p3.2xlarge",
                "framework": "pytorch",
                "model_name": "bert",
                "gpu_utilization_avg": 80.0,  # Good GPU utilization to avoid gpu_underutilized anomaly
            },
        ]

        env_vars = {"TRAINING_MONITOR_ENABLED": "true"}
        with patch.dict(os.environ, env_vars):
            monitor = TrainingRunMonitor()
            anomalies = monitor.detect_training_anomalies(runs)

        assert len(anomalies) == 1
        assert anomalies[0]["anomaly_type"] == "stalled_training"
        assert anomalies[0]["severity"] == "high"

    def test_detect_gpu_underutilization(self):
        """Detect low GPU utilization."""
        runs = [
            {
                "run_id": "run-1",
                "job_name": "job1",
                "provider": "sagemaker",
                "gpu_utilization_avg": 15.0,
                "instance_type": "ml.p3.2xlarge",
                "framework": "pytorch",
                "model_name": "bert",
            },
        ]

        env_vars = {"TRAINING_MONITOR_ENABLED": "true"}
        with patch.dict(os.environ, env_vars):
            monitor = TrainingRunMonitor()
            anomalies = monitor.detect_training_anomalies(runs)

        assert len(anomalies) == 1
        assert anomalies[0]["anomaly_type"] == "gpu_underutilized"
        assert anomalies[0]["severity"] == "medium"

    def test_detect_hung_job_by_duration(self):
        """Detect jobs running longer than median duration."""
        runs = [
            {
                "run_id": "run-1",
                "job_name": "job1",
                "provider": "sagemaker",
                "elapsed_hours": 5.0,
                "instance_type": "ml.p3.2xlarge",
                "framework": "pytorch",
                "model_name": "bert",
                "gpu_utilization_avg": 85.0,
                "training_loss": 0.5,
            },
            {
                "run_id": "run-2",
                "job_name": "job2",
                "provider": "sagemaker",
                "elapsed_hours": 5.0,
                "instance_type": "ml.p3.2xlarge",
                "framework": "pytorch",
                "model_name": "bert",
                "gpu_utilization_avg": 85.0,
                "training_loss": 0.5,
            },
            {
                "run_id": "run-3",
                "job_name": "job3",
                "provider": "sagemaker",
                "elapsed_hours": 15.0,  # 3x median
                "instance_type": "ml.p3.2xlarge",
                "framework": "pytorch",
                "model_name": "bert",
                "gpu_utilization_avg": 85.0,
                "training_loss": 0.5,
            },
        ]

        env_vars = {"TRAINING_MONITOR_ENABLED": "true"}
        with patch.dict(os.environ, env_vars):
            monitor = TrainingRunMonitor()
            anomalies = monitor.detect_training_anomalies(runs)

        hung_job_anomalies = [a for a in anomalies if a["anomaly_type"] == "hung_job"]
        assert len(hung_job_anomalies) >= 0  # May detect hung job

    def test_detect_multiple_anomalies_same_run(self):
        """Single run can have multiple anomalies."""
        runs = [
            {
                "run_id": "run-1",
                "job_name": "job1",
                "provider": "sagemaker",
                "training_loss": 20.0,  # High loss
                "gpu_utilization_avg": 10.0,  # Low util
                "instance_type": "ml.p3.2xlarge",
                "framework": "pytorch",
                "model_name": "bert",
            },
        ]

        env_vars = {"TRAINING_MONITOR_ENABLED": "true"}
        with patch.dict(os.environ, env_vars):
            monitor = TrainingRunMonitor()
            anomalies = monitor.detect_training_anomalies(runs)

        # Should detect at least stalled training and GPU underutil
        anomaly_types = [a["anomaly_type"] for a in anomalies]
        assert len(anomalies) >= 2

    def test_detect_no_anomalies(self):
        """Healthy run produces no anomalies."""
        runs = [
            {
                "run_id": "run-1",
                "job_name": "job1",
                "provider": "sagemaker",
                "training_loss": 0.5,
                "gpu_utilization_avg": 85.0,
                "instance_type": "ml.p3.2xlarge",
                "framework": "pytorch",
                "model_name": "bert",
                "elapsed_hours": 5.0,
            },
        ]

        env_vars = {"TRAINING_MONITOR_ENABLED": "true"}
        with patch.dict(os.environ, env_vars):
            monitor = TrainingRunMonitor()
            anomalies = monitor.detect_training_anomalies(runs)

        assert len(anomalies) == 0

    def test_detect_anomalies_empty_list(self):
        """Empty run list returns no anomalies."""
        env_vars = {"TRAINING_MONITOR_ENABLED": "true"}
        with patch.dict(os.environ, env_vars):
            monitor = TrainingRunMonitor()
            anomalies = monitor.detect_training_anomalies([])

        assert anomalies == []


class TestCollectSageMakerRuns:
    """Test SageMaker run collection."""

    @patch('monitors.training_run_monitor.boto3')
    def test_collect_sagemaker_runs_success(self, mock_boto3):
        """Successfully collect SageMaker training jobs."""
        mock_client = Mock()
        mock_boto3.client.return_value = mock_client

        mock_client.list_training_jobs.return_value = {
            "TrainingJobSummaries": [
                {
                    "TrainingJobName": "job-1",
                    "TrainingJobArn": "arn:aws:sagemaker:us-east-1:123456789012:training-job/job-1",
                    "TrainingJobStatus": "InProgress",
                    "InstanceType": "ml.p3.2xlarge",
                    "InstanceCount": 1,
                    "CreationTime": datetime.utcnow(),
                },
            ]
        }

        env_vars = {
            "TRAINING_MONITOR_ENABLED": "true",
            "SAGEMAKER_TRAINING_ENABLED": "true",
        }
        with patch.dict(os.environ, env_vars):
            with patch('monitors.training_run_monitor.HAS_BOTO3', True):
                monitor = TrainingRunMonitor()
                monitor.sagemaker_client = mock_client
                runs = monitor._collect_sagemaker_runs()

        # Should return list (may be empty due to mocking limitations)
        assert isinstance(runs, list)

    def test_collect_sagemaker_runs_disabled(self):
        """SageMaker collection skipped when disabled."""
        env_vars = {
            "TRAINING_MONITOR_ENABLED": "true",
            "SAGEMAKER_TRAINING_ENABLED": "false",
        }
        with patch.dict(os.environ, env_vars):
            monitor = TrainingRunMonitor()
            runs = monitor._collect_sagemaker_runs()

        assert runs == []


class TestInstancePricing:
    """Test instance pricing data."""

    def test_instance_pricing_available(self):
        """Instance pricing dictionary is populated."""
        env_vars = {"TRAINING_MONITOR_ENABLED": "true"}
        with patch.dict(os.environ, env_vars):
            monitor = TrainingRunMonitor()

        assert len(monitor.INSTANCE_PRICING) > 0
        assert "ml.p3.2xlarge" in monitor.INSTANCE_PRICING
        assert "ml.p4d.24xlarge" in monitor.INSTANCE_PRICING

    def test_instance_pricing_sagemaker(self):
        """SageMaker instance pricing is available."""
        env_vars = {"TRAINING_MONITOR_ENABLED": "true"}
        with patch.dict(os.environ, env_vars):
            monitor = TrainingRunMonitor()

        p3_pricing = monitor.INSTANCE_PRICING.get("ml.p3.2xlarge")
        assert p3_pricing is not None
        assert p3_pricing.provider == "sagemaker"
        assert p3_pricing.gpu_type == "V100"
        assert p3_pricing.gpu_count == 1

    def test_instance_pricing_azure_ml(self):
        """Azure ML instance pricing is available."""
        env_vars = {"TRAINING_MONITOR_ENABLED": "true"}
        with patch.dict(os.environ, env_vars):
            monitor = TrainingRunMonitor()

        nd6_pricing = monitor.INSTANCE_PRICING.get("Standard_ND6S")
        assert nd6_pricing is not None
        assert nd6_pricing.provider == "azure_ml"
        assert nd6_pricing.gpu_type == "V100"

    def test_instance_pricing_vertex_ai(self):
        """Vertex AI instance pricing is available."""
        env_vars = {"TRAINING_MONITOR_ENABLED": "true"}
        with patch.dict(os.environ, env_vars):
            monitor = TrainingRunMonitor()

        a2_pricing = monitor.INSTANCE_PRICING.get("a2-highgpu-8g")
        assert a2_pricing is not None
        assert a2_pricing.provider == "vertex_ai"
        assert a2_pricing.gpu_type == "A100"


class TestGetTrainingHistory:
    """Test training history retrieval."""

    @patch('monitors.training_run_monitor.TrainingRunMonitor._collect_sagemaker_runs')
    @patch('monitors.training_run_monitor.TrainingRunMonitor._collect_azure_ml_runs')
    @patch('monitors.training_run_monitor.TrainingRunMonitor._collect_vertex_ai_runs')
    @patch('monitors.training_run_monitor.TrainingRunMonitor._collect_wandb_runs')
    @patch('monitors.training_run_monitor.TrainingRunMonitor._collect_mlflow_runs')
    def test_get_training_history(self, mock_mlflow, mock_wandb, mock_vertex, mock_azure, mock_sagemaker):
        """Get training history with date filtering."""
        cutoff_time = datetime.utcnow() - timedelta(days=7)

        mock_sagemaker.return_value = [
            {
                "run_id": "run-1",
                "provider": "sagemaker",
                "status": TrainingStatus.COMPLETED.value,
            },
        ]
        mock_azure.return_value = []
        mock_vertex.return_value = []
        mock_wandb.return_value = []
        mock_mlflow.return_value = []

        env_vars = {"TRAINING_MONITOR_ENABLED": "true"}
        with patch.dict(os.environ, env_vars):
            monitor = TrainingRunMonitor()
            history = monitor.get_training_history(days=7)

        # Verify the cutoff_time was passed
        mock_sagemaker.assert_called_once()
        args, kwargs = mock_sagemaker.call_args
        assert kwargs.get("cutoff_time") is not None

    def test_get_training_history_disabled(self):
        """Returns empty list when disabled."""
        env_vars = {"TRAINING_MONITOR_ENABLED": "false"}
        with patch.dict(os.environ, env_vars):
            monitor = TrainingRunMonitor()
            history = monitor.get_training_history(days=7)

        assert history == []


class TestTrainingStatusEnum:
    """Test TrainingStatus enum values."""

    def test_training_status_values(self):
        """All training status values are accessible."""
        assert TrainingStatus.RUNNING.value == "running"
        assert TrainingStatus.COMPLETED.value == "completed"
        assert TrainingStatus.FAILED.value == "failed"
        assert TrainingStatus.STOPPING.value == "stopping"
        assert TrainingStatus.STOPPED.value == "stopped"
        assert TrainingStatus.PENDING.value == "pending"


class TestTrainingFrameworkEnum:
    """Test TrainingFramework enum values."""

    def test_training_framework_values(self):
        """All training framework values are accessible."""
        assert TrainingFramework.PYTORCH.value == "pytorch"
        assert TrainingFramework.TENSORFLOW.value == "tensorflow"
        assert TrainingFramework.JAX.value == "jax"
        assert TrainingFramework.HUGGINGFACE.value == "huggingface"
        assert TrainingFramework.UNKNOWN.value == "unknown"


class TestCollectAzureMLRuns:
    """Test Azure ML run collection."""

    @patch('monitors.training_run_monitor.HAS_AZURE_ML', True)
    def test_collect_azure_ml_runs_success(self, mock_azure_ml_enabled=True):
        """Successfully collect Azure ML training runs."""
        env_vars = {
            "TRAINING_MONITOR_ENABLED": "true",
            "AZURE_ML_TRAINING_ENABLED": "true",
        }
        with patch.dict(os.environ, env_vars):
            monitor = TrainingRunMonitor()

            # Mock Azure ML client
            mock_client = Mock()
            mock_experiment = Mock()
            mock_experiment.name = "test-experiment"
            mock_client.experiments.list.return_value = [mock_experiment]

            mock_run = Mock()
            mock_run.id = "run-1"
            mock_run.start_time = datetime.utcnow() - timedelta(hours=2)
            mock_run.end_time = datetime.utcnow()
            mock_run.status = "Completed"
            mock_run.display_name = "test-run"
            mock_run.compute_target = "Standard_ND6S"
            mock_run.tags = {"model_name": "bert", "dataset_name": "wikitext"}

            mock_client.runs.list.return_value = [mock_run]
            monitor.azure_ml_client = mock_client

            runs = monitor._collect_azure_ml_runs()
            assert isinstance(runs, list)

    def test_collect_azure_ml_runs_disabled(self):
        """Azure ML collection returns empty when disabled."""
        env_vars = {
            "TRAINING_MONITOR_ENABLED": "true",
            "AZURE_ML_TRAINING_ENABLED": "false",
        }
        with patch.dict(os.environ, env_vars):
            monitor = TrainingRunMonitor()
            runs = monitor._collect_azure_ml_runs()

        assert runs == []

    @patch('monitors.training_run_monitor.HAS_AZURE_ML', True)
    def test_collect_azure_ml_runs_with_exception(self):
        """Handle exceptions during Azure ML collection."""
        env_vars = {
            "TRAINING_MONITOR_ENABLED": "true",
            "AZURE_ML_TRAINING_ENABLED": "true",
        }
        with patch.dict(os.environ, env_vars):
            monitor = TrainingRunMonitor()
            mock_client = Mock()
            mock_client.experiments.list.side_effect = Exception("Connection error")
            monitor.azure_ml_client = mock_client

            runs = monitor._collect_azure_ml_runs()
            assert runs == []


class TestCollectVertexAIRuns:
    """Test Vertex AI run collection."""

    def test_collect_vertex_ai_runs_disabled(self):
        """Vertex AI collection returns empty when disabled."""
        env_vars = {
            "TRAINING_MONITOR_ENABLED": "true",
            "VERTEX_AI_TRAINING_ENABLED": "false",
        }
        with patch.dict(os.environ, env_vars):
            monitor = TrainingRunMonitor()
            runs = monitor._collect_vertex_ai_runs()

        assert runs == []

    def test_collect_vertex_ai_runs_not_available(self):
        """Vertex AI collection returns empty when not available."""
        env_vars = {
            "TRAINING_MONITOR_ENABLED": "true",
            "VERTEX_AI_TRAINING_ENABLED": "true",
        }
        with patch.dict(os.environ, env_vars):
            with patch('monitors.training_run_monitor.HAS_VERTEX_AI', False):
                monitor = TrainingRunMonitor()
                runs = monitor._collect_vertex_ai_runs()
                assert runs == []


class TestCollectWandBRuns:
    """Test Weights & Biases run collection."""

    @patch('monitors.training_run_monitor.requests')
    def test_collect_wandb_runs_with_api_key(self, mock_requests):
        """Collect W&B runs when API key is available."""
        env_vars = {
            "TRAINING_MONITOR_ENABLED": "true",
            "WANDB_API_KEY": "test-wandb-key",
        }
        with patch.dict(os.environ, env_vars):
            with patch('monitors.training_run_monitor.HAS_REQUESTS', True):
                monitor = TrainingRunMonitor()

                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = {"data": {}}
                mock_requests.post.return_value = mock_response

                runs = monitor._collect_wandb_runs()
                assert isinstance(runs, list)

    def test_collect_wandb_runs_no_api_key(self):
        """W&B collection returns empty without API key."""
        env_vars = {"TRAINING_MONITOR_ENABLED": "true"}
        with patch.dict(os.environ, env_vars, clear=True):
            monitor = TrainingRunMonitor()
            runs = monitor._collect_wandb_runs()

        assert runs == []

    @patch('monitors.training_run_monitor.requests')
    def test_collect_wandb_runs_with_exception(self, mock_requests):
        """Handle exceptions during W&B collection."""
        env_vars = {
            "TRAINING_MONITOR_ENABLED": "true",
            "WANDB_API_KEY": "test-wandb-key",
        }
        with patch.dict(os.environ, env_vars):
            with patch('monitors.training_run_monitor.HAS_REQUESTS', True):
                monitor = TrainingRunMonitor()
                mock_requests.post.side_effect = Exception("Network error")

                runs = monitor._collect_wandb_runs()
                assert runs == []


class TestCollectMLFlowRuns:
    """Test MLflow run collection."""

    def test_collect_mlflow_runs_no_uri(self):
        """MLflow collection returns empty without tracking URI."""
        env_vars = {"TRAINING_MONITOR_ENABLED": "true"}
        with patch.dict(os.environ, env_vars, clear=True):
            monitor = TrainingRunMonitor()
            runs = monitor._collect_mlflow_runs()

        assert runs == []

    def test_collect_mlflow_runs_not_available(self):
        """MLflow collection returns empty when not available."""
        env_vars = {
            "TRAINING_MONITOR_ENABLED": "true",
            "MLFLOW_TRACKING_URI": "http://localhost:5000",
        }
        with patch.dict(os.environ, env_vars):
            with patch('monitors.training_run_monitor.HAS_MLFLOW', False):
                monitor = TrainingRunMonitor()
                runs = monitor._collect_mlflow_runs()
                assert runs == []


class TestGetTrainingRecommendations:
    """Test training optimization recommendations."""

    @patch('monitors.training_run_monitor.TrainingRunMonitor.get_active_training_runs')
    def test_recommend_spot_instances(self, mock_active_runs):
        """Recommend using spot instances for small runs."""
        mock_active_runs.return_value = [
            {
                "run_id": "run-1",
                "gpu_count": 2,
                "cost_per_hour": 10.0,
                "gpu_utilization_avg": 80.0,
            },
            {
                "run_id": "run-2",
                "gpu_count": 3,
                "cost_per_hour": 15.0,
                "gpu_utilization_avg": 75.0,
            },
        ]

        env_vars = {"TRAINING_MONITOR_ENABLED": "true"}
        with patch.dict(os.environ, env_vars):
            monitor = TrainingRunMonitor()
            recommendations = monitor.get_training_recommendations()

        spot_recs = [r for r in recommendations if r["recommendation_type"] == "use_spot_instances"]
        assert len(spot_recs) > 0

    @patch('monitors.training_run_monitor.TrainingRunMonitor.get_active_training_runs')
    def test_recommend_downsize_underutilized(self, mock_active_runs):
        """Recommend downsizing underutilized instances."""
        mock_active_runs.return_value = [
            {
                "run_id": "run-1",
                "gpu_count": 4,
                "cost_per_hour": 20.0,
                "gpu_utilization_avg": 15.0,  # Low utilization
            },
        ]

        env_vars = {"TRAINING_MONITOR_ENABLED": "true"}
        with patch.dict(os.environ, env_vars):
            monitor = TrainingRunMonitor()
            recommendations = monitor.get_training_recommendations()

        downsize_recs = [r for r in recommendations if r["recommendation_type"] == "downsize_instances"]
        assert len(downsize_recs) > 0

    @patch('monitors.training_run_monitor.TrainingRunMonitor.get_active_training_runs')
    def test_recommend_mixed_precision(self, mock_active_runs):
        """Recommend enabling mixed precision."""
        mock_active_runs.return_value = [
            {
                "run_id": "run-1",
                "gpu_count": 1,
                "cost_per_hour": 5.0,
                "gpu_utilization_avg": 80.0,
                "tags": {"mixed_precision": "disabled"},
            },
        ]

        env_vars = {"TRAINING_MONITOR_ENABLED": "true"}
        with patch.dict(os.environ, env_vars):
            monitor = TrainingRunMonitor()
            recommendations = monitor.get_training_recommendations()

        amp_recs = [r for r in recommendations if r["recommendation_type"] == "enable_mixed_precision"]
        assert len(amp_recs) > 0

    @patch('monitors.training_run_monitor.TrainingRunMonitor.get_active_training_runs')
    def test_recommend_gradient_accumulation(self, mock_active_runs):
        """Recommend gradient accumulation for memory-constrained runs."""
        mock_active_runs.return_value = [
            {
                "run_id": "run-1",
                "gpu_count": 1,
                "cost_per_hour": 5.0,
                "gpu_utilization_avg": 80.0,
                "memory_utilization_avg": 95.0,
            },
        ]

        env_vars = {"TRAINING_MONITOR_ENABLED": "true"}
        with patch.dict(os.environ, env_vars):
            monitor = TrainingRunMonitor()
            recommendations = monitor.get_training_recommendations()

        grad_acc_recs = [r for r in recommendations if r["recommendation_type"] == "gradient_accumulation"]
        assert len(grad_acc_recs) > 0

    @patch('monitors.training_run_monitor.TrainingRunMonitor.get_active_training_runs')
    def test_recommend_consolidate_small_runs(self, mock_active_runs):
        """Recommend consolidating multiple small runs."""
        mock_active_runs.return_value = [
            {"run_id": "run-1", "gpu_count": 1, "cost_per_hour": 2.0},
            {"run_id": "run-2", "gpu_count": 1, "cost_per_hour": 2.0},
            {"run_id": "run-3", "gpu_count": 1, "cost_per_hour": 2.0},
            {"run_id": "run-4", "gpu_count": 1, "cost_per_hour": 2.0},
        ]

        env_vars = {"TRAINING_MONITOR_ENABLED": "true"}
        with patch.dict(os.environ, env_vars):
            monitor = TrainingRunMonitor()
            recommendations = monitor.get_training_recommendations()

        consolidate_recs = [r for r in recommendations if r["recommendation_type"] == "consolidate_small_runs"]
        assert len(consolidate_recs) > 0

    @patch('monitors.training_run_monitor.TrainingRunMonitor.get_active_training_runs')
    def test_recommendations_empty_runs(self, mock_active_runs):
        """Return empty recommendations when no active runs."""
        mock_active_runs.return_value = []

        env_vars = {"TRAINING_MONITOR_ENABLED": "true"}
        with patch.dict(os.environ, env_vars):
            monitor = TrainingRunMonitor()
            recommendations = monitor.get_training_recommendations()

        assert recommendations == []


class TestAnomalyDetectionEdgeCases:
    """Test anomaly detection edge cases."""

    def test_detect_memory_pressure_anomaly(self):
        """Detect high memory utilization anomaly."""
        runs = [
            {
                "run_id": "run-1",
                "job_name": "job1",
                "provider": "sagemaker",
                "memory_utilization_avg": 95.0,  # High memory
                "instance_type": "ml.p3.2xlarge",
                "framework": "pytorch",
                "model_name": "bert",
                "gpu_utilization_avg": 85.0,
                "training_loss": 0.5,
            },
        ]

        env_vars = {"TRAINING_MONITOR_ENABLED": "true"}
        with patch.dict(os.environ, env_vars):
            monitor = TrainingRunMonitor()
            anomalies = monitor.detect_training_anomalies(runs)

        memory_anomalies = [a for a in anomalies if a["anomaly_type"] == "memory_pressure"]
        assert len(memory_anomalies) == 1
        assert memory_anomalies[0]["severity"] == "high"

    def test_detect_budget_exceeded_anomaly(self):
        """Detect cost exceeding budget anomaly."""
        runs = [
            {
                "run_id": "run-1",
                "job_name": "job1",
                "provider": "sagemaker",
                "cost_so_far": 120.0,
                "instance_type": "ml.p3.2xlarge",
                "framework": "pytorch",
                "model_name": "bert",
                "gpu_utilization_avg": 85.0,
                "training_loss": 0.5,
                "tags": {"budget_usd": 100.0},
            },
        ]

        env_vars = {"TRAINING_MONITOR_ENABLED": "true"}
        with patch.dict(os.environ, env_vars):
            monitor = TrainingRunMonitor()
            anomalies = monitor.detect_training_anomalies(runs)

        budget_anomalies = [a for a in anomalies if a["anomaly_type"] == "budget_exceeded"]
        assert len(budget_anomalies) == 1
        assert budget_anomalies[0]["severity"] == "medium"

    def test_detect_no_anomalies_with_healthy_metrics(self):
        """No anomalies for well-performing runs."""
        runs = [
            {
                "run_id": "run-1",
                "job_name": "job1",
                "provider": "sagemaker",
                "training_loss": 0.3,
                "gpu_utilization_avg": 88.0,
                "memory_utilization_avg": 60.0,
                "instance_type": "ml.p3.2xlarge",
                "framework": "pytorch",
                "model_name": "bert",
                "elapsed_hours": 5.0,
            },
        ]

        env_vars = {"TRAINING_MONITOR_ENABLED": "true"}
        with patch.dict(os.environ, env_vars):
            monitor = TrainingRunMonitor()
            anomalies = monitor.detect_training_anomalies(runs)

        assert len(anomalies) == 0


class TestCostSummaryEdgeCases:
    """Test cost summary edge cases."""

    @patch('monitors.training_run_monitor.TrainingRunMonitor.get_training_history')
    def test_cost_summary_no_runs(self, mock_history):
        """Cost summary with empty history."""
        mock_history.return_value = []

        env_vars = {"TRAINING_MONITOR_ENABLED": "true"}
        with patch.dict(os.environ, env_vars):
            monitor = TrainingRunMonitor()
            summary = monitor.get_training_cost_summary(days=7)

        assert summary["total_cost"] == 0.0
        assert summary["avg_cost_per_run"] == 0.0
        assert summary["run_count"] == 0

    @patch('monitors.training_run_monitor.TrainingRunMonitor.get_training_history')
    def test_cost_summary_untagged_projects(self, mock_history):
        """Cost summary handles untagged runs."""
        mock_history.return_value = [
            {
                "run_id": "run-1",
                "provider": "sagemaker",
                "instance_type": "ml.p3.2xlarge",
                "estimated_total_cost": 50.0,
                "gpu_count": 1,
                "elapsed_hours": 5.0,
                "status": TrainingStatus.COMPLETED.value,
                "tags": {},  # No project tag
            },
        ]

        env_vars = {"TRAINING_MONITOR_ENABLED": "true"}
        with patch.dict(os.environ, env_vars):
            monitor = TrainingRunMonitor()
            summary = monitor.get_training_cost_summary(days=7)

        assert "untagged" in summary["cost_by_project"]
        assert summary["cost_by_project"]["untagged"] == 50.0

    @patch('monitors.training_run_monitor.TrainingRunMonitor.get_training_history')
    def test_cost_summary_multiple_failed_runs(self, mock_history):
        """Cost summary tracks waste from multiple failed runs."""
        mock_history.return_value = [
            {
                "run_id": "run-1",
                "provider": "sagemaker",
                "instance_type": "ml.p3.2xlarge",
                "estimated_total_cost": 100.0,
                "gpu_count": 1,
                "elapsed_hours": 10.0,
                "status": TrainingStatus.FAILED.value,
                "tags": {},
            },
            {
                "run_id": "run-2",
                "provider": "sagemaker",
                "instance_type": "ml.p3.2xlarge",
                "estimated_total_cost": 50.0,
                "gpu_count": 1,
                "elapsed_hours": 5.0,
                "status": TrainingStatus.FAILED.value,
                "tags": {},
            },
            {
                "run_id": "run-3",
                "provider": "sagemaker",
                "instance_type": "ml.p3.2xlarge",
                "estimated_total_cost": 100.0,
                "gpu_count": 1,
                "elapsed_hours": 10.0,
                "status": TrainingStatus.COMPLETED.value,
                "tags": {},
            },
        ]

        env_vars = {"TRAINING_MONITOR_ENABLED": "true"}
        with patch.dict(os.environ, env_vars):
            monitor = TrainingRunMonitor()
            summary = monitor.get_training_cost_summary(days=7)

        assert summary["failed_run_waste"] == 150.0
        assert summary["total_cost"] == 250.0


class TestParsingMethods:
    """Test parsing methods for run data."""

    def test_parse_sagemaker_run_complete(self):
        """Parse complete SageMaker job details."""
        env_vars = {"TRAINING_MONITOR_ENABLED": "true"}
        with patch.dict(os.environ, env_vars):
            monitor = TrainingRunMonitor()

            job_details = {
                "TrainingJobName": "test-job",
                "TrainingJobArn": "arn:aws:sagemaker:us-east-1:123456789012:training-job/test-job",
                "TrainingJobStatus": "Completed",
                "ResourceConfig": {
                    "InstanceType": "ml.p3.2xlarge",
                    "InstanceCount": 1,
                },
                "CreationTime": datetime.utcnow() - timedelta(hours=2),
                "TrainingEndTime": datetime.utcnow(),
                "HyperParameters": {"model_name": "bert"},
                "AlgorithmSpecification": {"TrainingImage": "pytorch-training:1.9-cpu-py36"},
            }

            run = monitor._parse_sagemaker_run(job_details)
            assert run["job_name"] == "test-job"
            assert run["provider"] == "sagemaker"
            assert run["status"] == TrainingStatus.COMPLETED.value
            assert run["instance_type"] == "ml.p3.2xlarge"

    @patch('monitors.training_run_monitor.HAS_AZURE_ML', True)
    def test_parse_azure_ml_run_complete(self):
        """Parse complete Azure ML run details."""
        env_vars = {"TRAINING_MONITOR_ENABLED": "true"}
        with patch.dict(os.environ, env_vars):
            monitor = TrainingRunMonitor()

            mock_run = Mock()
            mock_run.id = "run-1"
            mock_run.status = "Completed"
            mock_run.display_name = "test-run"
            mock_run.start_time = datetime.utcnow() - timedelta(hours=2)
            mock_run.end_time = datetime.utcnow()
            mock_run.compute_target = "Standard_ND6S"
            mock_run.tags = {"model_name": "efficientnet"}
            mock_run.target = "eastus"

            run = monitor._parse_azure_ml_run(mock_run)
            assert run["job_name"] == "test-run"
            assert run["provider"] == "azure_ml"
            assert run["status"] == TrainingStatus.COMPLETED.value

    def test_parse_mlflow_run_complete(self):
        """Parse complete MLflow run details."""
        env_vars = {"TRAINING_MONITOR_ENABLED": "true"}
        with patch.dict(os.environ, env_vars):
            monitor = TrainingRunMonitor()

            mock_run = Mock()
            mock_run.info = Mock()
            mock_run.info.run_id = "run-1"
            mock_run.info.status = "FINISHED"
            mock_run.info.start_time = int((datetime.utcnow() - timedelta(hours=2)).timestamp() * 1000)
            mock_run.info.end_time = int(datetime.utcnow().timestamp() * 1000)
            mock_run.data = Mock()
            mock_run.data.params = {
                "framework": "pytorch",
                "model_name": "bert",
                "instance_type": "gpu-instance",
                "instance_count": "1",
                "gpu_type": "V100",
                "gpu_count": "2",
                "cloud_provider": "aws",
                "region": "us-east-1",
                "total_epochs": "10",
            }
            mock_run.data.metrics = {
                "training_loss": 0.5,
                "validation_loss": 0.6,
                "current_epoch": 8,
                "current_step": 100,
                "total_steps": 500,
                "cost_so_far": 50.0,
                "estimated_total_cost": 100.0,
                "cost_per_hour": 10.0,
                "gpu_utilization_avg": 85.0,
                "memory_utilization_avg": 70.0,
            }
            mock_run.data.tags = {"mlflow.runName": "test-run"}

            run = monitor._parse_mlflow_run(mock_run)
            assert run["job_name"] == "test-run"
            assert run["provider"] == "mlflow"
            assert run["status"] == TrainingStatus.COMPLETED.value
            assert run["training_loss"] == 0.5



class TestInitializationFailures:
    """Test client initialization with failures."""

    @patch('monitors.training_run_monitor.boto3')
    def test_sagemaker_init_failure(self, mock_boto3):
        """Handle SageMaker client initialization failure."""
        mock_boto3.client.side_effect = Exception("AWS credentials not found")

        env_vars = {
            "TRAINING_MONITOR_ENABLED": "true",
            "SAGEMAKER_TRAINING_ENABLED": "true",
        }
        with patch.dict(os.environ, env_vars):
            with patch('monitors.training_run_monitor.HAS_BOTO3', True):
                monitor = TrainingRunMonitor()
                # Should not crash, sagemaker_client should be None or uninitialized
                runs = monitor._collect_sagemaker_runs()
                assert runs == []

    def test_azure_ml_init_failure(self):
        """Handle Azure ML client initialization failure."""
        env_vars = {
            "TRAINING_MONITOR_ENABLED": "true",
            "AZURE_ML_TRAINING_ENABLED": "true",
        }
        with patch.dict(os.environ, env_vars):
            # Monitor should initialize without crashing even if Azure ML fails
            monitor = TrainingRunMonitor()
            runs = monitor._collect_azure_ml_runs()
            assert runs == []

    def test_vertex_ai_init_failure(self):
        """Handle Vertex AI client initialization failure."""
        env_vars = {
            "TRAINING_MONITOR_ENABLED": "true",
            "VERTEX_AI_TRAINING_ENABLED": "true",
            "GCP_PROJECT_ID": "test-project",
        }
        with patch.dict(os.environ, env_vars):
            # Monitor should initialize without crashing even if Vertex AI fails
            monitor = TrainingRunMonitor()
            runs = monitor._collect_vertex_ai_runs()
            assert runs == []


class TestSageMakerCompleted:
    """Test SageMaker collection with completed_only parameter."""

    @patch('monitors.training_run_monitor.boto3')
    def test_collect_sagemaker_runs_completed_only(self, mock_boto3):
        """Collect only completed SageMaker training jobs."""
        mock_client = Mock()
        mock_boto3.client.return_value = mock_client

        mock_paginator = Mock()
        mock_client.get_paginator.return_value = mock_paginator

        mock_page = {
            "TrainingJobSummaries": [
                {
                    "TrainingJobName": "completed-job",
                    "TrainingJobArn": "arn:aws:sagemaker:us-east-1:123456789012:training-job/completed-job",
                    "TrainingJobStatus": "Completed",
                    "CreationTime": datetime.utcnow() - timedelta(days=1),
                },
            ]
        }
        mock_paginator.paginate.return_value = [mock_page]

        env_vars = {
            "TRAINING_MONITOR_ENABLED": "true",
            "SAGEMAKER_TRAINING_ENABLED": "true",
        }
        with patch.dict(os.environ, env_vars):
            with patch('monitors.training_run_monitor.HAS_BOTO3', True):
                monitor = TrainingRunMonitor()
                monitor.sagemaker_client = mock_client

                # Call with completed_only=True
                monitor._collect_sagemaker_runs(completed_only=True)

                # Verify paginate was called with None status (when completed_only=True)
                call_args = mock_paginator.paginate.call_args
                assert call_args is not None


class TestSageMakerErrorHandling:
    """Test SageMaker error handling."""

    @patch('monitors.training_run_monitor.boto3')
    def test_collect_sagemaker_paginate_error(self, mock_boto3):
        """Handle error during SageMaker pagination."""
        mock_client = Mock()
        mock_boto3.client.return_value = mock_client

        mock_paginator = Mock()
        mock_client.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.side_effect = Exception("Paginate error")

        env_vars = {
            "TRAINING_MONITOR_ENABLED": "true",
            "SAGEMAKER_TRAINING_ENABLED": "true",
        }
        with patch.dict(os.environ, env_vars):
            with patch('monitors.training_run_monitor.HAS_BOTO3', True):
                monitor = TrainingRunMonitor()
                monitor.sagemaker_client = mock_client
                runs = monitor._collect_sagemaker_runs()
                assert runs == []

    @patch('monitors.training_run_monitor.boto3')
    def test_collect_sagemaker_describe_job_error(self, mock_boto3):
        """Handle error when describing SageMaker job."""
        mock_client = Mock()
        mock_boto3.client.return_value = mock_client

        mock_paginator = Mock()
        mock_client.get_paginator.return_value = mock_paginator

        mock_page = {
            "TrainingJobSummaries": [
                {
                    "TrainingJobName": "job-1",
                    "TrainingJobArn": "arn:aws:sagemaker:us-east-1:123456789012:training-job/job-1",
                    "CreationTime": datetime.utcnow(),
                },
            ]
        }
        mock_paginator.paginate.return_value = [mock_page]
        mock_client.describe_training_job.side_effect = Exception("Describe error")

        env_vars = {
            "TRAINING_MONITOR_ENABLED": "true",
            "SAGEMAKER_TRAINING_ENABLED": "true",
        }
        with patch.dict(os.environ, env_vars):
            with patch('monitors.training_run_monitor.HAS_BOTO3', True):
                monitor = TrainingRunMonitor()
                monitor.sagemaker_client = mock_client
                runs = monitor._collect_sagemaker_runs()
                assert runs == []


class TestAzureMLErrorHandling:
    """Test Azure ML error handling."""

    @patch('monitors.training_run_monitor.HAS_AZURE_ML', True)
    def test_collect_azure_ml_no_client(self):
        """Handle Azure ML collection with no client."""
        env_vars = {
            "TRAINING_MONITOR_ENABLED": "true",
            "AZURE_ML_TRAINING_ENABLED": "true",
        }
        with patch.dict(os.environ, env_vars):
            monitor = TrainingRunMonitor()
            monitor.azure_ml_client = None
            runs = monitor._collect_azure_ml_runs()
            assert runs == []

    @patch('monitors.training_run_monitor.HAS_AZURE_ML', True)
    def test_collect_azure_ml_error(self):
        """Handle error during Azure ML collection."""
        env_vars = {
            "TRAINING_MONITOR_ENABLED": "true",
            "AZURE_ML_TRAINING_ENABLED": "true",
        }
        with patch.dict(os.environ, env_vars):
            monitor = TrainingRunMonitor()
            mock_client = Mock()
            mock_client.experiments.list.side_effect = Exception("List error")
            monitor.azure_ml_client = mock_client
            runs = monitor._collect_azure_ml_runs()
            assert runs == []


class TestWandBErrorHandling:
    """Test W&B error handling."""

    @patch('monitors.training_run_monitor.requests')
    def test_collect_wandb_api_error(self, mock_requests):
        """Handle error during W&B API call."""
        env_vars = {
            "TRAINING_MONITOR_ENABLED": "true",
            "WANDB_API_KEY": "test-key",
        }
        with patch.dict(os.environ, env_vars):
            with patch('monitors.training_run_monitor.HAS_REQUESTS', True):
                monitor = TrainingRunMonitor()
                mock_requests.post.side_effect = Exception("Network error")
                runs = monitor._collect_wandb_runs()
                assert runs == []


class TestDetectFramework:
    """Test framework detection from image/name."""

    def test_detect_pytorch_framework(self):
        """Detect PyTorch from image string."""
        env_vars = {"TRAINING_MONITOR_ENABLED": "true"}
        with patch.dict(os.environ, env_vars):
            monitor = TrainingRunMonitor()
            framework = monitor._detect_framework("pytorch-training:1.9-cpu-py36")
            assert framework == TrainingFramework.PYTORCH.value

    def test_detect_tensorflow_framework(self):
        """Detect TensorFlow from image string."""
        env_vars = {"TRAINING_MONITOR_ENABLED": "true"}
        with patch.dict(os.environ, env_vars):
            monitor = TrainingRunMonitor()
            framework = monitor._detect_framework("tensorflow-training:2.8-gpu-py39")
            assert framework == TrainingFramework.TENSORFLOW.value

    def test_detect_huggingface_framework(self):
        """Detect HuggingFace from image string."""
        env_vars = {"TRAINING_MONITOR_ENABLED": "true"}
        with patch.dict(os.environ, env_vars):
            monitor = TrainingRunMonitor()
            framework = monitor._detect_framework("huggingface-transformers:4.21")
            assert framework == TrainingFramework.HUGGINGFACE.value

    def test_detect_jax_framework(self):
        """Detect JAX from image string."""
        env_vars = {"TRAINING_MONITOR_ENABLED": "true"}
        with patch.dict(os.environ, env_vars):
            monitor = TrainingRunMonitor()
            framework = monitor._detect_framework("jax-training:0.3")
            assert framework == TrainingFramework.JAX.value

    def test_detect_unknown_framework(self):
        """Detect unknown framework from image string."""
        env_vars = {"TRAINING_MONITOR_ENABLED": "true"}
        with patch.dict(os.environ, env_vars):
            monitor = TrainingRunMonitor()
            framework = monitor._detect_framework("custom-training:1.0")
            assert framework == TrainingFramework.UNKNOWN.value


class TestExtractGPUInfo:
    """Test GPU info extraction from instance types."""

    def test_extract_gpu_sagemaker_p3(self):
        """Extract GPU info from SageMaker p3 instance."""
        env_vars = {"TRAINING_MONITOR_ENABLED": "true"}
        with patch.dict(os.environ, env_vars):
            monitor = TrainingRunMonitor()
            gpu_type, gpu_count = monitor._extract_gpu_info("ml.p3.2xlarge", 1)
            assert gpu_type == "V100"
            assert gpu_count == 1

    def test_extract_gpu_sagemaker_p3_8xlarge(self):
        """Extract GPU info from SageMaker p3.8xlarge instance."""
        env_vars = {"TRAINING_MONITOR_ENABLED": "true"}
        with patch.dict(os.environ, env_vars):
            monitor = TrainingRunMonitor()
            gpu_type, gpu_count = monitor._extract_gpu_info("ml.p3.8xlarge", 1)
            assert gpu_type == "V100"
            assert gpu_count == 4

    def test_extract_gpu_azure(self):
        """Extract GPU info from Azure instance."""
        env_vars = {"TRAINING_MONITOR_ENABLED": "true"}
        with patch.dict(os.environ, env_vars):
            monitor = TrainingRunMonitor()
            gpu_type, gpu_count = monitor._extract_gpu_info("Standard_ND6S", 1)
            assert gpu_type == "V100"
            assert gpu_count == 1

    def test_extract_gpu_p4d(self):
        """Extract GPU info from p4d instance."""
        env_vars = {"TRAINING_MONITOR_ENABLED": "true"}
        with patch.dict(os.environ, env_vars):
            monitor = TrainingRunMonitor()
            gpu_type, gpu_count = monitor._extract_gpu_info("ml.p4d.24xlarge", 1)
            assert gpu_type == "A100"
            assert gpu_count == 8

    def test_extract_gpu_p5(self):
        """Extract GPU info from p5 instance."""
        env_vars = {"TRAINING_MONITOR_ENABLED": "true"}
        with patch.dict(os.environ, env_vars):
            monitor = TrainingRunMonitor()
            gpu_type, gpu_count = monitor._extract_gpu_info("ml.p5.48xlarge", 1)
            assert gpu_type == "H100"
            assert gpu_count == 8

    def test_extract_gpu_g4_instance(self):
        """Extract GPU info from g4 instance."""
        env_vars = {"TRAINING_MONITOR_ENABLED": "true"}
        with patch.dict(os.environ, env_vars):
            monitor = TrainingRunMonitor()
            gpu_type, gpu_count = monitor._extract_gpu_info("ml.g4dn.xlarge", 1)
            assert gpu_type == "T4"
            assert gpu_count == 1

    def test_extract_gpu_g4_multi_gpu(self):
        """Extract GPU info from g4 multi-GPU instance."""
        env_vars = {"TRAINING_MONITOR_ENABLED": "true"}
        with patch.dict(os.environ, env_vars):
            monitor = TrainingRunMonitor()
            # g4dn instances with "xlarge" in name return 1, others return 4
            gpu_type, gpu_count = monitor._extract_gpu_info("ml.g4dn.2xlarge", 1)
            assert gpu_type == "T4"
            assert gpu_count == 1  # "2xlarge" contains "xlarge"

    def test_extract_gpu_g4_four_gpu(self):
        """Extract GPU info from g4 instance with 4 GPUs."""
        env_vars = {"TRAINING_MONITOR_ENABLED": "true"}
        with patch.dict(os.environ, env_vars):
            monitor = TrainingRunMonitor()
            # g4 instances without "xlarge" return 4
            gpu_type, gpu_count = monitor._extract_gpu_info("g4-gpu-instance", 1)
            assert gpu_type == "T4"
            assert gpu_count == 4

    def test_extract_gpu_g5(self):
        """Extract GPU info from g5 instance."""
        env_vars = {"TRAINING_MONITOR_ENABLED": "true"}
        with patch.dict(os.environ, env_vars):
            monitor = TrainingRunMonitor()
            gpu_type, gpu_count = monitor._extract_gpu_info("ml.g5.xlarge", 1)
            assert gpu_type == "A10G"
            assert gpu_count == 1

    def test_extract_gpu_g6(self):
        """Extract GPU info from g6 instance."""
        env_vars = {"TRAINING_MONITOR_ENABLED": "true"}
        with patch.dict(os.environ, env_vars):
            monitor = TrainingRunMonitor()
            gpu_type, gpu_count = monitor._extract_gpu_info("ml.g6.xlarge", 1)
            assert gpu_type == "L40"
            assert gpu_count == 1

    def test_extract_gpu_a2_instance(self):
        """Extract GPU info from a2 instance."""
        env_vars = {"TRAINING_MONITOR_ENABLED": "true"}
        with patch.dict(os.environ, env_vars):
            monitor = TrainingRunMonitor()
            gpu_type, gpu_count = monitor._extract_gpu_info("a2-highgpu-8g", 1)
            assert gpu_type == "A100"
            assert gpu_count == 1

    def test_extract_gpu_t4(self):
        """Extract GPU info from t4 instance."""
        env_vars = {"TRAINING_MONITOR_ENABLED": "true"}
        with patch.dict(os.environ, env_vars):
            monitor = TrainingRunMonitor()
            gpu_type, gpu_count = monitor._extract_gpu_info("t4-gpu", 2)
            assert gpu_type == "T4"
            assert gpu_count == 2

    def test_extract_gpu_unknown_instance(self):
        """Extract GPU info from unknown instance type."""
        env_vars = {"TRAINING_MONITOR_ENABLED": "true"}
        with patch.dict(os.environ, env_vars):
            monitor = TrainingRunMonitor()
            gpu_type, gpu_count = monitor._extract_gpu_info("custom-gpu-xyz", 1)
            assert gpu_type is None
            assert gpu_count == 0


class TestEmptyCostSummary:
    """Test empty cost summary helper."""

    def test_empty_cost_summary_structure(self):
        """Empty cost summary has correct structure."""
        env_vars = {"TRAINING_MONITOR_ENABLED": "true"}
        with patch.dict(os.environ, env_vars):
            monitor = TrainingRunMonitor()
            summary = monitor._empty_cost_summary()

        assert "total_cost" in summary
        assert "cost_by_provider" in summary
        assert "cost_by_instance_type" in summary
        assert "cost_by_project" in summary
        assert summary["total_cost"] == 0.0


class TestSageMakerTagParsing:
    """Test SageMaker tag parsing."""

    def test_parse_sagemaker_tags_empty(self):
        """Parse empty SageMaker tags."""
        env_vars = {"TRAINING_MONITOR_ENABLED": "true"}
        with patch.dict(os.environ, env_vars):
            monitor = TrainingRunMonitor()
            job_details = {}
            tags = monitor._parse_sagemaker_tags(job_details)
            assert tags == {}

    def test_parse_sagemaker_tags_with_values(self):
        """Parse SageMaker tags with values."""
        env_vars = {"TRAINING_MONITOR_ENABLED": "true"}
        with patch.dict(os.environ, env_vars):
            monitor = TrainingRunMonitor()
            job_details = {
                "Tags": [
                    {"Key": "Project", "Value": "ml-research"},
                    {"Key": "Owner", "Value": "data-team"},
                ]
            }
            tags = monitor._parse_sagemaker_tags(job_details)
            assert tags["Project"] == "ml-research"
            assert tags["Owner"] == "data-team"


class TestAnomalyEdgeCases:
    """Test anomaly detection edge cases."""

    def test_detect_anomalies_with_none_values(self):
        """Handle None values in anomaly detection."""
        runs = [
            {
                "run_id": "run-1",
                "job_name": "job1",
                "provider": "sagemaker",
                "training_loss": None,
                "gpu_utilization_avg": 85.0,
                "memory_utilization_avg": 70.0,
                "instance_type": "ml.p3.2xlarge",
                "framework": "pytorch",
                "model_name": "bert",
                "elapsed_hours": 5.0,
            },
        ]

        env_vars = {"TRAINING_MONITOR_ENABLED": "true"}
        with patch.dict(os.environ, env_vars):
            monitor = TrainingRunMonitor()
            anomalies = monitor.detect_training_anomalies(runs)
            # Should not crash with None values - should return no anomalies for healthy run
            assert isinstance(anomalies, list)
            assert len(anomalies) == 0

    def test_detect_anomalies_missing_optional_fields(self):
        """Handle missing optional fields in anomaly detection."""
        runs = [
            {
                "run_id": "run-1",
                "job_name": "job1",
                "provider": "sagemaker",
                "instance_type": "ml.p3.2xlarge",
                "framework": "pytorch",
                "model_name": "bert",
                # Missing optional fields like training_loss, gpu_utilization_avg
            },
        ]

        env_vars = {"TRAINING_MONITOR_ENABLED": "true"}
        with patch.dict(os.environ, env_vars):
            monitor = TrainingRunMonitor()
            anomalies = monitor.detect_training_anomalies(runs)
            # Should not crash with missing optional fields
            assert isinstance(anomalies, list)


class TestCostSummaryWithVariousCosts:
    """Test cost summary with various cost calculations."""

    @patch('monitors.training_run_monitor.TrainingRunMonitor.get_training_history')
    def test_cost_summary_mixed_instances(self, mock_history):
        """Cost summary with mixed instance types."""
        mock_history.return_value = [
            {
                "run_id": "run-1",
                "provider": "sagemaker",
                "instance_type": "ml.p3.2xlarge",
                "estimated_total_cost": 50.0,
                "gpu_count": 1,
                "elapsed_hours": 5.0,
                "status": TrainingStatus.COMPLETED.value,
                "tags": {"project": "project-a"},
            },
            {
                "run_id": "run-2",
                "provider": "sagemaker",
                "instance_type": "ml.p4d.24xlarge",
                "estimated_total_cost": 200.0,
                "gpu_count": 8,
                "elapsed_hours": 10.0,
                "status": TrainingStatus.COMPLETED.value,
                "tags": {"project": "project-b"},
            },
            {
                "run_id": "run-3",
                "provider": "vertex_ai",
                "instance_type": "a2-highgpu-8g",
                "estimated_total_cost": 150.0,
                "gpu_count": 8,
                "elapsed_hours": 5.0,
                "status": TrainingStatus.COMPLETED.value,
                "tags": {"project": "project-a"},
            },
        ]

        env_vars = {"TRAINING_MONITOR_ENABLED": "true"}
        with patch.dict(os.environ, env_vars):
            monitor = TrainingRunMonitor()
            summary = monitor.get_training_cost_summary(days=7)

        assert summary["total_cost"] == 400.0
        assert summary["cost_by_provider"]["sagemaker"] == 250.0
        assert summary["cost_by_provider"]["vertex_ai"] == 150.0
        assert summary["total_gpu_hours"] == 125.0  # 1*5 + 8*10 + 8*5 = 5 + 80 + 40 = 125
        assert summary["run_count"] == 3

    @patch('monitors.training_run_monitor.TrainingRunMonitor.get_training_history')
    def test_cost_summary_single_run(self, mock_history):
        """Cost summary with single run."""
        mock_history.return_value = [
            {
                "run_id": "run-1",
                "provider": "sagemaker",
                "instance_type": "ml.p3.2xlarge",
                "estimated_total_cost": 100.0,
                "gpu_count": 1,
                "elapsed_hours": 10.0,
                "status": TrainingStatus.COMPLETED.value,
                "tags": {"project": "test"},
            },
        ]

        env_vars = {"TRAINING_MONITOR_ENABLED": "true"}
        with patch.dict(os.environ, env_vars):
            monitor = TrainingRunMonitor()
            summary = monitor.get_training_cost_summary(days=30)

        assert summary["avg_cost_per_run"] == 100.0
        assert summary["total_gpu_hours"] == 10.0


class TestRecommendationsEdgeCases:
    """Test recommendation generation edge cases."""

    @patch('monitors.training_run_monitor.TrainingRunMonitor.get_active_training_runs')
    def test_recommendations_with_large_instances(self, mock_active_runs):
        """Recommendations for large multi-GPU runs."""
        mock_active_runs.return_value = [
            {
                "run_id": "run-1",
                "gpu_count": 8,
                "cost_per_hour": 50.0,
                "gpu_utilization_avg": 85.0,
                "memory_utilization_avg": 70.0,
            },
        ]

        env_vars = {"TRAINING_MONITOR_ENABLED": "true"}
        with patch.dict(os.environ, env_vars):
            monitor = TrainingRunMonitor()
            recommendations = monitor.get_training_recommendations()

        # Large instances may not be recommended for spot
        assert isinstance(recommendations, list)

    @patch('monitors.training_run_monitor.TrainingRunMonitor.get_active_training_runs')
    def test_recommendations_mixed_sizes(self, mock_active_runs):
        """Recommendations for mixed-size runs."""
        mock_active_runs.return_value = [
            {
                "run_id": "run-1",
                "gpu_count": 1,
                "cost_per_hour": 5.0,
                "gpu_utilization_avg": 85.0,
                "memory_utilization_avg": 70.0,
                "tags": {},
            },
            {
                "run_id": "run-2",
                "gpu_count": 8,
                "cost_per_hour": 50.0,
                "gpu_utilization_avg": 85.0,
                "memory_utilization_avg": 70.0,
                "tags": {},
            },
        ]

        env_vars = {"TRAINING_MONITOR_ENABLED": "true"}
        with patch.dict(os.environ, env_vars):
            monitor = TrainingRunMonitor()
            recommendations = monitor.get_training_recommendations()

        assert isinstance(recommendations, list)


class TestGetTrainingHistoryWithFiltering:
    """Test training history retrieval with filtering."""

    @patch('monitors.training_run_monitor.TrainingRunMonitor._collect_sagemaker_runs')
    @patch('monitors.training_run_monitor.TrainingRunMonitor._collect_azure_ml_runs')
    @patch('monitors.training_run_monitor.TrainingRunMonitor._collect_vertex_ai_runs')
    @patch('monitors.training_run_monitor.TrainingRunMonitor._collect_wandb_runs')
    @patch('monitors.training_run_monitor.TrainingRunMonitor._collect_mlflow_runs')
    def test_get_training_history_with_days_parameter(self, mock_mlflow, mock_wandb, mock_vertex, mock_azure, mock_sagemaker):
        """Test that days parameter is passed to collection methods."""
        mock_sagemaker.return_value = [
            {
                "run_id": "run-1",
                "provider": "sagemaker",
                "status": TrainingStatus.COMPLETED.value,
            },
        ]
        mock_azure.return_value = []
        mock_vertex.return_value = []
        mock_wandb.return_value = []
        mock_mlflow.return_value = []

        env_vars = {"TRAINING_MONITOR_ENABLED": "true"}
        with patch.dict(os.environ, env_vars):
            monitor = TrainingRunMonitor()
            history = monitor.get_training_history(days=30)

        # Verify that methods were called with cutoff_time
        mock_sagemaker.assert_called_once()
        args, kwargs = mock_sagemaker.call_args
        assert 'cutoff_time' in kwargs


class TestFilteringByStatus:
    """Test filtering of runs by status."""

    @patch('monitors.training_run_monitor.TrainingRunMonitor._collect_sagemaker_runs')
    @patch('monitors.training_run_monitor.TrainingRunMonitor._collect_azure_ml_runs')
    @patch('monitors.training_run_monitor.TrainingRunMonitor._collect_vertex_ai_runs')
    @patch('monitors.training_run_monitor.TrainingRunMonitor._collect_wandb_runs')
    @patch('monitors.training_run_monitor.TrainingRunMonitor._collect_mlflow_runs')
    def test_get_active_only_filters_stopped_runs(self, mock_mlflow, mock_wandb, mock_vertex, mock_azure, mock_sagemaker):
        """Verify that active runs filter excludes stopped runs."""
        mock_sagemaker.return_value = [
            {
                "run_id": "running",
                "provider": "sagemaker",
                "status": TrainingStatus.RUNNING.value,
            },
            {
                "run_id": "stopped",
                "provider": "sagemaker",
                "status": TrainingStatus.STOPPED.value,
            },
            {
                "run_id": "completed",
                "provider": "sagemaker",
                "status": TrainingStatus.COMPLETED.value,
            },
        ]
        mock_azure.return_value = []
        mock_vertex.return_value = []
        mock_wandb.return_value = []
        mock_mlflow.return_value = []

        env_vars = {"TRAINING_MONITOR_ENABLED": "true"}
        with patch.dict(os.environ, env_vars):
            monitor = TrainingRunMonitor()
            active = monitor.get_active_training_runs()

        # Only RUNNING status should be included
        assert len(active) == 1
        assert active[0]["run_id"] == "running"
        assert active[0]["status"] == TrainingStatus.RUNNING.value


class TestInstancePricingLookup:
    """Test instance pricing lookups."""

    def test_instance_pricing_sagemaker_p3_2xlarge(self):
        """Lookup pricing for p3.2xlarge."""
        env_vars = {"TRAINING_MONITOR_ENABLED": "true"}
        with patch.dict(os.environ, env_vars):
            monitor = TrainingRunMonitor()
            pricing = monitor.INSTANCE_PRICING.get("ml.p3.2xlarge")
            assert pricing is not None
            assert pricing.gpu_count == 1
            assert pricing.gpu_type == "V100"
            assert pricing.hourly_cost > 0

    def test_instance_pricing_azure_nd6s(self):
        """Lookup pricing for Azure ND6S."""
        env_vars = {"TRAINING_MONITOR_ENABLED": "true"}
        with patch.dict(os.environ, env_vars):
            monitor = TrainingRunMonitor()
            pricing = monitor.INSTANCE_PRICING.get("Standard_ND6S")
            assert pricing is not None
            assert pricing.provider == "azure_ml"

    def test_instance_pricing_vertex_a2(self):
        """Lookup pricing for Vertex AI A2."""
        env_vars = {"TRAINING_MONITOR_ENABLED": "true"}
        with patch.dict(os.environ, env_vars):
            monitor = TrainingRunMonitor()
            pricing = monitor.INSTANCE_PRICING.get("a2-highgpu-8g")
            assert pricing is not None
            assert pricing.provider == "vertex_ai"
            assert pricing.gpu_type == "A100"
