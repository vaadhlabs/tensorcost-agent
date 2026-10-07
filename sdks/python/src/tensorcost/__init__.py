"""TensorCost Python SDK — one-line wrapper for OpenAI, Anthropic, and
AWS Bedrock clients.

Usage:

    from openai import OpenAI
    from tensorcost import wrap

    client = wrap(OpenAI(api_key="..."))
    # use `client` exactly like a normal OpenAI client; observations
    # are sent fire-and-forget to TensorCost in the background.

Bedrock support (``boto3.client("bedrock-runtime")``) is observe-only;
``applied_mode=True`` raises ``MissingConfigError`` for Bedrock clients.

The SDK fails open by default: if TensorCost is unreachable, the
underlying provider call still completes normally.
"""

from ._wrap import wrap
from ._prewarm import prewarm, flush_observations
from ._config import MissingConfigError
from ._providers._detect import UnsupportedClientError
from ._errors import (
    TensorCostError,
    TensorCostNetworkError,
    TensorCostTimeoutError,
    TensorCostProxyError,
    TensorCostQuotaError,
    TensorCostProviderError,
    TensorCostRunBudgetExceededError,
    TensorCostPeriodBudgetExceededError,
    TensorCostComplianceDeniedError,
    TensorCostComplianceTeamMismatchError,
)
from ._telemetry import (
    LifecycleEvent,
    BeforeRequestEvent,
    AfterResponseEvent,
    OnRetryEvent,
    OnErrorEvent,
    OnFallbackEvent,
    LifecycleEventCallback,
)
from ._retry import RetryConfig

__version__ = "1.3.0"
__all__ = [
    "wrap",
    "prewarm",
    "flush_observations",
    "MissingConfigError",
    "UnsupportedClientError",
    # Error hierarchy
    "TensorCostError",
    "TensorCostNetworkError",
    "TensorCostTimeoutError",
    "TensorCostProxyError",
    "TensorCostQuotaError",
    "TensorCostProviderError",
    "TensorCostRunBudgetExceededError",
    "TensorCostPeriodBudgetExceededError",
    "TensorCostComplianceDeniedError",
    "TensorCostComplianceTeamMismatchError",
    # Telemetry types
    "LifecycleEvent",
    "BeforeRequestEvent",
    "AfterResponseEvent",
    "OnRetryEvent",
    "OnErrorEvent",
    "OnFallbackEvent",
    "LifecycleEventCallback",
    # Config types
    "RetryConfig",
    "__version__",
]
