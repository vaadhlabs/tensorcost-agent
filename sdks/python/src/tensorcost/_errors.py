"""Typed error hierarchy for the TensorCost Python SDK.

Every error thrown by the SDK's applied-mode request path is a subclass of
``TensorCostError``.  Customer code can ``isinstance``-branch on any of them
without importing raw exception types from the underlying HTTP layer.

Hierarchy::

    TensorCostError (base)
    ├── TensorCostNetworkError    couldn't reach the proxy at all
    ├── TensorCostTimeoutError    request or idle window exceeded limit
    ├── TensorCostProxyError      proxy responded with 5xx
    ├── TensorCostQuotaError      proxy responded with 429
    └── TensorCostProviderError   proxy forwarded an upstream provider error

All subclasses carry:

* ``root_cause``  — the underlying Exception, if any (avoids clash with
                    ``Exception.__cause__`` semantics)
* ``status``      — HTTP status code (None when the request never completed)
* ``request_id``  — value of the X-TC-Request-ID header the proxy stamps
* ``attempt``     — 1-based attempt number at the time the error occurred
"""

from __future__ import annotations

from typing import Optional


class TensorCostError(Exception):
    """Base class for all TensorCost SDK errors."""

    def __init__(
        self,
        message: str,
        *,
        root_cause: Optional[Exception] = None,
        status: Optional[int] = None,
        request_id: Optional[str] = None,
        attempt: int = 1,
    ) -> None:
        super().__init__(message)
        self.root_cause = root_cause
        self.status = status
        self.request_id = request_id
        self.attempt = attempt

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"message={str(self)!r}, "
            f"status={self.status!r}, "
            f"attempt={self.attempt!r})"
        )


class TensorCostNetworkError(TensorCostError):
    """The SDK could not establish a connection to the proxy at all."""


class TensorCostTimeoutError(TensorCostError):
    """A request or idle-window timeout fired before the operation completed.

    ``timeout_kind`` is ``"headers"`` when the proxy never returned response
    headers, ``"total"`` for a wall-clock limit, or ``"idle"`` for streaming idle.
    """

    def __init__(
        self,
        message: str,
        timeout_kind: str,
        *,
        root_cause: Optional[Exception] = None,
        status: Optional[int] = None,
        request_id: Optional[str] = None,
        attempt: int = 1,
    ) -> None:
        super().__init__(
            message,
            root_cause=root_cause,
            status=status,
            request_id=request_id,
            attempt=attempt,
        )
        self.timeout_kind = timeout_kind  # "headers" | "total" | "idle"


class TensorCostProxyError(TensorCostError):
    """The proxy returned a 5xx response (TensorCost infrastructure error)."""


class TensorCostQuotaError(TensorCostError):
    """The proxy returned 429 Too Many Requests.

    ``retry_after_ms`` is set when the response included a ``Retry-After``
    header (in milliseconds).
    """

    def __init__(
        self,
        message: str,
        retry_after_ms: Optional[int] = None,
        *,
        root_cause: Optional[Exception] = None,
        status: int = 429,
        request_id: Optional[str] = None,
        attempt: int = 1,
    ) -> None:
        super().__init__(
            message,
            root_cause=root_cause,
            status=status,
            request_id=request_id,
            attempt=attempt,
        )
        self.retry_after_ms = retry_after_ms


class TensorCostProviderError(TensorCostError):
    """The proxy reached the upstream provider, but the provider returned an
    error (4xx / 5xx from OpenAI, Anthropic, etc.).  ``status`` reflects the
    provider's HTTP status code."""


class TensorCostRunBudgetExceededError(TensorCostError):
    """Agent run spend cap reached (HTTP 403, ``code: RUN_BUDGET_EXCEEDED``).

    Terminal — must not trigger direct-to-provider fail-open.
    """

    def __init__(
        self,
        message: str,
        workflow_id: Optional[str] = None,
        cap_cents: Optional[int] = None,
        spent_cents: Optional[int] = None,
        *,
        root_cause: Optional[Exception] = None,
        status: int = 403,
        request_id: Optional[str] = None,
        attempt: int = 1,
    ) -> None:
        super().__init__(
            message,
            root_cause=root_cause,
            status=status,
            request_id=request_id,
            attempt=attempt,
        )
        self.workflow_id = workflow_id
        self.cap_cents = cap_cents
        self.spent_cents = spent_cents


class TensorCostPeriodBudgetExceededError(TensorCostError):
    """Daily/monthly agent or team cap reached (HTTP 403,
    ``code: PERIOD_BUDGET_EXCEEDED``). Terminal — must not fail-open.
    """

    def __init__(
        self,
        message: str,
        scope: Optional[str] = None,
        period: Optional[str] = None,
        cap_cents: Optional[int] = None,
        spent_cents: Optional[int] = None,
        *,
        root_cause: Optional[Exception] = None,
        status: int = 403,
        request_id: Optional[str] = None,
        attempt: int = 1,
    ) -> None:
        super().__init__(
            message,
            root_cause=root_cause,
            status=status,
            request_id=request_id,
            attempt=attempt,
        )
        self.scope = scope
        self.period = period
        self.cap_cents = cap_cents
        self.spent_cents = spent_cents


class TensorCostModelGovernanceDeniedError(TensorCostError):
    """Model denied by tenant governance (HTTP 403, ``MODEL_GOVERNANCE_DENIED``)."""

    def __init__(
        self,
        message: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        governance_status: Optional[str] = None,
        *,
        root_cause: Optional[Exception] = None,
        status: int = 403,
        request_id: Optional[str] = None,
        attempt: int = 1,
    ) -> None:
        super().__init__(
            message,
            root_cause=root_cause,
            status=status,
            request_id=request_id,
            attempt=attempt,
        )
        self.provider = provider
        self.model = model
        self.governance_status = governance_status


class TensorCostGuardrailHardStopError(TensorCostError):
    """Guardrail hard_stop refusal (HTTP 403, ``GUARDRAIL_HARD_STOP``)."""

    def __init__(
        self,
        message: str,
        policy_id: Optional[str] = None,
        *,
        root_cause: Optional[Exception] = None,
        status: int = 403,
        request_id: Optional[str] = None,
        attempt: int = 1,
    ) -> None:
        super().__init__(
            message,
            root_cause=root_cause,
            status=status,
            request_id=request_id,
            attempt=attempt,
        )
        self.policy_id = policy_id


class TensorCostComplianceDeniedError(TensorCostError):
    """Compliance policy refused this call (metadata or in-process DLP)."""


class TensorCostComplianceTeamMismatchError(TensorCostError):
    """Client team_id does not match token-bound team scope."""
