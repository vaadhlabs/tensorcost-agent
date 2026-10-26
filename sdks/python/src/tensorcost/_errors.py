"""Typed error hierarchy for the TensorCost Python SDK.

Every error thrown by the SDK's applied-mode request path is a subclass of
``TensorCostError``.  Customer code can ``isinstance``-branch on any of them
without importing raw exception types from the underlying HTTP layer.

Hierarchy::

    TensorCostError (base)
    ├── TensorCostNetworkError    couldn't reach the proxy at all          ┐ outages: the only
    ├── TensorCostTimeoutError    request or idle window exceeded limit    │ errors that may fall
    ├── TensorCostProxyError      proxy responded with 5xx                 │ back to the provider
    ├── TensorCostQuotaError      proxy 429, or 402 plan ceiling           ┘
    ├── TensorCostProviderError   proxy relayed an upstream provider 4xx
    │                             (re-raised as the provider SDK's own error
    │                             type when the client can build one)
    ├── TensorCostGatewayRejectedError  non-JSON 4xx from in front of TensorCost
    ├── TensorCostKeyRejectedError      the TensorCost key was rejected
    └── TensorCostRefusedError    TensorCost refused the call (terminal)
        ├── TensorCostRunBudgetExceededError
        ├── TensorCostRunStoppedError
        ├── TensorCostPeriodBudgetExceededError
        │   └── TensorCostBudgetHardCapError
        ├── TensorCostModelGovernanceDeniedError
        ├── TensorCostGuardrailHardStopError
        ├── TensorCostComplianceDeniedError
        └── TensorCostComplianceTeamMismatchError

A refusal code this SDK version doesn't model still arrives as a plain
``TensorCostRefusedError`` with its ``code``, ``reason`` and message intact.

All subclasses carry:

* ``root_cause``  — the underlying Exception, if any (avoids clash with
                    ``Exception.__cause__`` semantics)
* ``status``      — HTTP status code (None when the request never completed)
* ``request_id``  — value of the X-TC-Request-ID header the proxy stamps
* ``attempt``     — 1-based attempt number at the time the error occurred
"""

from __future__ import annotations

from typing import Any, Optional


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
    """The proxy returned 429 Too Many Requests (TensorCost's own rate limit),
    or 402 with no refusal body (the plan's monthly proxy-request ceiling;
    ``status`` says which). Neither carries a refusal, so both may fall back.

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
    """The proxy reached the upstream provider and relayed the provider's own
    4xx (OpenAI 400 invalid request, Anthropic 401, ...). ``status`` is the
    provider's status and ``body`` what it sent, parsed when it was JSON.
    Never falls back: the provider would only answer the same way again."""

    def __init__(
        self,
        message: str,
        *,
        body: Any = None,
        headers: Optional[dict] = None,
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
        self.body = body
        #: Response headers of the relayed error.
        self.headers = dict(headers or {})


class TensorCostGatewayRejectedError(TensorCostError):
    """Something in front of TensorCost turned the request away: a 4xx whose
    body isn't JSON (an HTML page from a firewall or load balancer, or
    nothing at all). Neither TensorCost nor the provider saw the call.
    Terminal: going around it would mean going around whatever put the
    firewall there, and the next call would most likely meet it too.
    """

    def __init__(
        self,
        status: int,
        body_text: Optional[str] = None,
        *,
        request_id: Optional[str] = None,
        attempt: int = 1,
    ) -> None:
        super().__init__(
            f"tensorcost: the request was rejected with {status} before it reached "
            "TensorCost or the provider (for example by a firewall or load balancer in "
            "front of the TensorCost proxy). Check the proxy URL and anything in front of it.",
            status=status,
            request_id=request_id,
            attempt=attempt,
        )
        #: The first 500 characters of what came back, if anything did.
        self.body_text = body_text


class TensorCostKeyRejectedError(TensorCostError):
    """The TensorCost API key was rejected (401 or 403): revoked, mistyped,
    or issued for a different tenant than the one configured. Not an outage,
    so it never fails open."""

    def __init__(
        self,
        status: int,
        *,
        request_id: Optional[str] = None,
        attempt: int = 1,
    ) -> None:
        super().__init__(
            f"tensorcost: the TensorCost key was rejected ({status}); check it's current "
            "and that it belongs to the configured tenant id",
            status=status,
            request_id=request_id,
            attempt=attempt,
        )


class TensorCostRefusedError(TensorCostError):
    """TensorCost answered and the answer is no: a budget, a stop, a policy.

    Every refusal is terminal. It is never retried and never falls back to
    the provider, because the proxy that sent it is healthy and the refusal
    is the product. ``code`` is the server's refusal code (None when it
    refused without naming one), ``reason`` its optional sub-reason, and
    ``detail`` the refusal object as the server sent it.

    Subclasses cover the codes this SDK version models. Anything else, a
    code added on the server after this release included, is raised as this
    class directly so it is still terminal.
    """

    _default_code: Optional[str] = None

    def __init__(
        self,
        message: str,
        *,
        code: Optional[str] = None,
        reason: Optional[str] = None,
        detail: Optional[dict] = None,
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
        self.code = code if code is not None else self._default_code
        self.reason = reason
        self.detail = detail


class TensorCostRunBudgetExceededError(TensorCostRefusedError):
    """Agent run spend cap reached (HTTP 403, ``code: RUN_BUDGET_EXCEEDED``).

    Terminal — must not trigger direct-to-provider fail-open.
    """

    _default_code = "RUN_BUDGET_EXCEEDED"

    def __init__(
        self,
        message: str,
        workflow_id: Optional[str] = None,
        cap_cents: Optional[int] = None,
        spent_cents: Optional[int] = None,
        *,
        detail: Optional[dict] = None,
        root_cause: Optional[Exception] = None,
        status: int = 403,
        request_id: Optional[str] = None,
        attempt: int = 1,
    ) -> None:
        super().__init__(
            message,
            detail=detail,
            root_cause=root_cause,
            status=status,
            request_id=request_id,
            attempt=attempt,
        )
        self.workflow_id = workflow_id
        self.cap_cents = cap_cents
        self.spent_cents = spent_cents


class TensorCostRunStoppedError(TensorCostRefusedError):
    """An operator stopped this agent run (HTTP 403, ``code: RUN_STOPPED``).

    Terminal for the same reason as the budget refusals: the stop comes from
    a healthy proxy, so falling back to the provider would undo it. Carries
    the who / when / why the proxy recorded.
    """

    _default_code = "RUN_STOPPED"

    def __init__(
        self,
        message: str,
        workflow_id: Optional[str] = None,
        stopped_by: Optional[str] = None,
        stopped_at: Optional[str] = None,
        reason: Optional[str] = None,
        *,
        detail: Optional[dict] = None,
        root_cause: Optional[Exception] = None,
        status: int = 403,
        request_id: Optional[str] = None,
        attempt: int = 1,
    ) -> None:
        super().__init__(
            message,
            reason=reason,
            detail=detail,
            root_cause=root_cause,
            status=status,
            request_id=request_id,
            attempt=attempt,
        )
        self.workflow_id = workflow_id
        self.stopped_by = stopped_by
        self.stopped_at = stopped_at

    @classmethod
    def from_proxy_detail(
        cls,
        detail: dict,
        *,
        request_id: Optional[str] = None,
        attempt: int = 1,
    ) -> "TensorCostRunStoppedError":
        """Build from the proxy's 403 ``error`` object (chat path and /admit)."""

        def _s(key: str) -> Optional[str]:
            v = detail.get(key)
            return v if isinstance(v, str) else None

        return cls(
            _s("message") or "tensorcost: agent run was stopped",
            workflow_id=_s("workflow_id"),
            stopped_by=_s("stopped_by"),
            stopped_at=_s("stopped_at"),
            reason=_s("reason"),
            detail=detail,
            status=403,
            request_id=request_id,
            attempt=attempt,
        )


class TensorCostPeriodBudgetExceededError(TensorCostRefusedError):
    """Daily/monthly agent or team cap reached (HTTP 403,
    ``code: PERIOD_BUDGET_EXCEEDED``). Terminal — must not fail-open.
    """

    _default_code = "PERIOD_BUDGET_EXCEEDED"

    def __init__(
        self,
        message: str,
        scope: Optional[str] = None,
        period: Optional[str] = None,
        cap_cents: Optional[int] = None,
        spent_cents: Optional[int] = None,
        *,
        code: Optional[str] = None,
        reason: Optional[str] = None,
        detail: Optional[dict] = None,
        root_cause: Optional[Exception] = None,
        status: int = 403,
        request_id: Optional[str] = None,
        attempt: int = 1,
    ) -> None:
        super().__init__(
            message,
            code=code,
            reason=reason,
            detail=detail,
            root_cause=root_cause,
            status=status,
            request_id=request_id,
            attempt=attempt,
        )
        self.scope = scope
        self.period = period
        self.cap_cents = cap_cents
        self.spent_cents = spent_cents


class TensorCostBudgetHardCapError(TensorCostPeriodBudgetExceededError):
    """A money budget's hard cap refused the call (HTTP 403).

    The server sends ``error.code: "PERIOD_BUDGET_EXCEEDED"`` with
    ``error.budget_code: "BUDGET_HARD_CAP"`` (that code is what made 1.4.0
    treat it as terminal), and older servers sent ``error.code:
    "BUDGET_HARD_CAP"``. Both map here. ``code`` keeps whichever the server
    sent; ``budget_code`` is always ``"BUDGET_HARD_CAP"``.

    It subclasses ``TensorCostPeriodBudgetExceededError`` because 1.4.0
    already raised that class for the current wire shape, so an existing
    ``except`` for it keeps working, and scope / period / cap_cents /
    spent_cents are filled the same way.

    ``reason == "spend_not_known_yet"`` (``retryable: true`` on the wire)
    means the budget hasn't been counted yet, usually for a few seconds after
    it is created or its scope changes, so the cap holds calls until the
    first count lands. That case is ``retryable``, and
    ``retry_after_seconds`` comes from the Retry-After header or the body's
    ``retry_after_seconds``. The SDK does not retry on its own.
    """

    _default_code = "BUDGET_HARD_CAP"
    budget_code = "BUDGET_HARD_CAP"

    def __init__(
        self,
        message: str,
        *,
        code: Optional[str] = None,
        budget_id: Optional[str] = None,
        budget_name: Optional[str] = None,
        burn_pct: Optional[float] = None,
        scope: Optional[str] = None,
        period: Optional[str] = None,
        cap_cents: Optional[float] = None,
        spent_cents: Optional[float] = None,
        reason: Optional[str] = None,
        retryable: Optional[bool] = None,
        retry_after_seconds: Optional[float] = None,
        detail: Optional[dict] = None,
        root_cause: Optional[Exception] = None,
        status: int = 403,
        request_id: Optional[str] = None,
        attempt: int = 1,
    ) -> None:
        super().__init__(
            message,
            scope=scope,
            period=period,
            cap_cents=cap_cents,
            spent_cents=spent_cents,
            code=code if code is not None else "BUDGET_HARD_CAP",
            reason=reason,
            detail=detail,
            root_cause=root_cause,
            status=status,
            request_id=request_id,
            attempt=attempt,
        )
        self.budget_id = budget_id
        self.budget_name = budget_name
        self.burn_pct = burn_pct
        self.retryable = retryable is True or reason == "spend_not_known_yet"
        self.retry_after_seconds = retry_after_seconds


class TensorCostModelGovernanceDeniedError(TensorCostRefusedError):
    """Model denied by tenant governance (HTTP 403, ``MODEL_GOVERNANCE_DENIED``)."""

    _default_code = "MODEL_GOVERNANCE_DENIED"

    def __init__(
        self,
        message: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        governance_status: Optional[str] = None,
        *,
        detail: Optional[dict] = None,
        root_cause: Optional[Exception] = None,
        status: int = 403,
        request_id: Optional[str] = None,
        attempt: int = 1,
    ) -> None:
        super().__init__(
            message,
            detail=detail,
            root_cause=root_cause,
            status=status,
            request_id=request_id,
            attempt=attempt,
        )
        self.provider = provider
        self.model = model
        self.governance_status = governance_status


class TensorCostGuardrailHardStopError(TensorCostRefusedError):
    """Guardrail hard_stop refusal (HTTP 403, ``GUARDRAIL_HARD_STOP``)."""

    _default_code = "GUARDRAIL_HARD_STOP"

    def __init__(
        self,
        message: str,
        policy_id: Optional[str] = None,
        *,
        reason: Optional[str] = None,
        detail: Optional[dict] = None,
        root_cause: Optional[Exception] = None,
        status: int = 403,
        request_id: Optional[str] = None,
        attempt: int = 1,
    ) -> None:
        super().__init__(
            message,
            reason=reason,
            detail=detail,
            root_cause=root_cause,
            status=status,
            request_id=request_id,
            attempt=attempt,
        )
        self.policy_id = policy_id


class TensorCostComplianceDeniedError(TensorCostRefusedError):
    """Compliance policy refused this call (metadata or in-process DLP)."""

    _default_code = "COMPLIANCE_DENIED"


class TensorCostComplianceTeamMismatchError(TensorCostRefusedError):
    """Client team_id does not match token-bound team scope."""

    _default_code = "COMPLIANCE_TEAM_MISMATCH"
