"""Reading a TensorCost refusal off an HTTP response.

The proxy and /admit send refusals in one of two shapes::

    {"error": {"code": "RUN_STOPPED", "message": "...", ...}}   (NestJS ForbiddenException body)
    {"code": "RUN_STOPPED", "message": "...", ...}              (flat)

Our codes are SCREAMING_SNAKE_CASE. That is what tells a TensorCost refusal
apart from a provider error the proxy relayed: OpenAI puts lowercase codes
in the same slot (``invalid_api_key``, ``model_not_found``), Vertex puts a
number there, Anthropic and Bedrock put nothing. A misread here only changes
which error class the caller sees, never whether the call falls back,
because no 4xx other than a bare 429 falls back at all.

``x-tc-decision: ...; action=refuse`` marks a refusal too, with or without a
code in the body.
"""

from __future__ import annotations

import json
import re
import time
from email.utils import parsedate_to_datetime
from typing import Any, Optional

from ._decision import parse_decision_header
from ._errors import (
    TensorCostBudgetHardCapError,
    TensorCostComplianceDeniedError,
    TensorCostComplianceTeamMismatchError,
    TensorCostGuardrailHardStopError,
    TensorCostModelGovernanceDeniedError,
    TensorCostPeriodBudgetExceededError,
    TensorCostRefusedError,
    TensorCostRunBudgetExceededError,
    TensorCostRunStoppedError,
)

_REFUSAL_CODE = re.compile(r"^[A-Z][A-Z0-9_]*$")

BUDGET_HARD_CAP_CODE = "BUDGET_HARD_CAP"


def parse_body(content: Any) -> Any:
    """Decode a response body as JSON when it is JSON, else return text (or None)."""
    if content is None:
        return None
    if isinstance(content, (bytes, bytearray)):
        content = bytes(content).decode("utf-8", errors="replace")
    if isinstance(content, str):
        if content.strip() == "":
            return None
        try:
            return json.loads(content)
        except (json.JSONDecodeError, ValueError):
            return content
    return content


def parse_refusal_body(body: Any) -> Optional[tuple[str, dict]]:
    """``(code, detail)`` when the decoded body is a TensorCost refusal, else None."""
    if not isinstance(body, dict):
        return None
    nested = body.get("error")
    if isinstance(nested, dict):
        code = nested.get("code")
        if isinstance(code, str) and _REFUSAL_CODE.match(code):
            return code, nested
    code = body.get("code")
    if isinstance(code, str) and _REFUSAL_CODE.match(code):
        return code, body
    return None


def decision_says_refuse(header: Optional[str]) -> bool:
    parsed = parse_decision_header(header)
    return parsed is not None and parsed.action == "refuse"


def _s(detail: dict, key: str) -> Optional[str]:
    v = detail.get(key)
    return v if isinstance(v, str) else None


def _n(detail: dict, key: str) -> Optional[float]:
    v = detail.get(key)
    if isinstance(v, bool):
        return None
    return v if isinstance(v, (int, float)) else None


def _retry_after_seconds(detail: dict, header: Optional[str]) -> Optional[float]:
    if header is not None and header.strip() != "":
        raw = header.strip()
        try:
            secs = float(raw)
            if secs >= 0:
                return int(secs) if secs.is_integer() else secs
        except ValueError:
            try:
                when = parsedate_to_datetime(raw).timestamp()
                return max(0, int(round(when - time.time())))
            except (TypeError, ValueError):
                pass
    for key in ("retry_after_seconds", "retry_after"):
        v = _n(detail, key)
        if v is not None and v >= 0:
            return v
    return None


def _budget_hard_cap_message(
    retryable: bool,
    budget_id: Optional[str],
    budget_name: Optional[str],
    retry_after_seconds: Optional[float],
) -> str:
    if budget_name and budget_id:
        which = f' (budget "{budget_name}", {budget_id})'
    elif budget_id:
        which = f" (budget {budget_id})"
    else:
        which = ""
    if retryable:
        wait = f"{retry_after_seconds:g} s" if retry_after_seconds is not None else "a few seconds"
        return f"tensorcost: the budget's spend hasn't been counted yet; retry in {wait}{which}"
    return f"tensorcost: the budget's hard cap is reached{which}"


def refusal_error(
    refusal: Optional[tuple[str, dict]],
    *,
    status: int,
    request_id: Optional[str],
    attempt: int = 1,
    retry_after_header: Optional[str] = None,
) -> TensorCostRefusedError:
    """Typed error for a refusal. Unknown codes, and refusals that name no
    code at all, come back as a plain ``TensorCostRefusedError``."""
    code, detail = refusal if refusal is not None else (None, {})
    common: dict[str, Any] = {
        "status": status,
        "request_id": request_id,
        "attempt": attempt,
        "detail": detail,
    }

    # budget_code is what the server keys a money budget on now (with code
    # PERIOD_BUDGET_EXCEEDED); code BUDGET_HARD_CAP is the older shape.
    if detail.get("budget_code") == BUDGET_HARD_CAP_CODE or code == BUDGET_HARD_CAP_CODE:
        reason = _s(detail, "reason")
        budget_id = _s(detail, "budget_id")
        budget_name = _s(detail, "budget_name")
        retryable = detail.get("retryable") is True or reason == "spend_not_known_yet"
        retry_after = _retry_after_seconds(detail, retry_after_header) if retryable else None
        return TensorCostBudgetHardCapError(
            _budget_hard_cap_message(retryable, budget_id, budget_name, retry_after),
            code=code,
            budget_id=budget_id,
            budget_name=budget_name,
            burn_pct=_n(detail, "burn_pct"),
            scope=_s(detail, "scope"),
            period=_s(detail, "period"),
            cap_cents=_n(detail, "cap_cents"),
            spent_cents=_n(detail, "spent_cents"),
            reason=reason,
            retryable=retryable,
            retry_after_seconds=retry_after,
            **common,
        )

    message = _s(detail, "message")
    if code == "RUN_STOPPED":
        return TensorCostRunStoppedError.from_proxy_detail(
            detail, request_id=request_id, attempt=attempt
        )
    if code == "RUN_BUDGET_EXCEEDED":
        return TensorCostRunBudgetExceededError(
            message or "tensorcost: agent run spend cap reached",
            workflow_id=_s(detail, "workflow_id"),
            cap_cents=_n(detail, "cap_cents"),
            spent_cents=_n(detail, "spent_cents"),
            **common,
        )
    if code == "PERIOD_BUDGET_EXCEEDED":
        return TensorCostPeriodBudgetExceededError(
            message or "tensorcost: agent period spend cap reached",
            scope=_s(detail, "scope"),
            period=_s(detail, "period"),
            cap_cents=_n(detail, "cap_cents"),
            spent_cents=_n(detail, "spent_cents"),
            **common,
        )
    if code == "MODEL_GOVERNANCE_DENIED":
        return TensorCostModelGovernanceDeniedError(
            message or "tensorcost: model denied by tenant governance",
            provider=_s(detail, "provider"),
            model=_s(detail, "model"),
            governance_status=_s(detail, "governance_status"),
            **common,
        )
    if code == "GUARDRAIL_HARD_STOP":
        return TensorCostGuardrailHardStopError(
            message or "tensorcost: guardrail hard_stop",
            policy_id=_s(detail, "policy_id"),
            reason=_s(detail, "reason"),
            **common,
        )
    if code == "COMPLIANCE_DENIED":
        return TensorCostComplianceDeniedError(
            message or "tensorcost: request refused by compliance policy",
            reason=_s(detail, "reason"),
            **common,
        )
    if code == "COMPLIANCE_TEAM_MISMATCH":
        return TensorCostComplianceTeamMismatchError(
            message or "tensorcost: SDK team scope does not match token binding",
            **common,
        )
    fallback = f"tensorcost: request refused ({code})" if code else "tensorcost: request refused"
    return TensorCostRefusedError(
        f"tensorcost: {message}" if message else fallback,
        code=code,
        reason=_s(detail, "reason"),
        **common,
    )


def provider_message_from(body: Any) -> Optional[str]:
    """Best-effort provider message out of a relayed error body."""
    if isinstance(body, dict):
        nested = body.get("error")
        msg = nested.get("message") if isinstance(nested, dict) else None
        if not isinstance(msg, str):
            msg = body.get("message")
        return msg[:500] if isinstance(msg, str) else None
    if isinstance(body, str) and body.strip():
        return body[:500]
    return None
