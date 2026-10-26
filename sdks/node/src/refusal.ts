/**
 * Reading a TensorCost refusal off an HTTP response.
 *
 * The proxy and /admit send their refusals in one of two shapes:
 *
 *   { "error": { "code": "RUN_STOPPED", "message": "...", ... } }   (NestJS ForbiddenException body)
 *   { "code": "RUN_STOPPED", "message": "...", ... }                (flat)
 *
 * Our codes are SCREAMING_SNAKE_CASE. That is what tells a TensorCost
 * refusal apart from a provider error the proxy relayed: OpenAI puts
 * lowercase codes in the same slot (`invalid_api_key`, `model_not_found`),
 * Vertex puts a number there, Anthropic and Bedrock put nothing. A
 * misread here only changes which error class the caller sees, never
 * whether the call falls back, because no 4xx other than a bare 429 falls
 * back at all.
 *
 * `x-tc-decision: ...; action=refuse` marks a refusal too, with or
 * without a code in the body.
 */

import {
  TensorCostBudgetHardCapError,
  TensorCostComplianceDeniedError,
  TensorCostComplianceTeamMismatchError,
  TensorCostGuardrailHardStopError,
  TensorCostModelGovernanceDeniedError,
  TensorCostPeriodBudgetExceededError,
  TensorCostRefusedError,
  TensorCostRunBudgetExceededError,
  TensorCostRunStoppedError,
} from "./errors.js";
import { parseDecisionHeader } from "./decision.js";

const REFUSAL_CODE = /^[A-Z][A-Z0-9_]*$/;

export const BUDGET_HARD_CAP_CODE = "BUDGET_HARD_CAP";

export interface ParsedRefusal {
  code: string | null;
  detail: Record<string, unknown>;
}

function asRecord(v: unknown): Record<string, unknown> | null {
  return v !== null && typeof v === "object" && !Array.isArray(v)
    ? (v as Record<string, unknown>)
    : null;
}

/** Parse a response body (already JSON-decoded, or raw text) for a refusal. */
export function parseRefusalBody(body: unknown): ParsedRefusal | null {
  let parsed: unknown = body;
  if (typeof body === "string") {
    try {
      parsed = JSON.parse(body);
    } catch {
      return null;
    }
  }
  const top = asRecord(parsed);
  if (!top) return null;
  const nested = asRecord(top["error"]);
  if (nested && typeof nested["code"] === "string" && REFUSAL_CODE.test(nested["code"])) {
    return { code: nested["code"], detail: nested };
  }
  if (typeof top["code"] === "string" && REFUSAL_CODE.test(top["code"])) {
    return { code: top["code"], detail: top };
  }
  return null;
}

/** True when the x-tc-decision header says the call was refused. */
export function decisionSaysRefuse(header: string | null | undefined): boolean {
  return parseDecisionHeader(header)?.action === "refuse";
}

const str = (v: unknown): string | null => (typeof v === "string" ? v : null);
const num = (v: unknown): number | null =>
  typeof v === "number" && Number.isFinite(v) ? v : null;

/** Seconds from the Retry-After header, else the body's retry_after_seconds / retry_after. */
function retryAfterSecondsFrom(
  detail: Record<string, unknown>,
  retryAfterHeader: string | null,
): number | null {
  if (retryAfterHeader !== null && retryAfterHeader.trim() !== "") {
    const secs = Number(retryAfterHeader.trim());
    if (Number.isFinite(secs) && secs >= 0) return secs;
    const date = Date.parse(retryAfterHeader);
    if (!Number.isNaN(date)) return Math.max(0, Math.ceil((date - Date.now()) / 1000));
  }
  const fromBody = num(detail["retry_after_seconds"]) ?? num(detail["retry_after"]);
  return fromBody !== null && fromBody >= 0 ? fromBody : null;
}

function budgetHardCapMessage(
  retryable: boolean,
  budgetId: string | null,
  budgetName: string | null,
  retryAfterSeconds: number | null,
): string {
  const which =
    budgetName && budgetId
      ? ` (budget "${budgetName}", ${budgetId})`
      : budgetId
        ? ` (budget ${budgetId})`
        : "";
  if (retryable) {
    const wait = retryAfterSeconds !== null ? `${retryAfterSeconds} s` : "a few seconds";
    return `tensorcost: the budget's spend hasn't been counted yet; retry in ${wait}${which}`;
  }
  return `tensorcost: the budget's hard cap is reached${which}`;
}

export interface RefusalContext {
  status: number;
  requestId: string | null;
  attempt: number;
  retryAfterHeader?: string | null;
}

/**
 * Build the typed error for a refusal. Unknown codes, and refusals that
 * name no code at all, come back as a plain TensorCostRefusedError.
 */
export function refusalError(
  refusal: ParsedRefusal | null,
  ctx: RefusalContext,
): TensorCostRefusedError {
  const detail = refusal?.detail ?? {};
  const code = refusal?.code ?? null;
  const opts = {
    status: ctx.status,
    requestId: ctx.requestId,
    attempt: ctx.attempt,
    detail,
  };
  const serverMessage = str(detail["message"]);
  const prefixed = (fallback: string): string =>
    serverMessage !== null ? `tensorcost: ${serverMessage}` : fallback;

  // budget_code is what the server keys a money budget on now (with
  // code PERIOD_BUDGET_EXCEEDED); code BUDGET_HARD_CAP is the older shape.
  if (detail["budget_code"] === BUDGET_HARD_CAP_CODE || code === BUDGET_HARD_CAP_CODE) {
    const reason = str(detail["reason"]);
    const budgetId = str(detail["budget_id"]);
    const budgetName = str(detail["budget_name"]);
    const retryable = detail["retryable"] === true || reason === "spend_not_known_yet";
    const retryAfterSeconds = retryable
      ? retryAfterSecondsFrom(detail, ctx.retryAfterHeader ?? null)
      : null;
    return new TensorCostBudgetHardCapError(
      budgetHardCapMessage(retryable, budgetId, budgetName, retryAfterSeconds),
      {
        code,
        budgetId,
        budgetName,
        burnPct: num(detail["burn_pct"]),
        scope: str(detail["scope"]),
        period: str(detail["period"]),
        capCents: num(detail["cap_cents"]),
        spentCents: num(detail["spent_cents"]),
        reason,
        retryable,
        retryAfterSeconds,
        detail,
      },
      opts,
    );
  }

  switch (code) {
    case "RUN_STOPPED":
      return TensorCostRunStoppedError.fromProxyBody(detail, opts);
    case "RUN_BUDGET_EXCEEDED":
      return new TensorCostRunBudgetExceededError(
        prefixed("tensorcost: agent run spend cap reached"),
        str(detail["workflow_id"]),
        num(detail["cap_cents"]),
        num(detail["spent_cents"]),
        opts,
      );
    case "PERIOD_BUDGET_EXCEEDED":
      return new TensorCostPeriodBudgetExceededError(
        prefixed("tensorcost: agent period spend cap reached"),
        str(detail["scope"]),
        str(detail["period"]),
        num(detail["cap_cents"]),
        num(detail["spent_cents"]),
        opts,
      );
    case "MODEL_GOVERNANCE_DENIED":
      return new TensorCostModelGovernanceDeniedError(
        prefixed("tensorcost: model denied by tenant governance"),
        str(detail["provider"]),
        str(detail["model"]),
        str(detail["governance_status"]),
        opts,
      );
    case "GUARDRAIL_HARD_STOP":
      return new TensorCostGuardrailHardStopError(
        prefixed("tensorcost: request blocked by hard_stop guardrail"),
        str(detail["policy_id"]),
        { ...opts, reason: str(detail["reason"]) },
      );
    case "COMPLIANCE_DENIED":
      // 1.4.0 put the code in `reason` when the server sent none; kept.
      return new TensorCostComplianceDeniedError(
        prefixed("tensorcost: request refused by compliance policy"),
        str(detail["reason"]) ?? code,
        opts,
      );
    case "COMPLIANCE_TEAM_MISMATCH":
      return new TensorCostComplianceTeamMismatchError(
        prefixed("tensorcost: SDK team scope does not match token binding"),
        opts,
      );
    default:
      return new TensorCostRefusedError(
        prefixed(
          code !== null
            ? `tensorcost: request refused (${code})`
            : "tensorcost: request refused",
        ),
        { code, reason: str(detail["reason"]), detail },
        opts,
      );
  }
}

/** Best-effort provider message out of a relayed error body, for the error text. */
export function providerMessageFrom(body: unknown): string | null {
  const top = asRecord(body);
  if (!top) return typeof body === "string" && body.trim() !== "" ? body.slice(0, 500) : null;
  const nested = asRecord(top["error"]);
  const msg = str(nested?.["message"]) ?? str(top["message"]);
  return msg !== null ? msg.slice(0, 500) : null;
}
