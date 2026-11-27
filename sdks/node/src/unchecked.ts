/**
 * Calls that went to the provider without TensorCost's check.
 *
 * Fail-open lets a call through when TensorCost cannot decide on it: /admit
 * or the proxy is down, the circuit is open, or TensorCost is pacing this
 * process (a bare 429) or the plan's ceiling is reached (a bare 402). Each
 * such call carries `decision_reason` on its observation, so the server can
 * count them. This module is the in-process side: a count per reason since
 * the process started, readable with `uncheckedCalls()`, and a warning per
 * reason at most once a minute, so it is never silent and never floods.
 */

export type UncheckedReason =
  | "admit_unavailable"
  | "admit_throttled"
  | "admit_plan_limit"
  | "proxy_unavailable"
  | "proxy_throttled"
  | "proxy_plan_limit"
  | "circuit_open";

export type UncheckedCalls = Record<UncheckedReason, number>;

const REASONS: readonly UncheckedReason[] = [
  "admit_unavailable",
  "admit_throttled",
  "admit_plan_limit",
  "proxy_unavailable",
  "proxy_throttled",
  "proxy_plan_limit",
  "circuit_open",
];

const WARN_INTERVAL_MS = 60_000;

const WHY: Record<UncheckedReason, string> = {
  admit_unavailable: "TensorCost's admission check could not be reached",
  admit_throttled: "TensorCost's admission check is rate limiting this process (429)",
  admit_plan_limit: "TensorCost's admission check says the plan's request ceiling is reached (402)",
  proxy_unavailable: "the TensorCost proxy could not be reached",
  proxy_throttled: "the TensorCost proxy is rate limiting this process (429)",
  proxy_plan_limit: "the TensorCost proxy says the plan's request ceiling is reached (402)",
  circuit_open: "the TensorCost proxy circuit is open",
};

const counts = new Map<UncheckedReason, number>();
const sinceWarn = new Map<UncheckedReason, number>();
const lastWarn = new Map<UncheckedReason, number>();

/** Count one call that went ahead unchecked, and warn about it at most once a minute per reason. */
export function recordUnchecked(reason: UncheckedReason): void {
  counts.set(reason, (counts.get(reason) ?? 0) + 1);
  sinceWarn.set(reason, (sinceWarn.get(reason) ?? 0) + 1);
  const now = Date.now();
  const last = lastWarn.get(reason);
  if (last !== undefined && now - last < WARN_INTERVAL_MS) return;
  const n = sinceWarn.get(reason) ?? 1;
  lastWarn.set(reason, now);
  sinceWarn.set(reason, 0);
  const window = last === undefined ? "" : " since the last warning";
  // eslint-disable-next-line no-console
  console.warn(
    `tensorcost: ${WHY[reason]}; ${n} call(s)${window} went to the provider ` +
      `without a budget or governance check (fail-open). Set failOpenEnabled: false ` +
      `to refuse them instead. uncheckedCalls() has the totals.`,
  );
}

/** Calls this process sent to the provider without TensorCost's check, by reason. */
export function uncheckedCalls(): UncheckedCalls {
  const out = {} as UncheckedCalls;
  for (const r of REASONS) out[r] = counts.get(r) ?? 0;
  return out;
}

/** Observation metadata for a call that went ahead unchecked. */
export function uncheckedMeta(reason: UncheckedReason): Record<string, string> {
  return { decision_action: "allow", decision_reason: reason };
}

export const __test__ = {
  reset(): void {
    counts.clear();
    sinceWarn.clear();
    lastWarn.clear();
  },
};
