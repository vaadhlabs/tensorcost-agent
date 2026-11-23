/**
 * Shared by the OpenAI and Anthropic wrappers for the TensorCost hops a
 * call waits on.
 */

import type { ObservationTransport } from "../transport.js";

/** Metadata on a call that went ahead because /admit couldn't be reached. */
export const ADMIT_UNAVAILABLE: Record<string, string> = Object.freeze({
  decision_action: "allow",
  decision_reason: "admit_unavailable",
}) as Record<string, string>;

/**
 * The TensorCost token for a hop on the customer's request path, waited on
 * for at most `budgetMs` (the hop's own budget). The `getToken` override is
 * for tests.
 */
export function callPathToken(
  opts: { transport: ObservationTransport; getToken?: () => Promise<string> },
  budgetMs: number,
): Promise<string> {
  return opts.getToken ? opts.getToken() : opts.transport.getToken({ budgetMs });
}
