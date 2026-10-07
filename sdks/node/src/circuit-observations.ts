/**
 * Emit circuit-breaker state transitions as observations (Phase 4).
 */

import { randomUUID } from "node:crypto";

import type { CircuitState } from "./circuit.js";
import type { ObservationTransport } from "./transport.js";
import type { Observation, Provider } from "./types.js";
import { SDK_VERSION } from "./version.js";

export function bindCircuitObservations(
  transport: ObservationTransport,
  ctx: {
    provider: Provider;
    tenantId: string | null;
    environment: string | null;
  },
): (from: CircuitState, to: CircuitState) => void {
  return (from, to) => {
    const obs: Observation = {
      sdk_version: SDK_VERSION,
      provider: ctx.provider,
      model: "circuit-breaker",
      operation: "chat.completions",
      modality: "text",
      request_at: new Date().toISOString(),
      response_at: new Date().toISOString(),
      input_tokens: null,
      output_tokens: null,
      cost_usd_cents: null,
      status: "success",
      error_message: null,
      correlation_id: randomUUID(),
      tenant_id: ctx.tenantId,
      environment: ctx.environment ?? undefined,
      metadata: {
        circuit_state: to,
        circuit_previous_state: from,
      },
    };
    try {
      transport.post(obs);
    } catch {
      /* fail-open */
    }
  };
}
