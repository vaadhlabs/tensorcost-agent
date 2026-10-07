import { AsyncLocalStorage } from "node:async_hooks";
import { createHash, randomUUID } from "node:crypto";
import type { Observation } from "./types.js";

/** Keep byte-identical to `@tensorcost/contracts` sdkModelCallSpanId. */
const SDK_MODEL_CALL_SPAN_NAMESPACE =
  "tensorcost:sdk-observation:model-call-span:v1";

export function sdkModelCallSpanId(correlationId: string): string {
  const hash = createHash("sha256")
    .update(`${SDK_MODEL_CALL_SPAN_NAMESPACE}\0${correlationId}`)
    .digest();
  hash[6] = (hash[6]! & 0x0f) | 0x40;
  hash[8] = (hash[8]! & 0x3f) | 0x80;
  const hex = hash.subarray(0, 16).toString("hex");
  return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20, 32)}`;
}

/** Per-call trace link for agent-sdk + sub_agent rollup dedup. */
export type TraceLinkFields = Pick<Observation, "trace_id" | "parent_span_id">;

/**
 * Link published by @tensorcost/agent-sdk around each provider call.
 * Mirrors packages/agent-sdk/src/tensorcost-link.ts — same symbol, same shape.
 */
interface ModelCallLink {
  correlationId: string;
  traceId: string;
  parentSpanId?: string | null;
  consumed?: boolean;
}

const MODEL_CALL_LINK_KEY = Symbol.for("tensorcost.modelCallLink.v1");

function linkStore(): AsyncLocalStorage<ModelCallLink> {
  const g = globalThis as Record<symbol, unknown>;
  let store = g[MODEL_CALL_LINK_KEY] as AsyncLocalStorage<ModelCallLink> | undefined;
  if (!store) {
    store = new AsyncLocalStorage<ModelCallLink>();
    g[MODEL_CALL_LINK_KEY] = store;
  }
  return store;
}

const UUID_RE =
  /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

function uuidOrUndefined(v: string | null | undefined): string | undefined {
  const t = v?.trim();
  return t && UUID_RE.test(t) ? t : undefined;
}

/**
 * Correlation id and trace link for one wrapped provider call.
 *
 * Precedence: explicit withMeta() values, then the link agent-sdk put in
 * scope around this call, then a fresh id with no link. The ambient link is
 * taken by the first wrapper that reads it, so a second provider call made
 * inside the same scope cannot reuse the agent span's id.
 *
 * Ids that are not UUIDs are dropped here: ai-service keys spans and trace
 * rows by uuid, and a link it cannot store must not cost the observation.
 */
export function resolveModelCallLink(explicit: {
  correlationId?: string;
  traceId?: string;
  parentSpanId?: string;
}): { correlationId: string; traceLink: TraceLinkFields } {
  let correlationId = uuidOrUndefined(explicit.correlationId);
  let traceId = uuidOrUndefined(explicit.traceId);
  let parentSpanId = uuidOrUndefined(explicit.parentSpanId);

  if (correlationId === undefined && traceId === undefined) {
    const ambient = linkStore().getStore();
    if (ambient && !ambient.consumed) {
      ambient.consumed = true;
      correlationId = uuidOrUndefined(ambient.correlationId);
      traceId = uuidOrUndefined(ambient.traceId);
      parentSpanId = uuidOrUndefined(ambient.parentSpanId ?? undefined);
    }
  }

  const traceLink: TraceLinkFields =
    traceId !== undefined
      ? {
          trace_id: traceId,
          ...(parentSpanId !== undefined ? { parent_span_id: parentSpanId } : {}),
        }
      : {};
  return { correlationId: correlationId ?? randomUUID(), traceLink };
}
