/**
 * Telemetry hook types for the TensorCost Node SDK.
 *
 * Customers pass `onLifecycleEvent` in WrapOptions to receive typed events
 * at each stage of a request lifecycle. The default is no-op — not providing
 * the callback adds zero overhead to the hot path.
 *
 * What's included in every event:
 *   - provider, model, operation — request identity
 *   - attemptNumber             — 1-based; increments on each retry
 *   - elapsedMs                 — wall-clock time since the first attempt began
 *
 * What is never included:
 *   - prompt / completion content
 *   - request or response bodies
 *   - credentials
 *
 * The five event kinds map to the lifecycle stages:
 *
 *   before_request   fires before each attempt (including retried ones)
 *   after_response   fires when a request completes successfully
 *   on_retry         fires when a request is going to be retried, with the
 *                    reason and how long the SDK will wait before the next
 *                    attempt
 *   on_error         fires when the SDK gives up (all retries exhausted or
 *                    a non-retriable error occurred)
 *   on_fallback      fires when the circuit breaker opens and the SDK routes
 *                    to the provider directly instead of the proxy
 */

export type Provider = "openai" | "anthropic";

/** Common fields shared by every lifecycle event. */
export interface TelemetryBase {
  provider: Provider;
  model: string;
  operation: string;
  attemptNumber: number;
  /** Wall-clock milliseconds since the first attempt for this call began. */
  elapsedMs: number;
}

export interface BeforeRequestEvent extends TelemetryBase {
  kind: "before_request";
}

export interface AfterResponseEvent extends TelemetryBase {
  kind: "after_response";
  status: number;
  /** Request ID from the X-TC-Request-ID header, when the proxy set it. */
  requestId: string | null;
}

export interface OnRetryEvent extends TelemetryBase {
  kind: "on_retry";
  reason: "5xx" | "network_error" | "quota_429";
  status: number | null;
  /** How many milliseconds the SDK will wait before the next attempt. */
  delayMs: number;
}

export interface OnErrorEvent extends TelemetryBase {
  kind: "on_error";
  errorClass: string;
  status: number | null;
}

export interface OnFallbackEvent extends TelemetryBase {
  kind: "on_fallback";
  /** Number of consecutive proxy failures that tripped the circuit. */
  consecutiveFailures: number;
}

export type LifecycleEvent =
  | BeforeRequestEvent
  | AfterResponseEvent
  | OnRetryEvent
  | OnErrorEvent
  | OnFallbackEvent;

export type LifecycleEventCallback = (event: LifecycleEvent) => void;

/** Call `cb` if defined — errors in the callback are always swallowed. */
export function emitEvent(
  cb: LifecycleEventCallback | undefined,
  event: LifecycleEvent,
): void {
  if (!cb) return;
  try {
    cb(event);
  } catch {
    // Customer callbacks must never blow up the SDK's control flow.
  }
}
