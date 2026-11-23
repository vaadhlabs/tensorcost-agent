/**
 * Circuit breaker for the fail-open fallback (Feature E).
 *
 * Tracks consecutive proxy failures. When the failure count hits
 * `openThreshold`, the circuit opens and all subsequent requests route
 * directly to the provider instead of through the proxy. The circuit
 * moves back to closed once `closeThreshold` consecutive probe calls
 * (which go through the proxy) succeed.
 *
 * State machine:
 *
 *   CLOSED ──(failures >= openThreshold)──> OPEN
 *   OPEN   ──(single-flight probe, at most
 *             once per probeIntervalMs)──> HALF_OPEN
 *   HALF_OPEN ──(proxy 2xx)──────────────> probeSuccesses++
 *              ──(proxy 5xx/err)──────────> back to OPEN (resets probes)
 *   HALF_OPEN ──(probeSuccesses >= closeThreshold)──> CLOSED
 *
 * This is intentionally simple and synchronous — the circuit lives on the
 * SDK client instance and is not shared across network calls.
 */

import { emitEvent } from "./telemetry.js";
import type { LifecycleEventCallback, Provider } from "./telemetry.js";

export type CircuitState = "CLOSED" | "OPEN" | "HALF_OPEN";

export interface CircuitBreakerConfig {
  /** Number of consecutive proxy failures before the circuit opens. Default: 3. */
  openThreshold: number;
  /** Number of consecutive successful probes before the circuit closes. Default: 5. */
  closeThreshold: number;
  /**
   * How long the circuit stays OPEN before a call may probe the proxy
   * again. 5000 in DEFAULT_CIRCUIT_CONFIG, which wrap() uses; a config
   * without it probes on the next call. Until 1.7.0 the next call always probed, so with
   * the proxy hung every call still waited out the proxy timeout (up to
   * timeoutMs, 60 s by default, for a buffered call) before falling back.
   */
  probeIntervalMs?: number;
}

export const DEFAULT_CIRCUIT_CONFIG: CircuitBreakerConfig = {
  openThreshold: 3,
  closeThreshold: 5,
  probeIntervalMs: 5_000,
};

export type CircuitTransitionListener = (
  from: CircuitState,
  to: CircuitState,
) => void;

export class CircuitBreaker {
  private state: CircuitState = "CLOSED";
  private consecutiveFailures = 0;
  private probeSuccesses = 0;
  /** True while a HALF_OPEN probe is in flight (single-flight). */
  private probeInFlight = false;
  /** When the circuit last went OPEN (Date.now()). */
  private openedAt = 0;
  readonly config: CircuitBreakerConfig;
  private readonly onTransition?: CircuitTransitionListener;

  constructor(
    config: CircuitBreakerConfig = DEFAULT_CIRCUIT_CONFIG,
    onTransition?: CircuitTransitionListener,
  ) {
    this.config = config;
    this.onTransition = onTransition;
  }

  private transition(to: CircuitState): void {
    if (to === this.state) return;
    const from = this.state;
    this.state = to;
    if (to === "OPEN") this.openedAt = Date.now();
    this.onTransition?.(from, to);
  }

  /** Current circuit state. */
  get currentState(): CircuitState {
    return this.state;
  }

  /** How many consecutive failures have been recorded. */
  get failures(): number {
    return this.consecutiveFailures;
  }

  /**
   * Returns true when calls should bypass the proxy and go directly to the
   * provider. The caller must always call `recordSuccess` or `recordFailure`
   * afterwards to keep the state machine moving.
   */
  shouldBypass(): boolean {
    if (this.state === "OPEN") return true;
    // Another caller is already probing — bypass until the probe settles.
    if (this.state === "HALF_OPEN" && this.probeInFlight) return true;
    return false;
  }

  /**
   * Returns true when this caller should run the single-flight HALF_OPEN
   * probe (OPEN → HALF_OPEN). Concurrent OPEN callers, and every caller
   * within probeIntervalMs of the circuit opening, get false and bypass.
   */
  shouldProbe(): boolean {
    const interval = this.config.probeIntervalMs ?? 0;
    if (this.state === "OPEN" && !this.probeInFlight && Date.now() - this.openedAt >= interval) {
      this.probeInFlight = true;
      this.transition("HALF_OPEN");
      return true;
    }
    return false;
  }

  /** Call after a proxy request succeeds. */
  recordSuccess(): void {
    if (this.state === "HALF_OPEN") {
      this.probeInFlight = false;
      this.probeSuccesses += 1;
      if (this.probeSuccesses >= this.config.closeThreshold) {
        this.reset();
      }
    } else if (this.state === "CLOSED") {
      this.consecutiveFailures = 0;
    }
  }

  /**
   * Clear a HALF_OPEN probe that never reached recordSuccess/recordFailure
   * (e.g. token exchange threw before the HTTP call). Idempotent once the
   * probe has settled.
   */
  releaseStuckProbe(): void {
    if (this.state === "HALF_OPEN" && this.probeInFlight) {
      this.probeSuccesses = 0;
      this.probeInFlight = false;
      this.transition("OPEN");
    }
  }

  /** Call after a proxy request fails (5xx or network error). */
  recordFailure(): void {
    if (this.state === "HALF_OPEN") {
      this.probeSuccesses = 0;
      this.probeInFlight = false;
      this.transition("OPEN");
    } else if (this.state === "CLOSED") {
      this.consecutiveFailures += 1;
      if (this.consecutiveFailures >= this.config.openThreshold) {
        this.transition("OPEN");
      }
    }
    // OPEN + another failure: stay open, no state change needed.
  }

  private reset(): void {
    this.consecutiveFailures = 0;
    this.probeSuccesses = 0;
    this.probeInFlight = false;
    this.transition("CLOSED");
  }
}

/**
 * The transition listener wrap() installs: the transition goes to the
 * caller's onLifecycleEvent, and opening or closing again is logged once
 * each (HALF_OPEN probes, which can flap with every call during an outage,
 * only reach the callback). Nothing is sent to TensorCost: until 1.7.0 each
 * transition was posted as an observation and stored as a 0-token spend
 * event, so an outage added phantom calls to the tenant's counts.
 */
export function reportCircuitTransitions(
  provider: Provider,
  onLifecycleEvent: LifecycleEventCallback | undefined,
  breaker: () => CircuitBreaker | undefined,
): CircuitTransitionListener {
  return (from, to) => {
    emitEvent(onLifecycleEvent, {
      kind: "on_circuit_state",
      provider,
      from,
      to,
      consecutiveFailures: breaker()?.failures ?? 0,
    });
    if (from === "CLOSED" && to === "OPEN") {
      // eslint-disable-next-line no-console
      console.warn(
        `tensorcost: the ${provider} proxy failed ${breaker()?.failures ?? "several"} times in a row; ` +
          "calls go straight to the provider until probes through the proxy succeed again",
      );
    } else if (to === "CLOSED") {
      // eslint-disable-next-line no-console
      console.warn(`tensorcost: the ${provider} proxy is answering again; calls go through it`);
    }
  };
}
