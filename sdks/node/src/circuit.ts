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
 *   OPEN   ──(single-flight probe)───────> HALF_OPEN
 *   HALF_OPEN ──(proxy 2xx)──────────────> probeSuccesses++
 *              ──(proxy 5xx/err)──────────> back to OPEN (resets probes)
 *   HALF_OPEN ──(probeSuccesses >= closeThreshold)──> CLOSED
 *
 * This is intentionally simple and synchronous — the circuit lives on the
 * SDK client instance and is not shared across network calls.
 */

export type CircuitState = "CLOSED" | "OPEN" | "HALF_OPEN";

export interface CircuitBreakerConfig {
  /** Number of consecutive proxy failures before the circuit opens. Default: 3. */
  openThreshold: number;
  /** Number of consecutive successful probes before the circuit closes. Default: 5. */
  closeThreshold: number;
}

export const DEFAULT_CIRCUIT_CONFIG: CircuitBreakerConfig = {
  openThreshold: 3,
  closeThreshold: 5,
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
   * probe (OPEN → HALF_OPEN). Concurrent OPEN callers get false and bypass.
   */
  shouldProbe(): boolean {
    if (this.state === "OPEN" && !this.probeInFlight) {
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
