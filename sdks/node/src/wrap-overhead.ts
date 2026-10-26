/**
 * Measures synchronous SDK wrap overhead in the customer process.
 *
 * All awaited I/O (layer fetch, admit, proxy, provider) must run inside
 * `withAwaitExcluded` so RTT is not counted. Sync work before/after those
 * hops — token extraction, observation enqueue — is included. The async
 * HTTP delivery inside `transport.post`'s microtask is excluded.
 */

import type { Observation } from "./types.js";
import type { ObservationTransport } from "./transport.js";

export class WrapOverhead {
  private readonly hookStart = performance.now();
  private accumulatedMs = 0;
  private activeStart: number | null = this.hookStart;

  /** Pause the sync clock before an awaited hop. */
  pauseForAwait(): void {
    if (this.activeStart === null) return;
    this.accumulatedMs += performance.now() - this.activeStart;
    this.activeStart = null;
  }

  /** Resume the sync clock after an awaited hop returns. */
  resumeAfterAwait(): void {
    if (this.activeStart !== null) return;
    this.activeStart = performance.now();
  }

  /** Milliseconds of synchronous wrap work (rounded, non-negative). */
  sdkAddedMs(): number {
    let total = this.accumulatedMs;
    if (this.activeStart !== null) {
      total += performance.now() - this.activeStart;
    }
    return Math.max(0, Math.round(total));
  }
}

/** Run awaited work with wrap clock paused — use for every external I/O hop. */
export async function withAwaitExcluded<T>(
  overhead: WrapOverhead,
  fn: () => Promise<T>,
): Promise<T> {
  overhead.pauseForAwait();
  try {
    return await fn();
  } finally {
    overhead.resumeAfterAwait();
  }
}

/**
 * Enqueue an observation with `sdk_added_ms` stamped. Fail-open — never throws.
 * Call after building the observation (token extraction included in overhead).
 */
export function postObservation(
  transport: ObservationTransport,
  overhead: WrapOverhead,
  obs: Observation,
): void {
  try {
    transport.post({ ...obs, sdk_added_ms: overhead.sdkAddedMs() });
  } catch {
    /* fail-open */
  }
}

/**
 * Off in 1.5.0. The server's ObservationDto reads `application` and `tags`,
 * but byoc-v0.1.40 (what design partners run) validates observations with
 * forbidNonWhitelisted and has neither field, so an observation carrying
 * them is rejected with a 400 and dropped. Turned on in a later release alongside the
 * server release that accepts them everywhere.
 */
let sendObservationScope = false;

/**
 * `application` / `tags` for an observation, from the wrap config. Before
 * this they only rode on proxied calls as x-tc-application / x-tc-tags.
 * Returns nothing while the flag above is off.
 */
export function observationScope(
  application: string | null | undefined,
  tags: readonly string[] | undefined,
): { application?: string; tags?: string[] } {
  if (!sendObservationScope) return {};
  return {
    ...(application ? { application } : {}),
    ...(tags && tags.length > 0 ? { tags: [...tags] } : {}),
  };
}

export const __test__ = {
  WrapOverhead,
  /** Internal: flips the held-back observation fields on, for tests only. */
  setObservationScope(on: boolean): void {
    sendObservationScope = on;
  },
};
