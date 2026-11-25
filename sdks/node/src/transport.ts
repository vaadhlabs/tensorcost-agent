/**
 * Fire-and-forget HTTP transport for observation events.
 *
 * Design:
 *
 * * `post(observation)` returns synchronously. Serialising, queueing and the
 *   fetch run on the microtask queue, so the customer's hot path never waits
 *   on network IO.
 * * Each observation is serialised once. A delivery that fails for a reason
 *   worth retrying (network error, timeout, 408, 429, 5xx, or a token
 *   exchange that hit one of those) goes back into a bounded in-memory
 *   buffer and is resent later with the same bytes, so the same
 *   correlation_id; ingestion dedupes on it, so a retry of a POST that
 *   actually landed is harmless.
 * * Backoff is transport-wide, not per observation: the first failure
 *   blocks sending for 1 s, each failed probe doubles that up to 60 s
 *   (with jitter), and while blocked only one request is in flight, so a
 *   recovering endpoint sees one probe rather than the whole buffer. A 429's
 *   Retry-After sets the wait instead. The first success clears it all.
 * * 400/401/403 (and any other 4xx but 408 and 429) are not retried: the
 *   same body would get the same answer. They warn once, then at most once
 *   a minute with a count, and are counted in `stats().rejected`.
 * * The buffer is capped by count and approximate bytes. When it is full
 *   the oldest observation goes (to the disk spool if one is configured,
 *   otherwise it is counted in `droppedCount`), with at most one warning a
 *   minute. An observation still undelivered after `maxRetryAgeMs` (an
 *   hour) is given up on and counted too.
 * * A short-lived JWT is fetched via
 *   `POST /api/inference-proxy/sdk-token/exchange` using the long-lived API
 *   key, cached in memory and refreshed within TOKEN_REFRESH_LEEWAY_MS of
 *   expiry.
 * * `stats()` counts what is buffered, what was delivered only after a
 *   retry, what was dropped and what was rejected. When delivery recovers
 *   after an outage that fact is logged once, with how many are waiting.
 * * Timers are unref'd: a transport waiting to retry never keeps the
 *   process alive. Exit draining is opt-out and lives in exit-flush.ts.
 * * All errors are swallowed and surfaced via console.warn when `failOpen`
 *   is true (the default). The customer's call site never sees one.
 */

import {
  TensorCostKeyRejectedError,
  TensorCostNetworkError,
  TensorCostProxyError,
  TensorCostQuotaError,
  TensorCostTimeoutError,
} from "./errors.js";
import { MissingConfigError } from "./config.js";
import { registerExitHooks, type ExitAware } from "./exit-flush.js";
import { Spool } from "./spool.js";
import type { Observation } from "./types.js";
import { SDK_VERSION } from "./version.js";
import { boundObservation } from "./observation-limits.js";
const TOKEN_REFRESH_LEEWAY_MS = 60_000;
/** A token this close to expiry is not handed out; the caller waits for a new one. */
const TOKEN_EXPIRY_MARGIN_MS = 5_000;
const DEFAULT_TOKEN_LIFETIME_MS = 15 * 60_000;
/** Concurrent observation POSTs; beyond this they wait in the buffer. */
const MAX_IN_FLIGHT = 1024;
const DEFAULT_TIMEOUT_MS = 2_000;
export const DEFAULT_MAX_BUFFERED_OBSERVATIONS = 10_000;
export const DEFAULT_MAX_BUFFERED_BYTES = 32 * 1024 * 1024;
export const DEFAULT_MAX_RETRY_AGE_MS = 60 * 60_000;
const BACKOFF_BASE_MS = 1_000;
const BACKOFF_MAX_MS = 60_000;
const WARN_INTERVAL_MS = 60_000;
/**
 * After a token exchange fails because TensorCost is unreachable, how long
 * callers on the request path are told so straight away instead of starting
 * (and waiting on) another exchange. Observations made meanwhile wait in the
 * retry buffer; the call is not slowed down by them.
 */
const TOKEN_OUTAGE_COOLDOWN_MS = 5_000;

export interface TransportOptions {
  baseUrl: string;
  apiKey: string;
  /** Required for token exchange against current inference-proxy DTO. */
  tenantId?: string | null;
  timeoutMs?: number;
  failOpen?: boolean;
  /** Optional fetch override for testing. Defaults to global fetch. */
  fetchImpl?: typeof fetch;
  /** Retry buffer cap by count. Default 10 000. */
  maxBufferedObservations?: number;
  /** Retry buffer cap by approximate serialised size. Default 32 MiB. */
  maxBufferedBytes?: number;
  /** Give up on an observation this long after it was first posted. Default 1 h. */
  maxRetryAgeMs?: number;
  /** Drain on beforeExit / SIGTERM. The transport default is off; wrap() turns it on. */
  flushOnExit?: boolean;
  /** Directory for the opt-in disk spool. Off when unset. */
  spoolDir?: string | null;
  spoolMaxBytes?: number;
  spoolSegmentBytes?: number;
  /** Jitter source, for tests. */
  random?: () => number;
}

export interface TransportStats {
  /** Observation POSTs on the wire now. */
  inFlight: number;
  /** Observations waiting in memory: new ones behind the in-flight cap, and retries. */
  buffered: number;
  bufferedBytes: number;
  /** Observations lost for good: buffer overflow without a spool, expiry, exit, spool overflow. */
  dropped: number;
  /** Observations written to the disk spool. */
  spooled: number;
  /** Observations held back by a failed delivery (theirs or the probe's) and then delivered. */
  deliveredAfterRetry: number;
  /** Observations TensorCost answered with a 4xx other than 408/429; never retried. */
  rejected: number;
}

interface TokenCache {
  token: string;
  expiresAt: number;
}

interface Entry {
  body: string;
  bytes: number;
  firstAt: number;
  /**
   * Held back by a failure: its own attempt failed, or it was waiting while
   * delivery was failing. Counted in deliveredAfterRetry when it lands.
   */
  held?: boolean;
}

type Outcome =
  | { kind: "ok" }
  | { kind: "retry"; detail: string; retryAfterMs: number | null }
  | { kind: "fatal"; status?: number; error?: unknown }
  | { kind: "aborted" };

export class ObservationTransport implements ExitAware {
  private readonly baseUrl: string;
  private readonly apiKey: string;
  private readonly tenantId: string | null;
  private readonly timeoutMs: number;
  private readonly failOpen: boolean;
  private readonly fetchImpl: typeof fetch;
  private readonly maxBuffered: number;
  private readonly maxBufferedBytes: number;
  private readonly maxRetryAgeMs: number;
  private readonly random: () => number;
  readonly flushOnExit: boolean;
  private readonly spool: Spool | null;

  /** Oldest first, by the time each observation was first posted. */
  private queue: Entry[] = [];
  private queueBytes = 0;
  private readonly inFlight = new Map<Entry, AbortController>();
  private consecutiveFailures = 0;
  private blockedUntil = 0;
  private timer: ReturnType<typeof setTimeout> | null = null;
  private timerAt = 0;
  /** Set by an exit drain; cleared by the next post(). */
  private dormant = false;
  private dropped = 0;
  private spooled = 0;
  private dropsSinceWarn = 0;
  private lastDropWarnAt = -Infinity;
  private lastFailWarnAt = -Infinity;
  private deliveredAfterRetry = 0;
  private rejected = 0;
  private rejectedSinceWarn = 0;
  private lastRejectWarnAt = -Infinity;
  private replay: Promise<void> = Promise.resolve();

  private tokenCache: TokenCache | null = null;
  /** In-flight token-exchange promise; coalesces concurrent refreshes. */
  private tokenInflight: Promise<string> | null = null;
  /** When the in-flight exchange started (Date.now()). */
  private tokenInflightStartedAt = 0;
  /** The last exchange that failed because TensorCost was unreachable. */
  private lastOutage: { err: Error; at: number } | null = null;
  private tokenCtrl: AbortController | null = null;

  constructor(opts: TransportOptions) {
    this.baseUrl = opts.baseUrl.replace(/\/+$/, "");
    this.apiKey = opts.apiKey;
    this.tenantId = opts.tenantId ?? null;
    this.timeoutMs = opts.timeoutMs ?? DEFAULT_TIMEOUT_MS;
    this.failOpen = opts.failOpen ?? true;
    this.fetchImpl = opts.fetchImpl ?? globalThis.fetch.bind(globalThis);
    this.maxBuffered = Math.max(1, opts.maxBufferedObservations ?? DEFAULT_MAX_BUFFERED_OBSERVATIONS);
    this.maxBufferedBytes = Math.max(1, opts.maxBufferedBytes ?? DEFAULT_MAX_BUFFERED_BYTES);
    this.maxRetryAgeMs = opts.maxRetryAgeMs ?? DEFAULT_MAX_RETRY_AGE_MS;
    this.random = opts.random ?? Math.random;
    this.flushOnExit = opts.flushOnExit ?? false;
    this.spool = opts.spoolDir
      ? new Spool(opts.spoolDir, {
          maxBytes: opts.spoolMaxBytes,
          segmentBytes: opts.spoolSegmentBytes,
          onLost: (lines, reason) => this.lose(lines, reason),
        })
      : null;
    if (this.flushOnExit || this.spool) registerExitHooks(this);
    if (this.spool && this.tenantId) this.replay = this.replaySpool();
  }

  /**
   * Enqueue an observation for fire-and-forget delivery. Returns
   * immediately (synchronously). Errors are logged via console.warn,
   * never thrown, when failOpen is true.
   */
  post(observation: Observation): void {
    // No tenant id, no token, so nothing can be delivered. wrap() has said so
    // once already; failing every POST would only repeat it per call.
    if (!this.tenantId && this.failOpen) return;
    this.dormant = false;
    queueMicrotask(() => {
      let body: string;
      try {
        // A long provider error must not cost the call its row (observation-limits.ts).
        body = JSON.stringify(boundObservation(observation));
      } catch (err) {
        // eslint-disable-next-line no-console
        console.warn(
          `tensorcost: observation could not be serialised: ${(err as Error).message}`,
        );
        return;
      }
      this.enqueue({ body, bytes: body.length, firstAt: Date.now() });
      this.pump();
    });
  }

  /**
   * Exchange or return the cached JWT for proxy/admit/sdk-layer calls.
   *
   * `budgetMs` is for callers on the customer's request path. They wait at
   * most that long, and no longer than that after the exchange in flight
   * started; the exchange itself carries on (with its own timeout) and
   * fills the cache for the next call. Giving up throws
   * TensorCostTimeoutError, which the fail-open rule reads as TensorCost
   * being unavailable. Within TOKEN_OUTAGE_COOLDOWN_MS of an exchange that
   * failed the same way, the call is told at once rather than made to wait
   * on another one.
   */
  async getToken(opts: { budgetMs?: number } = {}): Promise<string> {
    const cached = this.cachedToken();
    if (cached !== null) return cached;
    if (opts.budgetMs === undefined) return this.fetchToken();
    const now = Date.now();
    if (this.lastOutage && now - this.lastOutage.at < TOKEN_OUTAGE_COOLDOWN_MS) {
      throw this.lastOutage.err;
    }
    const exchange = this.fetchToken();
    const wait = Math.max(0, Math.min(opts.budgetMs, this.tokenInflightStartedAt + opts.budgetMs - now));
    let timer: ReturnType<typeof setTimeout> | undefined;
    try {
      return await Promise.race([
        exchange,
        new Promise<never>((_resolve, reject) => {
          timer = setTimeout(
            () =>
              reject(
                new TensorCostTimeoutError(
                  `tensorcost: token-exchange did not answer within ${opts.budgetMs}ms`,
                  "total",
                ),
              ),
            wait,
          );
        }),
      ]);
    } finally {
      clearTimeout(timer);
    }
  }

  /**
   * The cached token while it is still good. Inside the refresh leeway it is
   * still returned, and a refresh starts in the background, so the call
   * that notices the token is ageing doesn't wait for its replacement.
   */
  private cachedToken(): string | null {
    const c = this.tokenCache;
    if (!c) return null;
    const now = Date.now();
    if (now < c.expiresAt - TOKEN_REFRESH_LEEWAY_MS) return c.token;
    if (now < c.expiresAt - TOKEN_EXPIRY_MARGIN_MS) {
      const cooling = this.lastOutage && now - this.lastOutage.at < TOKEN_OUTAGE_COOLDOWN_MS;
      if (!this.tokenInflight && !cooling) void this.startExchange();
      return c.token;
    }
    return null;
  }

  /** Observations not yet delivered: in flight plus buffered. */
  get pendingCount(): number {
    return this.inFlight.size + this.queue.length;
  }

  /** Observations lost for good since this transport started. */
  get droppedCount(): number {
    return this.dropped;
  }

  stats(): TransportStats {
    return {
      inFlight: this.inFlight.size,
      buffered: this.queue.length,
      bufferedBytes: this.queueBytes,
      dropped: this.dropped,
      spooled: this.spooled,
      deliveredAfterRetry: this.deliveredAfterRetry,
      rejected: this.rejected,
    };
  }

  /**
   * Try to deliver everything queued or waiting to retry (Lambda shutdown,
   * test teardown). A pending backoff is cut short so one probe goes now.
   * Resolves when everything has landed, when the endpoint is still failing
   * and the next retry would fall past the deadline, or at the deadline.
   * Never throws; what is left stays buffered and keeps retrying. Resolves
   * to how many observations are still undelivered (0 when all landed).
   */
  async flush(timeoutMs = 5_000): Promise<number> {
    await this.drain(timeoutMs, false);
    return this.pendingCount;
  }

  // ------------------------------------------------------------------
  // Exit hooks (exit-flush.ts)

  needsExitFlush(): boolean {
    return !this.dormant && this.pendingCount > 0;
  }

  async flushForExit(timeoutMs: number): Promise<void> {
    await this.drain(timeoutMs, true);
  }

  spillRemainingSync(): void {
    const left = [...this.inFlight.keys(), ...this.queue];
    this.queue = [];
    this.queueBytes = 0;
    this.inFlight.clear();
    this.dormant = true;
    if (this.spool) {
      this.spool.appendSync(left.map((e) => e.body));
      this.spooled += left.length;
      return;
    }
    if (left.length === 0) return;
    this.dropped += left.length;
    // eslint-disable-next-line no-console
    console.warn(
      `tensorcost: ${left.length} observation(s) were not delivered before exit ` +
        "(set spoolDir to keep them on disk for the next start)",
    );
  }

  /** Test hook: resolves when queued spool writes are on disk. */
  async spoolIdle(): Promise<void> {
    await this.spool?.idle();
  }

  /** Test hook: resolves when the start-up spool replay has been read in. */
  async replayDone(): Promise<void> {
    await this.replay;
  }

  // ------------------------------------------------------------------
  // Internal

  private async drain(timeoutMs: number, forExit: boolean): Promise<void> {
    const deadline = Date.now() + timeoutMs;
    // Let post() calls made just before this one land in the queue.
    await new Promise<void>((resolve) => queueMicrotask(resolve));
    if (this.pendingCount === 0) {
      if (forExit) this.dormant = true;
      return;
    }
    this.dormant = false;
    if (this.inFlight.size === 0 && this.blockedUntil > Date.now()) {
      this.blockedUntil = 0; // probe now rather than wait out the backoff
    }
    this.pump();
    while (this.pendingCount > 0) {
      const now = Date.now();
      if (now >= deadline) break;
      if (this.inFlight.size === 0 && this.blockedUntil >= deadline) break;
      await sleep(Math.min(10, deadline - now));
    }
    if (!forExit) return;
    // Exit: stop sending, abort what is still open so its sockets and
    // timers release the loop, and leave the rest for spillRemainingSync.
    this.dormant = true;
    for (const ctrl of this.inFlight.values()) ctrl.abort();
    this.tokenCtrl?.abort();
    const settleBy = Date.now() + 50;
    while (this.inFlight.size > 0 && Date.now() < settleBy) await sleep(5);
  }

  private enqueue(entry: Entry): void {
    // Retries go back in first-posted order; new ones land at the end.
    let i = this.queue.length;
    while (i > 0 && this.queue[i - 1]!.firstAt > entry.firstAt) i -= 1;
    this.queue.splice(i, 0, entry);
    this.queueBytes += entry.bytes;
    if (this.consecutiveFailures > 0) entry.held = true;
    while (
      this.queue.length > 0 &&
      (this.queue.length > this.maxBuffered || this.queueBytes > this.maxBufferedBytes)
    ) {
      const oldest = this.queue.shift()!;
      this.queueBytes -= oldest.bytes;
      this.overflow(oldest);
    }
  }

  private overflow(entry: Entry): void {
    if (this.spool && this.spool.append(entry.body)) {
      this.spooled += 1;
      this.warnDrop("spooled to disk");
      return;
    }
    this.dropped += 1;
    this.warnDrop("dropped");
  }

  private lose(count: number, reason: string): void {
    this.dropped += count;
    this.dropsSinceWarn += count;
    this.warnDrop(reason, false);
  }

  private warnDrop(what: string, count = true): void {
    if (count) this.dropsSinceWarn += 1;
    const now = Date.now();
    if (now - this.lastDropWarnAt < WARN_INTERVAL_MS) return;
    this.lastDropWarnAt = now;
    // eslint-disable-next-line no-console
    console.warn(
      `tensorcost: observation buffer full (${this.queue.length} waiting); ` +
        `${this.dropsSinceWarn} oldest ${what === "spooled to disk" ? "spooled to disk" : `dropped (${what})`} ` +
        `since the last warning, ${this.dropped} dropped in total`,
    );
    this.dropsSinceWarn = 0;
  }

  private pump(): void {
    if (this.dormant) return;
    const now = Date.now();
    while (this.queue.length > 0 && now - this.queue[0]!.firstAt > this.maxRetryAgeMs) {
      const stale = this.queue.shift()!;
      this.queueBytes -= stale.bytes;
      this.expire();
    }
    if (this.queue.length === 0) return;
    if (now < this.blockedUntil) {
      this.schedule(this.blockedUntil);
      return;
    }
    const limit = this.consecutiveFailures > 0 ? 1 : MAX_IN_FLIGHT;
    while (this.inFlight.size < limit && this.queue.length > 0) {
      const entry = this.queue.shift()!;
      this.queueBytes -= entry.bytes;
      this.dispatch(entry);
    }
  }

  private expire(): void {
    this.dropped += 1;
    this.dropsSinceWarn += 1;
    this.warnDrop("undelivered after the retry window", false);
  }

  private schedule(at: number): void {
    if (this.timer && this.timerAt <= at) return;
    if (this.timer) clearTimeout(this.timer);
    this.timerAt = at;
    this.timer = setTimeout(() => {
      this.timer = null;
      this.pump();
    }, Math.max(0, at - Date.now()));
    // A pending retry never keeps the process alive on its own.
    (this.timer as { unref?: () => void }).unref?.();
  }

  private dispatch(entry: Entry): void {
    const ctrl = new AbortController();
    this.inFlight.set(entry, ctrl);
    void this.attempt(entry, ctrl).then((outcome) => {
      if (this.inFlight.get(entry) !== ctrl) return; // spilled at exit
      this.inFlight.delete(entry);
      this.settle(entry, outcome);
      this.pump();
    });
  }

  private settle(entry: Entry, outcome: Outcome): void {
    const now = Date.now();
    switch (outcome.kind) {
      case "ok":
        if (entry.held) this.deliveredAfterRetry += 1;
        if (this.consecutiveFailures > 0) {
          // eslint-disable-next-line no-console
          console.warn(
            `tensorcost: observation delivery recovered; sending ${this.queue.length} buffered ` +
              `observation(s) (${this.deliveredAfterRetry} delivered after a retry so far)`,
          );
        }
        this.consecutiveFailures = 0;
        this.blockedUntil = 0;
        return;
      case "aborted":
        this.enqueue(entry);
        return;
      case "fatal":
        if (outcome.status !== undefined) {
          this.warnRejected(outcome.status);
          return;
        }
        if (!this.failOpen) throw outcome.error;
        // eslint-disable-next-line no-console
        console.warn(
          `tensorcost: observation delivery failed: ${(outcome.error as Error)?.message ?? String(outcome.error)}`,
        );
        return;
      case "retry": {
        if (now - entry.firstAt > this.maxRetryAgeMs) {
          this.expire();
          return;
        }
        if (outcome.retryAfterMs !== null) {
          this.consecutiveFailures = Math.max(1, this.consecutiveFailures);
          this.blockedUntil = Math.max(this.blockedUntil, now + outcome.retryAfterMs);
        } else if (now >= this.blockedUntil) {
          // Only a request sent while we were not already backing off
          // escalates; the rest of an outage's first wave just requeues.
          this.consecutiveFailures += 1;
          this.blockedUntil = now + this.backoffMs(this.consecutiveFailures);
        }
        entry.held = true;
        for (const waiting of this.queue) waiting.held = true;
        this.enqueue(entry);
        if (now - this.lastFailWarnAt >= WARN_INTERVAL_MS) {
          this.lastFailWarnAt = now;
          // eslint-disable-next-line no-console
          console.warn(
            `tensorcost: observation delivery failed (${outcome.detail}); ` +
              `${this.queue.length} buffered, retrying`,
          );
        }
        return;
      }
    }
  }

  /** A 4xx TensorCost won't change its mind about: warn on the first, then once a minute. */
  private warnRejected(status: number): void {
    this.rejected += 1;
    this.rejectedSinceWarn += 1;
    const now = Date.now();
    if (now - this.lastRejectWarnAt < WARN_INTERVAL_MS) return;
    this.lastRejectWarnAt = now;
    // eslint-disable-next-line no-console
    console.warn(
      `tensorcost: observation POST failed status=${status}; not retried ` +
        `(${this.rejectedSinceWarn} rejected since the last warning, ${this.rejected} in total)`,
    );
    this.rejectedSinceWarn = 0;
  }

  private backoffMs(failures: number): number {
    const ceiling = Math.min(BACKOFF_MAX_MS, BACKOFF_BASE_MS * 2 ** (failures - 1));
    return Math.round(ceiling * (0.5 + this.random() * 0.5));
  }

  private async attempt(entry: Entry, ctrl: AbortController): Promise<Outcome> {
    let token: string;
    try {
      token = await this.fetchToken();
    } catch (err) {
      if (ctrl.signal.aborted) return { kind: "aborted" };
      return classifyTokenError(err);
    }
    if (ctrl.signal.aborted) return { kind: "aborted" };
    let timedOut = false;
    const timer = setTimeout(() => {
      timedOut = true;
      ctrl.abort();
    }, this.timeoutMs);
    try {
      const resp = await this.fetchImpl(`${this.baseUrl}/api/inference-proxy/observation`, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${token}`,
          "Content-Type": "application/json",
          "User-Agent": SDK_VERSION,
        },
        body: entry.body,
        signal: ctrl.signal,
      });
      void resp.body?.cancel().catch(() => {});
      if (resp.ok) return { kind: "ok" };
      if (isRetryableStatus(resp.status)) {
        return {
          kind: "retry",
          detail: `status=${resp.status}`,
          retryAfterMs:
            resp.status === 429 || resp.status === 503
              ? parseRetryAfter(resp.headers.get("retry-after"), this.maxRetryAgeMs)
              : null,
        };
      }
      return { kind: "fatal", status: resp.status };
    } catch (err) {
      if (ctrl.signal.aborted && !timedOut) return { kind: "aborted" };
      const detail = timedOut
        ? `timed out after ${this.timeoutMs}ms`
        : (err as Error)?.message ?? String(err);
      return { kind: "retry", detail, retryAfterMs: null };
    } finally {
      clearTimeout(timer);
    }
  }

  private async replaySpool(): Promise<void> {
    const spool = this.spool!;
    try {
      // Read a segment at a time while there is room, so a large spool
      // does not push the buffer straight back out to disk.
      while (this.queue.length < this.maxBuffered / 2) {
        const lines = await spool.claimNext();
        if (!lines) return;
        const now = Date.now();
        for (const body of lines) this.enqueue({ body, bytes: body.length, firstAt: now });
        this.pump();
      }
    } catch {
      /* replay is best effort */
    }
  }

  private async fetchToken(): Promise<string> {
    const cached = this.cachedToken();
    if (cached !== null) return cached;
    return this.tokenInflight ?? this.startExchange();
  }

  private startExchange(): Promise<string> {
    this.tokenInflightStartedAt = Date.now();
    this.tokenInflight = this.exchangeToken()
      .then(
        (token) => {
          this.lastOutage = null;
          return token;
        },
        (err: unknown) => {
          if (isOutage(err)) this.lastOutage = { err: err as Error, at: Date.now() };
          throw err;
        },
      )
      .finally(() => {
        this.tokenInflight = null;
      });
    // A bounded caller may have stopped waiting; nobody else may be listening.
    this.tokenInflight.catch(() => undefined);
    return this.tokenInflight;
  }

  private async exchangeToken(): Promise<string> {
    if (!this.tenantId) {
      // wrap() refuses govern/steer/route without a tenant id, so this is
      // only reached by observation delivery (fail-open, logged) in observe.
      throw new MissingConfigError(
        "tensorcost: tenant_id required for token exchange (pass tenantId to wrap() or set TENSORCOST_TENANT_ID)",
      );
    }
    const url = `${this.baseUrl}/api/inference-proxy/sdk-token/exchange`;
    const controller = new AbortController();
    this.tokenCtrl = controller;
    const timer = setTimeout(() => controller.abort(), this.timeoutMs);
    try {
      // Outages are typed so the wrappers' fail-open predicate can tell
      // "TensorCost is down" (network, timeout, 5xx, 429) from "TensorCost
      // said no" (any other status), which is not a reason to go around it.
      let resp: Response;
      try {
        resp = await this.fetchImpl(url, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "User-Agent": SDK_VERSION,
          },
          body: JSON.stringify({
            tenant_id: this.tenantId,
            sdk_long_lived_token: this.apiKey,
          }),
          signal: controller.signal,
        });
      } catch (err) {
        const e = err as Error;
        if (e?.name === "AbortError" || e?.name === "TimeoutError") {
          throw new TensorCostTimeoutError(
            `tensorcost: token-exchange timed out after ${this.timeoutMs}ms`,
            "total",
            { cause: e },
          );
        }
        throw new TensorCostNetworkError(
          `tensorcost: token-exchange network error: ${e?.message ?? String(err)}`,
          { cause: e },
        );
      }
      if (resp.status >= 500) {
        throw new TensorCostProxyError(`token-exchange failed status=${resp.status}`, {
          status: resp.status,
        });
      }
      if (resp.status === 429) {
        throw new TensorCostQuotaError(`token-exchange failed status=${resp.status}`, null);
      }
      if (resp.status === 401 || resp.status === 403) {
        throw new TensorCostKeyRejectedError(resp.status);
      }
      if (!resp.ok) {
        throw new Error(`token-exchange failed status=${resp.status}`);
      }
      const body = (await resp.json()) as {
        token?: string;
        access_token?: string;
        expires_in?: number;
      };
      const token = body.token ?? body.access_token;
      if (!token) {
        throw new Error("token-exchange response missing token field");
      }
      const lifetimeMs =
        typeof body.expires_in === "number"
          ? body.expires_in * 1000
          : DEFAULT_TOKEN_LIFETIME_MS;
      this.tokenCache = {
        token,
        expiresAt: Date.now() + lifetimeMs,
      };
      return token;
    } finally {
      clearTimeout(timer);
    }
  }
}

export const __test__ = {
  TOKEN_OUTAGE_COOLDOWN_MS,
  SDK_VERSION,
  MAX_IN_FLIGHT,
};

/** The exchange failed because TensorCost was unreachable, not because it said no. */
function isOutage(err: unknown): boolean {
  return (
    err instanceof TensorCostNetworkError ||
    err instanceof TensorCostTimeoutError ||
    err instanceof TensorCostProxyError ||
    err instanceof TensorCostQuotaError
  );
}

function isRetryableStatus(status: number): boolean {
  return status === 408 || status === 429 || status >= 500;
}

function classifyTokenError(err: unknown): Outcome {
  if (isOutage(err)) {
    return { kind: "retry", detail: (err as Error).message, retryAfterMs: null };
  }
  return { kind: "fatal", error: err };
}

/** Retry-After as delta-seconds or an HTTP-date; null when absent or unreadable. */
export function parseRetryAfter(value: string | null, maxMs: number): number | null {
  if (value === null) return null;
  const trimmed = value.trim();
  if (trimmed === "") return null;
  let ms: number;
  if (/^\d+(\.\d+)?$/.test(trimmed)) {
    ms = Number(trimmed) * 1000;
  } else {
    const at = Date.parse(trimmed);
    if (Number.isNaN(at)) return null;
    ms = at - Date.now();
  }
  return Math.min(Math.max(0, ms), maxMs);
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
