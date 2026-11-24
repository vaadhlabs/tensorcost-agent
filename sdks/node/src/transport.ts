/**
 * Fire-and-forget HTTP transport for observation events.
 *
 * Design:
 *
 * * `post(observation)` returns synchronously the moment the observation
 *   is enqueued. The actual fetch runs on the microtask queue via
 *   `queueMicrotask`, so the customer's hot path is never blocked on
 *   network IO.
 * * If the in-flight queue saturates (>= MAX_QUEUE_DEPTH), we drop the
 *   oldest observation rather than block. Losing one observation is
 *   strictly preferable to slowing down the customer's request loop.
 * * A short-lived JWT is fetched once via
 *   `POST /api/inference-proxy/sdk-token/exchange` using the long-lived
 *   API key.
 *   The token is cached in-memory and refreshed when within
 *   TOKEN_REFRESH_LEEWAY_MS of expiry.
 * * All errors during posting are swallowed when `failOpen` is true
 *   (the default) and surfaced via console.warn. The customer's call
 *   site never sees an exception.
 */

import {
  TensorCostKeyRejectedError,
  TensorCostNetworkError,
  TensorCostProxyError,
  TensorCostQuotaError,
  TensorCostTimeoutError,
} from "./errors.js";
import { MissingConfigError } from "./config.js";
import type { Observation } from "./types.js";
import { SDK_VERSION } from "./version.js";
import { boundObservation } from "./observation-limits.js";
const TOKEN_REFRESH_LEEWAY_MS = 60_000;
/** A token this close to expiry is not handed out; the caller waits for a new one. */
const TOKEN_EXPIRY_MARGIN_MS = 5_000;
const DEFAULT_TOKEN_LIFETIME_MS = 15 * 60_000;
const MAX_QUEUE_DEPTH = 1024;
const DEFAULT_TIMEOUT_MS = 2_000;
/**
 * After a token exchange fails because TensorCost is unreachable, how long
 * callers on the request path are told so straight away instead of starting
 * (and waiting on) another exchange. Observations made meanwhile are lost
 * either way; the call is not slowed down by them.
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
}

interface TokenCache {
  token: string;
  expiresAt: number;
}

export class ObservationTransport {
  private readonly baseUrl: string;
  private readonly apiKey: string;
  private readonly tenantId: string | null;
  private readonly timeoutMs: number;
  private readonly failOpen: boolean;
  private readonly fetchImpl: typeof fetch;

  private pending = 0;
  private tokenCache: TokenCache | null = null;
  /** In-flight token-exchange promise; coalesces concurrent refreshes. */
  private tokenInflight: Promise<string> | null = null;
  /** When the in-flight exchange started (Date.now()). */
  private tokenInflightStartedAt = 0;
  /** The last exchange that failed because TensorCost was unreachable. */
  private lastOutage: { err: Error; at: number } | null = null;

  constructor(opts: TransportOptions) {
    this.baseUrl = opts.baseUrl.replace(/\/+$/, "");
    this.apiKey = opts.apiKey;
    this.tenantId = opts.tenantId ?? null;
    this.timeoutMs = opts.timeoutMs ?? DEFAULT_TIMEOUT_MS;
    this.failOpen = opts.failOpen ?? true;
    this.fetchImpl = opts.fetchImpl ?? globalThis.fetch.bind(globalThis);
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
    if (this.pending >= MAX_QUEUE_DEPTH) {
      // Backpressure: drop *this* observation (the "oldest" semantically
      // is whichever is in-flight; in a microtask-dispatched queue we
      // cannot pull in-flight work back, so we drop the newest enqueue
      // attempt — the practical effect of "drop oldest when full" once
      // the queue stays saturated.).
      // eslint-disable-next-line no-console
      console.warn(
        "tensorcost: observation queue saturated; dropping observation",
      );
      return;
    }
    this.pending += 1;
    // A long provider error must not cost the call its row (observation-limits.ts).
    const bounded = boundObservation(observation);
    queueMicrotask(() => {
      void this.deliver(bounded).finally(() => {
        this.pending -= 1;
      });
    });
  }

  private async deliver(observation: Observation): Promise<void> {
    try {
      const token = await this.fetchToken();
      const url = `${this.baseUrl}/api/inference-proxy/observation`;
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), this.timeoutMs);
      try {
        const resp = await this.fetchImpl(url, {
          method: "POST",
          headers: {
            Authorization: `Bearer ${token}`,
            "Content-Type": "application/json",
            "User-Agent": SDK_VERSION,
          },
          body: JSON.stringify(observation),
          signal: controller.signal,
        });
        if (!resp.ok) {
          // eslint-disable-next-line no-console
          console.warn(
            `tensorcost: observation POST failed status=${resp.status}`,
          );
        }
      } finally {
        clearTimeout(timer);
      }
    } catch (err) {
      if (!this.failOpen) {
        throw err;
      }
      // eslint-disable-next-line no-console
      console.warn(
        `tensorcost: observation delivery failed: ${(err as Error).message}`,
      );
    }
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
   * on another one. Through 1.7.0 every call waited out the full 2 s
   * exchange timeout, twice, whenever TensorCost was unreachable.
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

  /** In-flight observation POST count (microtask queue + network). */
  get pendingCount(): number {
    return this.pending;
  }

  /**
   * Wait for queued observations to finish (Lambda shutdown / test teardown).
   * Does not throw when the deadline is hit — best-effort only.
   */
  async flush(timeoutMs = 5_000): Promise<void> {
    const deadline = Date.now() + timeoutMs;
    while (this.pending > 0 && Date.now() < deadline) {
      await new Promise((resolve) => setTimeout(resolve, 10));
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

/** The exchange failed because TensorCost was unreachable, not because it said no. */
function isOutage(err: unknown): boolean {
  return (
    err instanceof TensorCostNetworkError ||
    err instanceof TensorCostTimeoutError ||
    err instanceof TensorCostProxyError ||
    err instanceof TensorCostQuotaError
  );
}

export const __test__ = {
  TOKEN_OUTAGE_COOLDOWN_MS,
  SDK_VERSION,
  MAX_QUEUE_DEPTH,
};
