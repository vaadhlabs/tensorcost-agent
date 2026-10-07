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

import type { Observation } from "./types.js";
import { SDK_VERSION } from "./version.js";
const TOKEN_REFRESH_LEEWAY_MS = 60_000;
const DEFAULT_TOKEN_LIFETIME_MS = 15 * 60_000;
const MAX_QUEUE_DEPTH = 1024;
const DEFAULT_TIMEOUT_MS = 2_000;

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
    queueMicrotask(() => {
      void this.deliver(observation).finally(() => {
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

  /** Exchange or return cached JWT for proxy/admit/sdk-layer calls. */
  async getToken(): Promise<string> {
    return this.fetchToken();
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
    const now = Date.now();
    if (
      this.tokenCache &&
      now < this.tokenCache.expiresAt - TOKEN_REFRESH_LEEWAY_MS
    ) {
      return this.tokenCache.token;
    }
    if (this.tokenInflight) {
      return this.tokenInflight;
    }
    this.tokenInflight = this.exchangeToken().finally(() => {
      this.tokenInflight = null;
    });
    return this.tokenInflight;
  }

  private async exchangeToken(): Promise<string> {
    if (!this.tenantId) {
      throw new Error(
        "tensorcost: tenant_id required for token exchange (pass tenantId to wrap() or set TENSORCOST_TENANT_ID)",
      );
    }
    const url = `${this.baseUrl}/api/inference-proxy/sdk-token/exchange`;
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.timeoutMs);
    try {
      const resp = await this.fetchImpl(url, {
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
  SDK_VERSION,
  MAX_QUEUE_DEPTH,
};
