/**
 * Retry utilities for the TensorCost applied-mode request path.
 *
 * Retries use jittered exponential backoff:
 *
 *   delay = min(maxDelayMs, baseDelayMs * 2^attempt) * (0.5 + Math.random() * 0.5)
 *
 * where `attempt` is 0-indexed. With defaults (baseDelayMs=500, maxDelayMs=30 000),
 * the sequence of *maximum* delays is:
 *   attempt 1: 500ms
 *   attempt 2: 1 000ms
 *   attempt 3: 2 000ms
 *   attempt 4: 4 000ms
 *   attempt 5: 8 000ms
 * The jitter factor halves the minimum so actual delays are 50–100% of those
 * maximums. This matches the posture used by the OpenAI Node SDK.
 *
 * Rules:
 *   - Never retry 4xx responses (except 429).
 *   - Always retry 5xx responses and network errors.
 *   - On 429: honour the `Retry-After` response header when present
 *     (overrides the backoff delay).
 *   - Non-idempotent verbs (POST without `stream:false` is still idempotent
 *     at the proxy level — the proxy is stateless) default to idempotent for
 *     all chat/completions calls.
 */

export interface RetryConfig {
  /** Maximum number of total attempts (1 = no retries). Default: 5. */
  maxAttempts: number;
  /** Base backoff interval in milliseconds. Default: 500. */
  baseDelayMs: number;
  /** Cap on any single backoff interval. Default: 30 000 (30s). */
  maxDelayMs: number;
}

export const DEFAULT_RETRY_CONFIG: RetryConfig = {
  maxAttempts: 5,
  baseDelayMs: 500,
  maxDelayMs: 30_000,
};

/**
 * Compute the delay before the next attempt.
 *
 * When `retryAfterMs` is provided (from a 429 Retry-After header) it
 * overrides the exponential calculation but is still capped at maxDelayMs.
 */
export function computeDelay(
  attempt: number, // 0-indexed
  config: RetryConfig,
  retryAfterMs?: number,
): number {
  if (retryAfterMs !== null && retryAfterMs !== undefined && retryAfterMs >= 0) {
    return Math.min(retryAfterMs, config.maxDelayMs);
  }
  const exponential = config.baseDelayMs * Math.pow(2, attempt);
  const capped = Math.min(exponential, config.maxDelayMs);
  // Jitter: uniform in [0.5 * capped, 1.0 * capped]
  return capped * (0.5 + Math.random() * 0.5);
}

/** True for HTTP status codes the SDK should retry on. */
export function isRetryableStatus(status: number): boolean {
  return status === 429 || status >= 500;
}

/** Extract Retry-After delay in milliseconds from a fetch Response. */
export function retryAfterMs(resp: Response): number | undefined {
  const header = resp.headers.get("retry-after");
  if (!header) return undefined;
  // RFC 7231: numeric seconds OR HTTP-date
  const secs = parseFloat(header);
  if (!isNaN(secs)) return Math.ceil(secs * 1000);
  const date = Date.parse(header);
  if (!isNaN(date)) return Math.max(0, date - Date.now());
  return undefined;
}

/** Sleep for `ms` milliseconds (abortable). */
export function sleep(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(new DOMException("Aborted", "AbortError"));
      return;
    }
    const timer = setTimeout(resolve, ms);
    signal?.addEventListener("abort", () => {
      clearTimeout(timer);
      reject(new DOMException("Aborted", "AbortError"));
    });
  });
}
