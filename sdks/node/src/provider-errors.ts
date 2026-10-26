/**
 * Rethrowing a relayed provider error as the provider SDK's own type.
 *
 * When the proxy relays an OpenAI 400, a caller written against the openai
 * package expects `openai.BadRequestError`, not a TensorCost class. Both
 * openai@4 and @anthropic-ai/sdk build their status errors through the
 * client's `makeStatusError(status, body, message, headers)` (which calls
 * `APIError.generate`), so the SDK asks the client to build it, exactly as
 * the client would have for a direct call. Clients without that method, or
 * whose factory rejects both header shapes below, get the
 * TensorCostProviderError as is.
 */

import { TensorCostProviderError } from "./errors.js";

type MakeStatusError = (
  status: number,
  body: unknown,
  message: string | undefined,
  headers: unknown,
) => unknown;

function statusErrorFactory(client: unknown): MakeStatusError | null {
  if (!client || typeof client !== "object") return null;
  const c = client as Record<string, unknown>;
  if (typeof c["makeStatusError"] === "function") {
    return (c["makeStatusError"] as MakeStatusError).bind(client);
  }
  // Older shapes expose the error class on the constructor instead.
  const ctor = (client as { constructor?: Record<string, unknown> }).constructor;
  const apiError = ctor?.["APIError"] as { generate?: MakeStatusError } | undefined;
  if (apiError && typeof apiError.generate === "function") {
    return apiError.generate.bind(apiError);
  }
  return null;
}

/**
 * The two header shapes the factories expect. openai@4 reads
 * `headers["x-request-id"]` off a plain object; openai v5/v6 and current
 * @anthropic-ai/sdk call `headers.get(...)` on a `Headers` instance and throw
 * on a plain object. The plain object goes first (openai@4 would take a
 * `Headers` too but lose the request id), then a `Headers` instance.
 */
function headerShapes(headers: Record<string, string>): unknown[] {
  const shapes: unknown[] = [headers];
  if (typeof Headers === "function") shapes.push(new Headers(headers));
  return shapes;
}

/** The provider SDK's error for `err`, or `err` itself when one can't be built. */
export function providerNativeError(client: unknown, err: unknown): unknown {
  if (!(err instanceof TensorCostProviderError) || err.status === null) return err;
  const make = statusErrorFactory(client);
  if (!make) return err;
  const body = err.body !== null && typeof err.body === "object" ? err.body : undefined;
  for (const headers of headerShapes(err.headers)) {
    let native: unknown;
    try {
      native = make(err.status, body, undefined, headers);
    } catch {
      continue;
    }
    if (native instanceof Error) {
      Object.defineProperty(native, "tensorcost", {
        value: err,
        enumerable: false,
        configurable: true,
      });
      return native;
    }
  }
  return err;
}
