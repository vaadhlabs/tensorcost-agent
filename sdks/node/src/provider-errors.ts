/**
 * Rethrowing a relayed provider error as the provider SDK's own type.
 *
 * When the proxy relays an OpenAI 400, a caller written against the openai
 * package expects `openai.BadRequestError`, not a TensorCost class. Both
 * openai@4 and @anthropic-ai/sdk build their status errors through the
 * client's `makeStatusError(status, body, message, headers)` (which calls
 * `APIError.generate`), so the SDK asks the client to build it, exactly as
 * the client would have for a direct call. Clients without that method get
 * the TensorCostProviderError as is.
 */

import { TensorCostProviderError } from "./errors.js";

type MakeStatusError = (
  status: number,
  body: unknown,
  message: string | undefined,
  headers: Record<string, string>,
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

/** The provider SDK's error for `err`, or `err` itself when one can't be built. */
export function providerNativeError(client: unknown, err: unknown): unknown {
  if (!(err instanceof TensorCostProviderError) || err.status === null) return err;
  const make = statusErrorFactory(client);
  if (!make) return err;
  try {
    const native = make(
      err.status,
      err.body !== null && typeof err.body === "object" ? err.body : undefined,
      undefined,
      err.headers,
    );
    if (native instanceof Error) {
      Object.defineProperty(native, "tensorcost", {
        value: err,
        enumerable: false,
        configurable: true,
      });
      return native;
    }
  } catch {
    /* fall through to the TensorCost error */
  }
  return err;
}
