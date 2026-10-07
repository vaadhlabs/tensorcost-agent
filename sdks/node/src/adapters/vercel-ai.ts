import type { WrapOptions } from "../types.js";
import { wrap } from "../wrap.js";

/** Thin passthrough — wraps the provider client passed to Vercel AI SDK. */
export function wrapVercelAi<T>(client: T, options?: WrapOptions): T {
  return wrap(client, options);
}
