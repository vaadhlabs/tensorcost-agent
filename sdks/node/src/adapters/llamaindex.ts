import type { WrapOptions } from "../types.js";
import { wrap } from "../wrap.js";

/** Thin passthrough — wraps the underlying LLM client instance. */
export function wrapLlamaIndex<T>(client: T, options?: WrapOptions): T {
  return wrap(client, options);
}
