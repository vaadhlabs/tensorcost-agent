import type { WrapOptions } from "../types.js";
import { wrap } from "../wrap.js";

/** Thin passthrough — wraps the underlying OpenAI/Anthropic client. */
export function wrapLangChain<T>(client: T, options?: WrapOptions): T {
  return wrap(client, options);
}
