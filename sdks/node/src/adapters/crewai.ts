import type { WrapOptions } from "../types.js";
import { wrap } from "../wrap.js";

/** Thin passthrough — wraps the LLM used by CrewAI agents. */
export function wrapCrewAI<T>(client: T, options?: WrapOptions): T {
  return wrap(client, options);
}
