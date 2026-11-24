/**
 * The proxy's caps on observation prose, applied before the POST.
 *
 * `error_message` is built as `${err.name}: ${err.message}`, and a provider's
 * message (an Azure content-filter refusal, an OpenAI rate-limit notice) runs
 * well past 200 characters. Proxies before the fix answered such an
 * observation 400, and a 4xx is final here, so the call's spend row was
 * lost. Newer proxies cut the message themselves; cutting it here as well
 * means an older proxy keeps the row too, and nothing is sent that the
 * proxy would only throw away.
 *
 * Keys (model, correlation_id, agent_id, customer, environment ...) are left
 * as the caller set them: cut short, one would name something else.
 */
export const ERROR_MESSAGE_MAX = 200;

const ELLIPSIS = "…";

/**
 * `s` cut to at most `max` UTF-16 code units (the unit the proxy measures),
 * ending in an ellipsis when cut. Never splits a surrogate pair.
 */
export function truncateFreeText(s: string, max: number): string {
  if (s.length <= max) return s;
  let cut = max - ELLIPSIS.length;
  const last = s.charCodeAt(cut - 1);
  if (last >= 0xd800 && last <= 0xdbff) cut -= 1;
  return s.slice(0, cut) + ELLIPSIS;
}

/** The observation with its error_message within the proxy's cap; the same object when it already is. */
export function boundObservation<T extends { error_message?: string | null }>(obs: T): T {
  const msg = obs.error_message;
  if (typeof msg !== "string" || msg.length <= ERROR_MESSAGE_MAX) return obs;
  return { ...obs, error_message: truncateFreeText(msg, ERROR_MESSAGE_MAX) };
}
