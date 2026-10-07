/**
 * Parse x-tc-decision response header for observation metadata.
 */

export interface ParsedDecision {
  layer: string;
  action: string;
  reason?: string;
}

export function parseDecisionHeader(
  header: string | null | undefined,
): ParsedDecision | null {
  if (!header || header.trim() === "") return null;
  let layer: string | undefined;
  let action: string | undefined;
  let reason: string | undefined;
  for (const segment of header.split(";")) {
    const trimmed = segment.trim();
    const eq = trimmed.indexOf("=");
    if (eq <= 0) continue;
    const key = trimmed.slice(0, eq).trim();
    const value = trimmed.slice(eq + 1).trim();
    if (key === "layer") layer = value;
    else if (key === "action") action = value;
    else if (key === "reason") reason = value;
  }
  if (!layer || !action) return null;
  return { layer, action, reason };
}

export function decisionMetadata(
  header: string | null | undefined,
): Record<string, string> | undefined {
  const parsed = parseDecisionHeader(header);
  if (!parsed) return undefined;
  const meta: Record<string, string> = {
    decision_action: parsed.action,
  };
  if (parsed.reason) {
    meta.decision_reason = parsed.reason;
  }
  return meta;
}
