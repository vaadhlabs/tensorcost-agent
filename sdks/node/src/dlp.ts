/**
 * In-process DLP — prompt text never leaves the customer process.
 */

import type { CompiledComplianceSnapshot } from "./layer.js";

export type { CompiledComplianceSnapshot };

export type DlpOutcome = "none" | "redacted" | "refused";

const DETECTOR_PATTERNS: Record<string, RegExp> = {
  phi_ssn: /\b\d{3}-\d{2}-\d{4}\b/,
  phi_mrn: /\bMRN[:\s#-]*\d{6,12}\b/i,
  phi_dob: /\b(DOB|date of birth)[:\s]*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b/i,
  pci_pan: /\b(?:\d[ -]*?){13,19}\b/,
  pci_cvv: /\b(CVV|CVC)[:\s#-]*\d{3,4}\b/i,
};

export interface DlpScanResult {
  outcome: DlpOutcome;
  matchedDetectors: string[];
}

export function extractAnthropicMessagesText(firstArg: unknown): string {
  if (!firstArg || typeof firstArg !== "object") return "";
  const messages = (firstArg as { messages?: unknown }).messages;
  if (!Array.isArray(messages)) return "";
  const parts: string[] = [];
  for (const msg of messages) {
    if (!msg || typeof msg !== "object") continue;
    const content = (msg as { content?: unknown }).content;
    if (typeof content === "string") parts.push(content);
    else if (Array.isArray(content)) {
      for (const block of content) {
        if (!block || typeof block !== "object") continue;
        const text = (block as { text?: unknown }).text;
        if (typeof text === "string") parts.push(text);
      }
    }
  }
  return parts.join("\n");
}

export function extractOpenAiMessagesText(args: unknown[]): string {
  const first = args[0];
  if (!first || typeof first !== "object") return "";
  const messages = (first as { messages?: unknown }).messages;
  if (!Array.isArray(messages)) return "";
  const parts: string[] = [];
  for (const msg of messages) {
    if (!msg || typeof msg !== "object") continue;
    const content = (msg as { content?: unknown }).content;
    if (typeof content === "string") parts.push(content);
    else if (Array.isArray(content)) {
      for (const block of content) {
        if (block && typeof block === "object" && typeof (block as { text?: unknown }).text === "string") {
          parts.push((block as { text: string }).text);
        }
      }
    }
  }
  return parts.join("\n");
}

export function redactTextForCompliance(text: string, detectors: readonly string[]): string {
  let out = text;
  for (const detector of detectors) {
    const pattern = DETECTOR_PATTERNS[detector];
    if (pattern) {
      out = out.replace(pattern, "[REDACTED]");
    }
  }
  return out;
}

export function applyDlpRedactionToOpenAiArgs(
  args: unknown[],
  matchedDetectors: readonly string[],
): void {
  if (args.length === 0 || !args[0] || typeof args[0] !== "object") return;
  const req = args[0] as { messages?: unknown };
  if (!Array.isArray(req.messages)) return;
  for (const msg of req.messages) {
    if (!msg || typeof msg !== "object") continue;
    const content = (msg as { content?: unknown }).content;
    if (typeof content === "string") {
      (msg as { content: string }).content = redactTextForCompliance(
        content,
        matchedDetectors,
      );
    } else if (Array.isArray(content)) {
      for (const block of content) {
        if (
          block &&
          typeof block === "object" &&
          typeof (block as { text?: unknown }).text === "string"
        ) {
          (block as { text: string }).text = redactTextForCompliance(
            (block as { text: string }).text,
            matchedDetectors,
          );
        }
      }
    }
  }
}

export function scanTextForCompliance(
  text: string,
  snapshot: CompiledComplianceSnapshot | null | undefined,
): DlpScanResult {
  if (!snapshot?.dlp?.enabled || snapshot.dlp.detectors.length === 0) {
    return { outcome: "none", matchedDetectors: [] };
  }

  const matched: string[] = [];
  for (const detector of snapshot.dlp.detectors) {
    const pattern = DETECTOR_PATTERNS[detector];
    if (pattern && pattern.test(text)) {
      matched.push(detector);
    }
  }

  if (matched.length === 0) {
    return { outcome: "none", matchedDetectors: [] };
  }

  if (snapshot.mode === "enforce" && snapshot.dlp.action === "refuse") {
    return { outcome: "refused", matchedDetectors: matched };
  }

  if (snapshot.dlp.action === "redact") {
    return { outcome: "redacted", matchedDetectors: matched };
  }

  return { outcome: "none", matchedDetectors: matched };
}
