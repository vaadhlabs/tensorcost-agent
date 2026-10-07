/**
 * Client-side compliance — team narrow-check + in-process DLP before provider calls.
 */

import { resolveEffectiveTeam } from "./compliance-team.js";
import {
  applyDlpRedactionToOpenAiArgs,
  scanTextForCompliance,
  type DlpScanResult,
} from "./dlp.js";
import {
  TensorCostComplianceDeniedError,
  TensorCostComplianceTeamMismatchError,
} from "./errors.js";
import type { CompiledComplianceSnapshot } from "./layer.js";

export interface ComplianceStamp {
  compliance_frameworks?: string[];
  dlp_action?: string;
}

export function assertTeamScope(args: {
  tokenTeamId?: string | null;
  clientTeamId?: string | null;
}): void {
  const resolved = resolveEffectiveTeam(args);
  if (resolved.error === "compliance_team_mismatch") {
    throw new TensorCostComplianceTeamMismatchError(
      "tensorcost: SDK team scope does not match token binding",
    );
  }
}

export function enforceInProcessDlp(
  text: string,
  snapshot: CompiledComplianceSnapshot | null | undefined,
  args?: unknown[],
): DlpScanResult {
  const result = scanTextForCompliance(text, snapshot);
  if (
    result.outcome === "refused" &&
    snapshot?.mode === "enforce"
  ) {
    throw new TensorCostComplianceDeniedError(
      "tensorcost: in-process DLP refused this call",
      "dlp_refused",
    );
  }
  if (
    result.outcome === "redacted" &&
    args &&
    result.matchedDetectors.length > 0
  ) {
    applyDlpRedactionToOpenAiArgs(args, result.matchedDetectors);
  }
  return result;
}

export function complianceStampFrom(
  snapshot: CompiledComplianceSnapshot | null | undefined,
  dlp: DlpScanResult,
): ComplianceStamp {
  const stamp: ComplianceStamp = {};
  if (snapshot?.audit?.stamp && snapshot.frameworks.length > 0) {
    stamp.compliance_frameworks = [...snapshot.frameworks];
  }
  if (dlp.outcome !== "none") {
    stamp.dlp_action = dlp.outcome;
  } else if (snapshot?.dlp?.enabled && dlp.matchedDetectors.length > 0) {
    stamp.dlp_action = "none";
  }
  return stamp;
}
