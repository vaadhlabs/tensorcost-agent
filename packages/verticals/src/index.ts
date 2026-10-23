export type VerticalName =
  | "ai"
  | "aiMl"
  | "fleet"
  | "controls"
  | "reporting"
  | "billing"
  | "devTools";

const ENV_BY_VERTICAL: Record<VerticalName, string> = {
  ai: "TC_VERTICAL_AI",
  aiMl: "TC_VERTICAL_AI_ML",
  fleet: "TC_VERTICAL_FLEET",
  controls: "TC_VERTICAL_CONTROLS",
  reporting: "TC_VERTICAL_REPORTING",
  billing: "TC_VERTICAL_BILLING",
  devTools: "TC_VERTICAL_DEVTOOLS",
};

/** When env is unset (local docker-compose), all verticals are enabled. */
export function isVerticalEnabled(name: VerticalName, env: NodeJS.ProcessEnv = process.env): boolean {
  const key = ENV_BY_VERTICAL[name];
  const raw = env[key];
  if (raw === undefined || raw === "") return true;
  return raw === "true" || raw === "1";
}

export function installedVerticals(env: NodeJS.ProcessEnv = process.env): Record<VerticalName, boolean> {
  return {
    ai: isVerticalEnabled("ai", env),
    aiMl: isVerticalEnabled("aiMl", env),
    fleet: isVerticalEnabled("fleet", env),
    controls: isVerticalEnabled("controls", env),
    reporting: isVerticalEnabled("reporting", env),
    billing: isVerticalEnabled("billing", env),
    devTools: isVerticalEnabled("devTools", env),
  };
}

/** Map backend service registry name → vertical gate (undefined = core, always on). */
export function serviceVerticalGate(serviceName: string): VerticalName | undefined {
  switch (serviceName) {
    case "ai-service":
    case "inference-proxy-service":
      return "ai";
    case "ml-service":
      return "aiMl";
    case "gpu-service":
      return "fleet";
    case "alert-service":
    case "enforcement-service":
      return "controls";
    case "export-service":
    case "schedule-service":
      return "reporting";
    case "billing-service":
      return "billing";
    case "devtools-service":
    case "smoke-service":
      return "devTools";
    default:
      return undefined;
  }
}

export function isServiceDeployed(serviceName: string, env: NodeJS.ProcessEnv = process.env): boolean {
  const gate = serviceVerticalGate(serviceName);
  if (!gate) return true;
  if (gate === "aiMl" && !isVerticalEnabled("ai", env)) return false;
  return isVerticalEnabled(gate, env);
}
