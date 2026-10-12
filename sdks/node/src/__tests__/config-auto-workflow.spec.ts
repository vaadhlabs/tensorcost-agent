import { describe, expect, it, afterEach } from "vitest";
import { resolveConfig } from "../config.js";

describe("resolveConfig auto workflow_id", () => {
  const savedEnv = { ...process.env };

  afterEach(() => {
    for (const key of Object.keys(process.env)) {
      if (!(key in savedEnv)) delete process.env[key];
    }
    for (const [key, value] of Object.entries(savedEnv)) {
      if (value === undefined) delete process.env[key];
      else process.env[key] = value;
    }
  });

  it("does not mint workflow_id unless auto workflow is enabled", () => {
    process.env.TENSORCOST_API_KEY = "tc-key";
    delete process.env.TENSORCOST_WORKFLOW_ID;
    delete process.env.TENSORCOST_AUTO_WORKFLOW_ID;

    const cfg = resolveConfig({ agentId: "support-bot" });
    expect(cfg.workflowId).toBeNull();
  });

  it("mints when autoWorkflowId is true", () => {
    process.env.TENSORCOST_API_KEY = "tc-key";
    delete process.env.TENSORCOST_WORKFLOW_ID;

    const cfg = resolveConfig({
      agentId: "support-bot",
      autoWorkflowId: true,
    });
    expect(cfg.workflowId).toMatch(
      /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i,
    );
  });

  it("mints when TENSORCOST_AUTO_WORKFLOW_ID=1", () => {
    process.env.TENSORCOST_API_KEY = "tc-key";
    process.env.TENSORCOST_AUTO_WORKFLOW_ID = "1";

    const cfg = resolveConfig({ agentId: "support-bot" });
    expect(cfg.workflowId).not.toBeNull();
  });
});
