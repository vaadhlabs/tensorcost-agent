/**
 * A deliberately runaway agent, for watching TensorCost run caps and stops
 * work against your own sandbox.
 *
 * Every step it sends a bigger prompt than the one before (the context grows
 * by `--growth` each time), so its input tokens, and the spend on its run,
 * climb until something stops it. It makes its calls through the TensorCost
 * proxy with wrap({ maxLayer: "route" }), tagged with a run id and an agent
 * id. When an operator stops the run, or the run hits its cap, the proxy
 * answers 403, the agent prints the refusal and exits 3. A 403 that is not a
 * TensorCost policy refusal (a WAF or load balancer, say) prints
 * `403 UNKNOWN: <message>` and exits 1, so a misconfiguration never looks
 * like a stop.
 *
 * Calls are OpenAI chat completions only: that is the one shape the proxy
 * routes today.
 *
 * Point --provider-url at the sandbox endpoint and nothing is sent to a real
 * model provider: the proxy answers the call itself. The tenant needs the
 * sandbox flag on. See examples/README.md.
 *
 * Inside your own project, import from the published package instead:
 *   import { wrap } from "@tensorcost/sdk";
 */

import { randomUUID } from "node:crypto";
import { fileURLToPath } from "node:url";

import {
  wrap,
  TensorCostError,
  TensorCostRunBudgetExceededError,
  TensorCostPeriodBudgetExceededError,
  TensorCostRunStoppedError,
  TensorCostModelGovernanceDeniedError,
  TensorCostGuardrailHardStopError,
  TensorCostComplianceDeniedError,
  TensorCostComplianceTeamMismatchError,
} from "../src/index.js";

/**
 * Only the OpenAI shape. The proxy routes /v1/chat/completions; the Anthropic
 * SDK posts /v1/messages, which the proxy does not route, so every call
 * would fail with 404 before a cap or a stop could ever apply.
 */
export type RunawayProvider = "openai";

export interface RunawayOptions {
  apiKey: string;
  /** TensorCost backend: token exchange and observation intake. */
  baseUrl: string;
  /** Inference proxy base. Defaults to baseUrl. */
  proxyUrl: string;
  /** The SDK exchanges the API key for a token per tenant. */
  tenantId: string;
  provider: RunawayProvider;
  providerUrl: string;
  model: string;
  workflowId: string;
  agentId: string;
  /** Context multiplier per step. */
  growth: number;
  delayMs: number;
  maxSteps: number;
}

export interface RunawayDeps {
  /** Stands in for globalThis.fetch for the duration of the run. */
  fetch?: typeof fetch;
  sleep: (ms: number) => Promise<void>;
  out: (line: string) => void;
  err: (line: string) => void;
}

export const EXIT_OK = 0;
export const EXIT_ERROR = 1;
export const EXIT_REFUSED = 3;

const DEFAULTS = {
  baseUrl: "https://api.tensorcost.com",
  provider: "openai" as RunawayProvider,
  agentId: "research-agent",
  growth: 1.5,
  delayMs: 2000,
  maxSteps: 200,
};

/** Characters in the first step's prompt body. */
const FIRST_STEP_CHARS = 400;
/** The prompt stops growing here so a long run cannot exhaust memory. */
const MAX_PROMPT_CHARS = 1_000_000;
/** Rough chars-per-token, used only to size a refused call that has no usage. */
const CHARS_PER_TOKEN = 4;

const FILLER =
  "Notes gathered so far: the agent keeps every earlier finding in its context and adds more. ";

const USAGE = `usage: runaway-agent --model <id> [options]

  --api-key <key>        TensorCost API key              (TENSORCOST_API_KEY)
  --base-url <url>       TensorCost backend              (TENSORCOST_BASE_URL)
  --proxy-url <url>      inference proxy, if not the base (TENSORCOST_PROXY_URL)
  --tenant-id <id>       tenant for token exchange       (TENSORCOST_TENANT_ID)
  --provider <name>      openai (the only shape the proxy routes)
  --provider-url <url>   provider endpoint, e.g. the sandbox (TENSORCOST_PROVIDER_URL)
  --model <id>           model to call                   (required)
  --workflow-id <id>     run id                          (default runaway-<uuid>)
  --agent-id <id>        agent label                     (default research-agent)
  --growth <n>           context multiplier per step     (default 1.5)
  --delay-ms <n>         pause between steps             (default 2000)
  --max-steps <n>        stop after this many steps      (default 200)`;

type Env = Record<string, string | undefined>;

class UsageError extends Error {}

function parseFlags(argv: readonly string[]): Map<string, string> {
  const flags = new Map<string, string>();
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i]!;
    if (!arg.startsWith("--")) throw new UsageError(`unexpected argument ${arg}`);
    const eq = arg.indexOf("=");
    if (eq !== -1) {
      flags.set(arg.slice(2, eq), arg.slice(eq + 1));
      continue;
    }
    const next = argv[i + 1];
    if (next === undefined || next.startsWith("--")) {
      throw new UsageError(`${arg} needs a value`);
    }
    flags.set(arg.slice(2), next);
    i += 1;
  }
  return flags;
}

function positive(name: string, raw: string | undefined, fallback: number): number {
  if (raw === undefined) return fallback;
  const n = Number(raw);
  if (!Number.isFinite(n) || n <= 0) throw new UsageError(`${name} must be a positive number`);
  return n;
}

function nonNegative(name: string, raw: string | undefined, fallback: number): number {
  if (raw === undefined) return fallback;
  const n = Number(raw);
  if (!Number.isFinite(n) || n < 0) throw new UsageError(`${name} must be zero or more`);
  return n;
}

/** Flags first, then the environment, then the defaults. */
export function parseOptions(argv: readonly string[], env: Env): RunawayOptions {
  const f = parseFlags(argv);
  const pick = (flag: string, envName: string): string | undefined =>
    f.get(flag) ?? env[envName] ?? undefined;

  const model = f.get("model");
  if (!model) throw new UsageError("--model is required");
  const apiKey = pick("api-key", "TENSORCOST_API_KEY");
  if (!apiKey) throw new UsageError("--api-key (or TENSORCOST_API_KEY) is required");
  const providerUrl = pick("provider-url", "TENSORCOST_PROVIDER_URL");
  if (!providerUrl) throw new UsageError("--provider-url (or TENSORCOST_PROVIDER_URL) is required");
  const tenantId = pick("tenant-id", "TENSORCOST_TENANT_ID");
  if (!tenantId) throw new UsageError("--tenant-id (or TENSORCOST_TENANT_ID) is required");

  const provider = f.get("provider") ?? DEFAULTS.provider;
  if (provider === "anthropic") {
    throw new UsageError(
      "--provider anthropic is not supported: the TensorCost proxy routes only " +
        "OpenAI-style /v1/chat/completions, and the Anthropic shape posts /v1/messages, " +
        "which it would answer with 404. Use --provider openai (the default) with a " +
        "model the sandbox or your OpenAI-compatible endpoint serves.",
    );
  }
  if (provider !== "openai") {
    throw new UsageError("--provider must be openai");
  }

  const baseUrl = (pick("base-url", "TENSORCOST_BASE_URL") ?? DEFAULTS.baseUrl).replace(/\/+$/, "");
  const proxyUrl = (pick("proxy-url", "TENSORCOST_PROXY_URL") ?? baseUrl).replace(/\/+$/, "");

  return {
    apiKey,
    baseUrl,
    proxyUrl,
    tenantId,
    provider,
    providerUrl,
    model,
    workflowId: f.get("workflow-id") ?? `runaway-${randomUUID()}`,
    agentId: f.get("agent-id") ?? DEFAULTS.agentId,
    growth: positive("--growth", f.get("growth"), DEFAULTS.growth),
    delayMs: nonNegative("--delay-ms", f.get("delay-ms"), DEFAULTS.delayMs),
    maxSteps: positive("--max-steps", f.get("max-steps"), DEFAULTS.maxSteps),
  };
}

/**
 * The client wrap() decorates. In route mode every call goes through the
 * proxy; this client's own create() is only what wrap() would fall back to
 * if the proxy were unreachable, and this example never wants that: a run
 * that cannot reach TensorCost should end, not go around it.
 */
function sealedClient(providerUrl: string) {
  const refuse = async (): Promise<never> => {
    throw new Error("direct provider calls are disabled in this example");
  };
  const common = {
    baseURL: providerUrl,
    // wrap() requires an apiKey to forward. The sandbox ignores it; a real
    // provider endpoint would need a real key, which this example never sends.
    apiKey: "not-used",
    defaultHeaders: {} as Record<string, string>,
  };
  return { ...common, chat: { completions: { create: refuse } } };
}

interface Reply {
  usage?: { prompt_tokens?: number; input_tokens?: number };
}

function promptFor(step: number, growth: number): string {
  const chars = Math.min(MAX_PROMPT_CHARS, Math.round(FIRST_STEP_CHARS * growth ** (step - 1)));
  return FILLER.repeat(Math.ceil(chars / FILLER.length)).slice(0, chars);
}

function bodyFor(o: RunawayOptions, step: number): Record<string, unknown> {
  const messages = [
    { role: "user", content: `Step ${step}. Keep researching.\n${promptFor(step, o.growth)}` },
  ];
  return { model: o.model, messages };
}

const withoutPrefix = (m: string): string => m.replace(/^tensorcost: /, "");

/**
 * The code and message of a TensorCost policy refusal, however the SDK
 * surfaced it, or null for anything else. Only the SDK's typed refusals
 * count: they are built from the proxy's own 403 body. A bare 403 from
 * something in front of the proxy is not one of them.
 */
function refusalOf(e: unknown): { code: string; message: string } | null {
  if (e instanceof TensorCostRunStoppedError) {
    return { code: e.code, message: e.message };
  }
  const typed: Array<[new (...a: never[]) => Error, string]> = [
    [TensorCostRunBudgetExceededError, "RUN_BUDGET_EXCEEDED"],
    [TensorCostPeriodBudgetExceededError, "PERIOD_BUDGET_EXCEEDED"],
    [TensorCostModelGovernanceDeniedError, "MODEL_GOVERNANCE_DENIED"],
    [TensorCostGuardrailHardStopError, "GUARDRAIL_HARD_STOP"],
    [TensorCostComplianceDeniedError, "COMPLIANCE_DENIED"],
    [TensorCostComplianceTeamMismatchError, "COMPLIANCE_TEAM_MISMATCH"],
  ];
  for (const [cls, code] of typed) {
    if (e instanceof cls) return { code, message: withoutPrefix(e.message) };
  }
  return null;
}

/** HTTP status of the proxy answer behind an error, including a wrapped one. */
function providerStatusOf(e: unknown): number | null {
  if (!(e instanceof TensorCostError)) return null;
  if (e.status !== null) return e.status;
  return e.rootCause ? providerStatusOf(e.rootCause) : null;
}

function pad(n: number, width: number): string {
  return String(n).padStart(width, " ");
}

/**
 * Runs the loop and returns the process exit code: 0 after max-steps,
 * 3 when TensorCost refused a call (a stop or a cap), 1 for anything else,
 * including a 403 that did not come from a TensorCost policy.
 */
export async function runRunaway(o: RunawayOptions, deps: RunawayDeps): Promise<number> {
  const realFetch = globalThis.fetch;
  if (deps.fetch) globalThis.fetch = deps.fetch;
  try {
    const client = wrap(sealedClient(o.providerUrl), {
      apiKey: o.apiKey,
      baseUrl: o.baseUrl,
      proxyUrl: o.proxyUrl,
      tenantId: o.tenantId,
      maxLayer: "route",
      agentId: o.agentId,
      workflowId: o.workflowId,
      failOpenEnabled: false,
      retry: { maxAttempts: 1 },
    });

    const api = client as {
      chat: { completions: { create: (b: unknown) => Promise<Reply> } };
    };
    const call = (body: Record<string, unknown>): Promise<Reply> => api.chat.completions.create(body);

    for (let step = 1; step <= o.maxSteps; step += 1) {
      const body = bodyFor(o, step);
      try {
        const reply = await call(body);
        const used = reply.usage?.prompt_tokens ?? reply.usage?.input_tokens;
        const tokens = used ?? Math.ceil(JSON.stringify(body).length / CHARS_PER_TOKEN);
        deps.out(`step ${pad(step, 3)}  input_tokens=${used === undefined ? "~" : ""}${tokens}  status=200`);
      } catch (e) {
        const estimate = Math.ceil(JSON.stringify(body).length / CHARS_PER_TOKEN);
        const refusal = refusalOf(e);
        if (refusal) {
          deps.out(`step ${pad(step, 3)}  input_tokens=~${estimate}  status=403`);
          deps.out(`403 ${refusal.code}: ${refusal.message}`);
          return EXIT_REFUSED;
        }
        // The SDK can wrap the proxy's answer, so read the status through it.
        const status = providerStatusOf(e) ?? "error";
        deps.out(`step ${pad(step, 3)}  input_tokens=~${estimate}  status=${status}`);
        if (providerStatusOf(e) === 403) {
          // A 403 the SDK could not read as a TensorCost refusal: a WAF, a
          // load balancer, a bad key. Say so plainly, and do not exit 3.
          const message = e instanceof Error ? withoutPrefix(e.message) : String(e);
          deps.out(`403 UNKNOWN: ${message}`);
          deps.err(
            "The 403 did not come from a TensorCost stop or cap. Check the API key, " +
              "the proxy URL and anything in front of the proxy (a WAF or load balancer).",
          );
          return EXIT_ERROR;
        }
        deps.err(e instanceof Error ? e.message : String(e));
        if (providerStatusOf(e) === 400) {
          // The SDK keeps the status of a 400 but not its body. With the
          // sandbox URL the usual cause is the tenant flag being off.
          deps.err(
            "The proxy rejected the request with 400. If --provider-url is the sandbox, " +
              "check that inference-proxy.sandbox.enabled is on for your tenant.",
          );
        }
        return EXIT_ERROR;
      }
      if (step < o.maxSteps) await deps.sleep(o.delayMs);
    }
    return EXIT_OK;
  } finally {
    globalThis.fetch = realFetch;
  }
}

/** Parse argv and env, then run. Usage mistakes exit 1 with the usage text. */
export async function main(
  argv: readonly string[],
  env: Env,
  deps: RunawayDeps,
): Promise<number> {
  let options: RunawayOptions;
  try {
    options = parseOptions(argv, env);
  } catch (e) {
    if (!(e instanceof UsageError)) throw e;
    deps.err(`${e.message}\n\n${USAGE}`);
    return EXIT_ERROR;
  }
  return runRunaway(options, deps);
}

if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) {
  void main(process.argv.slice(2), process.env, {
    sleep: (ms) => new Promise((resolve) => setTimeout(resolve, ms)),
    out: (line) => console.log(line),
    err: (line) => console.error(line),
  }).then((code) => {
    process.exitCode = code;
  });
}
