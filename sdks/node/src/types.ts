/**
 * Internal type defs for the TensorCost Node SDK.
 *
 * Kept intentionally narrow — the wrap() entry-point is generic over
 * the customer's client type, so most provider shapes are duck-typed
 * at runtime rather than expressed in TypeScript.
 */

import type { LifecycleEventCallback } from "./telemetry.js";
import type { RetryConfig } from "./retry.js";
import type { ControlLayer, DataGrant } from "./layer.js";

export type { ControlLayer, DataGrant } from "./layer.js";

export type Provider = "openai" | "anthropic" | "bedrock" | "vertex";

export type Operation =
  | "chat.completions"
  | "completions"
  | "messages"
  | "bedrock.invoke_model"
  | "bedrock.converse"
  | "bedrock.invoke_model_stream"
  | "bedrock.converse_stream"
  | "vertex.generate_content"
  | "vertex.generate_content_stream";

export type ObservationStatus = "success" | "error";

export type ObservationModality =
  | "text"
  | "image"
  | "audio"
  | "video"
  | "mixed";

/**
 * Wire shape posted to `POST /api/proxy/observation`. Mirrors the
 * Python SDK's `Observation` model byte-for-byte so the two SDKs
 * are interchangeable from the backend's point of view.
 */
export interface Observation {
  sdk_version: string;
  provider: Provider;
  model: string;
  operation: Operation;
  /** Defaults to "text" when omitted on the wire. */
  modality?: ObservationModality;
  request_at: string;
  response_at: string;
  input_tokens: number | null;
  output_tokens: number | null;
  cost_usd_cents: number | null;
  status: ObservationStatus;
  error_message: string | null;
  correlation_id: string;
  /**
   * Agent trace + sub_agent parent for proxied calls. When both are set,
   * ai-service inserts the model_call span under the agent trace so run
   * rollup can subtract proxied child spend from sub_agent totals.
   */
  trace_id?: string;
  parent_span_id?: string;
  tenant_id?: string | null;
  /**
   * Phase A4 batch 3 — per-environment data scoping. Optional. Stamped
   * onto every observation envelope when the SDK was constructed with
   * `wrap(client, { environment: 'staging' })`. Forwarded verbatim by
   * the inference-proxy ObservationService onto the Kafka envelope;
   * ai-service falls back to 'production' on omit.
   */
  environment?: string;
  /**
   * Phase A4 batch 4 — provider-connection identifier. Optional.
   * Stamped onto every observation envelope when the SDK was
   * constructed with `wrap(client, { connectionId: '...' })`.
   * Forwarded verbatim by the inference-proxy onto the Kafka
   * envelope; ai-service uses it as the second tier of its
   * env-resolution chain (see ProxyObservationConsumer) when the SDK
   * did not set `environment`.
   */
  connection_id?: string;
  /**
   * SDK metadata bag. Merged from wrap-time defaults and per-call
   * withMeta() overrides. The backend stores this verbatim in
   * trace_span.metadata; two well-known keys are indexed for
   * group-by queries:
   *   - deployment            — git SHA, CI run id, or any operator marker
   *   - prompt_template_id    — per-call-site template name/version
   */
  metadata?: Record<string, unknown>;
  /**
   * Per-agent-run cost attribution — agent identity. Optional. Stamped
   * onto the observation envelope when the SDK was constructed with
   * `wrap(client, { agentId: '...' })`, or overridden per-call via
   * `.withMeta({ agentId: '...' })`. Forwarded verbatim by
   * inference-proxy-service's ObservationService onto the Kafka/event-bus
   * envelope; ai-service writes it straight into
   * `ai.ai_spend_events.agent_id` (migration 011 — the column has existed
   * since April 2026 as `agent_id text -- populated by tag propagation
   * (post-MVP)`; this is that propagation).
   *
   * A stable identifier for the *kind* of agent making calls (e.g.
   * "code-review-agent", "support-triage-bot") — set once per wrapped
   * client is the common case, hence the wrap-time default.
   *
   * Unlike `environment` / `connection_id`, this is NOT stored in the
   * `metadata` JSONB bag — it is a dedicated indexed column
   * (`spend_events_tenant_agent_time` partial index), so it is threaded
   * through as its own top-level envelope field the same way
   * `environment` / `connection_id` are.
   */
  agent_id?: string;
  /**
   * Per-agent-run cost attribution — run identity. Optional. Same
   * plumbing as `agent_id` (see above), landing in
   * `ai.ai_spend_events.workflow_id`.
   *
   * A unique identifier for ONE execution of an agent (e.g. a session id,
   * a job id) — this almost always varies call-to-call within a single
   * wrapped client's lifetime, so the primary way to set it is per-call
   * via `.withMeta({ workflowId: '...' })` rather than at `wrap()` time.
   * A wrap-time default is still supported for the case where one
   * process = one run (e.g. a short-lived batch job).
   *
   * "Cost of one agent run" is defined as
   * `SUM(total_cost_cents) WHERE tenant_id = ? AND workflow_id = ? AND
   * superseded_at IS NULL` — see ai-service's `agent-runs` module.
   */
  workflow_id?: string;
  /**
   * A-04 chargeback — end-customer cohort. Optional. Stamped on every
   * observation when configured via `wrap({ customer })` or per-call
   * `.withMeta({ customer })`. Lands in `ai.ai_spend_events.customer`.
   */
  customer?: string;
  /**
   * A-04 chargeback — product feature. Same plumbing as `customer`,
   * landing in `ai.ai_spend_events.feature`.
   */
  feature?: string;
  /** Compliance frameworks applied on this call (audit stamp). */
  compliance_frameworks?: string[];
  /** In-process DLP outcome: none | redacted | refused. */
  dlp_action?: string;
  /**
   * Synchronous SDK wrap overhead in milliseconds — customer-process work
   * only (middleware, token extraction, observation enqueue). Excludes
   * provider RTT and the async HTTP delivery of the observation POST.
   */
  sdk_added_ms?: number;
}

export interface WrapOptions {
  /** Long-lived TensorCost API key. Falls back to `TENSORCOST_API_KEY`. */
  apiKey?: string;
  /** Backend base URL. Falls back to `TENSORCOST_BASE_URL`, then default. */
  baseUrl?: string;
  /** Optional explicit tenant id, useful for multi-tenant test harnesses. */
  tenantId?: string;
  /**
   * Phase A4 batch 3 — environment tag. When present, the SDK reads
   * the value once at `wrap()` time and stamps every outbound
   * observation envelope with it. Falls back to
   * `TENSORCOST_ENVIRONMENT`. When omitted entirely the field is
   * absent from the envelope; the backend then falls back to the
   * column default ('production'). Free-text per
   * project_environment_scoping.md (1..64 chars).
   */
  environment?: string;
  /**
   * Phase A4 batch 4 — provider-connection identifier. When present,
   * the SDK stamps it on every outbound observation envelope. Falls
   * back to `TENSORCOST_CONNECTION_ID`. When the SDK is wrapping a
   * client that talks to ONE provider connection, set this so the
   * backend can resolve the connection's `environment` even if the
   * SDK was not given an explicit `environment`.
   *
   * Precedence (most-specific wins):
   *   1. SDK envelope `environment` (when set on the SDK or env var).
   *   2. SDK envelope `connectionId` → backend looks up
   *      `integration.cloud_account.environment` (60s LRU cache).
   *   3. Backend column default `'production'`.
   *
   * Customers using ONE SDK instance across multiple connections
   * should leave this unset and rely on `environment` per-init, OR
   * construct a separate `wrap(...)` per connection.
   */
  connectionId?: string;
  /**
   * Optional team scope for compliance overlays. Stamped as `x-tc-team-id`
   * on proxy/admit calls and as `teamId` query on sdk-layer fetch.
   * Falls back to `TENSORCOST_TEAM_ID`. Must match token-bound team when set.
   */
  teamId?: string;
  /**
   * When true (the default), any internal observation-pipeline error is
   * caught and logged via console.warn — the customer's call ALWAYS
   * returns normally. Set to false only in SDK tests / CI smoke runs.
   */
  failOpen?: boolean;
  /**
   * Control-layer ceiling for this wrapped client. A layer is a ceiling,
   * not a setting — effective layer is min(code.maxLayer, console published).
   *
   *   off     — no SDK wiring; client returned unchanged
   *   observe — telemetry only (default when maxLayer unset)
   *   govern  — metadata-only admit before direct provider call
   *   steer   — proxy in path; proxy may steer without full route
   *   route   — full applied-mode routing through inference-proxy
   *
   * Requires `proxyUrl` when maxLayer is steer or route.
   */
  maxLayer?: ControlLayer;
  /**
   * Code-side data-grant ceiling. Effective grants =
   * intersect(maxGrants, console.data_grants). Omit to allow all console grants.
   */
  maxGrants?: readonly DataGrant[];
  /**
   * @deprecated Use `maxLayer: 'route'` instead. When true and maxLayer is
   * unset, resolves to route and emits a console warning.
   */
  appliedMode?: boolean;
  /**
   * TensorCost inference-proxy base URL used in applied mode.
   * Falls back to `TENSORCOST_PROXY_URL`. When `appliedMode=true` and
   * neither is set, `wrap()` throws `TensorCostConfigError`.
   *
   * Example: `https://proxy.tensorcost.com`
   */
  proxyUrl?: string;

  // -------------------------------------------------------------------------
  // Hardening options (applied-mode only; all optional)

  /**
   * Retry configuration. Applies to applied-mode proxy requests only.
   * Observe-only telemetry posts do not retry (they are fire-and-forget).
   *
   * Defaults: maxAttempts=5, baseDelayMs=500, maxDelayMs=30 000.
   * Set `maxAttempts: 1` to disable retries entirely.
   */
  retry?: Partial<RetryConfig>;

  /**
   * Timeout for non-streaming proxy requests, in milliseconds.
   * Default: 60 000 (60s). Surfaces as `TensorCostTimeoutError` (kind: "total").
   */
  timeoutMs?: number;

  /**
   * Max wait for proxy response headers before fail-open to direct provider.
   * Default: 2 000 (2s). Surfaces as `TensorCostTimeoutError` (kind: "headers").
   */
  headersTimeoutMs?: number;

  /**
   * Idle timeout for streaming proxy requests (no bytes received for N ms).
   * Default: 30 000 (30s). Surfaces as `TensorCostTimeoutError` (kind: "idle").
   */
  idleTimeoutMs?: number;

  /**
   * Optional lifecycle event callback. Fires on `before_request`,
   * `after_response`, `on_retry`, `on_error`, and `on_fallback`.
   * The payload contains only request metadata — never prompt / completion
   * content or credentials. Errors in the callback are swallowed.
   */
  onLifecycleEvent?: LifecycleEventCallback;

  /**
   * Fail-open circuit breaker for the proxy. When the proxy returns 5xx
   * for `openThreshold` consecutive calls (default 3), the SDK routes
   * subsequent calls directly to the upstream provider, bypassing the
   * proxy. The circuit closes again after `closeThreshold` consecutive
   * successful probe calls (default 5).
   *
   * Default: true (enabled). Set to false when running in-VPC with a
   * sidecar proxy where direct-provider access is not available.
   *
   * When false, proxy 5xx errors surface as `TensorCostProxyError` with
   * no automatic fallback.
   *
   * The fallback requires that the customer's provider SDK client has its
   * original credentials intact (pre-wrap). If the client credentials are
   * not available, the SDK throws `TensorCostProxyError` instead.
   */
  failOpenEnabled?: boolean;

  /**
   * Provider API key used for the fail-open direct-provider fallback.
   * Only relevant when `failOpenEnabled=true` and `appliedMode=true`.
   *
   * In most cases the SDK reads the provider key directly from the wrapped
   * client (client.apiKey), so this is not needed. Supply it explicitly if
   * the client exposes the key under a non-standard property name, or if
   * you are using the SDK in a context where the client's key is unavailable.
   */
  providerApiKey?: string;

  // -------------------------------------------------------------------------
  // SDK metadata — stamped into every observation's metadata JSONB bag.
  // The backend indexes two well-known keys for group-by queries.

  /**
   * Deployment identifier stamped on every observation from this wrapped
   * client. Typically a git SHA, CI run id, or a human-readable release
   * label like `"2026-05-28-hotfix"`.
   *
   * Stored as `metadata.deployment` in the trace_span row. The dashboard's
   * "Group by deployment" view buckets spans by this value so Priya can
   * answer "did my latest deploy change cost?".
   *
   * Falls back to `TENSORCOST_DEPLOYMENT` env var. Leave unset if you don't
   * use deployment-level cost attribution.
   */
  deployment?: string;

  /**
   * Default prompt-template identifier for every call made through this
   * client. Stored as `metadata.prompt_template_id` on each observation.
   *
   * Per-call overrides are available via `.withMeta({ promptTemplateId })`.
   * The per-call value wins over this wrap-time default.
   *
   * Falls back to `TENSORCOST_PROMPT_TEMPLATE_ID` env var.
   */
  promptTemplateId?: string;

  // -------------------------------------------------------------------------
  // Per-agent-run cost attribution (see Observation.agent_id / .workflow_id
  // in types.ts for the full write-up).

  /**
   * Stable identifier for the kind of agent making calls through this
   * client (e.g. "code-review-agent"). Stamped on every observation as
   * the top-level `agent_id` field — NOT the metadata bag — and lands in
   * `ai.ai_spend_events.agent_id`.
   *
   * Per-call overrides are available via `.withMeta({ agentId })` on the
   * OpenAI and Anthropic wrappers. Bedrock has no per-call hook (Smithy
   * middleware is registered once at wrap() time), so for Bedrock this
   * wrap-time value is the only way to set `agent_id`.
   *
   * Falls back to `TENSORCOST_AGENT_ID` env var.
   */
  agentId?: string;
  /**
   * Identifier for ONE execution of an agent (a run/session id). Stamped
   * on every observation as the top-level `workflow_id` field and lands
   * in `ai.ai_spend_events.workflow_id`.
   *
   * Because a run id almost always changes between calls, the primary
   * way to set this is per-call via `.withMeta({ workflowId })`. This
   * wrap-time default exists for the one-process-per-run case (batch
   * jobs, Bedrock clients with no per-call hook).
   *
   * Falls back to `TENSORCOST_WORKFLOW_ID` env var.
   */
  workflowId?: string;
  /**
   * Routing-policy application scope. Sent as `x-tc-application` on proxy
   * requests. Falls back to `TC_APPLICATION` / `TENSORCOST_APPLICATION`.
   */
  application?: string;
  /**
   * Routing-policy tags. Sent as comma-separated `x-tc-tags` on proxy
   * requests. Falls back to `TC_TAGS` / `TENSORCOST_TAGS` (comma-separated).
   */
  tags?: string | string[];
  /**
   * A-04 chargeback — end-customer cohort (max 128 chars). Sent on
   * observations and as `x-tc-customer` in applied mode.
   * Falls back to `TENSORCOST_CUSTOMER`.
   */
  customer?: string;
  /**
   * A-04 chargeback — product feature (max 128 chars). Sent on
   * observations and as `x-tc-feature` in applied mode.
   * Falls back to `TENSORCOST_FEATURE`.
   */
  feature?: string;
}

export interface ResolvedConfig {
  apiKey: string;
  baseUrl: string;
  tenantId: string | null;
  /**
   * Phase A4 batch 3 — resolved environment tag. `null` when the
   * customer did not configure one; the SDK omits the `environment`
   * field from the envelope in that case so the backend's default
   * ('production') applies.
   */
  environment: string | null;
  /**
   * Phase A4 batch 4 — resolved provider-connection identifier.
   * `null` when the customer did not configure one; the SDK omits the
   * `connection_id` field from the envelope in that case.
   */
  connectionId: string | null;
  /** Resolved team scope for compliance. `null` when not configured. */
  teamId: string | null;
  failOpen: boolean;
  /** Resolved control-layer ceiling. */
  maxLayer: ControlLayer;
  /** @deprecated Derived: maxLayer is steer or route. */
  appliedMode: boolean;
  /**
   * Resolved proxy base URL. Required when maxLayer is steer or route.
   */
  proxyUrl: string | null;

  // Hardening fields — all have safe defaults in resolveConfig().
  retry: RetryConfig;
  timeoutMs: number;
  headersTimeoutMs: number;
  idleTimeoutMs: number;
  onLifecycleEvent: LifecycleEventCallback | undefined;
  failOpenEnabled: boolean;
  providerApiKey: string | null;

  // Metadata defaults resolved from wrap options / env vars.
  /** Resolved deployment tag. `null` when not configured. */
  deployment: string | null;
  /** Resolved default prompt-template id. `null` when not configured. */
  promptTemplateId: string | null;

  // Per-agent-run attribution defaults — see WrapOptions.agentId / .workflowId.
  /** Resolved wrap-time agent id default. `null` when not configured. */
  agentId: string | null;
  /** Resolved wrap-time workflow (run) id default. `null` when not configured. */
  workflowId: string | null;
  /** Resolved application scope for routing policies. `null` when unset. */
  application: string | null;
  /** Resolved routing tags (normalized list). Empty when unset. */
  tags: string[];
  /** A-04 — resolved customer cohort. `null` when unset. */
  customer: string | null;
  /** A-04 — resolved product feature. `null` when unset. */
  feature: string | null;
  /** Resolved code-side data grant ceiling. Empty = no ceiling. */
  maxGrants: DataGrant[];
}
