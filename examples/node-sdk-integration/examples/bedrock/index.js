/**
 * TensorCost SDK — AWS Bedrock integration example
 *
 * Bedrock support is observe-only. The SDK intercepts the `send()` call on a
 * BedrockRuntimeClient and records model, token counts, latency, and status
 * as a fire-and-forget observation.
 *
 * Applied mode (proxy routing) is NOT supported for Bedrock — wrap() will
 * throw immediately if you set { appliedMode: true } with a Bedrock client.
 *
 * Required env vars:
 *   TENSORCOST_API_KEY   — your TensorCost API key (from Settings → API Keys)
 *   AWS_REGION           — AWS region, e.g. us-east-1
 *
 * AWS credentials are resolved by the standard SDK credential chain:
 *   - IAM instance profile (EC2, ECS, Lambda) — preferred in production
 *   - ~/.aws/credentials or environment vars (dev)
 *
 * Optional:
 *   TENSORCOST_BASE_URL  — defaults to https://api.tensorcost.com
 *   TENSORCOST_ENVIRONMENT — e.g. "staging" or "production"
 */

import {
  BedrockRuntimeClient,
  InvokeModelCommand,
} from "@aws-sdk/client-bedrock-runtime";
import { wrap } from "@tensorcost/sdk";

// ── Before TensorCost ─────────────────────────────────────────────────────
// const client = new BedrockRuntimeClient({ region: process.env.AWS_REGION ?? "us-east-1" });

// ── After TensorCost (the only change) ───────────────────────────────────
const client = wrap(
  new BedrockRuntimeClient({ region: process.env.AWS_REGION ?? "us-east-1" }),
  // wrap() reads TENSORCOST_API_KEY from the environment automatically.
  // Bedrock does not support appliedMode — omit that option.
);

// Everything below is unchanged from your existing Bedrock code.
const body = JSON.stringify({
  anthropic_version: "bedrock-2023-05-31",
  max_tokens: 64,
  messages: [{ role: "user", content: "What is 2 + 2?" }],
});

const command = new InvokeModelCommand({
  modelId: "anthropic.claude-3-haiku-20240307-v1:0",
  body,
  contentType: "application/json",
  accept: "application/json",
});

const response = await client.send(command);

const responseBody = JSON.parse(Buffer.from(response.body).toString("utf-8"));
console.log("Response:", responseBody.content[0].text);
console.log("Usage:", responseBody.usage);
console.log(
  "\nThis call was observed by TensorCost. Check your dashboard at",
  "https://app.tensorcost.com to see it appear within ~2 minutes.",
);
