/**
 * TensorCost SDK — Anthropic integration example
 *
 * Shows the one-line change that enables TensorCost observability:
 *   Before: const client = new Anthropic({ apiKey: ... });
 *   After:  const client = wrap(new Anthropic({ apiKey: ... }));
 *
 * Required env vars:
 *   ANTHROPIC_API_KEY    — your Anthropic API key
 *   TENSORCOST_API_KEY   — your TensorCost API key (from Settings → API Keys)
 *
 * Optional:
 *   TENSORCOST_BASE_URL  — defaults to https://api.tensorcost.com
 *   TENSORCOST_ENVIRONMENT — e.g. "staging" or "production"
 */

import Anthropic from "@anthropic-ai/sdk";
import { wrap } from "@tensorcost/sdk";

// ── Before TensorCost ─────────────────────────────────────────────────────
// const client = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });

// ── After TensorCost (the only change) ───────────────────────────────────
const client = wrap(
  new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY }),
  // wrap() reads TENSORCOST_API_KEY from the environment automatically.
);

// Everything below is unchanged from your existing Anthropic code.
const message = await client.messages.create({
  model: "claude-3-5-haiku-20241022",
  max_tokens: 64,
  messages: [{ role: "user", content: "What is 2 + 2?" }],
});

console.log("Response:", message.content[0].text);
console.log("Usage:", message.usage);
console.log(
  "\nThis call was observed by TensorCost. Check your dashboard at",
  "https://app.tensorcost.com to see it appear within ~2 minutes.",
);
