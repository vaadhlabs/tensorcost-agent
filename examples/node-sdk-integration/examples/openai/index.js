/**
 * TensorCost SDK — OpenAI integration example
 *
 * Shows the one-line change that enables TensorCost observability:
 *   Before: const client = new OpenAI({ apiKey: ... });
 *   After:  const client = wrap(new OpenAI({ apiKey: ... }));
 *
 * Required env vars:
 *   OPENAI_API_KEY       — your OpenAI API key
 *   TENSORCOST_API_KEY   — your TensorCost API key (from Settings → API Keys)
 *
 * Optional:
 *   TENSORCOST_BASE_URL  — defaults to https://api.tensorcost.com
 *   TENSORCOST_ENVIRONMENT — e.g. "staging" or "production"
 */

import OpenAI from "openai";
import { wrap } from "@tensorcost/sdk";

// ── Before TensorCost ─────────────────────────────────────────────────────
// const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

// ── After TensorCost (the only change) ───────────────────────────────────
const client = wrap(
  new OpenAI({ apiKey: process.env.OPENAI_API_KEY }),
  // wrap() reads TENSORCOST_API_KEY from the environment automatically.
  // You can also pass it explicitly: { apiKey: "tc_live_xxx" }
);

// The rest of your code is completely unchanged.
const response = await client.chat.completions.create({
  model: "gpt-4o-mini",
  messages: [{ role: "user", content: "What is 2 + 2?" }],
  max_tokens: 64,
});

console.log("Response:", response.choices[0].message.content);
console.log("Usage:", response.usage);
console.log(
  "\nThis call was observed by TensorCost. Check your dashboard at",
  "https://app.tensorcost.com to see it appear within ~2 minutes.",
);
