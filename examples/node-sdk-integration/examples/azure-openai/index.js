/**
 * TensorCost SDK — Azure OpenAI integration example
 *
 * Azure OpenAI is OpenAI-API-compatible, so there is no separate "Azure
 * client" in @tensorcost/sdk to wrap: point the standard `openai` npm
 * client's `AzureOpenAI` class at your Azure endpoint/deployment, then
 * wrap it exactly like a plain OpenAI client.
 *
 * Because the underlying client class is still `openai`'s, observations
 * from this path land in TensorCost tagged with provider_id "openai"
 * (the SDK detects the client shape, not the backing endpoint). If you
 * need spend broken out per Azure resource, also set up the Azure
 * OpenAI direct-provider connection — see
 * ../../../direct-providers/azure-openai/.
 *
 * Required env vars:
 *   AZURE_OPENAI_ENDPOINT     — e.g. https://my-resource.openai.azure.com
 *   AZURE_OPENAI_API_KEY      — Azure OpenAI resource API key
 *   AZURE_OPENAI_API_VERSION  — e.g. 2024-10-21
 *   AZURE_OPENAI_DEPLOYMENT   — your chat-completions deployment name
 *   TENSORCOST_API_KEY        — your TensorCost API key (from Settings → API Keys)
 *
 * Optional:
 *   TENSORCOST_BASE_URL  — defaults to https://api.tensorcost.com
 *   TENSORCOST_ENVIRONMENT — e.g. "staging" or "production"
 */

import { AzureOpenAI } from "openai";
import { wrap } from "@tensorcost/sdk";

// ── Before TensorCost ─────────────────────────────────────────────────────
// const client = new AzureOpenAI({
//   endpoint: process.env.AZURE_OPENAI_ENDPOINT,
//   apiKey: process.env.AZURE_OPENAI_API_KEY,
//   apiVersion: process.env.AZURE_OPENAI_API_VERSION,
// });

// ── After TensorCost (the only change) ───────────────────────────────────
const client = wrap(
  new AzureOpenAI({
    endpoint: process.env.AZURE_OPENAI_ENDPOINT,
    apiKey: process.env.AZURE_OPENAI_API_KEY,
    apiVersion: process.env.AZURE_OPENAI_API_VERSION,
  }),
  // wrap() reads TENSORCOST_API_KEY from the environment automatically.
);

// Everything below is unchanged from your existing Azure OpenAI code.
// `model` is your Azure *deployment* name, not the underlying model id.
const response = await client.chat.completions.create({
  model: process.env.AZURE_OPENAI_DEPLOYMENT,
  messages: [{ role: "user", content: "What is 2 + 2?" }],
  max_tokens: 64,
});

console.log("Response:", response.choices[0].message.content);
console.log("Usage:", response.usage);
console.log(
  "\nThis call was observed by TensorCost (provider_id: openai). Check",
  "your dashboard at https://app.tensorcost.com to see it appear within ~2 minutes.",
);
