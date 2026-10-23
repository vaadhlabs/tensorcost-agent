/**
 * TensorCost SDK — Google Vertex AI integration example
 *
 * Vertex support is observe-only. The SDK patches models.generate_content on
 * a @google/genai Client constructed with vertexai: true.
 *
 * Required env vars:
 *   GOOGLE_CLOUD_PROJECT
 *   GOOGLE_CLOUD_LOCATION
 *   TENSORCOST_API_KEY
 *
 * Credentials: Application Default Credentials (gcloud auth application-default login)
 */

import { GoogleGenAI } from "@google/genai";
import { wrap } from "@tensorcost/sdk";

const project = process.env.GOOGLE_CLOUD_PROJECT;
const location = process.env.GOOGLE_CLOUD_LOCATION;
const model = process.env.VERTEX_MODEL ?? "gemini-2.0-flash";

if (!project || !location) {
  throw new Error("Set GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION");
}

const client = wrap(
  new GoogleGenAI({
    vertexai: true,
    project,
    location,
  }),
);

const response = await client.models.generateContent({
  model,
  contents: "What is 2 + 2? Reply with just the number.",
});

console.log("Response:", response.text);
const usage = response.usageMetadata;
if (usage) {
  console.log("Usage:", usage.promptTokenCount, "in /", usage.candidatesTokenCount, "out");
}

console.log(
  "\nThis call was observed by TensorCost. Check Router · Live in the console.",
);
