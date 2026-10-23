# Google Vertex AI — Node SDK example (observe-only)

Wraps `@google/genai` `GoogleGenAI({ vertexai: true, ... })` with
`@tensorcost/sdk` `wrap()`. Every `models.generateContent` call is observed
fire-and-forget.

**Observe-only.** Applied mode is not supported for Vertex.

Gemini Developer API clients (`vertexai: false`) are not wrapped in this release.

## Run

```bash
npm install
export GOOGLE_CLOUD_PROJECT=my-gcp-project
export GOOGLE_CLOUD_LOCATION=us-central1
gcloud auth application-default login
export TENSORCOST_API_KEY=tc_live_...
npm start
```

For dashboard Vertex **spend polling** (separate from SDK traces), see
[`../../../direct-providers/vertex/`](../../../direct-providers/vertex/).
