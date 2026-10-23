# Google Vertex AI — Python SDK example (observe-only)

Wraps a `google.genai.Client(vertexai=True, ...)` with `tensorcost.wrap()`.
Every `models.generate_content` call is observed fire-and-forget; nothing
else about the call changes.

**Observe-only.** Applied mode (proxy-routed inference) is not supported
for Vertex — `wrap(client, applied_mode=True)` raises
`MissingConfigError` immediately. Vertex uses Application Default
Credentials and Google-managed transport, which the SDK cannot intercept
the way it redirects an HTTP `base_url` for OpenAI/Anthropic clients.

Streaming calls (`models.generate_content_stream`) are observed too, but
the SDK does not read the stream — it emits a placeholder observation with
`input_tokens`/`output_tokens` set to `null` and a note that per-chunk
token accounting isn't implemented yet. Your iteration over the returned
stream is completely unaffected.

**Gemini Developer API** (`Client(api_key=...)`, `vertexai=False`) is not
wrapped in this release — only Vertex AI mode is supported.

## Run

```bash
pip install -r requirements.txt
export GOOGLE_CLOUD_PROJECT=my-gcp-project
export GOOGLE_CLOUD_LOCATION=us-central1
gcloud auth application-default login   # or use a attached SA on GCP
export TENSORCOST_API_KEY=tc_live_...
python main.py
```

## What to expect

```
Response: 4
Usage: 12 in / 1 out

This call was observed by TensorCost. Check Router · Live in the
console to see it appear within ~60 seconds.
```

## Dashboard Vertex spend (separate path)

SDK wrap observes individual inference calls. For **Vertex spend polling**
in the CFO dashboard, use the direct connection recipe:
[`../../../direct-providers/vertex/`](../../../direct-providers/vertex/).

See [`../../troubleshooting.md`](../../troubleshooting.md) if nothing
shows up in Router · Live.
