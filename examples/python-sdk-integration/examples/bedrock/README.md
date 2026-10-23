# AWS Bedrock — Python SDK example (observe-only)

Wraps a plain `boto3.client("bedrock-runtime")` with `tensorcost.wrap()`.
Every `invoke_model` and `converse` call is observed fire-and-forget;
nothing else about the call changes.

**Observe-only.** Applied mode (proxy-routed inference) is not supported
for Bedrock — `wrap(client, applied_mode=True)` raises
`MissingConfigError` immediately. Bedrock uses SigV4 request signing
inside `botocore`, which the SDK cannot intercept the way it redirects
an HTTP `base_url` for OpenAI/Anthropic-style clients.

Streaming calls (`invoke_model_with_response_stream`, `converse_stream`)
are observed too, but the SDK does not read the stream — it emits a
placeholder observation with `input_tokens`/`output_tokens` set to
`null` and a note that per-chunk token accounting isn't implemented yet.
Your iteration over the returned stream is completely unaffected.

## Run

```bash
pip install -r requirements.txt
export AWS_REGION=us-east-1
# AWS credentials via the standard boto3 chain (env vars, ~/.aws/credentials,
# or an IAM role) — nothing Bedrock-specific to configure here.
export TENSORCOST_API_KEY=tc_live_...
python main.py
```

## What to expect

```
Response: 4
Usage: {'input_tokens': 14, 'output_tokens': 5}

This call was observed by TensorCost. Check your dashboard at
https://app.tensorcost.com to see it appear within ~2 minutes.
```

## Token counts by model family

Bedrock does not standardize usage reporting across model families. The
SDK probes, in order:

1. **Anthropic-on-Bedrock** (`invoke_model`): response body JSON ->
   `usage.input_tokens` / `usage.output_tokens`
2. **Amazon Nova / Titan** (`invoke_model`): response body JSON ->
   `usage.inputTokens` / `usage.outputTokens`
3. **Converse API** (any model, via `converse`): top-level
   `usage.inputTokens` / `usage.outputTokens`

If none match, the observation is still recorded with `null` token
counts (latency and success/error status are unaffected) and a warning
is logged via `logging.getLogger("tensorcost")`.

See [`../../troubleshooting.md`](../../troubleshooting.md) if nothing
shows up in the dashboard.
