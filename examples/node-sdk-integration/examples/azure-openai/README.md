# Azure OpenAI — Node SDK example

Wraps `openai`'s `AzureOpenAI` class — the same npm package used for the
plain OpenAI example, configured for an Azure endpoint — with
`@tensorcost/sdk`'s `wrap()`. There is no separate Azure client in the
SDK; Azure OpenAI's data-plane API is OpenAI-compatible, so the same
`wrap()` call works unchanged.

**Observations from this path land with `provider_id: openai`**, not
`azure-openai` — the SDK detects the client shape it wrapped, not the
backing endpoint. If you need spend attributed per Azure resource, pair
this with the [Azure OpenAI direct-provider connection](../../../direct-providers/azure-openai/),
which polls Azure directly and reports as `azure-openai`.

## Run

```bash
npm install
export AZURE_OPENAI_ENDPOINT=https://my-resource.openai.azure.com
export AZURE_OPENAI_API_KEY=...
export AZURE_OPENAI_API_VERSION=2024-10-21
export AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini
export TENSORCOST_API_KEY=tc_live_...
npm start
```

## What to expect

```
Response: 4
Usage: { prompt_tokens: 14, completion_tokens: 1, total_tokens: 15 }

This call was observed by TensorCost (provider_id: openai). Check
your dashboard at https://app.tensorcost.com to see it appear within ~2 minutes.
```

See [`../../troubleshooting.md`](../../troubleshooting.md) if nothing
shows up in the dashboard.
