# Azure OpenAI — Python SDK example

Wraps `openai.AzureOpenAI()` — the standard OpenAI Python client's
Azure-flavored constructor — with `tensorcost.wrap()`. There is no
separate Azure client in the SDK; Azure OpenAI's data-plane API is
OpenAI-compatible, so the same `wrap()` call works unchanged.

**Observations from this path land with `provider_id: openai`**, not
`azure_openai` — the SDK detects the client class it wrapped, not the
backing endpoint. If you need spend attributed per Azure resource, pair
this with the [Azure OpenAI direct-provider connection](../../../direct-providers/azure-openai/),
which polls Azure directly and reports as `azure_openai`.

## Run

```bash
pip install -r requirements.txt
export AZURE_OPENAI_ENDPOINT=https://my-resource.openai.azure.com
export AZURE_OPENAI_API_KEY=...
export AZURE_OPENAI_API_VERSION=2024-10-21
export AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini
export TENSORCOST_API_KEY=tc_live_...
python main.py
```

## What to expect

```
Response: 4
Usage: CompletionUsage(completion_tokens=1, prompt_tokens=14, total_tokens=15)

This call was observed by TensorCost (provider_id: openai). Check your
dashboard at https://app.tensorcost.com to see it appear within ~2 minutes.
```

See [`../../troubleshooting.md`](../../troubleshooting.md#azure-openai-observations-show-provider_id-openai-not-azure_openai)
for more on the `provider_id` behavior.
