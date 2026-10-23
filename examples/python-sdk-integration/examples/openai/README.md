# OpenAI — Python SDK example

Wraps a plain `openai.OpenAI()` client with `tensorcost.wrap()`. Every
`chat.completions.create` call is observed fire-and-forget; nothing else
about the call changes.

## Run

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...
export TENSORCOST_API_KEY=tc_live_...
python main.py
```

## What to expect

```
Response: 4
Usage: CompletionUsage(completion_tokens=1, prompt_tokens=14, total_tokens=15)

This call was observed by TensorCost. Check your dashboard at
https://app.tensorcost.com to see it appear within ~2 minutes.
```

See [`../../troubleshooting.md`](../../troubleshooting.md) if nothing
shows up in the dashboard.
