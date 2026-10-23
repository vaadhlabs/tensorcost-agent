# Anthropic — Python SDK example

Wraps a plain `anthropic.Anthropic()` client with `tensorcost.wrap()`.
Every `messages.create` call is observed fire-and-forget; nothing else
about the call changes.

## Run

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
export TENSORCOST_API_KEY=tc_live_...
python main.py
```

## What to expect

```
Response: 4
Usage: Usage(input_tokens=14, output_tokens=5)

This call was observed by TensorCost. Check your dashboard at
https://app.tensorcost.com to see it appear within ~2 minutes.
```

See [`../../troubleshooting.md`](../../troubleshooting.md) if nothing
shows up in the dashboard.
