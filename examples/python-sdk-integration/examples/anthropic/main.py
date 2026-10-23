"""
TensorCost SDK — Anthropic integration example

Shows the one-line change that enables TensorCost observability:
    Before: client = Anthropic(api_key=...)
    After:  client = wrap(Anthropic(api_key=...))

Required env vars:
    ANTHROPIC_API_KEY    — your Anthropic API key
    TENSORCOST_API_KEY   — your TensorCost API key (from Settings -> API Keys)

Optional:
    TENSORCOST_BASE_URL    — defaults to https://api.tensorcost.com
    TENSORCOST_ENVIRONMENT — e.g. "staging" or "production"
"""

import os

from anthropic import Anthropic
from tensorcost import wrap

# -- Before TensorCost -------------------------------------------------------
# client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

# -- After TensorCost (the only change) --------------------------------------
client = wrap(
    Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"]),
    # wrap() reads TENSORCOST_API_KEY from the environment automatically.
)

# Everything below is unchanged from your existing Anthropic code.
message = client.messages.create(
    model="claude-3-5-haiku-20241022",
    max_tokens=64,
    messages=[{"role": "user", "content": "What is 2 + 2?"}],
)

print("Response:", message.content[0].text)
print("Usage:", message.usage)
print(
    "\nThis call was observed by TensorCost. Check your dashboard at "
    "https://app.tensorcost.com to see it appear within ~2 minutes."
)
