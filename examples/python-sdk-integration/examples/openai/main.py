"""
TensorCost SDK — OpenAI integration example

Shows the one-line change that enables TensorCost observability:
    Before: client = OpenAI(api_key=...)
    After:  client = wrap(OpenAI(api_key=...))

Required env vars:
    OPENAI_API_KEY       — your OpenAI API key
    TENSORCOST_API_KEY   — your TensorCost API key (from Settings -> API Keys)

Optional:
    TENSORCOST_BASE_URL    — defaults to https://api.tensorcost.com
    TENSORCOST_ENVIRONMENT — e.g. "staging" or "production"
"""

import os

from openai import OpenAI
from tensorcost import wrap

# -- Before TensorCost -------------------------------------------------------
# client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# -- After TensorCost (the only change) --------------------------------------
client = wrap(
    OpenAI(api_key=os.environ["OPENAI_API_KEY"]),
    # wrap() reads TENSORCOST_API_KEY from the environment automatically.
    # You can also pass it explicitly: api_key="tc_live_xxx"
)

# The rest of your code is completely unchanged.
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What is 2 + 2?"}],
    max_tokens=64,
)

print("Response:", response.choices[0].message.content)
print("Usage:", response.usage)
print(
    "\nThis call was observed by TensorCost. Check your dashboard at "
    "https://app.tensorcost.com to see it appear within ~2 minutes."
)
