"""
TensorCost SDK — Azure OpenAI integration example

Azure OpenAI is OpenAI-API-compatible, so there is no separate
"Azure client" in the tensorcost SDK to wrap: you point the standard
openai Python client's AzureOpenAI class at your Azure endpoint and
deployment, then wrap it exactly like a plain OpenAI client.

Because the underlying client class is still `openai`'s, observations
from this path land in TensorCost tagged with provider_id "openai" (the
SDK detects the client class, not the backing endpoint). If you need
spend broken out per Azure resource, also set up the Azure OpenAI
direct-provider connection — see ../../../direct-providers/azure-openai/.

Required env vars:
    AZURE_OPENAI_ENDPOINT     — e.g. https://my-resource.openai.azure.com
    AZURE_OPENAI_API_KEY      — Azure OpenAI resource API key
    AZURE_OPENAI_API_VERSION  — e.g. 2024-10-21
    AZURE_OPENAI_DEPLOYMENT   — your chat-completions deployment name
    TENSORCOST_API_KEY        — your TensorCost API key (from Settings -> API Keys)

Optional:
    TENSORCOST_BASE_URL    — defaults to https://api.tensorcost.com
    TENSORCOST_ENVIRONMENT — e.g. "staging" or "production"
"""

import os

from openai import AzureOpenAI
from tensorcost import wrap

# -- Before TensorCost -------------------------------------------------------
# client = AzureOpenAI(
#     azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
#     api_key=os.environ["AZURE_OPENAI_API_KEY"],
#     api_version=os.environ["AZURE_OPENAI_API_VERSION"],
# )

# -- After TensorCost (the only change) --------------------------------------
client = wrap(
    AzureOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    ),
    # wrap() reads TENSORCOST_API_KEY from the environment automatically.
)

# Everything below is unchanged from your existing Azure OpenAI code.
# `model` is your Azure *deployment* name, not the underlying model id.
response = client.chat.completions.create(
    model=os.environ["AZURE_OPENAI_DEPLOYMENT"],
    messages=[{"role": "user", "content": "What is 2 + 2?"}],
    max_tokens=64,
)

print("Response:", response.choices[0].message.content)
print("Usage:", response.usage)
print(
    "\nThis call was observed by TensorCost (provider_id: openai). "
    "Check your dashboard at https://app.tensorcost.com to see it "
    "appear within ~2 minutes."
)
