"""
TensorCost SDK — Google Vertex AI integration example

Vertex support is observe-only. The SDK monkey-patches
``models.generate_content`` / ``generate_content_stream`` on a
``google.genai.Client`` constructed with ``vertexai=True`` and records
model, token counts, latency, and status as a fire-and-forget observation.

Applied mode (proxy routing) is NOT supported for Vertex — wrap() raises
MissingConfigError immediately if you pass applied_mode=True with a
Vertex client.

Required env vars:
    GOOGLE_CLOUD_PROJECT  — GCP project id
    GOOGLE_CLOUD_LOCATION — region, e.g. us-central1
    TENSORCOST_API_KEY    — your TensorCost API key (from Keys & MCP)

Credentials are resolved by Application Default Credentials (ADC):
    - gcloud auth application-default login (local dev)
    - Workload identity / attached service account (GCE, GKE, Cloud Run)

Optional:
    TENSORCOST_BASE_URL    — defaults to https://api.tensorcost.com
    TENSORCOST_ENVIRONMENT — e.g. "staging" or "production"
    VERTEX_MODEL           — defaults to gemini-2.0-flash
"""

import os

from google import genai
from tensorcost import wrap

project = os.environ["GOOGLE_CLOUD_PROJECT"]
location = os.environ["GOOGLE_CLOUD_LOCATION"]
model = os.environ.get("VERTEX_MODEL", "gemini-2.0-flash")

# -- Before TensorCost -------------------------------------------------------
# client = genai.Client(vertexai=True, project=project, location=location)

# -- After TensorCost (the only change) --------------------------------------
client = wrap(
    genai.Client(vertexai=True, project=project, location=location),
    # wrap() reads TENSORCOST_API_KEY from the environment automatically.
    # Vertex does not support applied_mode — omit that option.
)

response = client.models.generate_content(
    model=model,
    contents="What is 2 + 2? Reply with just the number.",
)

print("Response:", response.text)
usage = getattr(response, "usage_metadata", None)
if usage is not None:
    print(
        "Usage:",
        getattr(usage, "prompt_token_count", None),
        "in /",
        getattr(usage, "candidates_token_count", None),
        "out",
    )

print(
    "\nThis call was observed by TensorCost. Check Router · Live in the "
    "console to see it appear within ~60 seconds."
)
