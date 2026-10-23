"""
TensorCost SDK — AWS Bedrock integration example

Bedrock support is observe-only. The SDK monkey-patches `invoke_model` /
`converse` (and their streaming counterparts) on a boto3 bedrock-runtime
client and records model, token counts, latency, and status as a
fire-and-forget observation.

Applied mode (proxy routing) is NOT supported for Bedrock — wrap() raises
MissingConfigError immediately if you pass applied_mode=True with a
Bedrock client.

Required env vars:
    AWS_REGION           — AWS region, e.g. us-east-1
    TENSORCOST_API_KEY   — your TensorCost API key (from Settings -> API Keys)

AWS credentials are resolved by the standard boto3 credential chain:
    - IAM instance profile / task role (EC2, ECS, Lambda) — preferred in production
    - ~/.aws/credentials or environment vars (dev)

Optional:
    TENSORCOST_BASE_URL    — defaults to https://api.tensorcost.com
    TENSORCOST_ENVIRONMENT — e.g. "staging" or "production"
"""

import json
import os

import boto3
from tensorcost import wrap

# -- Before TensorCost -------------------------------------------------------
# client = boto3.client("bedrock-runtime", region_name=os.environ.get("AWS_REGION", "us-east-1"))

# -- After TensorCost (the only change) --------------------------------------
client = wrap(
    boto3.client("bedrock-runtime", region_name=os.environ.get("AWS_REGION", "us-east-1")),
    # wrap() reads TENSORCOST_API_KEY from the environment automatically.
    # Bedrock does not support applied_mode — omit that option.
)

# Everything below is unchanged from your existing Bedrock code.
response = client.invoke_model(
    modelId="anthropic.claude-3-haiku-20240307-v1:0",
    body=json.dumps(
        {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "What is 2 + 2?"}],
        }
    ),
    contentType="application/json",
    accept="application/json",
)

body = json.loads(response["body"].read())
print("Response:", body["content"][0]["text"])
print("Usage:", body["usage"])
print(
    "\nThis call was observed by TensorCost. Check your dashboard at "
    "https://app.tensorcost.com to see it appear within ~2 minutes."
)
