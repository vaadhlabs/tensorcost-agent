"""End-to-end smoke suite for the Python production agent.

These tests assume a live `gpu-service` instance is reachable at
`GRPC_TARGET` and a clean Postgres at `E2E_DATABASE_URL`. The CI
workflow that drives them lives in the sibling repo at
`apps-new/.github/workflows/gpu-agent-e2e.yml` (FCR §7.5).

Local invocation:
    GRPC_TARGET=127.0.0.1:50051 \
    TENANT_ID=00000000-0000-0000-0000-0000000000e2 \
    AGENT_HOSTNAME=e2e-agent-1 \
    AGENT_KEY_ID=e2e-key \
    AGENT_HMAC_PEPPER=$(python -c "import secrets; print(secrets.token_hex(32))") \
    E2E_DATABASE_URL=postgres://postgres:postgres_password@127.0.0.1:5432/tensorcost \
    pytest tests/e2e/ -v
"""
