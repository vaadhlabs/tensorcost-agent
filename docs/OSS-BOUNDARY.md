# Open source boundary

**Repository:** [vaadhlabs/tensorcost-agent](https://github.com/vaadhlabs/tensorcost-agent) (Apache-2.0)

## In this repo (open)

| Area | Purpose |
|------|---------|
| `agent/` | GPU telemetry agent (NVML, exporters, optional control-plane upload) |
| `sdks/` | Node (`@tensorcost/sdk`) and Python (`tensorcost`) LLM wrappers |
| `packages/verticals` | Install-scope `TC_VERTICAL_*` helpers |
| `packages/cli`, `packages/mcp-framework`, `packages/throttler-storage-redis` | Integration libraries (migrate here from private monorepo over time) |
| `install/` | Customer-facing agent install (bare metal, Kubernetes, Docker Compose) |
| `examples/` | SDK provider walkthroughs |
| `tools/ci/` | Generic migration guardrails (schema map driven) |
| `docs/` | Ingest contract, install profiles schema, protos |

## Not in this repo (commercial / private)

- Control-plane microservices (cost, AI, GPU server, billing, enforcement, …)
- Multi-tenant shell UI and proprietary recommenders
- Full BYOC CloudFormation / CDK / platform Helm (product topology)
- Stripe, Marketplace metering, tenant entitlement source of truth
- Production `verticals.yaml` with real service names wired to deploy generators

Private installs consume **published** artifacts from this repo (npm, PyPI, GHCR) plus a license file and private deploy templates.

## Dependency direction

```
tensorcost-agent  ──publish──►  registries
       ▲
       │  version pin (npm / git tag)
private monorepo  ──depends──►  @tensorcost/*
```

Do not fork Apache-2.0 sources back into the monorepo long term.
