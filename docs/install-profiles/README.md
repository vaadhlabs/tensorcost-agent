# Install profiles (vertical flags)

Optional **groups** of control-plane services on a dedicated install (BYOC, demo, on-prem). This is separate from **tenant entitlements** on shared multi-tenant SaaS.

## Environment variables

| Vertical | Env var | When `false` |
|----------|---------|--------------|
| AI | `TC_VERTICAL_AI` | LLM ingest / proxy services off |
| AI-ML | `TC_VERTICAL_AI_ML` | ML worker off (requires AI) |
| Fleet | `TC_VERTICAL_FLEET` | GPU agent ingress / gRPC NLB off |
| Controls | `TC_VERTICAL_CONTROLS` | Alerts / enforcement off |
| Reporting | `TC_VERTICAL_REPORTING` | Export / schedule off |
| Billing | `TC_VERTICAL_BILLING` | Stripe bridge off (BYOC uses offline license) |
| DevTools | `TC_VERTICAL_DEVTOOLS` | Smoke / devtools off |

Unset vars default to **enabled** (local dev). Production dedicated stacks should set explicit booleans.

## Manifest shape

See [verticals.schema.json](./verticals.schema.json) and [example-verticals.yaml](./example-verticals.yaml) for a **sanitized** manifest (dummy service names). The real product map lives in the private TensorCost monorepo.

## Library

[`@tensorcost/verticals`](../../packages/verticals) reads the same env vars in Node services, gateways, and job runners.

## Migration guardrail

[`tools/ci/check-cross-schema-migrations.mjs`](../../tools/ci/check-cross-schema-migrations.mjs) fails CI when SQL migrations add unconditional `REFERENCES other_schema.table` — that breaks boots when the other vertical is not installed. Pass your repo root and a JSON schema map (see [schema-map.example.json](../../tools/ci/schema-map.example.json)).
