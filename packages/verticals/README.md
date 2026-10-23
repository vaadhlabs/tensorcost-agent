# @tensorcost/verticals

Apache-2.0 helpers for **install-scope** product verticals: read `TC_VERTICAL_*` environment variables and decide whether optional control-plane services are deployed on this stack.

This is **not** tenant entitlements (`feature.*` on shared SaaS). See [docs/install-profiles](../../docs/install-profiles/README.md).

## Install

```bash
npm install @tensorcost/verticals
```

## Usage

```typescript
import { installedVerticals, isServiceDeployed, isVerticalEnabled } from "@tensorcost/verticals";

if (isVerticalEnabled("fleet", process.env)) {
  // gpu-service task is expected to exist
}

if (!isServiceDeployed("gpu-service", process.env)) {
  return res.status(404).json({ code: "vertical_not_installed" });
}

const flags = installedVerticals(process.env);
// expose on /api/runtime-config for UI probes
```

When a `TC_VERTICAL_*` variable is **unset**, all verticals default to **enabled** (local docker-compose ergonomics). Set explicit `true` / `false` in production dedicated installs.

## License

Apache-2.0 — see [LICENSE](./LICENSE).
