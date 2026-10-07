# Releasing

Publishing is **tag-driven**. Push an annotated semver tag on `main`:

```bash
git tag v1.3.0
git push origin v1.3.0
```

That triggers [`.github/workflows/release.yml`](./.github/workflows/release.yml):

| Job | Output |
|-----|--------|
| **docker** | Multi-arch image to **GHCR** `ghcr.io/vaadhlabs/tensorcost-agent` (tags: semver, `latest` on release) |
| **npm** | **`@tensorcost/sdk`** on [npmjs.com](https://www.npmjs.com/package/@tensorcost/sdk) |
| **pypi** | **`tensorcost`** on [PyPI](https://pypi.org/project/tensorcost/) |

CI on every push does **not** publish; only `v*` tags do.

## One-time setup (repo admin)

### GitHub Container Registry (agent image)

No extra secret: `GITHUB_TOKEN` pushes to **`ghcr.io/vaadhlabs/tensorcost-agent`** for this repository.

After the first successful release, make the package **public**: GitHub → **Packages** → `tensorcost-agent` → **Package settings** → visibility **Public**.

To publish under `ghcr.io/tensorcost/agent` instead, transfer the repo to the **`tensorcost`** org (or add a second metadata image line) and re-run a tag.

### npm (`@tensorcost/sdk`)

1. Create an npm access token with **publish** rights to the `@tensorcost` scope (or use npm provenance + trusted publish if you migrate to that model).
2. Add repo secret: **Settings → Secrets → Actions → `NPM_TOKEN`**.

The release job runs `npm publish --provenance --access public` with `NODE_AUTH_TOKEN=${{ secrets.NPM_TOKEN }}`.

Bump **`sdks/node/package.json`** `version` (and changelog) before tagging so the tarball matches the git tag.

### PyPI (`tensorcost`)

Preferred: **[PyPI trusted publishing](https://docs.pypi.org/trusted-publishers/)** (OIDC, no long-lived API token).

1. On [pypi.org](https://pypi.org) → your `tensorcost` project → **Publishing** → **Add a new pending publisher**:
   - Owner: `vaadhlabs`
   - Repository: `tensorcost-agent`
   - Workflow: `release.yml`
   - Environment: (leave blank unless you add a GitHub Environment)
2. The workflow already sets `id-token: write` and uses `pypa/gh-action-pypi-publish`.

Alternative: store a **`PYPI_API_TOKEN`** in repo secrets and pass it to the publish action (not configured by default).

Bump **`sdks/python/pyproject.toml`** `version` before tagging.

## Dry run locally

```bash
# Agent image
docker build -f agent/Dockerfile agent --build-arg VARIANT=full -t tensorcost-agent:local

# Node SDK
cd sdks/node && npm run build && npm pack

# Python SDK
cd sdks/python && pip install build && python -m build
```

## Version alignment

Keep agent Helm chart / docs in sync manually for now. Tag **`v1.3.0`** should match SDK **`1.3.0`** in both `package.json` and `pyproject.toml` when you cut a coordinated release.
