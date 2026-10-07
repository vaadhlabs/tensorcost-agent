# Releasing the TensorCost Agent

## How releases work

Pushing a `v*` tag triggers `.github/workflows/publish-image.yml`, which builds two
image variants (`aws` and `full`) for `linux/amd64` and `linux/arm64` and pushes them
to `ghcr.io/tensorcost/agent`. A `workflow_dispatch` run on any branch builds the same
two variants but tags them `dev-<full-sha>` instead of semver, so you can pull and smoke-test
before cutting a real release.

## Tagging a release

```
git tag -a v0.4.1 -m "Release v0.4.1"
git push origin v0.4.1
```

That push triggers the workflow. Two minutes later (QEMU multi-arch, GHA cache warm) the
following tags are live:

| Tag | Image |
|-----|-------|
| `0.4.1`, `0.4`, `0`, `latest` | `ghcr.io/tensorcost/agent` — AWS + SageMaker variant |
| `0.4.1-full`, `0.4-full`, `0-full`, `latest-full` | `ghcr.io/tensorcost/agent` — all cloud SDKs |

## Testing before tagging

1. Trigger `workflow_dispatch` on your branch via the GitHub Actions UI (or `gh workflow run publish-image.yml`).
2. Copy the `dev-<sha>` URI from the workflow summary.
3. Pull and run locally:
   ```
   docker pull ghcr.io/tensorcost/agent:dev-<sha>
   docker run --rm \
     -e BACKEND_API_KEY=smoke \
     -e TENANT_ID=smoke \
     -e COMM_MODE=http \
     -e BACKEND_API_URL=http://localhost:9999 \
     ghcr.io/tensorcost/agent:dev-<sha> \
     python -c "import sys; sys.path.insert(0,'/app/src/proto'); import agent_pb2; print('OK')"
   ```

## Permissions setup (one-time, per repo)

The workflow uses the built-in `GITHUB_TOKEN` — no org-level PAT or separate secret needed.
You do need to enable write permissions once:

`Settings → Actions → General → Workflow permissions → Read and write permissions`

That's it. GHCR automatically creates the `ghcr.io/tensorcost/agent` package on first push
and ties it to the repository.

## Extending to other variants

The `gcp`, `azure`, and `node` variants are deliberately excluded from v0 to keep build time
under the 40-minute job cap. To add one, duplicate the `meta-*` + `build-and-push` step pair
in `publish-image.yml`, set `build-args: VARIANT=<name>`, and apply a matching `suffix` in the
metadata flavor block (e.g. `suffix=-gcp`).

## Deferred work

- **ECR Public mirror** — separate workflow that assumes AWS OIDC role and `docker push` to
  `public.ecr.aws/tensorcost/agent`. Needed when BYOC customers want to avoid GHCR auth.
- **SBOM generation** — `anchore/sbom-action` post-push, attach to the GitHub release.
- **Vulnerability scan** — `aquasecurity/trivy-action` on the built digest, fail on CRITICAL.
- **gcp / azure / node variants** — extend once the workflow is validated on aws + full.
