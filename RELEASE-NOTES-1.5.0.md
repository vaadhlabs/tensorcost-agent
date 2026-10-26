# SDK 1.5.0: release notes for the owner

Node `@tensorcost/sdk` and Python `tensorcost`, both 1.5.0. What changed for users is in `sdks/node/CHANGELOG.md` and `sdks/python/CHANGELOG.md`. This page covers what you do to ship it. None of these commands has been run.

## Before you publish: BYOC partners

`wrap()` now throws `MissingConfigError` at govern, steer and route when no tenant id is set. Partners on byoc-v0.1.40 can't see `TENSORCOST_TENANT_ID` in their console yet, because the console snippet fix isn't in that release. A partner who upgrades without the id fails at startup.

Do one of these before they upgrade:

- send each BYOC partner their tenant id and the line `export TENSORCOST_TENANT_ID=<id>`, or
- ship a BYOC release with the console snippet fix first, so Developers › SDK setup shows the id.

Nothing else in 1.5.0 needs a newer server. The SDK sends nothing new, and it reads both money-budget wire shapes.

## Publish

`main` is protected (PR, one review, required CI), so don't push to it directly.

1. Push the release commits on a branch and open a PR to `main`. Merge it once CI is green.
2. Tag the merge commit on `main`, then push the tag:

   ```bash
   git tag -a v1.5.0 -m "SDK 1.5.0"
   git push origin v1.5.0
   ```

3. Approve the `release` environment in GitHub Actions. The tag runs `.github/workflows/release.yml`:
   - `npm-sdk` publishes `@tensorcost/sdk@1.5.0`;
   - `pypi` publishes `tensorcost==1.5.0`;
   - `docker` builds `ghcr.io/vaadhlabs/tensorcost-agent` tagged `1.5.0`, `1.5` and `latest`;
   - `npm-verticals` skips if its version is already on npm.

## After it's live: retire 1.4.0

npm:

```bash
npm deprecate @tensorcost/sdk@1.4.0 "1.4.0 can route around a money budget's hard cap when fail-open is on (the default). Upgrade to 1.5.0; see the CHANGELOG for the tenant id requirement."
```

PyPI: you may also yank 1.4.0 (pypi.org → tensorcost → Manage → 1.4.0 → Options → Yank) with the same reason. Yanking stops `pip install tensorcost` from picking 1.4.0, but anyone who pins `==1.4.0` still gets it.
