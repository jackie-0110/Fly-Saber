# CI and container delivery

## What runs automatically

- Pull requests and pushes to `main`: Python 3.11/3.12 tests, Python lint/format/type
  checks, frontend lint/format/build, desktop/mobile Chromium tests, and Docker
  build/startup checks.
- Version tags matching `v*`: rerun the complete CI workflow, then publish the
  combined frontend/API image to GitHub Container Registry (GHCR).
- Manual **Publish container** dispatch on `main`: run the same checks and publish
  an image tagged with the full commit SHA. Publishing is disabled for other branches.
- Weekly: Dependabot opens dependency update PRs.

The delivery boundary is a tested image in GHCR. No production host is configured
in this repository yet. Deployment to Vultr requires provisioning that host and
connecting its credentials or a deployment agent; this workflow does not restart
an external server.

## Build and run locally

Run from the repository root with Docker available:

```bash
make docker-build
docker run --rm -p 8000:8000 fly-saber:local
```

Open `http://localhost:8000` for the UI. The same service handles `/api/*` and
`/ws/session`, so the frontend and backend share an origin. The image builds the
frontend and installs Python dependencies from the committed lockfiles, runs as
an unprivileged user, and exposes port 8000. Its health check requests `/api/games`.
The Docker context allowlist excludes secrets, local environments, raw data,
checkpoints, and build output.

The image includes configs, the demo audio allowlist and the trained brains in
`bundles/`, each with the subgraph it was trained on. Provide normalized maps and
licensed audio through a volume on the deployment host when available:

```bash
docker run --rm -p 8000:8000 \
  --mount type=bind,src=/absolute/path/to/data/maps,dst=/app/data/maps,readonly \
  fly-saber:local
```

That maps directory should contain `demo_allowlist.json`, `normalized/`, and
`audio/` as described in [PROTOCOL.md](PROTOCOL.md). Without downloaded maps the
UI and API still start, and the map list is empty. `/api/brains` lists the scripted
expert plus every brain mode in `configs/brains.yaml` that has a bundle.

Startup loads each bundle and compiles every brain for every map size before the
server listens, so the first response takes tens of seconds, longer on a small
host. The health check allows 180 s.

CI's container smoke test mounts `docker/smoke/normalized` (one tiny map), checks
that `/api/brains` lists `real`, and plays one session with it through
`docker/smoke_session.py`.

## Publish a release

After merging and checking `main`, create a version tag on the intended commit:

```bash
git switch main
git pull --ff-only
git tag v0.1.0
git push origin v0.1.0
```

The publish workflow uses the repository's built-in `GITHUB_TOKEN` with
`packages: write`; no registry password secret is needed. GitHub Actions and
package publishing must be enabled in the repository/organization. For an
existing GHCR package, ensure this repository has permission to write it.

For this repository, `v0.1.0` publishes:

```text
ghcr.io/jackie-0110/fly-saber:v0.1.0
ghcr.io/jackie-0110/fly-saber:0.1.0
ghcr.io/jackie-0110/fly-saber:0.1
ghcr.io/jackie-0110/fly-saber:sha-<full-commit-sha>
```

Docker metadata also adds `latest` for stable semantic versions. Prerelease tags
do not advance `latest`. Prefer the full SHA tag or image digest for deployment
and rollback. Use the previous tested image to roll back; the pipeline does not
delete previous release images. GHCR packages may initially be private; configure
package visibility or authenticate the deployment host before pulling.

## Diagnose failures

Open the failing job in GitHub Actions. Download `python-results-*` for pytest
JUnit output, `browser-results` for Playwright HTML/JUnit reports and failure
traces/screenshots, or `frontend-dist` for the production assets. Container startup
logs are printed even when the smoke test fails. A release cannot publish unless
the reusable CI workflow succeeds.

To inspect a downloaded Playwright trace:

```bash
cd frontend
npx playwright show-trace /path/to/trace.zip
```
