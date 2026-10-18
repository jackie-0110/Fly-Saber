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

The delivery boundary is a tested image in GHCR. CI does not restart an external
server; the Vultr host is deployed by hand with `infra/deploy/deploy_vultr.sh` (below).

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

CI's container smoke test mounts `infra/docker/smoke/normalized` (one tiny map), checks
that `/api/brains` lists `real`, and plays one session with it through
`infra/docker/smoke_session.py`.

## Deploy on Vultr

The backend (FastAPI and the trained brains in `bundles/`) runs on a Vultr Cloud
Compute instance; the Cloudflare Worker serves the front end and proxies `/api/*` and
`/ws/*` to it (`frontend/README.md`). A CPU instance is enough: one brain takes about
6 ms per 30 Hz frame, two (Versus) about 8 ms. Suggested: 2 to 4 vCPU, 8 GB RAM,
Ubuntu, a US East region.

On the new server, once:

```bash
curl -fsSL https://get.docker.com | sh            # Docker
git clone https://github.com/jackie-0110/Fly-Saber.git && cd Fly-Saber   # or SSH with a deploy key
bash infra/deploy/deploy_vultr.sh
```

The script builds the image (trained brains included), downloads the allowlisted demo
songs into the Docker volume `fly-saber-maps` on first run, starts the container with
`--restart unless-stopped` on port 80, and waits until `/api/games` answers. Then:

1. Open TCP 80 in the Vultr firewall group (or `ufw allow 80/tcp`).
2. Give the backend a hostname: Workers may not fetch a bare IP (error 1003). We use a
   Cloudflare Tunnel (Zero Trust → Networks → Tunnels) with public hostname
   `api.<your domain>` → `http://localhost:80`; run its `cloudflared service install
   <token>` command on the server once. Error 1033 on that hostname means cloudflared is
   not running.
3. Set the `flysaber` Worker secret `BACKEND_URL=https://api.<your domain>`, with no port.
   Deployed Workers ignore non-standard ports in `fetch()`, so a backend on `:8000` is
   fetched on port 80 and the site shows "Backend disconnected".

Or run everything from your laptop, including the Worker secret and an end-to-end check:
`bash infra/deploy/go_live.sh` (defaults: server `64.177.52.103`, site
`flysaber.jackie-courtney.com`; see the header of the script for overrides).

To update: `git pull && bash infra/deploy/deploy_vultr.sh` (songs are kept). Logs:
`docker logs -f fly-saber`. Never rebuild the brain subgraph on the server; the
bundles carry the exact files the checkpoints were trained on.

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
