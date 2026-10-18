# Contributing

Use a branch for each change and open a pull request into `main`. Keep PRs focused,
explain the resulting behavior, and include validation results in the PR template.
Bug and feature forms are available under GitHub Issues.

Branch with your owner prefix (`p1/`, `p2/`, `p3/`); `main` always runs the demo.
Don't use em dashes in user-facing text or docs.

## Setup and checks

Use Python 3.11 (the default in `.python-version`), uv, and Node 22 (`nvm use`).

```bash
make setup
cd frontend
npx playwright install --with-deps chromium
cd ..
make check
```

`make check` runs Python lint, format, strict type checking, all pytest tests,
frontend lint and formatting, a TypeScript/production build, and Playwright tests.
Chromium installation is a one-time setup step, repeated when Playwright changes.

For a narrower feedback loop:

| Command | Checks |
| --- | --- |
| `make test-python` | Python unit, simulation, training, REST, and WebSocket tests |
| `make test-frontend` | Production frontend build and desktop/mobile browser tests |
| `make lint` | Python Ruff/mypy and frontend Oxlint/Prettier |
| `make fmt` | Fix Python lint/formatting and format the frontend |
| `make build` | Frontend TypeScript check and Vite production build |
| `cd frontend && npm run test:ui` | Interactive Playwright runner; run `make build` first |

Tests use small synthetic graphs/maps and local fixtures. No connectome download,
GPU, credentials, audio files, or training checkpoint is required. Browser tests
exercise the built app on desktop and mobile Chromium. Optional Google Fonts are
stubbed during these tests so external font availability cannot fail the suite.
Update the playback expectations when a live session is implemented.

## Pull request checks

The [CI workflow](.github/workflows/ci.yml) runs for every pull request and push to
`main`, and can also be dispatched manually. It checks Python 3.11 and 3.12, the
frontend, and the combined Docker image. Browser traces/screenshots, JUnit reports,
and built frontend assets are retained as Actions artifacts for 14 days.

A repository administrator can require **CI passed** in the `main` branch ruleset
after this workflow has run once. This PR adds the check but does not change
repository settings. The final check also fails when a prerequisite fails or is
skipped, so a failed build cannot look successful.

Actions are pinned to commit SHAs and PR checks have read-only repository access.
Dependabot proposes weekly updates for Actions, npm, uv, and Docker dependencies.
Review and merge those updates through the same tests. Never commit `.env`, raw
connectomes, checkpoints, replays, or unlicensed audio.

## Agent tooling

The installed Impeccable skills and editor hooks are documented in
[Agent tooling](docs/AGENT_TOOLING.md), including engine setup and local cache handling.

## Container delivery

See [docs/CI_CD.md](docs/CI_CD.md) for image builds, release publishing, and deployment.
