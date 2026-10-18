# syntax=docker/dockerfile:1
FROM node:25-bookworm-slim AS frontend
WORKDIR /frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.14-slim-bookworm AS python-builder
COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY brain/ brain/
COPY fly_saber/ fly_saber/
COPY training/ training/
COPY server/ server/
RUN uv sync --locked --no-dev

FROM python:3.14-slim-bookworm AS runtime
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    JAX_PLATFORMS=cpu
WORKDIR /app
RUN groupadd --gid 10001 fly_saber \
    && useradd --uid 10001 --gid fly_saber --create-home fly_saber
COPY --from=python-builder /app/.venv /app/.venv
COPY brain/ brain/
COPY fly_saber/ fly_saber/
COPY training/ training/
COPY server/ server/
COPY configs/ configs/
# Trained flies with the exact subgraphs they were trained on (never rebuilt here).
COPY bundles/ bundles/
# Songs: the allowlist is in the image; maps and audio are downloaded into a volume on
# the host with scripts/fetch_maps.py (scripts/deploy_vultr.sh), so data/maps is writable.
COPY scripts/fetch_maps.py scripts/fetch_maps.py
COPY data/maps/demo_allowlist.json data/maps/demo_allowlist.json
RUN chown -R fly_saber:fly_saber data
COPY --from=frontend /frontend/dist frontend/dist/
USER fly_saber
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/games', timeout=3)"]
CMD ["uvicorn", "server.app:serve", "--factory", "--host", "0.0.0.0", "--port", "8000"]
