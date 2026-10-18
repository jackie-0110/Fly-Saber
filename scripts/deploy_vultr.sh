#!/bin/bash
# Deploy Fly Saber's backend (FastAPI + trained brains) on a fresh Vultr server.
# Owner: Person 1. Run on the server, from the repo root, as a user that can run docker:
#   bash scripts/deploy_vultr.sh
# Safe to rerun after `git pull`: rebuilds the image and restarts the container, keeping
# the downloaded songs. Then set the Cloudflare secret BACKEND_URL=http://<server ip>:8000.
set -euo pipefail
cd "$(dirname "$0")/.."

IMAGE=fly-saber
CONTAINER=fly-saber
SONGS=fly-saber-maps   # Docker volume for data/maps: normalized charts and licensed audio
PORT=${PORT:-8000}

echo "== building image (a few minutes the first time)"
docker build -f docker/Dockerfile -t "$IMAGE" .

echo "== songs"
docker volume create "$SONGS" >/dev/null
if docker run --rm -v "$SONGS":/app/data/maps "$IMAGE" \
    sh -c 'ls data/maps/normalized/*.json >/dev/null 2>&1'; then
    echo "already downloaded"
else
    # Allowlisted demo songs (Creative Commons audio) from BeatSaver, normalized for the server.
    docker run --rm -v "$SONGS":/app/data/maps "$IMAGE" python scripts/fetch_maps.py
fi

echo "== starting server on port $PORT"
docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
docker run -d --name "$CONTAINER" --restart unless-stopped \
    -p "$PORT":8000 -v "$SONGS":/app/data/maps "$IMAGE" >/dev/null

echo "== waiting for it to answer"
for _ in $(seq 1 60); do
    if curl -fsS "http://127.0.0.1:$PORT/api/games" >/dev/null 2>&1; then
        echo "up: http://$(curl -fsS -4 ifconfig.me 2>/dev/null || echo '<server ip>'):$PORT"
        echo "songs: $(curl -fsS "http://127.0.0.1:$PORT/api/maps" | python3 -c 'import json,sys; print(len(json.load(sys.stdin)))') playable maps"
        echo "next: in Cloudflare, set the flysaber Worker secret BACKEND_URL to that address"
        exit 0
    fi
    sleep 2
done
echo "server did not answer in 2 minutes; logs:" >&2
docker logs --tail 50 "$CONTAINER" >&2
exit 1
