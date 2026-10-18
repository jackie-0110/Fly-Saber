#!/bin/bash
# Deploy Fly Saber's backend (FastAPI + trained brains) on a fresh Vultr server.
# Owner: Person 1. Run on the server, from the repo root, as a user that can run docker:
#   bash scripts/deploy_vultr.sh
# Safe to rerun after `git pull`: rebuilds the image and restarts the container, keeping
# the downloaded songs. Then set the Cloudflare secret BACKEND_URL=http://<server ip>, or run
# scripts/go_live.sh from your laptop to do all of this plus the Worker secret.
#
# Port 80 by default: deployed Cloudflare Workers ignore non-standard ports in fetch(), so a
# backend on :8000 is unreachable through the Worker. PORT=8000 still works for direct use.
set -euo pipefail
cd "$(dirname "$0")/.."

IMAGE=fly-saber
CONTAINER=fly-saber
SONGS=fly-saber-maps   # Docker volume for data/maps: normalized charts and licensed audio
PORT=${PORT:-80}
# Live sessions at once; past this, players see "the server is full". One brain costs
# about 6 ms per 33 ms frame, so a 2 to 4 vCPU instance handles 6 comfortably.
MAX_SESSIONS=${MAX_SESSIONS:-6}

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

echo "== starting server on port $PORT ($MAX_SESSIONS live sessions)"
docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
docker run -d --name "$CONTAINER" --restart unless-stopped -e MAX_SESSIONS="$MAX_SESSIONS" \
    -p "$PORT":8000 -v "$SONGS":/app/data/maps "$IMAGE" >/dev/null

echo "== waiting for it to answer"
for _ in $(seq 1 60); do
    if curl -fsS "http://127.0.0.1:$PORT/api/games" >/dev/null 2>&1; then
        address="http://$(curl -fsS -4 ifconfig.me 2>/dev/null || echo '<server ip>')"
        [ "$PORT" = 80 ] || address="$address:$PORT"
        echo "up: $address"
        echo "songs: $(curl -fsS "http://127.0.0.1:$PORT/api/maps" | python3 -c 'import json,sys; print(len(json.load(sys.stdin)))') playable maps"
        echo "next: set the flysaber Worker secret BACKEND_URL=$address (scripts/go_live.sh does it)"
        exit 0
    fi
    sleep 2
done
echo "server did not answer in 2 minutes; logs:" >&2
docker logs --tail 50 "$CONTAINER" >&2
exit 1
