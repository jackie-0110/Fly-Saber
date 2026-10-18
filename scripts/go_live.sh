#!/bin/bash
# Put Fly Saber live end to end: backend on the Vultr server, Cloudflare Worker pointed at it.
# Run from your laptop, at the repo root:
#   bash scripts/go_live.sh
# Needs ssh access to the server, Node/npm, and a Cloudflare login (wrangler opens a browser
# the first time). Safe to rerun.
#
# Deployed Workers ignore non-standard ports in fetch(): BACKEND_URL=http://<ip>:8000 is
# fetched on port 80. So the backend is published on port 80 and the Worker gets no port.
#
# Overrides: SERVER_IP, SSH_TARGET, REMOTE_DIR, SITE, WORKER, BACKEND_URL.
# Skip steps: SKIP_SERVER=1 (server already on port 80), SKIP_WORKER=1 (secret already set).
# The server redeploys the code it already has. PULL=1 also runs `git pull` there first,
# which needs GitHub credentials on the server (set them up there, never in this script).
set -euo pipefail
cd "$(dirname "$0")/.."

SERVER_IP=${SERVER_IP:-64.177.52.103}
SSH_TARGET=${SSH_TARGET:-root@$SERVER_IP}
REMOTE_DIR=${REMOTE_DIR:-Fly-Saber}
SITE=${SITE:-https://flysaber.jackie-courtney.com}
WORKER=${WORKER:-flysaber}
# Workers cannot fetch bare IPs (Cloudflare error 1003), so the backend needs a hostname.
# api.jackie-courtney.com is a Cloudflare Tunnel to the server's port 80; cloudflared must
# be running on the server (error 1033 means it is not).
BACKEND_HOST=${BACKEND_HOST:-api.jackie-courtney.com}
BACKEND_URL=${BACKEND_URL:-https://$BACKEND_HOST}
PULL=${PULL:-0}

step() { printf '\n== %s\n' "$*"; }
fail() {
    printf '\n!! %s\n' "$*" >&2
    exit 1
}

if [ "${SKIP_SERVER:-0}" != 1 ]; then
    step "server: update and serve the backend on port 80 ($SSH_TARGET:$REMOTE_DIR)"
    # shellcheck disable=SC2087  # expand REMOTE_DIR locally on purpose
    ssh "$SSH_TARGET" bash -s <<EOF
set -euo pipefail
cd "$REMOTE_DIR"
if [ "$PULL" = 1 ]; then git pull --ff-only; fi
SUDO=\$([ "\$(id -u)" = 0 ] || echo sudo)
if command -v ufw >/dev/null && \$SUDO ufw status | grep -q 'Status: active'; then
    \$SUDO ufw allow 80/tcp >/dev/null && echo "ufw: port 80 open"
fi
PORT=80 bash scripts/deploy_vultr.sh
EOF
fi

step "check: backend answers from outside at $BACKEND_URL"
if ! curl -fsS --max-time 10 "$BACKEND_URL/api/games"; then
    curl -sS --max-time 10 "$BACKEND_URL/api/games" | grep -q 1033 &&
        fail "Tunnel not connected (error 1033): run the 'cloudflared service install <token>'
   command from Zero Trust -> Networks -> Tunnels on the server, with the public hostname
   $BACKEND_HOST -> http://localhost:80. Then rerun with SKIP_SERVER=1."
    fail "No answer at $BACKEND_URL/api/games. Check the backend answers on the server:
   curl http://$SERVER_IP/api/games (open TCP 80 in the Vultr firewall if it times out)."
fi
echo

if [ "${SKIP_WORKER:-0}" != 1 ]; then
    step "cloudflare: set $WORKER secret BACKEND_URL=$BACKEND_URL"
    (
        cd frontend
        [ -x node_modules/.bin/wrangler ] || npm ci
        npx wrangler whoami >/dev/null 2>&1 || npx wrangler login
        # Piped value: wrangler stores it without prompting and deploys a new version.
        printf '%s' "$BACKEND_URL" | npx wrangler secret put BACKEND_URL --name "$WORKER"
    )
fi

step "verify: $SITE/api/games through the Worker"
body=$(mktemp)
trap 'rm -f "$body"' EXIT
code=000
for _ in $(seq 1 12); do
    code=$(curl -sS -o "$body" -w '%{http_code}' --max-time 15 "$SITE/api/games" || true)
    [ "$code" = 200 ] && break
    sleep 5
done
echo "HTTP $code: $(cat "$body")"
case "$code" in
200) echo "live: open $SITE; the header badge should read 'Backend connected'." ;;
503) fail "The Worker still has no BACKEND_URL. Check the secret is on the production
   Worker '$WORKER' (dashboard -> Workers -> $WORKER -> Settings -> Variables and Secrets)." ;;
403) fail "Cloudflare error 1003: the Worker's BACKEND_URL is still a bare IP. Set it to
   $BACKEND_URL (dashboard -> Workers -> $WORKER -> Settings -> Variables and Secrets)." ;;
502) fail "The Worker cannot reach $BACKEND_URL even though it answers directly. Check the
   secret is exactly $BACKEND_URL, then 'npx wrangler tail $WORKER' while reloading." ;;
*) fail "Unexpected response; check 'npx wrangler tail $WORKER' while reloading the site." ;;
esac
