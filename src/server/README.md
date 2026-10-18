# server/

**Owner: Person 2**, except `narrator.py` and `replay.py` (Person 3). Spec sections 9 and 13.

| Path | Owner | What |
|---|---|---|
| `app.py` | Person 2 | FastAPI app, REST endpoints, `WS /ws/session` |
| `session.py` | Person 2 | Steps env and brain at 30 Hz, Versus backend |
| `protocol.py` | Person 2 | Message types; must match `docs/reference/PROTOCOL.md` |
| `catalog.py` | Person 2 | Normalized maps and allowlisted audio on disk |
| `narrator.py` | Person 3 | Gemini narrator (optional, app works without it) |
| `replay.py` | Person 3 | Replay recording and serving (demo safety net) |

Run: `make serve` (port 8000).
