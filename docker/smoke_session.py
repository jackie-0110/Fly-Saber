"""Play the trained fly ("real") through one WS session on docker/smoke's map; exit 1 unless
it ends.

CI runs it inside the image: docker exec -i <container> python - < docker/smoke_session.py
Pass a WS URL to check a server elsewhere.
"""

import asyncio
import json
import sys

from websockets.asyncio.client import connect

URL = sys.argv[1] if len(sys.argv) > 1 else "ws://127.0.0.1:8000/ws/session"


async def play() -> None:
    async with connect(URL) as ws:
        start = {
            "type": "start",
            "game": "saber",
            "brains": ["real"],
            "map_id": "smoke1_hard",
            "seed": 0,
        }
        await ws.send(json.dumps(start))
        ready = json.loads(await ws.recv())
        if ready["type"] != "ready" or ready["n_neurons"] == 0:
            sys.exit(f"expected ready from a trained brain, got {ready}")
        await ws.send(json.dumps({"type": "play", "song_t": 0.0}))
        # Past the song end, so the server catches up without waiting on the wall clock.
        await ws.send(json.dumps({"type": "clock", "song_t": 5.0}))
        frames = 0
        while True:
            message = json.loads(await asyncio.wait_for(ws.recv(), timeout=60))
            if message["type"] == "error":
                sys.exit(f"server error: {message['message']}")
            frames += message["type"] == "frame"
            if message["type"] == "ended":
                print(
                    f"real played {frames} frames, n_neurons={ready['n_neurons']}, "
                    f"score={message['score']}"
                )
                return


asyncio.run(play())
