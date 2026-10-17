"""FastAPI app, REST endpoints and WS /ws/session (spec 13).

Owner: Person 2. See FLY_SABER_HANDOFF.md for the spec.
"""

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.responses import FileResponse

from server.catalog import Catalog
from server.protocol import Brain, Game, MapSummary
from server.session import BRAINS, GAMES, FakeSaberSim, Session, SimFactory


def create_app(maps_dir: Path = Path("data/maps"), make_sim: SimFactory = FakeSaberSim) -> FastAPI:
    app = FastAPI(title="Fly Saber")
    catalog = Catalog(maps_dir)

    @app.get("/api/games")
    def games() -> list[Game]:
        return GAMES

    @app.get("/api/brains")
    def brains() -> list[Brain]:
        return BRAINS

    @app.get("/api/maps")
    def maps() -> list[MapSummary]:
        return catalog.list_maps()

    @app.get("/api/maps/{map_id}")
    def map_detail(map_id: str) -> dict[str, Any]:
        map_data = catalog.get_map(map_id)
        if map_data is None:
            raise HTTPException(404, f"unknown map {map_id}")
        return map_data

    @app.get("/api/maps/{map_id}/audio")
    def map_audio(map_id: str) -> FileResponse:
        path = catalog.audio_path(map_id)
        if path is None:
            raise HTTPException(404, f"no allowlisted audio for {map_id}")
        return FileResponse(path, media_type="audio/ogg")

    @app.websocket("/ws/session")
    async def session(websocket: WebSocket) -> None:
        await websocket.accept()
        await Session(websocket, catalog, make_sim).run()

    return app


app = create_app()
