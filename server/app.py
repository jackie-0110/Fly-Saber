"""FastAPI app, REST endpoints and WS /ws/session (spec 13).

Owner: Person 2. See FLY_SABER_HANDOFF.md for the spec.
"""

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from fly_saber.config import load_config
from fly_saber.env import SaberConfig, SaberEnv
from fly_saber.experts import saber_expert
from fly_saber.eye import EyeConfig
from server.catalog import Catalog
from server.protocol import Brain, Game, MapSummary
from server.saber_sim import LiveSaber
from server.session import BRAINS, GAMES, Session, SimFactory


def expert_sims() -> LiveSaber:
    """The real env played by the scripted expert."""
    env = SaberEnv.create(load_config(SaberConfig, "saber"), load_config(EyeConfig, "eye"))
    return LiveSaber(env, saber_expert)


def warm_up(catalog: Catalog, make_sim: SimFactory) -> None:
    """Compile the sim for every map size now; a first session would stall the event loop."""
    for summary in catalog.list_maps():
        map_data = catalog.get_map(summary.map_id)
        if map_data is not None:
            sim = make_sim(map_data, BRAINS[0].id, 0)
            sim.reset(0.0)
            sim.step()


def create_app(
    maps_dir: Path = Path("data/maps"),
    make_sim: SimFactory | None = None,
    frontend_dir: Path | None = None,
) -> FastAPI:
    app = FastAPI(title="Fly Saber")
    make_sim = make_sim or expert_sims()
    catalog = Catalog(maps_dir)
    warm_up(catalog, make_sim)

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

    if frontend_dir is not None:
        app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")

    return app


_frontend = Path("frontend/dist")
app = create_app(frontend_dir=_frontend if _frontend.is_dir() else None)
