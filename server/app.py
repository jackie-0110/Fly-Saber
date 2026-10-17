"""FastAPI app, REST endpoints and WS /ws/session (spec 13).

Owner: Person 2. See FLY_SABER_HANDOFF.md for the spec.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from fly_saber.config import CONFIG_DIR
from server.catalog import Catalog
from server.protocol import Brain, Game, MapSummary
from server.saber_sim import brain_sims, expert_sims
from server.session import GAMES, LiveBrain, Session

EXPERT = Brain(
    id="expert",
    label="Scripted expert",
    description="Perfect player that reads the map; stands in until trained brains load.",
)


@dataclass(frozen=True)
class BrainSpec:
    id: str
    label: str
    description: str
    checkpoint: str


def brain_specs() -> list[BrainSpec]:
    raw = yaml.safe_load((CONFIG_DIR / "server.yaml").read_text())
    return [BrainSpec(**b) for b in raw["brains"]]


def serving_brains(specs: list[BrainSpec]) -> list[LiveBrain]:
    """The expert, plus every brain whose checkpoint exists."""
    brains = [LiveBrain(EXPERT, expert_sims())]
    for spec in specs:
        if Path(spec.checkpoint).is_file():
            info = Brain(id=spec.id, label=spec.label, description=spec.description)
            brains.append(LiveBrain(info, brain_sims(Path(spec.checkpoint))))
    return brains


def warm_up(catalog: Catalog, brains: list[LiveBrain]) -> None:
    """Compile every player's sim for every map size now; a first session would stall the
    event loop."""
    for summary in catalog.list_maps():
        map_data = catalog.get_map(summary.map_id)
        if map_data is None:
            continue
        for brain in brains:
            sim = brain.make_sim(map_data, 0)
            sim.reset(0.0)
            sim.step()


def create_app(
    maps_dir: Path = Path("data/maps"),
    brains: list[LiveBrain] | None = None,
    frontend_dir: Path | None = None,
) -> FastAPI:
    """`brains` defaults to the expert alone."""
    app = FastAPI(title="Fly Saber")
    brains = brains or [LiveBrain(EXPERT, expert_sims())]
    by_id = {b.info.id: b for b in brains}
    catalog = Catalog(maps_dir)
    warm_up(catalog, brains)

    @app.get("/api/games")
    def games() -> list[Game]:
        return GAMES

    @app.get("/api/brains")
    def brain_list() -> list[Brain]:
        return [b.info for b in brains]

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
        await Session(websocket, catalog, by_id).run()

    if frontend_dir is not None:
        app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")

    return app


def serve() -> FastAPI:
    """The production app, for `uvicorn server.app:serve --factory`."""
    frontend = Path("frontend/dist")
    return create_app(
        brains=serving_brains(brain_specs()),
        frontend_dir=frontend if frontend.is_dir() else None,
    )
