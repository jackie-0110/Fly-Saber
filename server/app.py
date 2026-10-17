"""FastAPI app, REST endpoints and WS /ws/session (spec 13).

Owner: Person 2. See FLY_SABER_HANDOFF.md for the spec.
"""

import json
import logging
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

log = logging.getLogger(__name__)
EXPERT = Brain(
    id="expert",
    label="Scripted expert",
    description="Perfect player that reads the map; stands in until trained brains load.",
)


def brain_modes() -> list[Brain]:
    """The demo's brain modes (Person 1, configs/brains.yaml), playable or not."""
    raw = yaml.safe_load((CONFIG_DIR / "brains.yaml").read_text())
    return [Brain(id=m["id"], label=m["label"], description=m["description"]) for m in raw["modes"]]


def serving_brains(modes: list[Brain], bundles_dir: Path) -> list[LiveBrain]:
    """The expert, plus every mode with a trained bundle at bundles_dir/<id>/
    (training/bundle.py). A bundle that fails to load is logged and left out."""
    brains = [LiveBrain(EXPERT, expert_sims())]
    for mode in modes:
        bundle = bundles_dir / mode.id
        manifest = bundle / "MANIFEST.json"
        if not manifest.is_file():
            continue
        checkpoint = json.loads(manifest.read_text())["checkpoint"]
        if checkpoint is None:  # view-only mode: wiring to show, nothing to play
            continue
        try:
            sims = brain_sims(bundle / checkpoint, bundle / "subgraph")
        except (OSError, ValueError, KeyError) as error:
            log.warning("not serving brain %s: %s", mode.id, error)
            continue
        brains.append(LiveBrain(mode, sims))
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
        brains=serving_brains(brain_modes(), Path("bundles")),
        frontend_dir=frontend if frontend.is_dir() else None,
    )
