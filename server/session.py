"""Live session stepping env and brain at 30 Hz (spec 13).

Owner: Person 2. See FLYSABER_HANDOFF.md for the spec.
"""

import asyncio
import uuid
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from fastapi import WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from server.catalog import Catalog
from server.protocol import (
    Brain,
    ClientMessage,
    Clock,
    Ended,
    Error,
    Event,
    Frame,
    Game,
    Message,
    Pause,
    Play,
    Ready,
    RenderState,
    Reset,
    SlotInfo,
    Start,
    client_message_adapter,
)

STEP_HZ = 30
GAMES = [Game(id="saber", label="Fly Saber")]
BRAINS = [
    Brain(
        id="expert",
        label="Scripted expert",
        description="Perfect player that reads the map; stands in until trained brains load.",
    ),
]


@dataclass(frozen=True)
class NoteEvent:
    t: float
    kind: Literal["hit", "miss", "bad_cut", "bomb"]
    note_id: int


@dataclass(frozen=True)
class StepOutput:
    render_state: RenderState
    events: list[NoteEvent]
    done: bool


class Sim(Protocol):
    """One env plus brain, stepped at STEP_HZ from a song time."""

    def reset(self, song_t: float) -> RenderState: ...

    def step(self) -> StepOutput: ...


SimFactory = Callable[[dict[str, Any], str, int], Sim]


@dataclass
class SlotRun:
    sim: Sim
    render_state: RenderState
    done: bool = False


class Session:
    """One WebSocket: start, play, clock, pause, reset; one or two slots (Versus)."""

    def __init__(self, websocket: WebSocket, catalog: Catalog, make_sim: SimFactory) -> None:
        self.websocket = websocket
        self.catalog = catalog
        self.make_sim = make_sim
        self.session_id = uuid.uuid4().hex
        self.start: Start | None = None
        self.slots: list[SlotRun] = []
        self.ticker: asyncio.Task[None] | None = None
        self.anchor_song_t = 0.0
        self.anchor_wall_t = 0.0
        self.paused = False

    async def run(self) -> None:
        try:
            while True:
                raw = await self.websocket.receive_text()
                try:
                    message = client_message_adapter.validate_json(raw)
                except ValidationError as error:
                    await self.send(Error(message=str(error)))
                    continue
                await self.handle(message)
        except WebSocketDisconnect:
            pass
        finally:
            await self.stop_ticker()

    async def handle(self, message: ClientMessage) -> None:
        match message:
            case Start():
                await self.load(message)
            case Reset() if self.start:
                await self.load(self.start)
            case Play() if self.start:
                await self.play(message.song_t)
            case Clock():
                self.anchor_song_t, self.anchor_wall_t = message.song_t, self.now()
            case Pause():
                self.anchor_song_t, self.anchor_wall_t = self.song_t(), self.now()
                self.paused = message.paused
            case Reset() | Play():
                await self.send(Error(message="send start first"))
            case _:
                await self.send(Error(message=f"{message.type} is not implemented yet"))

    async def load(self, start: Start) -> None:
        await self.stop_ticker()
        map_data = self.catalog.get_map(start.map_id)
        if start.game not in {g.id for g in GAMES}:
            await self.send(Error(message=f"unknown game {start.game}"))
            return
        if not set(start.brains) <= {b.id for b in BRAINS}:
            await self.send(Error(message=f"unknown brain in {start.brains}"))
            return
        if map_data is None:
            await self.send(Error(message=f"unknown map {start.map_id}"))
            return
        self.start = start
        self.slots = []
        for brain in start.brains:
            sim = self.make_sim(map_data, brain, start.seed)
            self.slots.append(SlotRun(sim, sim.reset(0.0)))
        await self.send(
            Ready(
                session_id=self.session_id,
                game=start.game,
                map_id=start.map_id,
                slots=[SlotInfo(slot=i, brain=b) for i, b in enumerate(start.brains)],
                step_hz=STEP_HZ,
                n_neurons=0,
                cell_types=[],
            )
        )

    async def play(self, song_t: float) -> None:
        await self.stop_ticker()
        for run in self.slots:
            run.render_state, run.done = run.sim.reset(song_t), False
        self.anchor_song_t, self.anchor_wall_t = song_t, self.now()
        self.paused = False
        self.ticker = asyncio.create_task(self.tick())

    async def tick(self) -> None:
        """Step each slot up to the client's song clock, then send its latest frame."""
        next_tick = self.now()
        while not all(run.done for run in self.slots):
            next_tick += 1 / STEP_HZ
            await asyncio.sleep(max(0.0, next_tick - self.now()))
            target = self.song_t()
            for slot, run in enumerate(self.slots):
                stepped = False
                # Nearest step, not floor: ticks land on step boundaries and jitter either way.
                while not run.done and run.render_state.t + 0.5 / STEP_HZ <= target:
                    out = run.sim.step()
                    run.render_state, run.done, stepped = out.render_state, out.done, True
                    for e in out.events:
                        await self.send(Event(slot=slot, t=e.t, kind=e.kind, note_id=e.note_id))
                if stepped:
                    await self.send(Frame(slot=slot, render_state=run.render_state))
                    if run.done:
                        await self.send(Ended(slot=slot, score=run.render_state.score))

    async def stop_ticker(self) -> None:
        if self.ticker is None:
            return
        self.ticker.cancel()
        with suppress(asyncio.CancelledError, WebSocketDisconnect):
            await self.ticker
        self.ticker = None

    def song_t(self) -> float:
        if self.paused:
            return self.anchor_song_t
        return self.anchor_song_t + self.now() - self.anchor_wall_t

    def now(self) -> float:
        return asyncio.get_running_loop().time()

    async def send(self, message: Message) -> None:
        await self.websocket.send_text(message.model_dump_json())
