"""Live session stepping env and brain at 30 Hz (spec 13).

Owner: Person 2. See docs/archive/project/FLY_SABER_HANDOFF.md for the spec.
"""

import asyncio
import uuid
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Literal, Protocol, TypeVar

from fastapi import WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from server.catalog import Catalog
from server.protocol import (
    EYE_IMAGE,
    NEURON_ACTIVITY,
    TYPE_ACTIVITY,
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
    SetInterventions,
    SlotInfo,
    Start,
    binary_message,
    client_message_adapter,
)

STEP_HZ = 30
T = TypeVar("T")
GAMES = [Game(id="saber", label="Fly Saber")]


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

    def set_interventions(self, controls: Mapping[str, Any], eye: Mapping[str, Any]) -> None: ...

    def activity(self) -> tuple[bytes, bytes] | None:
        """The last step's per-neuron and per-cell-type payloads (PROTOCOL.md section 6);
        None for a player without a brain."""
        ...

    def eye_image(self) -> bytes | None:
        """Left/right uint16 counts followed by processed uint8 luminance; None without a brain."""
        ...


class SimFactory(Protocol):
    """Makes one player's sims; n_neurons is 0 for players without a brain."""

    n_neurons: int
    cell_types: list[str]

    def __call__(self, map_data: dict[str, Any], seed: int) -> Sim: ...

    def eye_lattice(self) -> dict[str, list[list[float]]]: ...


@dataclass(frozen=True)
class LiveBrain:
    info: Brain
    make_sim: SimFactory


@dataclass
class SlotRun:
    sim: Sim
    render_state: RenderState
    done: bool = False
    neurons_sent_step: int | None = None
    eye_sent_step: int | None = None


class Session:
    """One WebSocket: start, play, clock, pause, reset; one or two slots (Versus)."""

    def __init__(
        self,
        websocket: WebSocket,
        catalog: Catalog,
        brains: Mapping[str, LiveBrain],
        seats: asyncio.Semaphore,
    ) -> None:
        """`seats` is shared by all sessions: a session holds one from its first `start`
        until it disconnects, so at most that many sessions run sims at once."""
        self.websocket = websocket
        self.catalog = catalog
        self.brains = brains
        self.seats = seats
        self.seated = False
        # One thread per session: its sim work runs in order, off the event loop, and
        # in parallel with other sessions (JAX releases the GIL while it computes).
        self.worker = ThreadPoolExecutor(max_workers=1)
        self.session_id = uuid.uuid4().hex
        self.start: Start | None = None
        self.slots: list[SlotRun] = []
        self.interventions: dict[int, SetInterventions] = {}
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
            self.worker.shutdown(wait=False, cancel_futures=True)
            if self.seated:
                self.seats.release()

    async def handle(self, message: ClientMessage) -> None:
        match message:
            case Start():
                self.interventions = {}
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
            case SetInterventions() if self.start:
                await self.set_interventions(message)
            case Reset() | Play() | SetInterventions():
                await self.send(Error(message="send start first"))
            case _:
                await self.send(Error(message=f"{message.type} is not implemented yet"))

    async def load(self, start: Start) -> None:
        await self.stop_ticker()
        map_data = self.catalog.get_map(start.map_id)
        if start.game not in {g.id for g in GAMES}:
            await self.send(Error(message=f"unknown game {start.game}"))
            return
        if not set(start.brains) <= set(self.brains):
            await self.send(Error(message=f"unknown brain in {start.brains}"))
            return
        if map_data is None:
            await self.send(Error(message=f"unknown map {start.map_id}"))
            return
        if not self.seated:
            if self.seats.locked():
                await self.send(Error(message="the server is full; try again in a minute"))
                return
            await self.seats.acquire()
            self.seated = True
        factories = [self.brains[b].make_sim for b in start.brains]
        try:
            self.slots = await self.on_worker(self.make_slots, factories, map_data, start.seed)
        except ValueError as error:  # a map the sim cannot play, e.g. too many notes
            await self.send(Error(message=str(error)))
            return
        self.start = start
        # Versus pairs brains on one subgraph, or a brain with the brainless expert.
        neural = max(factories, key=lambda f: f.n_neurons)
        await self.send(
            Ready(
                session_id=self.session_id,
                game=start.game,
                map_id=start.map_id,
                slots=[SlotInfo(slot=i, brain=b) for i, b in enumerate(start.brains)],
                step_hz=STEP_HZ,
                n_neurons=neural.n_neurons,
                cell_types=neural.cell_types,
            )
        )

    async def set_interventions(self, message: SetInterventions) -> None:
        """Replace one slot's brain and eye controls, live (PROTOCOL.md section 3)."""
        if message.slot >= len(self.slots):
            await self.send(Error(message=f"no slot {message.slot} in this session"))
            return
        sim = self.slots[message.slot].sim
        try:
            await self.on_worker(
                sim.set_interventions, message.interventions, message.eye.model_dump()
            )
        except ValueError as error:
            await self.send(Error(message=str(error)))
            return
        self.interventions[message.slot] = message

    def make_slots(
        self, factories: list[SimFactory], map_data: dict[str, Any], seed: int
    ) -> list[SlotRun]:
        slots = []
        for make_sim in factories:
            sim = make_sim(map_data, seed)
            slots.append(SlotRun(sim, sim.reset(0.0)))
        for slot, message in self.interventions.items():  # kept through reset
            slots[slot].sim.set_interventions(message.interventions, message.eye.model_dump())
        return slots

    def reset_slots(self, song_t: float) -> None:
        for run in self.slots:
            run.render_state, run.done = run.sim.reset(song_t), False
            run.neurons_sent_step = run.eye_sent_step = None

    async def play(self, song_t: float) -> None:
        await self.stop_ticker()
        await self.on_worker(self.reset_slots, song_t)
        self.anchor_song_t, self.anchor_wall_t = song_t, self.now()
        self.paused = False
        self.ticker = asyncio.create_task(self.tick())

    async def tick(self) -> None:
        """Step each slot up to the client's song clock, then send its latest frame."""
        next_tick = self.now()
        while not all(run.done for run in self.slots):
            next_tick += 1 / STEP_HZ
            await asyncio.sleep(max(0.0, next_tick - self.now()))
            for message in await self.on_worker(self.advance, self.song_t()):
                await self.send(message)

    def advance(self, target: float) -> list[Message | bytes]:
        """Step each slot up to song time `target`; returns the messages to send."""
        messages: list[Message | bytes] = []
        for slot, run in enumerate(self.slots):
            stepped = False
            # Nearest step, not floor: ticks land on step boundaries and jitter either way.
            while not run.done and run.render_state.t + 0.5 / STEP_HZ <= target:
                out = run.sim.step()
                run.render_state, run.done, stepped = out.render_state, out.done, True
                messages += [
                    Event(slot=slot, t=e.t, kind=e.kind, note_id=e.note_id) for e in out.events
                ]
            if stepped:
                messages.append(Frame(slot=slot, render_state=run.render_state))
                messages += activity(slot, run)
                step, last_eye = run.render_state.step, run.eye_sent_step
                if last_eye is None or not 0 < step - last_eye < 2:
                    image = run.sim.eye_image()
                    if image is not None:
                        messages.append(binary_message(EYE_IMAGE, slot, step, image))
                        run.eye_sent_step = step
                if run.done:
                    messages.append(Ended(slot=slot, score=run.render_state.score))
        return messages

    async def stop_ticker(self) -> None:
        if self.ticker is None:
            return
        self.ticker.cancel()
        with suppress(asyncio.CancelledError, WebSocketDisconnect):
            await self.ticker
        self.ticker = None

    async def on_worker(self, fn: Callable[..., T], *args: Any) -> T:
        """Run sim work on this session's thread. A cancelled caller does not stop it, but
        later work still waits for it, so sims never change under a running step."""
        return await asyncio.get_running_loop().run_in_executor(self.worker, fn, *args)

    def song_t(self) -> float:
        if self.paused:
            return self.anchor_song_t
        return self.anchor_song_t + self.now() - self.anchor_wall_t

    def now(self) -> float:
        return asyncio.get_running_loop().time()

    async def send(self, message: Message | bytes) -> None:
        if isinstance(message, bytes):
            await self.websocket.send_bytes(message)
        else:
            await self.websocket.send_text(message.model_dump_json())


def activity(slot: int, run: SlotRun) -> list[bytes]:
    """Binary activity for the slot's latest step: cell-type means with every frame (30 Hz),
    every neuron at most every other step (15 Hz)."""
    payloads = run.sim.activity()
    if payloads is None:
        return []
    neurons, types = payloads
    step = run.render_state.step
    messages = [binary_message(TYPE_ACTIVITY, slot, step, types)]
    last = run.neurons_sent_step
    if last is None or not 0 < step - last < 2:  # a seek can move the step back
        run.neurons_sent_step = step
        messages.append(binary_message(NEURON_ACTIVITY, slot, step, neurons))
    return messages
