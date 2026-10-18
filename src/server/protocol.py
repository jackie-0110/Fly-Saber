"""WebSocket and REST message types, mirrors docs/reference/PROTOCOL.md sections 2, 3 and 6 (spec 13).

Owner: Person 2. See docs/archive/project/FLY_SABER_HANDOFF.md for the spec.
"""

import struct
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

Cell = Annotated[int, Field(ge=0, le=11)]
Cut = Annotated[int, Field(ge=0, le=3)]
Color = Annotated[int, Field(ge=0, le=1)]
Slot = Annotated[int, Field(ge=0, le=1)]
Fraction = Annotated[float, Field(ge=0.0, le=1.0)]


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# REST


class Game(Message):
    id: str
    label: str


class Brain(Message):
    id: str
    label: str
    description: str


class SongSummary(Message):
    title: str
    artist: str
    mapper: str
    license: str | None


class DifficultySummary(Message):
    name: str


class MapSummary(Message):
    map_id: str
    split: Literal["train", "heldout"]
    song: SongSummary
    difficulty: DifficultySummary
    note_count: int


# render_state


class Saber(Message):
    color: Color
    cell: Cell | None
    cut: Cut | None


class Score(Message):
    score: Annotated[int, Field(ge=0, le=100)]
    combo: int
    max_combo: int
    hits: int
    misses: int
    accuracy: Annotated[float, Field(ge=0.0, le=1.0)]


class RenderState(Message):
    t: float
    step: int
    sabers: list[Saber]
    score: Score


# Client to server


class Start(Message):
    type: Literal["start"] = "start"
    game: str
    brains: Annotated[list[str], Field(min_length=1, max_length=2)]
    map_id: str
    seed: int


class Play(Message):
    type: Literal["play"] = "play"
    song_t: float


class Clock(Message):
    type: Literal["clock"] = "clock"
    song_t: float


class Pause(Message):
    type: Literal["pause"] = "pause"
    paused: bool


class Reset(Message):
    type: Literal["reset"] = "reset"


class EyeSettings(Message):
    """Live eye controls (fly_saber.eye.EyeControls); the defaults see normally."""

    blind_left: Fraction = 0.0
    blind_right: Fraction = 0.0
    blur_deg: Annotated[float, Field(ge=0.0)] = 0.0
    facet_dropout: Fraction = 0.0
    noise: Annotated[float, Field(ge=0.0)] = 0.0
    seed: Annotated[int, Field(ge=0, lt=2**32)] = 0


class SetInterventions(Message):
    type: Literal["set_interventions"] = "set_interventions"
    slot: Slot
    interventions: dict[str, Any] = {}
    """Brain controls, PROTOCOL.md section 4; validated by brain.model.controls.from_controls."""
    eye: EyeSettings = EyeSettings()


class HumanAction(Message):
    type: Literal["human_action"] = "human_action"
    color: Color
    cell: Cell | None
    cut: Cut | None


class Narrate(Message):
    type: Literal["narrate"] = "narrate"
    slot: Slot


ClientMessage = Annotated[
    Start | Play | Clock | Pause | Reset | SetInterventions | HumanAction | Narrate,
    Field(discriminator="type"),
]
client_message_adapter: TypeAdapter[ClientMessage] = TypeAdapter(ClientMessage)


# Server to client


class SlotInfo(Message):
    slot: Slot
    brain: str


class Ready(Message):
    type: Literal["ready"] = "ready"
    session_id: str
    game: str
    map_id: str
    slots: list[SlotInfo]
    step_hz: int
    n_neurons: int
    cell_types: list[str]


class Frame(Message):
    type: Literal["frame"] = "frame"
    slot: Slot
    render_state: RenderState


class Event(Message):
    type: Literal["event"] = "event"
    slot: Slot
    t: float
    kind: Literal["hit", "miss", "bad_cut", "bomb"]
    note_id: int


class Ended(Message):
    type: Literal["ended"] = "ended"
    slot: Slot
    score: Score


class Narration(Message):
    type: Literal["narration"] = "narration"
    slot: Slot
    text: str


class Error(Message):
    type: Literal["error"] = "error"
    message: str


# Binary messages: activity (section 6) and processed eye luminance (section 3).

NEURON_ACTIVITY, TYPE_ACTIVITY, EYE_IMAGE = 1, 2, 3


def binary_message(kind: int, slot: int, step: int, payload: bytes) -> bytes:
    """Header (kind, slot, 2 reserved zero bytes, little-endian uint32 sim step), then payload."""
    return struct.pack("<BBHI", kind, slot, 0, step) + payload
