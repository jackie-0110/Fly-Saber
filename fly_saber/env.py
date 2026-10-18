"""Fly Saber env, pure-functional JAX (spec 7.2).

Owner: Person 2. See FLY_SABER_HANDOFF.md for the spec.

Actions are (n_sabers, 2) ints: [cell, cut]. Cell is layer * 4 + lane (0 to 11)
or IDLE_CELL; cut is a sim direction 0 to 3 or NO_CUT. Saber 0 is red, 1 blue.
"""

import dataclasses
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from fly_saber.eye import (
    EyeConfig,
    EyeControls,
    Patches,
    apply_eye_controls,
    eye_controls,
    lattice,
    render,
)
from fly_saber.maps.adapt import ANY_DIRECTION, NoteArrays

N_LANES = 4
N_LAYERS = 3
IDLE_CELL = N_LANES * N_LAYERS
NO_CUT = 4

# Indexed by sim direction (fly_saber.maps.adapt): 0 up, 1 down, 2 left, 3 right, 4 any.
# Unit vector toward the side of the block where each cut starts; that half is drawn bright.
CUT_START_SIDE = jnp.array([[0.0, -1.0], [0.0, 1.0], [1.0, 0.0], [-1.0, 0.0], [0.0, 0.0]])


class Event(IntEnum):
    NONE = 0
    HIT = 1
    MISS = 2
    BAD_CUT = 3
    BOMB = 4


@dataclass(frozen=True)
class SaberConfig:
    lane_spacing_m: float
    layer_spacing_m: float
    block_size_m: float
    eye_distance_m: float
    bright_luminance: float
    dim_luminance: float
    bomb_luminance: float
    red_chroma: float
    blue_chroma: float
    step_hz: int
    note_speed_mps: float
    timing_window_sec: float
    spawn_distance_m: float
    max_visible_notes: int
    two_sabers: bool
    score_direction: bool
    reward_hit: float
    reward_miss: float
    reward_bad_cut: float
    reward_bomb: float
    reward_empty_cut: float

    @property
    def n_sabers(self) -> int:
        return 2 if self.two_sabers else 1


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SaberState:
    notes: NoteArrays
    judged: jax.Array
    """(N,) hit, cut wrong, bombed, or passed."""
    removed: jax.Array
    """(N,) cut by a saber, so no longer drawn."""
    start_sec: jax.Array
    end_sec: jax.Array
    speed_mps: jax.Array
    step: jax.Array
    saber_cell: jax.Array
    saber_cut: jax.Array
    hits: jax.Array
    misses: jax.Array
    bad_cuts: jax.Array
    bombs: jax.Array
    combo: jax.Array
    max_combo: jax.Array
    eye: EyeControls
    """Replace live with `set_eye_controls`."""


def set_eye_controls(state: SaberState, **values: Any) -> SaberState:
    """`state` with the named `EyeControls` fields replaced; takes effect from the next frame."""
    return dataclasses.replace(state, eye=eye_controls(state.eye, **values))


def block_patches(
    lane: jax.Array,
    layer: jax.Array,
    distance_m: jax.Array,
    direction: jax.Array,
    is_bomb: jax.Array,
    cfg: SaberConfig,
) -> Patches:
    """Two patches per note (bright and dim halves), (2 * notes,) rows.

    `distance_m` is how far in front of the hit plane each note is. Bombs are
    one dark full-size patch; their second patch has zero size.
    """
    half = cfg.block_size_m / 2.0
    center = jnp.stack(
        [
            (lane - (N_LANES - 1) / 2.0) * cfg.lane_spacing_m,
            (layer - (N_LAYERS - 1) / 2.0) * cfg.layer_spacing_m,
            distance_m + cfg.eye_distance_m,
        ],
        axis=-1,
    )
    side = jnp.where(is_bomb[:, None], 0.0, CUT_START_SIDE[direction])
    offset = jnp.concatenate([side * half / 2.0, jnp.zeros_like(distance_m)[:, None]], axis=-1)
    bright_half_size = half - jnp.abs(side) * half / 2.0
    dim_half_size = bright_half_size * jnp.abs(side).sum(axis=-1, keepdims=True)
    bright = jnp.where(is_bomb, cfg.bomb_luminance, cfg.bright_luminance)
    return Patches(
        center=jnp.concatenate([center + offset, center - offset]),
        half_size=jnp.concatenate([bright_half_size, dim_half_size]),
        luminance=jnp.concatenate([bright, jnp.full_like(bright, cfg.dim_luminance)]),
    )


@dataclass(frozen=True)
class SaberEnv:
    """Gymnax-style env: `reset` and `step` are pure and safe to jit and vmap."""

    cfg: SaberConfig
    eye_cfg: EyeConfig
    directions: jax.Array
    left_eye: jax.Array
    """(n,) bool, which eye each direction belongs to."""

    @classmethod
    def create(
        cls,
        cfg: SaberConfig,
        eye_cfg: EyeConfig,
        directions: np.ndarray | None = None,
        left_eye: np.ndarray | None = None,
    ) -> "SaberEnv":
        """`directions` are the eye's (n, 2) viewing angles; the hex lattice by default.

        Pass the brain's eye column angles and sides (training/brain_policy.py) so
        observations line up with the brain's eye input. The side is not the sign of
        the azimuth: each eye sees `binocular_overlap_deg` past the midline.
        """
        if (directions is None) != (left_eye is None):
            raise ValueError("pass directions and left_eye together")
        if directions is None or left_eye is None:
            directions, n_left = lattice(eye_cfg)
            left_eye = np.arange(len(directions)) < n_left
        return cls(
            cfg,
            eye_cfg,
            jnp.asarray(directions, dtype=jnp.float32),
            jnp.asarray(left_eye, dtype=bool),
        )

    def time(self, state: SaberState) -> jax.Array:
        return state.start_sec + state.step / self.cfg.step_hz

    def reset(
        self,
        notes: NoteArrays,
        speed_mps: jax.Array,
        start_sec: jax.Array,
        end_sec: jax.Array,
        eye: EyeControls | None = None,
    ) -> tuple[jax.Array, SaberState]:
        """Start at `start_sec` with notes outside [start_sec, end_sec) already judged.

        The episode ends two timing windows after the last note in play, which can be
        up to that much past `end_sec`, so every note in play is judged.
        """
        if notes.time_sec.shape[-1] < self.cfg.max_visible_notes:
            raise ValueError("pad notes to at least max_visible_notes rows")
        window = self.cfg.timing_window_sec
        in_play = notes.valid & (notes.time_sec >= start_sec) & (notes.time_sec < end_sec)
        last_note = jnp.max(jnp.where(in_play, notes.time_sec, start_sec))
        zero = jnp.zeros((), jnp.int32)
        state = SaberState(
            notes=notes,
            judged=~in_play,
            removed=~in_play,
            start_sec=jnp.asarray(start_sec, jnp.float32),
            end_sec=last_note + 2 * window,
            speed_mps=jnp.asarray(speed_mps, jnp.float32),
            step=zero,
            saber_cell=jnp.full((self.cfg.n_sabers,), IDLE_CELL, jnp.int32),
            saber_cut=jnp.full((self.cfg.n_sabers,), NO_CUT, jnp.int32),
            hits=zero,
            misses=zero,
            bad_cuts=zero,
            bombs=zero,
            combo=zero,
            max_combo=zero,
            eye=eye_controls() if eye is None else eye,
        )
        return self.observe(state), state

    def step(
        self, state: SaberState, action: jax.Array
    ) -> tuple[jax.Array, SaberState, jax.Array, jax.Array, jax.Array]:
        """Judge cuts at the current time, then advance one frame.

        Returns (observation, state, reward, done, events), where events is (N,)
        `Event` codes for notes judged this step.
        """
        cfg, notes = self.cfg, state.notes
        t = self.time(state)
        cell, cut = action[:, 0], action[:, 1]
        cutting = (cut != NO_CUT) & (cell != IDLE_CELL)

        note_cell = notes.layer * N_LANES + notes.lane
        open_notes = ~state.judged & (jnp.abs(notes.time_sec - t) <= cfg.timing_window_sec)
        touches = open_notes[:, None] & (note_cell[:, None] == cell[None, :]) & cutting[None, :]

        saber_color = jnp.arange(cfg.n_sabers)
        color_ok = (notes.color[:, None] == saber_color[None, :]) | (not cfg.two_sabers)
        direction_ok = (
            (notes.direction[:, None] == cut[None, :])
            | (notes.direction[:, None] == ANY_DIRECTION)
            | (not cfg.score_direction)
        )
        bomb = (touches & notes.is_bomb[:, None]).any(axis=1)
        good = touches & ~notes.is_bomb[:, None] & color_ok & direction_ok
        hit = good.any(axis=1)
        bad = (touches & ~notes.is_bomb[:, None]).any(axis=1) & ~hit
        passed = ~state.judged & ~touches.any(axis=1)
        passed &= notes.time_sec < t - cfg.timing_window_sec
        miss = passed & ~notes.is_bomb
        empty_cut = cutting & ~touches.any(axis=0)

        n_hit, n_miss, n_bad, n_bomb = (x.sum(dtype=jnp.int32) for x in (hit, miss, bad, bomb))
        failed = (n_miss + n_bad + n_bomb) > 0
        combo = jnp.where(failed, 0, state.combo + n_hit)
        reward = (
            cfg.reward_hit * n_hit
            + cfg.reward_miss * n_miss
            + cfg.reward_bad_cut * n_bad
            + cfg.reward_bomb * n_bomb
            + cfg.reward_empty_cut * empty_cut.sum()
        )
        events = jnp.select(
            [hit, miss, bad, bomb], [Event.HIT, Event.MISS, Event.BAD_CUT, Event.BOMB], Event.NONE
        ).astype(jnp.int8)

        state = dataclasses.replace(
            state,
            judged=state.judged | hit | bad | bomb | passed,
            removed=state.removed | hit | bad | bomb,
            step=state.step + 1,
            saber_cell=cell,
            saber_cut=jnp.where(cutting, cut, NO_CUT),
            hits=state.hits + n_hit,
            misses=state.misses + n_miss,
            bad_cuts=state.bad_cuts + n_bad,
            bombs=state.bombs + n_bomb,
            combo=combo,
            max_combo=jnp.maximum(state.max_combo, combo),
        )
        done = self.time(state) > state.end_sec
        return self.observe(state), state, reward, done, events

    def observe(self, state: SaberState) -> jax.Array:
        """(2, n_ommatidia): channel 0 luminance for R1-R6, channel 1 color for R7/R8."""
        cfg, notes = self.cfg, state.notes
        t = self.time(state)
        first = jnp.searchsorted(notes.time_sec, t - cfg.eye_distance_m / state.speed_mps)

        def window(x: jax.Array) -> jax.Array:
            return jax.lax.dynamic_slice_in_dim(x, first, cfg.max_visible_notes)

        distance = (window(notes.time_sec) - t) * state.speed_mps
        is_bomb = window(notes.is_bomb)
        visible = (
            window(notes.valid)
            & ~window(state.removed)
            & (distance <= cfg.spawn_distance_m)
            & (distance + cfg.eye_distance_m > cfg.block_size_m / 2.0)
        )
        patches = block_patches(
            window(notes.lane), window(notes.layer), distance, window(notes.direction), is_bomb, cfg
        )
        background = self.eye_cfg.background_luminance
        shown = jnp.concatenate([visible, visible])
        chroma = jnp.select(
            [window(notes.color) == 0, window(notes.color) == 1],
            [cfg.red_chroma, cfg.blue_chroma],
            background,
        )
        luminance = dataclasses.replace(
            patches, luminance=jnp.where(shown, patches.luminance, background)
        )
        color = dataclasses.replace(
            patches, luminance=jnp.where(shown, jnp.concatenate([chroma, chroma]), background)
        )
        blur = state.eye.blur_deg
        image = jnp.stack(
            [
                render(self.directions, luminance, self.eye_cfg, blur),
                render(self.directions, color, self.eye_cfg, blur),
            ]
        )
        return apply_eye_controls(image, state.eye, self.left_eye, state.step, background)

    def render_state(self, state: SaberState) -> dict[str, Any]:
        """The `render_state` contract (docs/PROTOCOL.md section 2) for one unbatched state."""
        judged = int(state.hits + state.misses + state.bad_cuts)
        accuracy = int(state.hits) / judged if judged else 1.0
        t = float(self.time(state))
        return {
            "t": round(t, 4),
            "step": round(t * self.cfg.step_hz),
            "sabers": [
                {
                    "color": color,
                    "cell": None if int(cell) == IDLE_CELL else int(cell),
                    "cut": None if int(cut) == NO_CUT else int(cut),
                }
                for color, (cell, cut) in enumerate(
                    zip(state.saber_cell, state.saber_cut, strict=True)
                )
            ],
            "score": {
                "score": round(100 * accuracy),
                "combo": int(state.combo),
                "max_combo": int(state.max_combo),
                "hits": int(state.hits),
                "misses": int(state.misses),
                "accuracy": round(accuracy, 4),
            },
        }
