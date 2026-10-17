# Data Notes

What the real files actually contain, and every deviation from the spec. Inspect before coding against anything (spec ground rule 2).

## MaleCNS v1.0 (Person 1)

## FlyWire (Person 1)

## Ommatidia to photoreceptor mapping (Person 2)

## Beat Saber map formats, v2 and v3 (Person 2)

Inspected 5 BeatSaver levels of Kevin MacLeod songs (keys 53af, 2dd26, 304be, 4fd1b, 4f4de): 11 Standard difficulties, difficulty formats 2.0.0, 2.2.0, 2.6.0, and 3.3.0.

- **BeatSaver API has no license field.** Licensing comes from the song itself, so demo maps are songs with a known Creative Commons license (Kevin MacLeod, CC BY 4.0), recorded in `demo_allowlist.json`. Each allowlisted song was matched by title in incompetech's catalog (`pieces.json`) and links to its incompetech page; Bathroom Ballet (4fd1b) is not in the catalog and was dropped. The audio is the mapper's upload, so its length can differ from incompetech's by a few seconds (trimmed intros or tails).
- `song.duration_sec` comes from BeatSaver's `metadata.duration`, the length of the uploaded audio.
- Every Info file seen is the v2 layout (`_version` 2.x, `_difficultyBeatmapSets`), even when the difficulty files are v3. The filename case varies (`info.dat` and `Info.dat`). Info v4 is rejected.
- Files may start with a UTF-8 BOM; read with `utf-8-sig`.
- v2 notes: `_type` 0 red, 1 blue, 3 bomb. v2 walls: `_type` 0 full height, 1 crouch; we convert to the v3 `(layer, height)` the game uses, `(0, 5)` and `(2, 3)`.
- v3 notes: `colorNotes {b, x, y, c, d, a}`, bombs in a separate `bombNotes {b, x, y}`, walls `obstacles {b, x, y, d, w, h}`. v3 writers may omit zero-valued fields, so every read defaults to 0. The angle offset `a` was 0 in every file and is ignored.
- Cut directions 0 to 8 all occur, diagonals included, so `adapt.py` must reduce them for the 4-direction sim.
- Many walls are decorative, a few hundredths of a beat long; kept in the normalized file, `adapt.py` decides.
- No inspected map has BPM changes. v3 `bpmEvents {b, m}` are supported. v2 `_BPMChanges` (top level or `_customData`) have editor-specific semantics, so maps using them are rejected rather than guessed. All `_songTimeOffset` values were 0; nonzero is rejected.
- Note jump speed and offset come from Info. Spawn distance follows the game's half-jump rule (4 beats, halved until within 18 m, plus offset, minimum 0.25 beats). An NJS of 0 falls back to the game's per-difficulty default.
- Notes outside the 4x3 grid or with cut directions above 8 (mapping-extension maps) are dropped.
