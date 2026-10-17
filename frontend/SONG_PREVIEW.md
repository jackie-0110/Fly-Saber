# Song browser and 3D chart preview

The frontend renders existing song charts and plays audio without a fly session. This is also the presentation foundation for real-time inference: the audio clock supplies song time while inference will supply saber actions and judged outcomes. A fast inference loop does not require a different chart renderer.

## User flow

1. Choose a song from the header or open the Songs sidebar tab.
2. Search community maps, paste a map link/ID, or import a map ZIP.
3. Select a difficulty, play the track, and scrub the note-density timeline.
4. Orbit, pan, zoom, or choose a camera preset independently of playback.

Song information appears above the viewport. Audio starts after a user gesture. Selecting a new song pauses the old one; a failed or cancelled load retains the previous song. Difficulty changes preserve song time and playback. Restart pauses and returns to zero. Pointer scrubbing temporarily pauses audio and resumes only if it was playing before the drag.

There is no allowlist filter in the browser. Community downloads and local imports stay in memory; song files are not bundled, uploaded to the app server, or retained across refreshes. Plain audio files are rejected because they do not describe blocks. Automatic chart generation is outside this feature.

## Supported charts

- Info.dat v2 with beatmap v2/v3; Info.dat v4 with indexed beatmap data and audio sample-region timing.
- Standard, OneSaber, NoArrows, and Lawless four-lane charts. Rotation modes are not flattened into an inaccurate preview.
- Four lanes, three note layers, nine cut directions, bombs, and walls. Omitted v3/v4 fields default to zero.
- BPM changes in v3 charts and audio timing regions in v4 maps. Initial BPM in the header is metadata, not the playback timebase.
- A malformed difficulty is omitted with a compatibility note when other difficulties still load. Off-grid objects are counted and reported.

Arc guides, chain slices, custom extension behavior, and environment/lightshow events are not rendered. Compatibility notes appear for arcs/chains and declared map requirements. Legacy nonzero song offsets and editor-specific v2 BPM changes are rejected rather than assigning wrong timestamps, matching the Python pipeline's restrictions.

ZIPs and individual expanded files are capped at 64 MB, total expanded bytes at 192 MB, entries at 5,000, chart notes at 100,000, walls at 20,000, and decoded audio duration at 30 minutes. ZIP parsing uses fflate's asynchronous API. Files resolve inside the archive, case-insensitively, relative to its single Info.dat root. Covers are optional raster images. Community asset URLs must use HTTPS on the provider's CDN.

## Code and timebase

| File | Responsibility |
| --- | --- |
| `src/game/chart.ts` | Typed chart, tempo conversion, sample timing, note normalization, binary search |
| `src/game/songs.ts` | Community API, streamed downloads, bounded ZIP extraction, metadata and chart import |
| `src/game/transport.ts` | Web Audio playback, pause, seek, rate, volume, clock snapshots |
| `src/game/store.ts` | Cancellable song loading, selected difficulty, UI state |
| `src/game/game-scene.tsx` | Instanced notes/walls, track, neutral fly mesh, independent camera |
| `src/components/song-library.tsx` | Search, pagination, map imports, difficulty selection |
| `src/components/song-header.tsx` | Current song metadata |
| `src/components/session-timeline.tsx` | Playback, seek, density, time, volume |

Note `time_sec` is the instant it crosses the hit plane. Notes stay sorted and IDs are their sorted indices, matching `docs/PROTOCOL.md`. Standard community map IDs retain `{key}_{difficulty}`; other characteristics include their mode to avoid collisions.

The display looks along negative Z, with the hit plane at Z = 0:

```text
x = (lane - 1.5) * 1.2
y = layer * 1.1 + 0.45
z = -(note.time_sec - songTime) * note_speed_mps
```

Each frame reads `transport.getTime()` without integrating frame deltas. A seek reconstructs the visible note window using binary search, including when seeking backward. The audio buffer is reused; a fresh AudioBufferSourceNode starts at the chosen offset. Pause/seek/load invalidate pending asynchronous play attempts so a late audio resume cannot restart an old track.

React receives timeline snapshots around 15 Hz; the scene reads the audio clock at render rate. Paused scenes render on demand, including camera changes and seeks. Device pixel ratio, note/wall instances, and archive sizes are bounded. GPU resources are disposed on unmount. The fly and sabers stay in a neutral pose without inferred actions, hits, or fabricated scores.

Playback speed changes Web Audio's buffer rate, including pitch, for community previews. Live server songs stay at 1× because the protocol does not negotiate speed.

## Live sessions

Live songs load a normalized chart from `GET /api/maps/{id}` and audio from its server audio endpoint. `session-client.ts` opens `/ws/session`, sends `start`, waits for a matching `ready`, then sends `play` when Web Audio starts and `clock` every second. Saber actions, scores, note removals and final results come from server messages. Community maps remain local previews; the server cannot play an arbitrary browser-loaded chart.

Pause sends the current clock and freezes the existing run. Resume preserves score and brain state. A seek sends a new `play` at the selected audio time, resetting score, events and recurrent state there. It does not restore the brain's earlier history. Changing maps/brains tears down the old socket; stale callbacks cannot update its replacement. Errors stop audio and allow a fresh connection on play.

The scene retains mapper speed/spawn distance; the server currently uses the fixed speed in `configs/saber.yaml`. Timing agrees at the hit plane, but spacing may differ. Chart cut directions retain all nine values while the sim adapts to four. Neuron activity still requires the binary stream contract.

## Validation

`npm run test:unit` covers tempo boundaries, v2/v3/v4 notes, metadata indices, ZIP roots/BOMs/case, invalid inputs, cancellation, seeks, playback rate, end-of-song behavior, and asynchronous play races, alongside the authentic brain tests.

`npm run build`, `npm run lint`, `npm run format:check`, and `npm test` validate the production UI on desktop and mobile Chromium. Browser tests use an original silent map fixture with no third-party service dependency. A separate real community search/download check verified CORS, ZIP parsing, Ogg decoding, and playback.

Map format references: [Info](https://bsmg.wiki/mapping/map-format/info.html), [beatmaps](https://bsmg.wiki/mapping/map-format/beatmap.html), and [audio timing](https://bsmg.wiki/mapping/map-format/audio.html). See [AudioBufferSourceNode](https://developer.mozilla.org/en-US/docs/Web/API/AudioBufferSourceNode) for playback and [Drei controls](https://drei.docs.pmnd.rs/controls/introduction) for camera navigation.
