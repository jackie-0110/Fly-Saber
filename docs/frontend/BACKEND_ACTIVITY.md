# Pending integration: neuron firing activity

The interactive brain currently loads real neuron positions, identities, and groups from the committed JSON exports. Point colors identify cell types. There is no activity animation or live backend connection.

**Person 3 will need to receive firing/activity data from the backend later.** Before driving point brightness:

- Match each activity value to the original neuron array index in `neurons.json`. Filtering only changes displayed points; it must never change the underlying index mapping.
- Verify the asset identity/version, neuron count, and cell-type order against the session. `schema_version: 1` is a format version, not a unique identity for the subgraph. Agree on an asset identifier or hash before streaming.
- Use the binary header, session slot, and simulation step defined in `docs/PROTOCOL.md`. Check the actual backend payload and quantization/scale semantics before implementing the decoder; do not infer hertz from raw rates.
- Reject incompatible lengths, stale sessions, and out-of-order updates. Keep activity buffers outside React state, preserve the camera and selection during updates, and keep original backend indices when applying group/type filters.
- Leave anatomy visible when disconnected, mark activity as unavailable or stale, and never substitute random firing patterns.

Relevant files: `frontend/src/features/brain/data.ts` validates and preserves the exported index mapping; `frontend/src/features/brain/brain-scene.tsx` renders the cloud; `frontend/src/features/brain/components/brain-preview.tsx` displays the pending-backend note and selection metadata. Live activity and lesion/opto interventions remain separate future integrations.

## Additional asset needed for neuron shapes

The current export has one location per neuron, not its branching morphology. Inspection found 19,454 neuron locations at source z below 400 micrometers and eight distant `ascending_neuron` somas at z above 500 micrometers. All eight have `position_source: soma`; they are supplied coordinates, not points introduced by the renderer. The 1,310 descending neurons also have only a single location each, so their long axons are not visible.

A continuous ventral nerve cord view or long neuron paths needs an additional authentic asset: selected neuron skeletons (for example SWC), sampled morphology points, or an anatomical mesh with source/version and coordinate units. Preserve body-ID/index mapping and use the same coordinate transform. Do not connect isolated soma points with invented axons or fill the missing cord with synthetic data.
